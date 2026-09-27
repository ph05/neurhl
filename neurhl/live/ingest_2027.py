"""NeurHL LIVE: nightly ingest of completed 2026-27 games into the NeurHL-G tables.

Turns every completed regular-season game of the live season into the same
per-season tables the historical pipeline built for 2008-2026, so that
neurhl/data/build_g_state.py (which discovers seasons from games_ctx_*.parquet)
and neurhl/sim/g_live.py see 2026-27 games as history:

  events_S, shifts_S, stints_S            per-game rows (tensorize_events, build_shifts, build_stints)
  games_ctx_S, player_games_S             tensorize_games, with labels from the play-by-play itself
                                          and rest/travel from sim/schedule_context (build_travel rules)
  mp_shots_S                              parse_mp_shots (MoneyPuck in-season zip)
  xg_shots_S                              walk-forward xG: GBM FROZEN before the season (fit on
                                          seasons <= S-2, isotonic seed on S-1) + the A9 sequential
                                          in-season calibrator (train_xg.seq_calibrate)
  goalie_games_S                          build_goalie_games (walk-forward carry replayed from 2008)
  stream_S                                build_stream (per-event xG)
  pgx_S, tgx_S                            build_xg_games
  usage_S, onice_rates_S, absences_S      build_usage, build_onice_rates, build_absences
  rr_S                                    build_right_rail.parse (only when right-rail files exist;
                                          tgx is NOT rewritten with official PP counts, see README)

Every builder function is imported and called, never re-derived, except the
two per-season `main()` bodies that mix in sources a live season does not have
(tensorize_games: games.csv / travel_games.csv / the EDA game-summary cache;
build_onice_rates: its loop has no function) which are mirrored here line for
line and checked by the parity test (neurhl/live/test_ingest_parity.py).

Incremental and idempotent: raw files are fetched only for games not on disk,
per-game work (events, rosters, shifts, stints) is done only for new games and
merged into the season files, and the cheap season-level tables are rebuilt
from those files so walk-forward quantities are always recomputed in order.
A game is ingested only when its play-by-play is final, its shift chart is
complete and MoneyPuck has its shots; missing sources defer the game for up to
--max-defer-days after its date, after which it is ingested degraded (logged,
and re-ingested automatically once the source appears).

Writes only files for season S. In the real tensors directory it refuses to
write any season <= 2026 and any symlink.

Usage (repo root):
  uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow \
     --with requests --with scikit-learn==1.9.1 --with scipy python neurhl/live/ingest_2027.py
  options: --season S  --tensors DIR  --raw DIR  --results CSV  --no-fetch  --through D
           --include-playoffs  --force  --no-right-rail  --max-defer-days N  --workers N
           --freeze-xg-only
Exit status: 0 success (including "no completed 2027 games" / "no new games"), 1 failure.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

NRL = Path(__file__).resolve().parents[1]
if str(NRL) not in sys.path:
    sys.path.insert(0, str(NRL))
import common  # noqa: E402
import manifest as MAN  # noqa: E402
import data.tensorize_events as TE  # noqa: E402
import data.tensorize_games as TG  # noqa: E402
import data.build_shifts as BSH  # noqa: E402
import data.build_stints as BST  # noqa: E402
import data.build_usage as BU  # noqa: E402
import data.build_absences as BA  # noqa: E402
import data.build_goalie_games as GG  # noqa: E402
import data.build_stream as BSTR  # noqa: E402
import data.build_xg_games as XGG  # noqa: E402
import data.build_right_rail as BRR  # noqa: E402
import data.parse_mp_shots as PMS  # noqa: E402
import models.xg as XG  # noqa: E402

REAL_TENSORS = common.TENSORS.resolve()
LIVE_SEASON = 2027
FIRST_HIST = 2008
UA = common.UA
PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
SHIFT_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}"
RR_URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/right-rail"
MP_URLS = ("https://moneypuck.com/moneypuck/playerData/shots/shots_{y}.zip",
           "https://peter-tanner.com/moneypuck/downloads/shots_{y}.zip")
FINAL_STATES = ("OFF", "FINAL")
SHIFT_MIN_RATIO = 5.4     # per side: summed shift seconds / game seconds (2026: min 5.68)
MP_MIN_RATIO = 0.9        # MoneyPuck shots / pbp unblocked attempts (2026: min 0.968)
UNBLOCKED = ("shot-on-goal", "missed-shot", "goal")
MODULES = (TE, TG, BSH, BST, BU, BA, GG, BSTR, XGG, BRR, PMS, XG)
LAST_TABLES = ("games_ctx", "player_games", "usage", "onice_rates", "goalie_games",
               "pgx", "tgx")


def log(msg: str) -> None:
    print(f"[ingest {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, a):
        self.S = a.season
        self.out = Path(a.tensors).resolve()
        self.raw = Path(a.raw).resolve()
        self.real = self.out == REAL_TENSORS
        if self.real and self.S <= 2026:
            raise SystemExit(f"refusing to build season {self.S} into the real tensors dir")
        self.cache = self.out / f"_ingest_{self.S}"      # created on first write
        self.maps = json.loads((self.out / "maps.json").read_text())
        self.xg_era_through = a.xg_era_through
        self.allow_late_freeze = a.allow_late_freeze
        self.xg_ckpt = Path(a.xg_ckpt) if a.xg_ckpt else (
            common.CKPT / f"xg_live_v{self.S}.pkl" if self.real
            else self.out.parent / f"xg_live_v{self.S}.pkl")
        for m in MODULES:            # every builder reads/writes through this dir
            if hasattr(m, "TENSORS"):
                m.TENSORS = self.out
            if hasattr(m, "RAW"):
                m.RAW = self.raw

    def path(self, fam: str, ext: str = "parquet") -> Path:
        return self.out / f"{fam}_{self.S}.{ext}"

    def guard(self, p: Path) -> None:
        p = Path(p)
        if p.is_symlink():
            raise RuntimeError(f"refusing to write through symlink {p}")
        if p.resolve().parent != self.out and self.cache not in p.resolve().parents:
            raise RuntimeError(f"refusing to write outside {self.out}: {p}")
        if self.real and f"_{self.S}" not in p.name:
            raise RuntimeError(f"refusing to write non-{self.S} file {p}")

    def write(self, df: pd.DataFrame, fam: str, sources=(), cfg=None) -> Path:
        p = self.path(fam)
        self.guard(p)
        tmp = p.with_name(p.name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
        if cfg is not None:
            self.guard(MAN.manifest_path(p))
            MAN.write_manifest(p, sources, cfg, {"n_rows": len(df), "built_by_live": True})
        return p


# ------------------------------------------------------------------ helpers
def mmss(s: str) -> int:
    m, ss = s.split(":")
    return int(m) * 60 + int(ss)


def read_gz(p: Path):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def write_gz(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, p)


def shift_ratio(recs, game_sec: int) -> float:
    """Min over the two teams of summed shift seconds / game seconds."""
    tot: dict = {}
    for r in recs:
        if r.get("typeCode") != 517 or not r.get("startTime") or not r.get("endTime"):
            continue
        if r["period"] >= 5:
            continue
        d = mmss(r["endTime"]) - mmss(r["startTime"])
        if d > 0:
            tot[r["teamId"]] = tot.get(r["teamId"], 0) + d
    if len(tot) < 2 or game_sec <= 0:
        return 0.0
    return min(tot.values()) / game_sec


def pbp_facts(g: dict) -> dict:
    """Game length (periods <= 4) and unblocked-attempt count from raw pbp."""
    plays = [p for p in g.get("plays", [])
             if p.get("periodDescriptor", {}).get("number", 0) <= 4]
    last = max(((p["periodDescriptor"]["number"] - 1) * 1200
                + mmss(p.get("timeInPeriod", "0:00")) for p in plays), default=0)
    n = sum(1 for p in plays if p.get("typeDescKey") in UNBLOCKED)
    return {"game_sec": last, "n_unblocked": n}


def http_json(url: str, tries: int = 4):
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"fetch failed: {url}")


# --------------------------------------------------------- game discovery
def completed_games(ctx: Ctx, a) -> pd.DataFrame:
    """game_id, date, home, away, game_type of completed games of season S."""
    through = a.through or dt.date.today().isoformat()
    if a.results:
        p = Path(a.results)
        if not p.exists():
            return pd.DataFrame(columns=["game_id", "date", "home", "away", "game_type"])
        r = pd.read_csv(p)
        if not len(r):
            return pd.DataFrame(columns=["game_id", "date", "home", "away", "game_type"])
        r = r[(r.game_id // 1_000_000 == ctx.S - 1) & (r.date.astype(str) <= through)]
        r = r.assign(game_type=(r.game_id // 10_000 % 100).astype(int))
        return r[["game_id", "date", "home", "away", "game_type"]].drop_duplicates("game_id")
    # no results file: every game with raw pbp on disk (sandbox / parity mode)
    rows = []
    srcs = ["pbp"] + (["pbp_po"] if a.include_playoffs else [])
    for src in srcs:
        for p in sorted((ctx.raw / src / str(ctx.S)).glob("*.json.gz")):
            g = read_gz(p)
            if g.get("gameDate", "") > through:
                continue
            rows.append({"game_id": g["id"], "date": g.get("gameDate", ""),
                         "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                         "game_type": g["gameType"]})
    return pd.DataFrame(rows, columns=["game_id", "date", "home", "away", "game_type"])


def pbp_path(ctx: Ctx, gid: int) -> Path:
    src = "pbp" if (gid // 10_000) % 100 == 2 else "pbp_po"
    return ctx.raw / src / str(ctx.S) / f"{gid}.json.gz"


def shift_path(ctx: Ctx, gid: int) -> Path:
    return ctx.raw / "shifts" / str(ctx.S) / f"{gid}.json.gz"


# ------------------------------------------------------------------ fetch
def fetch_raw(ctx: Ctx, games: pd.DataFrame, a, state: dict) -> dict:
    """Fetch missing pbp/shift/right-rail files. Returns {gid: status} where
    status is 'ready', 'degraded:<what>' or 'pending:<what>'. A shift chart is
    written only once it is complete, or when the game is overdue (then it is
    re-fetched on later runs while the game stays degraded)."""
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    redo = {int(g) for g, v in state.get("degraded", {}).items() if "shifts" in v}
    status = {}
    for r in games.sort_values("game_id").itertuples():
        gid = int(r.game_id)
        overdue = (today - dt.date.fromisoformat(str(r.date)[:10])).days > a.max_defer_days
        pp = pbp_path(ctx, gid)
        if not pp.exists() and a.fetch:
            g = http_json(PBP_URL.format(gid=gid))
            if g and g.get("plays") and g.get("gameState") in FINAL_STATES:
                write_gz(pp, g)
                time.sleep(0.3)
        if not pp.exists():
            status[gid] = "pending:pbp"
            continue
        g = read_gz(pp)
        if g.get("gameState") not in FINAL_STATES:
            status[gid] = "pending:pbp_not_final"
            continue
        facts = pbp_facts(g)
        state.setdefault("facts", {})[str(gid)] = facts
        sp = shift_path(ctx, gid)
        miss = []
        if not sp.exists() or gid in redo:
            recs = read_gz(sp) if sp.exists() else []
            old_ratio = shift_ratio(recs, facts["game_sec"])
            if a.fetch:
                js = http_json(SHIFT_URL.format(gid=gid))
                new = (js or {}).get("data", []) or []
                if shift_ratio(new, facts["game_sec"]) > old_ratio:
                    recs = new
                time.sleep(0.3)
            ok = shift_ratio(recs, facts["game_sec"]) >= SHIFT_MIN_RATIO
            if recs and (ok or overdue) and (not sp.exists() or
                                             shift_ratio(recs, facts["game_sec"]) > old_ratio):
                write_gz(sp, recs)
            if not ok:
                miss.append("shifts")
        if a.fetch and a.right_rail and r.game_type == 2:
            rp = ctx.raw / "right_rail" / str(ctx.S) / f"{gid}.json.gz"
            if not rp.exists():
                js = http_json(RR_URL.format(gid=gid))
                if js:
                    write_gz(rp, js)
                time.sleep(0.3)
        if miss and not overdue:
            status[gid] = "pending:" + ",".join(miss)
        elif miss:
            status[gid] = "degraded:" + ",".join(miss)
        else:
            status[gid] = "ready"
    return status


def fetch_mp_zip(ctx: Ctx, a) -> Path:
    """MoneyPuck keys seasons by start year: season_end S is shots_{S-1}.zip."""
    y = ctx.S - 1
    dest = ctx.raw / "mp_shots" / f"shots_{y}.zip"
    if not a.fetch or ctx.S <= 2026:        # history zips are read-only
        return dest
    import requests
    for url in MP_URLS:
        try:
            with requests.get(url.format(y=y), headers=UA, timeout=180, stream=True) as r:
                if r.status_code != 200:
                    log(f"MoneyPuck HTTP {r.status_code} at {url.format(y=y)}")
                    continue
                tmp = dest.with_name(dest.name + ".part")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                if tmp.stat().st_size < 10_000:
                    tmp.unlink()
                    continue
                os.replace(tmp, dest)
                return dest
        except requests.RequestException as e:
            log(f"MoneyPuck fetch error {e!r}")
    log("WARNING: MoneyPuck zip not refreshed; using the copy on disk (if any)")
    return dest


def load_mp(ctx: Ctx, zp: Path) -> pd.DataFrame:
    if not zp.exists():
        return pd.DataFrame()
    return PMS.parse_season(zp)


# ------------------------------------------------------- per-game (pass 1)
_H0 = TE.COLS.index("h_on0")


def _winit(maps, eids):
    TE._M = maps
    BST._EIDS = eids


def game_job(args):
    """One game -> events rows, roster rows, shift rows, summary row.
    Calls the builders' own per-game functions."""
    pp, sp = args
    g = read_gz(pp)
    gid = g["id"]
    ev = TE.tensorize_game((pp, sp))
    excluded = ev is None                    # TE.EXCLUDE_GAMES: no events and no
    ev = ev or []                            # shifts, but the game, its roster and
    roster = TG.roster_toi_one((pp, sp))     # its labels stay (as tensorize_games)
    home_pids = set()
    for row in ev:
        home_pids.update(int(v) for v in row[_H0:_H0 + 7] if v)
    shifts = None
    if sp.exists() and not excluded:
        BSH._GOALIES = {r["playerId"] for r in g.get("rosterSpots", [])
                        if r.get("positionCode") == "G"}
        shifts = BSH._json_game((gid, sp, home_pids))
    oc = g.get("gameOutcome", {}) or {}
    summ = {"game_id": gid, "game_type": g["gameType"], "date": g.get("gameDate", ""),
            "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
            "home_score": g["homeTeam"].get("score"), "away_score": g["awayTeam"].get("score"),
            "last_period_type": oc.get("lastPeriodType", "")}
    return {"gid": gid, "ev": ev, "roster": roster, "shifts": shifts, "summ": summ}


def stint_job(args):
    return BST._game(args)


def run_pool(fn, jobs, workers, maps, eids):
    if not jobs:
        return []
    if len(jobs) < 40 or workers <= 1:
        _winit(maps, eids)
        return [fn(j) for j in jobs]
    with Pool(workers, initializer=_winit, initargs=(maps, eids)) as pool:
        return list(pool.imap(fn, jobs, chunksize=8))


# ---------------------------------------------- table casts (as the builders)
EV_DT = [("game_id", "uint32"), ("game_type", "uint8"), ("event_idx", "uint16"),
         ("period", "uint8"), ("t", "uint16"), ("dt", "uint16"), ("event_type", "uint8"),
         ("zone", "uint8"), ("shot_type", "uint8"), ("strength", "uint16"),
         ("home_event", "int8"), ("x", "int16"), ("y", "int16"), ("dir", "int8"),
         ("has_coord", "uint8"), ("xn", "int16"), ("yn", "int16"), ("score_h", "uint8"),
         ("score_a", "uint8"), ("p1", "uint32"), ("p2", "uint32"), ("p3", "uint32"),
         ("goalie", "uint32"), ("venue", "uint8")]


def events_frame(rows) -> pd.DataFrame:
    """tensorize_events.main: rows -> typed frame."""
    df = pd.DataFrame(rows, columns=TE.COLS)
    for c, t in EV_DT:
        df[c] = df[c].astype(t)
    for i in range(7):
        df[f"h_on{i}"] = df[f"h_on{i}"].astype("uint32")
        df[f"a_on{i}"] = df[f"a_on{i}"].astype("uint32")
    return df


def shifts_frame(arrs) -> pd.DataFrame:
    """build_shifts.build: stacked arrays -> typed frame."""
    if not arrs:
        return pd.DataFrame({c: pd.Series(dtype="int64") for c in BSH.COLS}).astype(SH_DT)
    df = pd.DataFrame(np.vstack(arrs), columns=BSH.COLS)
    return df.astype(SH_DT)


SH_DT = {"game_id": "uint32", "player_id": "uint32", "is_home": "bool",
         "is_goalie": "bool", "period": "uint8", "t0": "uint16", "t1": "uint16",
         "src": "uint8"}
ROSTER_COLS = ["game_id", "game_type", "player_id", "is_home", "pos_group", "toi_sec",
               "goalie_start"]


def merge_season(old: pd.DataFrame | None, new: pd.DataFrame, gids: set) -> pd.DataFrame:
    """Replace the rows of `gids` in `old` by `new`; keep game_id order, in-game order."""
    parts = []
    if old is not None and len(old):
        parts.append(old[~old.game_id.isin(list(gids))])
    if len(new):
        parts.append(new)
    if not parts:
        return new
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("game_id", kind="stable").reset_index(drop=True)


def read_or_none(p: Path):
    return pd.read_parquet(p) if p.exists() else None


# --------------------------------------------------- per-game stage (driver)
def ingest_games(ctx: Ctx, gids: list, a) -> dict:
    """Per-game work for `gids`, merged into events/shifts/stints and the roster
    and summary caches. Returns counts."""
    eids = BST.event_ids(ctx.maps)
    jobs = [(pbp_path(ctx, g), shift_path(ctx, g)) for g in sorted(gids)]
    res = [r for r in run_pool(game_job, jobs, a.workers, ctx.maps, eids) if r]
    done = {r["gid"] for r in res}
    ev_rows, ro_rows, sh_arrs, summ = [], [], [], []
    for r in res:
        ev_rows.extend(r["ev"])
        ro_rows.extend(r["roster"])
        if r["shifts"] is not None and len(r["shifts"]):
            sh_arrs.append(r["shifts"])
        summ.append(r["summ"])
    ev_new = events_frame(ev_rows)
    sh_new = shifts_frame(sh_arrs)
    ro_new = pd.DataFrame(ro_rows, columns=ROSTER_COLS)
    su_new = pd.DataFrame(summ)

    # stints for the new games (build_stints.main's job construction)
    ev_s = ev_new[["game_id", "game_type", "t", "event_type", "home_event", "strength",
                   "score_h", "score_a", "zone"]]
    ev_s = ev_s[ev_s.t.notna()]
    sh_g = dict(list(sh_new.groupby("game_id", sort=False)))
    sj = []
    for gid, e in ev_s.groupby("game_id", sort=True):
        s = sh_g.get(gid)
        if s is None or not len(s):
            continue
        sj.append((int(gid), ctx.S, int(e.game_type.iloc[0]),
                   (s.player_id.to_numpy(np.int64), s.is_home.to_numpy(bool),
                    s.is_goalie.to_numpy(bool), s.t0.to_numpy(np.int32),
                    s.t1.to_numpy(np.int32)),
                   (e.t.to_numpy(np.int32), e.event_type.to_numpy(np.int16),
                    e.home_event.to_numpy(np.int8), e.strength.to_numpy(np.int32),
                    e.score_h.to_numpy(np.int16), e.score_a.to_numpy(np.int16),
                    e.zone.to_numpy(np.int16))))
    st_rows = []
    for r in run_pool(stint_job, sj, a.workers, ctx.maps, eids):
        if r:
            st_rows.extend(r)
    st_new = pd.DataFrame(st_rows, columns=BST.COLS)
    st_new["src"] = BSH.SRC_JSON

    gset = set(gids)
    ev = merge_season(read_or_none(ctx.path("events")), ev_new, gset)
    sh = merge_season(read_or_none(ctx.path("shifts")), sh_new, gset)
    st = merge_season(read_or_none(ctx.path("stints")), st_new, gset)
    ro = merge_season(read_or_none(ctx.cache / "roster.parquet"), ro_new, gset)
    su = merge_season(read_or_none(ctx.cache / "games.parquet"), su_new, gset)
    ctx.write(ev, "events")
    ctx.write(sh, "shifts", [ctx.path("events")], BSH.CFG)
    ctx.write(st, "stints", [ctx.path("shifts"), ctx.path("events")], BST.CFG)
    for df, name in ((ro, "roster.parquet"), (su, "games.parquet")):
        p = ctx.cache / name
        ctx.guard(p)
        df.to_parquet(p, index=False)
    return {"games": len(done), "events": len(ev_new), "shifts": len(sh_new),
            "stints": len(st_new)}


# ------------------------------------------------ season stage: games_ctx
def era_row(S: int) -> dict:
    """tensorize_games.era_table row for S; for a season not yet in games.csv,
    the same prior-season construction (as sim/g_live.era_2027)."""
    era = TG.era_table()
    if S in era.index:
        return {c: float(era.loc[S, c]) for c in TG.ERA_COLS}
    games = pd.read_csv(common.PROC / "games.csv")
    ts = pd.read_csv(common.PROC / "team_seasons.csv")
    r = games[(games.game_type == "R") & (games.season_end == S - 1)]
    return {"prior_gpg": float((r.home_g + r.away_g).mean()),
            "prior_ot_share": float(r.went_ot.astype(bool).mean()),
            "prior_so_share": float(r.went_so.astype(bool).mean()),
            "prior_margin_abs": float(r.margin.abs().mean()),
            "prior_parity": float(ts[ts.season_end == S - 1].pts_pct.std()),
            "flag_3v3": float(S >= 2016), "flag_covid": float(S in (2020, 2021)),
            "season_scaled": (S - 2006) / 20.0}


def build_games_tables(ctx: Ctx, known: pd.DataFrame):
    """games_ctx_S and player_games_S, mirroring tensorize_games.main for one
    season. Differences by necessity (no games.csv / travel_games.csv rows exist
    for a live season): labels (goals, OT/SO) come from the final play-by-play;
    rest/travel from sim/schedule_context.build over the season's completed
    regular-season games in date order (build_travel's rules). Playoff games get
    NaN travel, as travel_games.csv has regular-season rows only."""
    from sim.schedule_context import build as sched_ctx
    maps = ctx.maps
    su = pd.read_parquet(ctx.cache / "games.parquet")
    ro = pd.read_parquet(ctx.cache / "roster.parquet")
    ev = pd.read_parquet(ctx.path("events"), columns=["game_id", "event_type", "strength",
                                                      "home_event", "p1", "p2", "p3"])
    # --- player-game table (tensorize_games.main, verbatim logic)
    pg = ro.copy()
    et_inv = {v: k for k, v in maps["event_type"].items()}
    ev["etn"] = ev.event_type.map(et_inv)

    def count(mask, col):
        c = ev[mask].groupby(["game_id", col]).size()
        c.index.names = ["game_id", "player_id"]
        return c

    stats = pd.DataFrame({
        "goals": count(ev.etn == TG.GOAL_T, "p1"),
        "a1": count(ev.etn == TG.GOAL_T, "p2"),
        "a2": count(ev.etn == TG.GOAL_T, "p3"),
        "sog": count(ev.etn.isin([TG.SOG_T, TG.GOAL_T]), "p1"),
        "att": count(ev.etn.isin([TG.SOG_T, TG.GOAL_T, TG.MISS_T, TG.BLOCK_T]), "p1"),
        "blocks": count(ev.etn == TG.BLOCK_T, "p2"),
        "pen": count(ev.etn == TG.PEN_T, "p1"),
        "fo_w": count(ev.etn == TG.FO_T, "p1"),
        "fo_l": count(ev.etn == TG.FO_T, "p2"),
    }).fillna(0).astype(int).reset_index()
    stats = stats[stats.player_id > 0]
    stats["assists"] = stats.a1 + stats.a2
    pg = pg.merge(stats.drop(columns=["a1", "a2"]),
                  on=["game_id", "player_id"], how="left").fillna(0)
    for c in ("goals", "sog", "att", "blocks", "pen", "fo_w", "fo_l", "assists"):
        pg[c] = pg[c].astype("int16")

    # --- game context table
    sat = (ev[(ev.strength == 1551)
              & ev.etn.isin([TG.SOG_T, TG.GOAL_T, TG.MISS_T, TG.BLOCK_T])]
           .groupby(["game_id", "home_event"]).size().unstack(fill_value=0))
    m = su.copy()
    m["home_g"] = m.home_score.astype("int64")
    m["away_g"] = m.away_score.astype("int64")
    extra = m.last_period_type.isin(["OT", "SO"])
    home_won = m.home_g > m.away_g
    m["outcome4"] = np.select(
        [home_won & ~extra, ~home_won & ~extra, home_won & extra],
        [0, 1, 2], default=3).astype("uint8")
    # regular-season schedule of all completed games (ingested or not), for
    # days_in and rest/travel; teams keyed by franchise code like games.csv
    reg = pd.concat([m.loc[m.game_type == 2, ["game_id", "date", "home", "away"]],
                     known.loc[known.game_type == 2, ["game_id", "date", "home", "away"]]],
                    ignore_index=True).drop_duplicates("game_id")
    reg["home"] = reg.home.replace(TG.REMAP)
    reg["away"] = reg.away.replace(TG.REMAP)
    first_date = pd.to_datetime(reg.date).min()
    m["days_in"] = (pd.to_datetime(m.date) - first_date).dt.days
    m["home_idx"] = m.home.map(maps["team"]).fillna(0).astype("uint8")
    m["away_idx"] = m.away.map(maps["team"]).fillna(0).astype("uint8")
    # (int64 unless a game has no events, then float64 -- as tensorize_games)
    m["sat5_h"] = m.game_id.map(sat.get(1, pd.Series(dtype=int))).fillna(0)
    m["sat5_a"] = m.game_id.map(sat.get(0, pd.Series(dtype=int))).fillna(0)
    tv = sched_ctx(reg, ctx.S).set_index("game_id")
    for c in ("home_rest", "away_rest", "home_km3d", "away_km3d", "home_dtz", "away_dtz"):
        m[c] = m.game_id.map(tv[c]).astype("float64")
        m.loc[m.game_type != 2, c] = np.nan
    er = era_row(ctx.S)
    for c in TG.ERA_COLS:
        m[c] = er[c]
    m["game_id"] = m.game_id.astype("int64")
    m["game_type"] = m.game_type.astype("int64")
    keep = (["game_id", "game_type", "date", "home_idx", "away_idx",
             "home_g", "away_g", "outcome4", "sat5_h", "sat5_a", "days_in",
             "home_rest", "away_rest", "home_km3d", "away_km3d",
             "home_dtz", "away_dtz"] + TG.ERA_COLS)
    gc = m[keep].sort_values("game_id", kind="stable").reset_index(drop=True)
    ctx.write(pg, "player_games")
    ctx.write(gc, "games_ctx")
    return gc, pg


# ------------------------------------------------------ season stage: xG
def era_features(S: int) -> pd.DataFrame:
    """train_xg.main's covariate table, over seasons 2008..S."""
    rng = range(FIRST_HIST, S + 1)
    era = XG.era_covariates(rng)
    era = era.join(XG.rink_offsets(rng), how="outer")
    era = era.join(XG.rink_disagreement(rng), how="outer")
    era = era.join(XG.regime_covariates(rng), how="outer")
    return era


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_xg(ctx: Ctx, era: pd.DataFrame) -> dict:
    """train_xg.fit_vantage(V=S) up to the calibrator: GBM on seasons < S-1,
    isotonic seed = its predictions on S-1. Saved once; never refit on S."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    import sklearn
    import train.train_xg as TX
    S = ctx.S
    tr = XG.load_shots(range(FIRST_HIST, S))
    Xtr, ytr = XG.build_features(tr, era), tr[XG.TARGET].to_numpy()
    m_fit = tr.season_end.to_numpy() != S - 1
    cal = ~m_fit
    if not m_fit.sum() or not cal.sum():
        raise RuntimeError("xG freeze needs seasons < S-1 and season S-1")
    dead = [c for c in Xtr.columns if c not in XG.CATS
            and Xtr.loc[m_fit, c].dropna().nunique() < 2]
    cats = {c: list(Xtr[c].cat.categories) for c in XG.CATS}
    Xtr = Xtr.drop(columns=dead)
    cat_mask = [c in XG.CATS for c in Xtr.columns]
    t0 = time.time()
    base = HistGradientBoostingClassifier(categorical_features=cat_mask, **TX.GBM)
    base.fit(Xtr[m_fit], ytr[m_fit])
    ck = {"vantage": S, "model": base, "columns": list(Xtr.columns), "cats": cats,
          "dead": dead, "seed_p": base.predict_proba(Xtr[cal])[:, 1],
          "seed_y": ytr[cal], "seed_gid": tr.game_id.to_numpy()[cal],
          "fit_seasons": sorted(set(tr.season_end[m_fit].tolist())),
          "seed_season": S - 1, "n_fit": int(m_fit.sum()), "gbm": TX.GBM,
          "seq_w": TX.SEQ_W, "seq_k": TX.SEQ_K, "sklearn": sklearn.__version__,
          "built_at": dt.datetime.now().isoformat(timespec="seconds")}
    ctx.xg_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ctx.xg_ckpt.exists():
        raise RuntimeError(f"{ctx.xg_ckpt} exists; the frozen xG model is never refit")
    with open(ctx.xg_ckpt, "wb") as f:
        pickle.dump(ck, f)
    Path(str(ctx.xg_ckpt) + ".sha256").write_text(
        f"{sha256(ctx.xg_ckpt)}  {ctx.xg_ckpt.name}\n")
    log(f"froze xG GBM for V={S}: fit on {ck['fit_seasons'][0]}-{ck['fit_seasons'][-1]} "
        f"({ck['n_fit']:,} shots, {time.time() - t0:.0f}s), seed {S - 1} "
        f"({int(cal.sum()):,} shots) -> {ctx.xg_ckpt}")
    return ck


def load_xg(ctx: Ctx) -> dict:
    want = Path(str(ctx.xg_ckpt) + ".sha256")
    if want.exists() and want.read_text().split()[0] != sha256(ctx.xg_ckpt):
        raise RuntimeError(f"{ctx.xg_ckpt} does not match its sha256 sidecar")
    with open(ctx.xg_ckpt, "rb") as f:
        ck = pickle.load(f)
    if ck["vantage"] != ctx.S:
        raise RuntimeError(f"{ctx.xg_ckpt} is for V={ck['vantage']}, not {ctx.S}")
    import sklearn
    if sklearn.__version__ != ck["sklearn"]:
        raise RuntimeError(f"{ctx.xg_ckpt} was frozen with scikit-learn {ck['sklearn']}, "
                           f"running {sklearn.__version__}; pin --with "
                           f"scikit-learn=={ck['sklearn']}")
    return ck


def freeze_era(ctx: Ctx) -> pd.DataFrame:
    """The covariate table the frozen GBM is fit with. It must end where
    train_xg.main's table ended (2026): models/xg.era_covariates and
    regime_covariates sort games without a games_ctx date (2007020003, a 2008
    TRAINING game) last, so that game's covariates are the trailing stats at the
    end of the table. Hence freeze BEFORE the season (mp_shots_S absent), which
    makes the table exactly train_xg's; a late freeze is refused."""
    if ctx.path("mp_shots").exists() and not ctx.allow_late_freeze:
        raise RuntimeError(
            f"refusing to freeze the V={ctx.S} xG GBM after {ctx.S} shots exist "
            f"({ctx.path('mp_shots').name}); the checkpoint {ctx.xg_ckpt} should have "
            "been frozen pre-season (--freeze-xg-only). --allow-late-freeze overrides.")
    return era_features(max(ctx.S, ctx.xg_era_through))


def build_xg(ctx: Ctx, era: pd.DataFrame) -> pd.DataFrame:
    """Score season-S shots with the frozen GBM and the A9 sequential
    calibrator exactly as train_xg.fit_vantage does (seq_calibrate imported)."""
    import train.train_xg as TX
    TX.TENSORS = ctx.out
    ck = load_xg(ctx) if ctx.xg_ckpt.exists() else freeze_xg(ctx, freeze_era(ctx))
    te = XG.load_shots([ctx.S])
    Xte = XG.build_features(te, era)
    for c in XG.CATS:
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=ck["cats"][c])
    Xte = Xte.drop(columns=ck["dead"])[ck["columns"]]
    yte = te[XG.TARGET].to_numpy()
    p_raw = ck["model"].predict_proba(Xte)[:, 1]
    p = TX.seq_calibrate(ck["seed_p"], ck["seed_y"], ck["seed_gid"], p_raw, yte,
                         te.game_id.to_numpy(), ctx.S, w=ck["seq_w"], k=ck["seq_k"])
    return pd.DataFrame({
        "game_id": te.game_id.to_numpy(), "is_home": te.isHomeTeam.to_numpy(),
        "goal": yte, "xg": p,
        "sk_h": te.homeSkatersOnIce.to_numpy(), "sk_a": te.awaySkatersOnIce.to_numpy(),
        "en": (te.homeEmptyNet | te.awayEmptyNet).to_numpy(),
        "time": te.time.to_numpy(), "period": te.period.to_numpy(),
        "shooter": te.shooterPlayerId.to_numpy(), "team_code": te.teamCode.to_numpy()})


# ---------------------------------------------------- season stage: others
def build_onice(ctx: Ctx) -> pd.DataFrame:
    """build_onice_rates.main's per-season body."""
    ON_H, ON_A = [f"h_on{i}" for i in range(7)], [f"a_on{i}" for i in range(7)]
    et = ctx.maps["event_type"]
    shot_types = [et[c] for c in ("shot-on-goal", "goal", "missed-shot", "blocked-shot")
                  if c in et]
    ev = pd.read_parquet(ctx.path("events"),
                         columns=["game_id", "game_type", "event_type", "home_event",
                                  "strength", "xn", "yn", "has_coord"] + ON_H + ON_A)
    ev = ev[(ev.game_type == 2) & (ev.home_event >= 0) & ev.event_type.isin(shot_types)]
    dist = np.hypot(89 - ev.xn.abs().clip(upper=100), ev.yn)
    is_close = (dist < 25) & ev.has_coord.astype(bool)
    is_5v5 = ev.strength == 1551
    home_shot = ev.home_event.to_numpy() == 1
    gid = ev.game_id.to_numpy()
    parts = []
    for cols, side_is_home in ((ON_H, True), (ON_A, False)):
        slots = ev[cols].to_numpy()
        forr = home_shot if side_is_home else ~home_shot
        for c in range(slots.shape[1]):
            pid = slots[:, c]
            keep = pid > 0
            if not keep.any():
                continue
            parts.append(pd.DataFrame({
                "game_id": gid[keep], "player_id": pid[keep], "is_home": side_is_home,
                "cf": forr[keep].astype(np.int32), "ca": (~forr[keep]).astype(np.int32),
                "cf5": (forr[keep] & is_5v5.to_numpy()[keep]).astype(np.int32),
                "ca5": (~forr[keep] & is_5v5.to_numpy()[keep]).astype(np.int32),
                "clf": (forr[keep] & is_close.to_numpy()[keep]).astype(np.int32),
                "cla": (~forr[keep] & is_close.to_numpy()[keep]).astype(np.int32)}))
    allr = pd.concat(parts, ignore_index=True)
    return (allr.groupby(["game_id", "player_id", "is_home"], as_index=False)
            [["cf", "ca", "cf5", "ca5", "clf", "cla"]].sum())


def build_goalies(ctx: Ctx) -> pd.DataFrame:
    """build_goalie_games: replay the walk-forward carry over 2008..S-1 from the
    existing tables, then build season S."""
    carry: dict = {}
    for s in range(FIRST_HIST, ctx.S):
        if not (ctx.out / f"player_games_{s}.parquet").exists():
            continue
        GG.build(s, carry)
    return GG.build(ctx.S, carry)


def build_rr(ctx: Ctx, gids: set) -> pd.DataFrame | None:
    files = sorted(p for p in (ctx.raw / "right_rail" / str(ctx.S)).glob("*.json.gz")
                   if int(p.name.split(".")[0]) in gids)
    if not files:
        return None
    return pd.DataFrame([BRR.parse(f) for f in files])


def official_pp(t: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:
    """build_right_rail.main's tgx update (applied only when the previous
    season's tgx carries it, so pp_opps keeps one definition across seasons)."""
    if "pp_opps_stint" not in t:
        t["pp_opps_stint"] = t.pp_opps
    off = pd.concat([rr[["game_id", "pp_opps_h"]].assign(is_home=1).rename(columns={"pp_opps_h": "o"}),
                     rr[["game_id", "pp_opps_a"]].assign(is_home=0).rename(columns={"pp_opps_a": "o"})])
    t = t.drop(columns=["pp_opps"]).merge(off, on=["game_id", "is_home"], how="left")
    t["pp_opps"] = t.o.fillna(t.pp_opps_stint).astype("float32")
    return t.drop(columns=["o"])


def season_tables(ctx: Ctx, known: pd.DataFrame, mp: pd.DataFrame, a) -> dict:
    t0 = time.time()
    gc, pg = build_games_tables(ctx, known)
    ingested = set(gc.game_id.astype(int))
    # MoneyPuck shots of ingested games only (parse_mp_shots allowlist)
    mps = mp[mp.game_id.isin(list(ingested))] if len(mp) else mp
    if len(mps):
        zp = ctx.raw / "mp_shots" / f"shots_{ctx.S - 1}.zip"
        ctx.write(mps.reset_index(drop=True), "mp_shots", [zp], PMS.CFG)
    else:
        log("WARNING: no MoneyPuck shots for any ingested game; xG tables will be empty")
    t1 = time.time()
    era = era_features(ctx.S)
    xg = build_xg(ctx, era) if len(mps) else pd.DataFrame(
        columns=["game_id", "is_home", "goal", "xg", "sk_h", "sk_a", "en", "time",
                 "period", "shooter", "team_code"])
    ctx.write(xg, "xg_shots")
    t2 = time.time()
    src = [ctx.path("player_games"), ctx.path("games_ctx"), ctx.path("events"),
           ctx.path("xg_shots")]
    ctx.write(build_goalies(ctx), "goalie_games", src, GG.CFG)
    t3 = time.time()
    st_src = [ctx.path("events")] + [p for p in (ctx.path("mp_shots"), ctx.path("xg_shots"))
                                     if p.exists()]
    stream = BSTR.build_season(ctx.S, ctx.maps)
    ctx.write(stream, "stream", st_src, BSTR.CFG)
    ctx.write(XGG.build_player(stream, XGG.goalie_ids(ctx.S), ctx.S), "pgx")
    ctx.write(XGG.build_team(stream, ctx.S), "tgx")
    u_src = [ctx.path("stints"), ctx.path("player_games"), ctx.path("games_ctx")]
    ctx.write(BU.build(ctx.S), "usage", u_src, BU.CFG)
    ctx.write(build_onice(ctx), "onice_rates")
    ctx.write(BA.build(ctx.S), "absences", [ctx.path("player_games"),
                                            ctx.path("games_ctx")], BA.CFG)
    rr = build_rr(ctx, ingested) if a.right_rail else None
    if rr is not None:
        ctx.write(rr, "rr")
        prev = ctx.out / f"tgx_{ctx.S - 1}.parquet"
        if prev.exists() and "pp_opps_stint" in pd.read_parquet(prev).columns:
            ctx.write(official_pp(pd.read_parquet(ctx.path("tgx")), rr), "tgx")
            log("tgx: pp_opps set to the official right-rail counts, as in "
                f"tgx_{ctx.S - 1} (stint count kept as pp_opps_stint)")
    t4 = time.time()
    return {"games_s": round(t1 - t0, 1), "xg_s": round(t2 - t1, 1),
            "goalies_s": round(t3 - t2, 1), "rest_s": round(t4 - t3, 1),
            "n_games": len(gc), "n_reg": int((gc.game_type == 2).sum()),
            "xg_shots": len(xg), "rr": 0 if rr is None else len(rr)}


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", type=int, default=LIVE_SEASON)
    ap.add_argument("--tensors", default=str(REAL_TENSORS))
    ap.add_argument("--raw", default=str(common.RAW))
    ap.add_argument("--results", default=str(common.NOUT / "live" / "results_2027.csv"),
                    help="completed games (score_live_2027.py); '' = every raw pbp file")
    ap.add_argument("--no-fetch", dest="fetch", action="store_false")
    ap.add_argument("--through", default=None, help="ingest games dated <= this")
    ap.add_argument("--today", default=None, help="override today's date (deferral clock)")
    ap.add_argument("--include-playoffs", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-ingest every game")
    ap.add_argument("--no-right-rail", dest="right_rail", action="store_false")
    ap.add_argument("--max-defer-days", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--xg-ckpt", default=None)
    ap.add_argument("--xg-era-through", type=int, default=0,
                    help="(tests) end season of the freeze-time covariate table")
    ap.add_argument("--allow-late-freeze", action="store_true",
                    help="(tests) allow freezing the xG GBM after season-S shots exist")
    ap.add_argument("--freeze-xg-only", action="store_true",
                    help="fit and save the frozen xG GBM for the season, then exit")
    a = ap.parse_args(argv)
    t_start = time.time()
    ctx = Ctx(a)
    S = ctx.S
    log(f"season {S}: tensors {ctx.out}{' (REAL)' if ctx.real else ''}, raw {ctx.raw}")

    if a.freeze_xg_only:
        if ctx.xg_ckpt.exists():
            log(f"frozen xG model already exists: {ctx.xg_ckpt} "
                f"(sha256 {sha256(ctx.xg_ckpt)[:16]})")
            return 0
        freeze_xg(ctx, freeze_era(ctx))
        return 0

    games = completed_games(ctx, a)
    if not len(games):
        log(f"no completed {S} games")
        return 0
    ctx.cache.mkdir(parents=True, exist_ok=True)
    state_p = ctx.cache / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    ingested_before = set()
    if (ctx.cache / "games.parquet").exists() and ctx.path("events").exists() \
            and not a.force:
        ingested_before = set(pd.read_parquet(ctx.cache / "games.parquet",
                                              columns=["game_id"]).game_id.astype(int))
    degraded_before = {int(k): v for k, v in state.get("degraded", {}).items()}
    inflight = {int(g) for g in state.get("inflight", [])} if state.get("dirty") else set()
    todo = games[~games.game_id.isin(list(ingested_before - set(degraded_before)
                                          - inflight))]
    log(f"{len(games)} completed games; {len(ingested_before)} already ingested; "
        f"{len(todo)} to check ({len(degraded_before)} degraded re-checks)")

    status = fetch_raw(ctx, todo, a, state)
    # MoneyPuck: one season zip; readiness per game by shot count vs pbp
    zp = fetch_mp_zip(ctx, a)
    zip_sha = sha256(zp) if zp.exists() else ""
    mp = load_mp(ctx, zp)
    mp_n = (mp[mp.period <= 4].groupby("game_id").size() if len(mp)
            else pd.Series(dtype=int))
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    date_of = dict(zip(games.game_id.astype(int), games.date.astype(str)))
    for gid, stt in list(status.items()):
        if stt.startswith("pending"):
            continue
        f = state["facts"][str(gid)]
        if f["n_unblocked"] and mp_n.get(gid, 0) < MP_MIN_RATIO * f["n_unblocked"]:
            overdue = (today - dt.date.fromisoformat(date_of[gid][:10])).days > a.max_defer_days
            if not a.fetch and not a.results:   # sandbox without MoneyPuck: no deferral
                overdue = True
            miss = [] if stt == "ready" else stt.split(":")[1].split(",")
            miss.append("mp")
            status[gid] = ("degraded:" if overdue else "pending:") + ",".join(miss)
    ready = sorted(g for g, s in status.items() if not s.startswith("pending"))
    newly = [g for g in ready if g not in ingested_before or
             degraded_before.get(g) != status[g].partition(":")[2]]
    # per-game rework: new games, plus games ingested without a complete shift
    # chart (their chart may have been re-fetched this run)
    per_game = sorted(set(g for g in ready if g not in ingested_before)
                      | {g for g in ready if "shifts" in degraded_before.get(g, "")})
    pend = {g: s for g, s in status.items() if s.startswith("pending")}
    for g, s in sorted(pend.items()):
        log(f"deferred {g} ({date_of[g]}): {s}")
    for g in ready:
        if status[g].startswith("degraded"):
            log(f"WARNING degraded ingest {g} ({date_of[g]}): {status[g]}")
    # a run that died mid-write leaves state["dirty"]: redo its games and the
    # season stage, whatever else happened since
    dirty = bool(state.get("dirty"))
    if dirty:
        redo = {int(g) for g in state.get("inflight", [])} & set(ready)
        per_game = sorted(set(per_game) | redo)
        log(f"previous run did not finish: redoing {len(redo)} games and the season tables")
    mp_changed = zip_sha != state.get("mp_zip_sha256", "")
    if not newly and not mp_changed and not dirty and not a.force and ctx.path("tgx").exists():
        log(f"no new {S} games (pending {len(pend)}); tables unchanged")
        state["last_run"] = dt.datetime.now().isoformat(timespec="seconds")
        state_p.write_text(json.dumps(state, indent=1))
        return 0

    if not ctx.xg_ckpt.exists():          # freeze before any season-S shot is written
        freeze_xg(ctx, freeze_era(ctx))
    state.update({"dirty": True, "inflight": sorted(set(per_game) |
                                                     set(state.get("inflight", [])))})
    state_p.write_text(json.dumps(state, indent=1))
    t0 = time.time()
    counts = ingest_games(ctx, per_game, a) if per_game else {"games": 0}
    log(f"per-game stage: {counts} in {time.time() - t0:.0f}s")
    info = season_tables(ctx, games, mp, a)
    log(f"season tables: {info}")
    deg = {str(g): status[g].partition(":")[2] for g in ready
           if status[g].startswith("degraded")}
    for g, v in degraded_before.items():      # still degraded and not re-checked
        if g not in status:
            deg[str(g)] = v
    state.update({"degraded": deg, "pending": {str(g): s for g, s in pend.items()},
                  "mp_zip_sha256": zip_sha,
                  "last_run": dt.datetime.now().isoformat(timespec="seconds"),
                  "n_ingested": info["n_games"], "dirty": False, "inflight": []})
    state_p.write_text(json.dumps(state, indent=1))
    log(f"done: {info['n_games']} games in {S} tables ({len(per_game)} new this run, "
        f"{len(pend)} deferred, {len(deg)} degraded) in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
