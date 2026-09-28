"""Score NeurHL-G's live 2026-27 forecasts as the season is played.

Implements the LIVE protocol in PLAN_NeurHL4.md for the files neurhl/live/
forecast.py writes and the publishing clone commits (neurhl/output/live/2027/
<date>/): pregame_<game_id>.csv, one game 45-75 minutes before its start, is
primary; morning.csv, every game that day at about 11:00 ET, is secondary.

  validity    a forecast counts only if the commit that first added its file
              to this repository's history (HEAD, plus origin/main once
              fetched) is dated before the game's scheduled start: the
              committer date, as review_tests_neurhl4 reads it (every parent
              of a merge is followed, so a merge cannot hide that commit),
              against the NHL API's startTimeUTC where it is known (online,
              cached), else the file's start_utc. A file is scored as first
              committed, so later edits or deletions change nothing. A morning
              row for a game that had started before the commit is invalid for
              that game only. Files not yet in the history (not pulled) never
              count. With a working gh CLI the GitHub Actions stamp of each
              counted commit is recorded too (neurhl/live/deadline.py); it is
              never required.
  pairing     a completed regular-season game without a valid pregame
              forecast is MISSED in the primary analysis for every model;
              likewise for morning. No backfill: a morning forecast never
              stands in for a missing pregame one.
  games       NeurHL-G, NeurHL-H, in-season Elo and the frozen 1.0 probability
              (descriptive) on the same games: per-game log loss and Brier;
              paired differences G - Elo, G - H, H - Elo with a week-block
              bootstrap 95% interval (ISO weeks of the ET game date, 9,999
              draws, seed 711). NeurHL-H can be missing: comparisons involving
              it use the games where it exists. Also by lineup source.
  stat sheet  descriptive MAE of team regulation goals (results), SOG and xGF
              (tgx_2027) and skater lines (player_games_2027) where the
              nightly ingest has built those tables; the A1 goal multiplier in
              force (goal_mult).

Interim scorecards are descriptive; inference is made once, after the last
regular-season game (2027-04-10). Reads results_2027.csv (written by
score_live_2027.py) and writes neurhl/output/live/scorecard_g_2027.json; NHL
API start times and stamps are cached in neurhl/data/tensors/_live_g_2027/.

Usage: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
         --with pyarrow --with requests python neurhl/eval/score_live_g_2027.py \
         [--offline] [--as-of YYYY-MM-DD] [--root DIR]
"""
import argparse
import datetime as dt
import importlib.util
import io
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NRL, PROJ, UA  # noqa: E402

ET = ZoneInfo("America/New_York")
LAST, N_GAMES = dt.date(2027, 4, 10), 1344
# paths relative to --root (default: this repository)
FORECASTS = "neurhl/output/live/2027"
RESULTS = "neurhl/output/live/results_2027.csv"
CARD = "neurhl/output/live/scorecard_g_2027.json"
TABLES = "neurhl/data/tensors"
CACHE = "neurhl/data/tensors/_live_g_2027"
SCHEDULES = "data/raw/nhl_sched_*_20262027.json"
SCORE_URL = "https://api-web.nhle.com/v1/score/{d}"
FILE_RE = re.compile(r"^(morning|pregame_\d+)\.csv$")
KINDS = ("pregame", "morning")                          # primary, secondary
MODELS = {"neurhl_g": "p_home_win_neurhl_g", "neurhl_h": "p_home_win_neurhl_h",
          "elo": "p_home_win_elo", "neurhl_1p0": "p_home_win_1p0"}
LABELS = {"neurhl_g": "NeurHL-G", "neurhl_h": "NeurHL-H", "elo": "Elo", "neurhl_1p0": "1.0"}
PAIRS = {"g_minus_elo": ("neurhl_g", "elo"), "g_minus_h": ("neurhl_g", "neurhl_h"),
         "h_minus_elo": ("neurhl_h", "elo")}
SOURCES = ["NHL_API", "DF_CONFIRMED", "DF_PROJECTED", "FALLBACK"]   # most certain first
STATS = ["goals_home", "goals_away", "goals_home_raw", "goals_away_raw", "sog_home",
         "sog_away", "xgf_home", "xgf_away", "goal_mult"]
DRAWS, SEED, MIN_BOOT = 9999, 711, 30


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def week_bootstrap(diffs: list, weeks, draws: int = DRAWS, seed: int = SEED) -> list:
    """95% intervals for the means of paired per-game differences: whole ISO
    weeks resampled with replacement, mean = resampled sum / resampled games
    (as score_live_2027.py). The same draws serve every difference passed."""
    keys, inv = np.unique(np.asarray(weeks), return_inverse=True)
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(draws, len(keys)))
    den = np.bincount(inv, minlength=len(keys))[idx].sum(1)
    out = []
    for d in diffs:
        num = np.bincount(inv, weights=np.asarray(d, float), minlength=len(keys))[idx].sum(1)
        boot = num / den
        out.append([float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))])
    return out


def load_results(root: Path, as_of: dt.date) -> pd.DataFrame:
    """Completed regular-season games dated before as_of (ET): home-win outcome,
    regulation goals (the OT/SO winner's extra goal removed, as A1 counts them)
    and the ISO week of the game date."""
    cols = {"game_id": int, "date": str, "y": float, "reg_h": int, "reg_a": int, "week": int}
    p = root / RESULTS
    r = pd.read_csv(p, dtype={"date": str}) if p.exists() else pd.DataFrame()
    if len(r):
        r = r[(r.game_id // 1_000_000 == 2026) & (r.game_id // 10_000 % 100 == 2)
              & (r.date < as_of.isoformat())].drop_duplicates("game_id").copy()
    if not len(r):
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})
    extra = r.last_period.isin(["OT", "SO"])
    hw = r.home_g > r.away_g
    r["y"] = hw.astype(float)
    r["reg_h"] = r.home_g - (extra & hw).astype(int)
    r["reg_a"] = r.away_g - (extra & ~hw).astype(int)
    iso = pd.to_datetime(r.date).dt.isocalendar()
    r["week"] = (iso.year * 100 + iso.week).astype(int)
    return r[list(cols)].sort_values(["date", "game_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ git
def git(root: Path, *args, text: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), "-c", "core.quotepath=off",
                           "-c", "log.showSignature=false", *args],
                          capture_output=True, text=text, **kw)


def history(root: Path) -> tuple:
    """{path: (sha, committer time)} of the first commit that added each file
    under FORECASTS, the paths edited or deleted since, and the root's prefix
    in the repository. Walks HEAD and, once fetched, origin/main (where the
    publishing clone pushes); --full-history follows every parent of a merge."""
    refs = ["HEAD"] + (["origin/main"] if git(root, "rev-parse", "-q", "--verify",
                                              "origin/main").returncode == 0 else [])
    p = git(root, "log", "--full-history", "--no-renames", "--relative", "--no-color",
            "--name-status", "--format=@@ %H %cI", *refs, "--", FORECASTS)
    first, edited, deleted = {}, set(), set()
    if p.returncode != 0:
        print(f"[score_live_g] no git history in {root} ({p.stderr.strip()[:120]}); "
              "no forecast can be validated")
        return first, edited, deleted, ""
    sha = when = None
    for line in p.stdout.splitlines():
        if line.startswith("@@ "):
            _, sha, iso = line.split()
            when = pd.Timestamp(iso).tz_convert("UTC")
        elif "\t" in line:
            st, path = line.split("\t", 1)
            if st == "A" and (path not in first or when < first[path][1]):
                first[path] = (sha, when)
            elif st == "M":
                edited.add(path)
            elif st == "D":
                deleted.add(path)
    return first, edited, deleted, git(root, "rev-parse", "--show-prefix").stdout.strip()


def blobs(root: Path, specs: list) -> dict:
    """{'<sha>:<path>': bytes}, read in one `git cat-file --batch` call."""
    if not specs:
        return {}
    out = git(root, "cat-file", "--batch", text=False,
              input="".join(f"{s}\n" for s in specs).encode()).stdout
    got, i = {}, 0
    for s in specs:
        j = out.find(b"\n", i)
        if j < 0:
            break
        head, i = out[i:j].split(), j + 1
        if len(head) == 3 and head[1] == b"blob":       # else "<spec> missing"
            n = int(head[2])
            got[s], i = out[i:i + n], i + n + 1
    return got


def candidates(root: Path, first: dict, prefix: str) -> pd.DataFrame:
    """Every row of every committed forecast file, as first committed."""
    files = sorted(f for f in first if FILE_RE.match(Path(f).name))
    spec = {f: f"{first[f][0]}:{prefix}{f}" for f in files}
    got = blobs(root, list(spec.values()))
    parts = []
    for f in files:
        try:
            d = pd.read_csv(io.BytesIO(got[spec[f]]))
        except Exception as e:  # noqa: BLE001
            print(f"[score_live_g] unreadable forecast {f} ({type(e).__name__}); skipped")
            continue
        kind = "morning" if Path(f).name == "morning.csv" else "pregame"
        parts.append(d.assign(kind=kind, file=f, sha=first[f][0], committed=first[f][1]))
    c = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["game_id", "date", "start_utc", "lineup_home", "lineup_away", "kind", "file",
                 "sha", "committed"])
    for col in [*MODELS.values(), *STATS]:
        c[col] = pd.to_numeric(c[col], errors="coerce") if col in c else np.nan
    for col in ("date", "start_utc", "lineup_home", "lineup_away"):
        if col not in c:
            c[col] = None
    c["game_id"] = pd.to_numeric(c.game_id, errors="coerce")
    c = c[c.game_id.notna()].astype({"game_id": int})
    c["committed"] = pd.to_datetime(c.committed, utc=True)
    return c.reset_index(drop=True)


def uncommitted(root: Path, first: dict) -> tuple:
    """Forecast files in the working tree but not in the history, and the games
    they cover by kind (for the MISSED reasons only; they are never scored)."""
    files, games = [], {k: set() for k in KINDS}
    base = root / FORECASTS
    for p in sorted(base.glob("*/*.csv")) if base.exists() else []:
        rel = p.relative_to(root).as_posix()
        if not FILE_RE.match(p.name) or rel in first:
            continue
        files.append(rel)
        try:
            ids = pd.read_csv(p, usecols=["game_id"]).game_id.astype(int)
        except Exception:  # noqa: BLE001
            continue
        games["morning" if p.name == "morning.csv" else "pregame"] |= set(ids)
    return files, games


# ------------------------------------------------------------------ start times, stamps
def api_starts(root: Path, dates: list, offline: bool) -> dict:
    """(date, game_id) -> scheduled start from the NHL score endpoint, cached in
    CACHE/api_starts_2027.csv; dates not in the cache are fetched unless offline."""
    path = root / CACHE / "api_starts_2027.csv"
    have = (pd.read_csv(path, dtype={"date": str}) if path.exists()
            else pd.DataFrame(columns=["date", "game_id", "start_utc"]))
    todo = sorted(set(dates) - set(have.date))
    if todo and not offline:
        import requests
        rows = []
        for ds in todo:
            try:
                js = requests.get(SCORE_URL.format(d=ds), headers=UA, timeout=30).json()
            except Exception as e:  # noqa: BLE001
                print(f"[score_live_g] NHL API unavailable ({type(e).__name__}); "
                      "file start times used where none is cached")
                break
            rows += [{"date": g.get("gameDate", ds), "game_id": int(g["id"]),
                      "start_utc": g["startTimeUTC"]} for g in js.get("games", [])
                     if g.get("gameType") == 2 and g.get("startTimeUTC")]
            time.sleep(0.5)
        if rows:
            new = pd.DataFrame(rows)
            have = (pd.concat([have, new], ignore_index=True) if len(have) else new) \
                .drop_duplicates(["date", "game_id"], keep="last").sort_values(["date", "game_id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            have.to_csv(path, index=False)
    return {(str(d), int(g)): s for d, g, s in zip(have.date, have.game_id, have.start_utc)}


def schedule_starts(root: Path) -> dict:
    """game_id -> startTimeUTC from the committed 2026-27 team schedules; used
    only for a forecast row with no start_utc and no NHL API start."""
    out = {}
    for f in sorted(root.glob(SCHEDULES)):
        try:
            games = json.loads(f.read_text()).get("games", [])
        except (OSError, ValueError, AttributeError):
            continue
        out.update({int(g["id"]): g.get("startTimeUTC") for g in games
                    if g.get("gameType") == 2 and g.get("id")})
    return out


def validate(cand: pd.DataFrame, res: pd.DataFrame, api: dict, root: Path) -> pd.DataFrame:
    """Candidate rows of completed games with their start (NHL API, else the
    file's start_utc, else the committed schedule) and validity."""
    c = cand[cand.game_id.isin(res.game_id)].copy()
    day = dict(zip(res.game_id, res.date))
    c["start_api"] = pd.Series(pd.to_datetime([api.get((day[g], g)) for g in c.game_id],
                                              utc=True), index=c.index)
    c["start_file"] = pd.to_datetime(c.start_utc, utc=True, errors="coerce")
    need = c.start_api.isna() & c.start_file.isna()
    sched = schedule_starts(root) if need.any() else {}
    c["start_sched"] = pd.to_datetime(c.game_id.map(sched), utc=True, errors="coerce")
    c["start"] = c.start_api.fillna(c.start_file).fillna(c.start_sched)
    c["start_src"] = np.select([c.start_api.notna(), c.start_file.notna(), c.start_sched.notna()],
                               ["api", "file", "schedule"], "none")
    c["on_time"] = c.start.notna() & (c.committed < c.start)
    c["complete"] = c[MODELS["neurhl_g"]].notna() & c[MODELS["elo"]].notna()
    c["valid"] = c.on_time & c.complete
    return c


def gh_stamps(root: Path, commits: dict, offline: bool, stamp_time=None) -> tuple:
    """sha -> GitHub Actions stamp time (neurhl/live/deadline.py, or the
    stamp_time given) for the counted commits ({sha: committer time}), cached
    in CACHE/stamps_2027.csv. A sha without a stamp is asked again until it has
    been checked two days after its commit. Informative only: gh may be missing
    or unauthenticated, and no validity decision depends on it."""
    path = root / CACHE / "stamps_2027.csv"
    have = ({r.sha: (r.stamp_utc, r.checked_utc) for r in
             pd.read_csv(path, dtype=str).fillna("").itertuples()} if path.exists() else {})
    todo = [s for s, t in sorted(commits.items()) if s not in have or
            (not have[s][0] and pd.Timestamp(have[s][1]) < t + dt.timedelta(days=2))]
    status = "offline: cached stamps only" if offline else "ok"
    if todo and not offline:
        now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        try:
            if stamp_time is None:
                spec = importlib.util.spec_from_file_location("deadline",
                                                              NRL / "live" / "deadline.py")
                D = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(D)
                stamp_time = D.stamp_time
            for s in todo:
                t = stamp_time(s)
                have[s] = (t.isoformat() if t is not None else "", now)
        except Exception as e:  # noqa: BLE001
            status = f"gh unavailable ({type(e).__name__}: {str(e)[:160]})"
        if have:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([(s, *v) for s, v in sorted(have.items())],
                         columns=["sha", "stamp_utc", "checked_utc"]).to_csv(path, index=False)
    return {s: v[0] for s, v in have.items() if v[0]}, status


# ------------------------------------------------------------------ scoring
def choose(c: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Per game, the valid forecast of this kind committed last before the start."""
    v = c[(c.kind == kind) & c.valid]
    return v.sort_values(["committed", "file"]).drop_duplicates("game_id", keep="last")


def missed(res: pd.DataFrame, c: pd.DataFrame, chosen: pd.DataFrame, kind: str,
           pending: dict) -> dict:
    """game_id -> why a completed game has no valid forecast of this kind."""
    out, have, ck = {}, set(chosen.game_id), c[c.kind == kind]
    for g in res.game_id:
        if g in have:
            continue
        k = ck[ck.game_id == g]
        if (k.on_time & ~k.complete).any():
            out[int(g)] = "incomplete"          # on time, but no G or Elo probability
        elif k.start.notna().any():
            out[int(g)] = "late"
        elif len(k):
            out[int(g)] = "no_start"
        else:
            out[int(g)] = "uncommitted" if g in pending[kind] else "none"
    return out


def metrics(p, y) -> dict:
    return {"n": int(len(p)), "log_loss": float(nll(p, y).mean()) if len(p) else None,
            "brier": float(((p - y) ** 2).mean()) if len(p) else None}


def paired(g: pd.DataFrame, a: str, b: str, boot: bool = True) -> dict:
    """Mean per-game difference a - b in log loss and Brier on the games where
    both exist; the week-block interval once there are more than 30 games."""
    ok = (g[MODELS[a]].notna() & g[MODELS[b]].notna()).to_numpy()
    h, y = g[ok], g.y.to_numpy(float)[ok]
    pa, pb = h[MODELS[a]].to_numpy(float), h[MODELS[b]].to_numpy(float)
    dl, db = nll(pa, y) - nll(pb, y), (pa - y) ** 2 - (pb - y) ** 2
    out = {"n": int(ok.sum()), "log_loss": float(dl.mean()) if len(h) else None,
           "brier": float(db.mean()) if len(h) else None}
    if boot:
        out.update(weeks=int(h.week.nunique()), ci95=None, ci95_brier=None)
        if len(h) > MIN_BOOT and out["weeks"] >= 2:
            out["ci95"], out["ci95_brier"] = week_bootstrap([dl, db], h.week.to_numpy())
    return out


def source_of(home, away) -> str:
    """A game's lineup source: the less certain of its two sides."""
    r = max(SOURCES.index(s) if s in SOURCES else len(SOURCES) for s in (home, away))
    return SOURCES[r] if r < len(SOURCES) else "UNKNOWN"


def by_source(g: pd.DataFrame) -> dict:
    out = {}
    for s in [*SOURCES, "UNKNOWN"]:
        h = g[g.source == s]
        if not len(h):
            continue
        y = h.y.to_numpy(float)
        e = {"n": int(len(h)), "log_loss": {}, "brier": {}}
        for m, col in MODELS.items():
            ok = h[col].notna().to_numpy()
            mm = metrics(h[col].to_numpy(float)[ok], y[ok])
            e["log_loss"][m], e["brier"][m] = mm["log_loss"], mm["brier"]
        e["paired"] = {k: paired(h, a, b, boot=False) for k, (a, b) in PAIRS.items()}
        out[s] = e
    return out


def actuals(root: Path) -> dict:
    """The ingest's 2026-27 tables where present (read only), and the games it
    ingested degraded (state.json), which the stat sheet leaves out."""
    T = root / TABLES
    out = {"degraded": set()}
    for key, name in (("tgx", "tgx_2027"), ("players", "player_games_2027")):
        p = T / f"{name}.parquet"
        try:
            out[key] = pd.read_parquet(p) if p.exists() else None
        except Exception as e:  # noqa: BLE001
            print(f"[score_live_g] {name} unreadable ({type(e).__name__}); skipped")
            out[key] = None
    try:
        st = json.loads((T / "_ingest_2027" / "state.json").read_text())
        out["degraded"] = {int(g) for g in st.get("degraded", {})}
    except (OSError, ValueError):
        pass
    return out


def _mae(pred, act) -> dict:
    pred, act = np.asarray(pred, float), np.asarray(act, float)
    ok = np.isfinite(pred) & np.isfinite(act)
    if not ok.any():
        return {"n_team_games": 0, "mae": None, "mean_pred": None, "mean_actual": None}
    return {"n_team_games": int(ok.sum()), "mae": float(np.abs(pred[ok] - act[ok]).mean()),
            "mean_pred": float(pred[ok].mean()), "mean_actual": float(act[ok].mean())}


def player_lines(g: pd.DataFrame, tabs: dict, root: Path, first: dict, prefix: str) -> dict:
    """Skater stat lines (*_players.csv, as first committed, before the start)
    against player_games_2027; skaters who did not dress are counted apart."""
    pg = tabs["players"]
    if pg is None:
        return {"available": False}
    want = {}
    for r in g.itertuples():
        f = r.file[:-4] + "_players.csv"
        if f in first and first[f][1] < r.start:
            want.setdefault(f, set()).add(int(r.game_id))
    spec = {f: f"{first[f][0]}:{prefix}{f}" for f in want}
    got = blobs(root, list(spec.values()))
    parts = []
    for f, ids in want.items():
        try:
            d = pd.read_csv(io.BytesIO(got[spec[f]]))
            parts.append(d[d.game_id.isin(ids)])
        except Exception:  # noqa: BLE001
            continue
    act = pg[(pg.pos_group != 2) & ~pg.game_id.isin(tabs["degraded"])]
    fc = pd.concat(parts, ignore_index=True) if parts else None
    if fc is None or not len(fc):
        return {"available": True, "n_skater_games": 0, "not_dressed": 0,
                "mae": {k: None for k in ("toi", "sog", "goals", "assists", "points")}}
    fc = fc[fc.game_id.isin(act.game_id)]
    j = fc.merge(act, on=["game_id", "player_id"])
    pairs = {"toi": (j.toi_ev + j.toi_pp + j.toi_sh, j.toi_sec / 60),   # minutes
             "sog": (j.sog_mean, j.sog), "goals": (j.goals_mean, j.goals),
             "assists": (j.assists_mean, j.assists), "points": (j.points_mean, j.goals + j.assists)}
    return {"available": True, "n_skater_games": int(len(j)), "not_dressed": int(len(fc) - len(j)),
            "mae": {k: float((p - a).abs().mean()) if len(j) else None for k, (p, a) in pairs.items()}}


def stat_sheet(g: pd.DataFrame, tabs: dict, lines: dict) -> dict:
    """Descriptive: team regulation goals (scaled by A1, and raw), SOG and xGF
    against the actuals; skater lines; the A1 multiplier in force."""
    act = np.r_[g.reg_h, g.reg_a]
    raw = _mae(np.r_[g.goals_home_raw, g.goals_away_raw], act)
    out = {"goals_reg": {**_mae(np.r_[g.goals_home, g.goals_away], act),
                         "mae_raw": raw["mae"], "mean_pred_raw": raw["mean_pred"]}}
    t = tabs["tgx"]
    for k, (h, a, col) in {"sog": ("sog_home", "sog_away", "sogf"),
                           "xgf": ("xgf_home", "xgf_away", "xgf_all")}.items():
        if t is None:
            out[k] = {"available": False}
            continue
        tt = t[~t.game_id.isin(tabs["degraded"])]
        side = {s: tt[tt.is_home == s].drop_duplicates("game_id").set_index("game_id")[col]
                for s in (1, 0)}
        out[k] = {"available": True, **_mae(np.r_[g[h], g[a]],
                                            np.r_[g.game_id.map(side[1]), g.game_id.map(side[0])])}
    out["players"] = lines
    gm = g[["date", "goal_mult"]].dropna()
    by = {}
    for d, s in gm.groupby("date").goal_mult:
        v = sorted({float(round(x, 4)) for x in s})
        by[str(d)] = v[0] if len(v) == 1 else v
    out["goal_mult"] = {"n": int(len(gm)),
                        "min": float(gm.goal_mult.min()) if len(gm) else None,
                        "max": float(gm.goal_mult.max()) if len(gm) else None,
                        "latest": by[max(by)] if by else None, "by_date": by}
    return out


def stamp_summary(g: pd.DataFrame, stamps: dict) -> dict:
    st = pd.to_datetime(g.sha.map(stamps), utc=True, errors="coerce")
    has = st.notna()
    late = has & (st >= g.start)
    return {"with_stamp": int(has.sum()), "stamp_before_start": int((has & ~late).sum()),
            "stamp_late_game_ids": sorted(int(x) for x in g.game_id[late]),
            "without_stamp": int((~has).sum())}


def analyse(g: pd.DataFrame, kind: str, lines: dict, tabs: dict, stamps: dict) -> dict:
    """Every metric on the games with a valid forecast of one kind."""
    y = g.y.to_numpy(float)
    out = {"forecast": kind, "role": "primary" if kind == "pregame" else "secondary",
           "n_games": int(len(g)), "home_win_rate": float(y.mean()) if len(g) else None,
           "models": {}}
    for m, col in MODELS.items():
        ok = g[col].notna().to_numpy()
        out["models"][m] = metrics(g[col].to_numpy(float)[ok], y[ok])
        if m in ("neurhl_h", "neurhl_1p0"):             # the two that can be missing
            out["models"][m]["missing_game_ids"] = sorted(int(x) for x in g.game_id[~ok])
    out["paired"] = {k: paired(g, a, b) for k, (a, b) in PAIRS.items()}
    out["by_lineup_source"] = by_source(g)
    out["stat_sheet"] = stat_sheet(g, tabs, lines)
    out["start_sources"] = {str(k): int(v) for k, v in sorted(Counter(g.start_src).items())}
    out["stamps"] = stamp_summary(g, stamps)
    return out


def show(card: dict) -> None:
    c = card["counts"]
    print(f"NeurHL-G live scorecard as of {card['as_of']} ({card['status']})")
    print(f"games completed {c['games_completed']} | valid pregame {c['valid']['pregame']}, "
          f"morning {c['valid']['morning']} | MISSED pregame {c['missed']['pregame']}, "
          f"morning {c['missed']['morning']}")
    if not c["games_completed"]:
        print("no completed 2026-27 regular-season games yet")
        return
    f = lambda x: f"{x:10.5f}" if x is not None else f"{'-':>10}"
    print(f"\n{'':10}{'pregame (primary)':>26}{'morning':>26}")
    print(f"{'model':<10}" + f"{'n':>6}{'log loss':>10}{'Brier':>10}" * 2)
    for m, lab in LABELS.items():
        print(f"{lab:<10}" + "".join(
            f"{card[k]['models'][m]['n']:>6}{f(card[k]['models'][m]['log_loss'])}"
            f"{f(card[k]['models'][m]['brier'])}" for k in KINDS))
    print("\npaired log-loss difference, mean [95% week-block interval]")
    print(f"{'':10}{'pregame (primary)':>34}{'morning':>34}")
    for p, lab in (("g_minus_elo", "G - Elo"), ("g_minus_h", "G - H"), ("h_minus_elo", "H - Elo")):
        cells = []
        for k in KINDS:
            e = card[k]["paired"][p]
            ci = f" [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]" if e["ci95"] else ""
            cells.append((f"{e['log_loss']:+.5f}{ci}" if e["n"] else "-").rjust(34))
        print(f"{lab:<10}" + "".join(cells))
    print(card["note"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score NeurHL-G's live 2026-27 forecasts "
                                             "(PLAN_NeurHL4 LIVE).")
    ap.add_argument("--offline", action="store_true",
                    help="no network: cached NHL API start times and stamps only")
    ap.add_argument("--as-of", help="score games dated before this day (ET), YYYY-MM-DD; "
                                    "default today")
    ap.add_argument("--root", default=str(PROJ),
                    help="repository to read forecasts, results and tables from and to "
                         "write the scorecard to (tests)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    as_of = dt.date.fromisoformat(a.as_of) if a.as_of else dt.datetime.now(ET).date()

    res = load_results(root, as_of)
    first, edited, deleted, prefix = history(root)
    cand = candidates(root, first, prefix)
    unc_files, pending = uncommitted(root, first)
    c = validate(cand, res, api_starts(root, sorted(set(res.date)), a.offline), root)
    chosen = {k: choose(c, k) for k in KINDS}
    stamps, stamp_status = gh_stamps(root, {s: t for k in KINDS for s, t in
                                            zip(chosen[k].sha, chosen[k].committed)}, a.offline)
    tabs = actuals(root)
    games = res.rename(columns={"date": "game_date"})

    final = as_of > LAST and len(res) >= N_GAMES
    card = {"as_of": as_of.isoformat(), "status": "final" if final else "interim",
            "note": ("final: the one inference PLAN_NeurHL4 LIVE declares" if final else
                     "interim: descriptive only; inference is made once, after the last "
                     "regular-season game (2027-04-10)"),
            "primary": "pregame", "games_completed": int(len(res))}
    miss, lineups = {}, {}
    for k in KINDS:
        g = chosen[k].merge(games, on="game_id")
        g["source"] = [source_of(h, a_) for h, a_ in zip(g.lineup_home, g.lineup_away)]
        miss[k] = missed(res, c, chosen[k], k, pending)
        n_g = Counter(g.source)
        n_s = Counter(s if s in SOURCES else "UNKNOWN" for s in [*g.lineup_home, *g.lineup_away])
        lineups[k] = {"games": {s: n_g[s] for s in [*SOURCES, "UNKNOWN"] if n_g[s]},
                      "sides": {s: n_s[s] for s in [*SOURCES, "UNKNOWN"] if n_s[s]}}
        card[k] = analyse(g, k, player_lines(g, tabs, root, first, prefix), tabs, stamps)
    fc_files = [f for f in first if FILE_RE.match(Path(f).name)]
    card["counts"] = {
        "games_completed": int(len(res)),
        "valid": {k: card[k]["n_games"] for k in KINDS},
        "missed": {k: len(miss[k]) for k in KINDS},
        "missed_by_reason": {k: dict(sorted(Counter(miss[k].values()).items())) for k in KINDS},
        "neurhl_h_missing": {k: len(card[k]["models"]["neurhl_h"]["missing_game_ids"])
                             for k in KINDS},
        "lineup_source": lineups,
        "forecast_files": {"committed": len(fc_files), "uncommitted": len(unc_files),
                           "edited_after_add": len(edited & set(fc_files)),
                           "deleted_after_add": len(deleted & set(fc_files))}}
    card["missed_game_ids"] = {k: sorted(miss[k]) for k in KINDS}
    late = c[~c.on_time & c.start.notna()].sort_values(["game_id", "kind", "file"])
    card["late_forecasts"] = [{"game_id": int(r.game_id), "forecast": r.kind, "file": r.file,
                               "committed_utc": r.committed.isoformat(),
                               "start_utc": r.start.isoformat(), "start_source": r.start_src}
                              for r in late.itertuples()]
    mm = c[(c.start_api - c.start_file).dt.total_seconds().abs() > 60]
    card["start_check"] = {
        "rows_by_source": {str(k): int(v) for k, v in sorted(Counter(c.start_src).items())},
        "mismatches": [{"game_id": int(r.game_id), "file": r.file,
                        "start_file": r.start_file.isoformat(), "start_api": r.start_api.isoformat()}
                       for r in mm.itertuples()]}
    card["stamps"] = {"status": stamp_status, "cached": len(stamps)}
    card["files"] = {"uncommitted": unc_files,         # not in the history: never scored
                     "edited_after_add": sorted(edited & set(fc_files)),   # scored as first added
                     "deleted_after_add": sorted(deleted & set(fc_files))}
    order = ["as_of", "status", "note", "primary", "games_completed", "counts",
             "missed_game_ids", *KINDS, "late_forecasts", "start_check", "stamps", "files"]
    card = {k: card[k] for k in order}

    out = root / CARD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=1))
    show(card)
    if unc_files:
        print(f"[score_live_g] WARNING: {len(unc_files)} forecast files are not in this "
              "repository's history (pull the published commits); they do not count")
    if card["late_forecasts"]:
        print(f"[score_live_g] {len(card['late_forecasts'])} forecast rows committed at or "
              "after their game's start (see late_forecasts)")
    late_push = {k: card[k]["stamps"]["stamp_late_game_ids"] for k in KINDS}
    if any(late_push.values()):
        print(f"[score_live_g] counted forecasts whose Actions stamp is at or after the start "
              f"(committed in time, pushed late): {late_push}")
    print(f"[score_live_g] stamps: {stamp_status}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
