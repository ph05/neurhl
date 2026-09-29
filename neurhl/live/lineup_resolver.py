"""NeurHL LIVE: resolve the dressed lineup of both teams for one game (PLAN_NeurHL4 LIVE).

resolve(game_id, date, home, away, snapshot_dir=None) returns

  {game_id, date, start_utc, as_of, home: SIDE, away: SIDE}

  SIDE = {team, forwards [ids], defense [ids], skaters [F then D, target 12 + 6],
          goalie (id or None), lineup_source, goalie_source, n_f, n_d, notes [str]}

All ids are NHL player ids. Sources, in priority order:

  lineup_source
    NHL_API       the game's boxscore (api-web.nhle.com/v1/gamecenter/{gid}/boxscore)
                  already lists the side's skaters (populated at/after puck drop).
    DF_CONFIRMED  DailyFaceoff lines, updated_at within 12 h of the start and a
                  source name without "project"/"offseason"/"camp" (heuristic:
                  DailyFaceoff's lines pages carry no explicit confirmed flag).
    DF_PROJECTED  any other DailyFaceoff lines page updated within 36 h of the start.
    FALLBACK      the team's dressed skaters in its most recent 2026-27 game (for
                  its first game: the latest roster snapshot's skaters ranked by
                  prior-season TOI, usage_2026.parquet toi_all), minus
                  status.unavailable(date) and DailyFaceoff IR/injured players,
                  topped up from the roster by prior TOI to 12 F / 6 D.
  goalie_source
    NHL_API       boxscore goalie flagged starter.
    DF_CONFIRMED  DailyFaceoff starting-goalies feed says Confirmed.
    DF_LIKELY     ... says Likely.
    DF_LINES      DailyFaceoff lines g1 (then g2); only when the lineup is DF_*.
    FALLBACK      roster goalie with the most 2026-27 starts for the team to date
                  (club-stats endpoint); prior-season starts (goalie_games_2026)
                  break ties and decide game 1.

The goalie feed is used whenever it lists the team as Confirmed/Likely for that
date, even if the team's lines page is stale. Unavailable players
(status.unavailable) and DailyFaceoff injured players are never returned by the
DailyFaceoff or fallback paths; the NHL_API path is ground truth and is not
filtered. A DailyFaceoff lineup that maps to fewer than 9 F or 4 D is rejected
(FALLBACK); smaller shortfalls are topped up from the roster by prior TOI.

DailyFaceoff snapshots: the latest file for the team under
data/raw/lineup_snapshots/<date>/ (daily, time = file mtime) or
<date>/<HHMM>/ (intraday, local time) taken no later than min(now, start).
With snapshot_dir, only that directory is read.

Never raises on missing data: every stage degrades to FALLBACK and says why in
`notes`.

CLI: python neurhl/live/lineup_resolver.py [--date YYYY-MM-DD] [--game GID]
     [--snapshot-dir DIR] [--rosters YYYY-MM-DD] [--no-api] [--json OUT] [--ids]
"""
import argparse
import glob
import importlib.util
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS, UA  # noqa: E402
# siblings: relative when imported as neurhl.live.lineup_resolver; plain when run as a script.
# (Never `import live.*`: common.py puts src/ first on sys.path and src/live.py shadows it.)
try:
    from .fetch_rosters import latest_rosters, load_august
    from .parse_dailyfaceoff import SNAP, SLUG_TO_NHL, parse_lines, parse_goalies, ev_role
    from .id_map import map_ids, load_aliases, load_fallback
    from .status import unavailable as _unavailable
except ImportError:
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    from fetch_rosters import latest_rosters, load_august  # noqa: E402
    from parse_dailyfaceoff import SNAP, SLUG_TO_NHL, parse_lines, parse_goalies, ev_role  # noqa: E402
    from id_map import map_ids, load_aliases, load_fallback  # noqa: E402
    # neurhl/status.py also exists; load the live one by path so the name cannot collide.
    _spec = importlib.util.spec_from_file_location("neurhl_live_status", _HERE / "status.py")
    _st = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_st)
    _unavailable = _st.unavailable

log = logging.getLogger("neurhl.live.lineup_resolver")

SEASON = "20262027"
API = "https://api-web.nhle.com/v1"
DF_WINDOW = timedelta(hours=36)
DF_CONFIRM_WINDOW = timedelta(hours=12)
N_F, N_D = 12, 6
MIN_DF_F, MIN_DF_D = 9, 4
LINEUP_SOURCES = ("NHL_API", "DF_CONFIRMED", "DF_PROJECTED", "FALLBACK")
GOALIE_SOURCES = ("NHL_API", "DF_CONFIRMED", "DF_LIKELY", "DF_LINES", "FALLBACK")
FWD_POS = {"C", "L", "R"}
_HHMM = re.compile(r"^\d{4}$")
_DATE = re.compile(r"^\d{4}-\d\d-\d\d$")


# ------------------------------------------------------------------ small helpers
def _utc(x) -> datetime | None:
    if x is None or x is pd.NaT:
        return None
    try:
        t = pd.Timestamp(x)
    except (ValueError, TypeError):
        return None
    if pd.isna(t):
        return None
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.to_pydatetime()


def _get_json(url: str, timeout: float = 20) -> dict | None:
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        log.info("GET %s failed: %s", url, e)
        return None


def _dedupe(ids) -> list[int]:
    seen, out = set(), []
    for i in ids:
        if i is None or pd.isna(i):
            continue
        i = int(i)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ------------------------------------------------------------------ static inputs (cached)
@lru_cache(maxsize=1)
def prior_toi() -> dict[int, int]:
    """player_id -> 2025-26 total toi_all (usage_2026.parquet)."""
    try:
        u = pd.read_parquet(TENSORS / "usage_2026.parquet", columns=["player_id", "toi_all"])
        return {int(k): int(v) for k, v in u.groupby("player_id")["toi_all"].sum().items()}
    except Exception as e:  # noqa: BLE001
        log.warning("usage_2026.parquet unreadable (%s); prior TOI treated as 0", e)
        return {}


@lru_cache(maxsize=1)
def prior_starts() -> dict[int, int]:
    """player_id -> 2025-26 goalie starts (goalie_games_2026.parquet)."""
    try:
        g = pd.read_parquet(TENSORS / "goalie_games_2026.parquet", columns=["player_id", "goalie_start"])
        return {int(k): int(v) for k, v in g.groupby("player_id")["goalie_start"].sum().items()}
    except Exception as e:  # noqa: BLE001
        log.warning("goalie_games_2026.parquet unreadable (%s); prior starts treated as 0", e)
        return {}


@lru_cache(maxsize=1)
def schedule() -> pd.DataFrame:
    """2026-27 regular season: game_id, date, home, away, start_utc.

    Same files and dedup as neurhl/sim/project_2027.load_schedule(), plus the start
    time (that module pulls in the simulation stack, so it is not imported here).
    """
    rows = []
    for f in sorted(glob.glob(str(RAW / f"nhl_sched_*_{SEASON}.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        for g in d.get("games", d if isinstance(d, list) else []):
            try:
                if int(g.get("gameType", 2)) != 2:
                    continue
                rows.append({"game_id": int(g["id"]), "date": g.get("gameDate", ""),
                             "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                             "start_utc": g.get("startTimeUTC")})
            except (KeyError, TypeError, ValueError):
                continue
    cols = ["game_id", "date", "home", "away", "start_utc"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(rows, columns=cols).drop_duplicates("game_id")
            .sort_values(["date", "game_id"]).reset_index(drop=True))


@lru_cache(maxsize=8)
def _rosters(on_or_before: str) -> tuple[str, pd.DataFrame]:
    try:
        return latest_rosters(on_or_before)
    except Exception as e:  # noqa: BLE001
        log.warning("no roster snapshot on/before %s (%s); using August files", on_or_before, e)
        return "2026-08", load_august()


@lru_cache(maxsize=1)
def _id_refs():
    return load_aliases(), load_fallback()


@lru_cache(maxsize=64)
def _lines(d: str) -> pd.DataFrame:
    return parse_lines(Path(d))


@lru_cache(maxsize=64)
def _goalies(d: str) -> pd.DataFrame:
    return parse_goalies(Path(d))


# ------------------------------------------------------------------ NHL API (cached per process)
@lru_cache(maxsize=16)
def score(date: str) -> list[dict]:
    js = _get_json(f"{API}/score/{date}")
    return (js or {}).get("games", []) or []


@lru_cache(maxsize=256)
def _boxscore_final(gid: int) -> dict | None:
    return _get_json(f"{API}/gamecenter/{gid}/boxscore")


def boxscore(gid: int, final: bool = False) -> dict | None:
    """Boxscore JSON; past (final) games are cached, the current game is always re-fetched."""
    return _boxscore_final(int(gid)) if final else _get_json(f"{API}/gamecenter/{gid}/boxscore")


@lru_cache(maxsize=64)
def club_starts(team: str) -> dict[int, int]:
    js = _get_json(f"{API}/club-stats/{team}/{SEASON}/2")
    out = {}
    for g in (js or {}).get("goalies", []) or []:
        try:
            out[int(g["playerId"])] = int(g.get("gamesStarted") or 0)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def box_side(box: dict | None, side: str) -> dict | None:
    """{'F': [...], 'D': [...], 'G': [...], 'starter': id|None, 'toi': {id: sec}} or None."""
    try:
        t = box["playerByGameStats"][f"{side}Team"]
    except (KeyError, TypeError):
        return None

    def sec(p):
        try:
            m, s = str(p.get("toi", "0:00")).split(":")
            return int(m) * 60 + int(s)
        except ValueError:
            return 0

    toi, out = {}, {"F": [], "D": [], "G": [], "starter": None}
    for key, grp in (("forwards", "F"), ("defense", "D"), ("goalies", "G")):
        for p in t.get(key) or []:
            try:
                pid = int(p["playerId"])
            except (KeyError, TypeError, ValueError):
                continue
            out[grp].append(pid)
            toi[pid] = sec(p)
            if grp == "G" and p.get("starter") and out["starter"] is None:
                out["starter"] = pid
    out["toi"] = toi
    if out["starter"] is None and out["G"]:
        out["starter"] = max(out["G"], key=lambda i: toi.get(i, 0))
    return out if (out["F"] or out["D"]) else None


# ------------------------------------------------------------------ DailyFaceoff snapshots
def _snap_time(f: Path, day: str) -> datetime:
    parent = f.parent.name
    if _HHMM.match(parent) and f.parent.parent.name == day:
        return datetime.strptime(f"{day} {parent}", "%Y-%m-%d %H%M").astimezone().astimezone(timezone.utc)
    return datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)


def latest_snapshot(fname: str, cutoff: datetime, game_date: str,
                    snapshot_dir: Path | None = None) -> tuple[Path | None, datetime | None]:
    """Newest <date>/[<HHMM>/]fname taken at or before cutoff (dates <= game_date)."""
    if snapshot_dir is not None:
        f = Path(snapshot_dir) / fname
        return (f.parent, datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)) if f.exists() else (None, None)
    if not SNAP.exists():
        return None, None
    days = sorted(p.name for p in SNAP.iterdir() if p.is_dir() and _DATE.match(p.name) and p.name <= game_date)
    for day in reversed(days[-7:]):
        dd = SNAP / day
        cands = [dd / fname] + [sub / fname for sub in dd.iterdir() if sub.is_dir() and _HHMM.match(sub.name)]
        best = None
        for f in cands:
            if not f.exists():
                continue
            t = _snap_time(f, day)
            if t <= cutoff and (best is None or t > best[1]):
                best = (f.parent, t)
        if best:
            return best
    return None, None


_CANON = {"UTA": "utah-mammoth"}   # snapshot files are always saved under the current Utah slug


def _slug_file(team: str) -> str:
    slug = _CANON.get(team) or next((s for s, t in SLUG_TO_NHL.items() if t == team), None)
    if slug is None:
        raise KeyError(f"unknown team {team!r}")
    return f"df_lines_{slug}.html.gz"


# ------------------------------------------------------------------ core
class _Ctx:
    def __init__(self, game_id, date, home, away, snapshot_dir, rosters_date, use_api, as_of):
        self.game_id, self.date, self.home, self.away = int(game_id), str(date), home, away
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.use_api = use_api
        self.notes_game: list[str] = []
        self.start = self._start()
        now = as_of or datetime.now(timezone.utc)
        self.as_of = now
        self.cutoff = min(now, self.start) if self.start else now
        self.rday, self.rosters = _rosters(rosters_date or self.date)
        try:
            self.unavail = _unavailable(self.date)
        except Exception as e:  # noqa: BLE001
            self.unavail = set()
            self.notes_game.append(f"status.unavailable failed: {e}")
        self.box = None
        if use_api:
            self.box = boxscore(self.game_id)

    def _start(self) -> datetime | None:
        if getattr(self, "use_api", True):
            for g in score(self.date):
                if int(g.get("id", 0)) == self.game_id:
                    return _utc(g.get("startTimeUTC"))
        s = schedule()
        m = s[s["game_id"] == self.game_id]
        if len(m):
            return _utc(m.iloc[0]["start_utc"])
        self.notes_game.append("start time unknown; assuming 23:00 UTC")
        return _utc(f"{self.date}T23:00:00Z")

    # roster helpers -------------------------------------------------
    def roster_team(self, team: str) -> pd.DataFrame:
        return self.rosters[self.rosters["team"] == team]

    def pos_of(self, pid: int, team: str) -> str | None:
        r = self.rosters[self.rosters["player_id"] == pid]
        if not len(r):
            return None
        p = r.iloc[0]["pos"]
        return "F" if p in FWD_POS else ("D" if p == "D" else ("G" if p == "G" else None))


def _df_injured(ctx: _Ctx, team: str, notes: list) -> set[int]:
    """NHL ids DailyFaceoff shows as injured for the team (ir group, or out/ir status)."""
    try:
        d, _ = latest_snapshot(_slug_file(team), ctx.cutoff, ctx.date, ctx.snapshot_dir)
        if d is None:
            return set()
        L = _lines(str(d))
        L = L[L["team_abbrev"] == team]
        inj = L[L["group"].eq("ir") | L["injury_status"].isin(["out", "ir"])]
        if not len(inj):
            return set()
        aliases, fb = _id_refs()
        m, _ = map_ids(inj, ctx.rosters, aliases, fb)
        return set(int(x) for x in m["player_id"].dropna())
    except Exception as e:  # noqa: BLE001
        notes.append(f"DF injury list unavailable ({type(e).__name__}: {e})")
        return set()


MP_OUT_STATUSES = {"IR", "IR-NR", "IR-LT", "O"}


def _mp_injured(ctx: _Ctx, team: str, notes: list) -> set[int]:
    """NHL ids MoneyPuck lists as out for this game (PLAN_NeurHL_1_1 A15): injured reserve or
    out, with a real return date after the game date, from the latest snapshot before the
    cutoff. Day-to-day players and placeholder dates are left to the other sources."""
    try:
        d, _ = latest_snapshot("mp_injuries.csv.gz", ctx.cutoff, ctx.date, ctx.snapshot_dir)
        if d is None:
            return set()
        m = pd.read_csv(Path(d) / "mp_injuries.csv.gz")
        ret = m.dateOfReturn.astype(str).str[:10]
        out = m[(m.teamCode == team) & m.playerInjuryStatus.isin(MP_OUT_STATUSES)
                & (ret > ctx.date) & (ret <= "2027-06-30")]
        ids = set(int(x) for x in out.playerId)
        if ids:
            notes.append(f"MoneyPuck injured excluded: {', '.join(out.playerName)}")
        return ids
    except Exception as e:  # noqa: BLE001
        notes.append(f"MoneyPuck injury list unavailable ({type(e).__name__}: {e})")
        return set()


def _top_up(ctx: _Ctx, team: str, F: list, D: list, exclude: set, notes: list) -> tuple[list, list]:
    toi = prior_toi()
    r = ctx.roster_team(team)
    have = set(F) | set(D)
    pool = r[~r["player_id"].isin(have | exclude)].copy()
    pool["toi"] = pool["player_id"].map(lambda i: toi.get(int(i), 0))
    pool = pool.sort_values(["toi", "player_id"], ascending=[False, True])
    addF = [int(i) for i in pool.loc[pool["pos"].isin(FWD_POS), "player_id"]][:max(0, N_F - len(F))]
    addD = [int(i) for i in pool.loc[pool["pos"].eq("D"), "player_id"]][:max(0, N_D - len(D))]
    if addF or addD:
        notes.append(f"topped up {len(addF)} F / {len(addD)} D from roster {ctx.rday} by prior TOI")
    F, D = F + addF, D + addD
    if len(F) < N_F or len(D) < N_D:
        notes.append(f"short: only {len(F)} F / {len(D)} D available")
    return F, D


def _df_lineup(ctx: _Ctx, team: str, exclude: set, notes: list):
    """(F, D, source, g_candidates) from DailyFaceoff lines, or None."""
    d, snap_t = latest_snapshot(_slug_file(team), ctx.cutoff, ctx.date, ctx.snapshot_dir)
    if d is None:
        notes.append("no DailyFaceoff lines snapshot")
        return None
    L = _lines(str(d))
    L = L[L["team_abbrev"] == team]
    if not len(L):
        notes.append(f"DF snapshot {d.relative_to(SNAP) if SNAP in d.parents else d}: team page unparseable")
        return None
    upd = _utc(L["updated_at"].iloc[0])
    where = d.relative_to(SNAP) if SNAP in d.parents else d
    if upd is None or ctx.start is None or upd < ctx.start - DF_WINDOW:
        notes.append(f"DF lines stale (updated {upd:%Y-%m-%d %H:%M}Z, snapshot {where})" if upd
                     else f"DF lines have no updated_at (snapshot {where})")
        return None
    aliases, fb = _id_refs()
    role = ev_role(L)
    ev = L[role.isin(["F", "D", "G"])].assign(role=role[role.isin(["F", "D", "G"])])
    ev = ev.assign(_ord=ev["group"].astype(str) + ev["slot"].astype(str))
    m, un = map_ids(ev, ctx.rosters, aliases, fb)
    if len(un):
        notes.append(f"DF unmatched: {', '.join(un['name'])}")
    inj_status = m["injury_status"].isin(["out", "ir"])
    m = m[m["player_id"].notna()]
    dropped = m[m["player_id"].astype(int).isin(exclude) | inj_status.reindex(m.index, fill_value=False)]
    if len(dropped):
        notes.append(f"DF dropped unavailable/injured: {', '.join(dropped['name'].drop_duplicates())}")
    m = m.drop(dropped.index)
    F = _dedupe(m.loc[m["role"].eq("F"), "player_id"])[:N_F]
    D = _dedupe(m.loc[m["role"].eq("D"), "player_id"])[:N_D]
    if len(F) < MIN_DF_F or len(D) < MIN_DF_D:
        notes.append(f"DF lines map to only {len(F)} F / {len(D)} D; rejected")
        return None
    src = str(L["source_name"].iloc[0] or "")
    confirmed = (upd >= ctx.start - DF_CONFIRM_WINDOW
                 and not re.search(r"project|offseason|camp", src, re.I))
    g = m[m["role"].eq("G")].sort_values("_ord")
    g_cands = _dedupe(g.loc[g["slot"].isin(["g1", "g2"]), "player_id"])
    notes.append(f"DF lines {where} '{src}' updated {upd:%m-%d %H:%M}Z")
    return F, D, ("DF_CONFIRMED" if confirmed else "DF_PROJECTED"), g_cands


def _df_goalie_feed(ctx: _Ctx, team: str, exclude: set, notes: list):
    """(goalie_id, 'DF_CONFIRMED'|'DF_LIKELY') from the starting-goalies feed, or None."""
    d, _ = latest_snapshot("df_goalies.html.gz", ctx.cutoff, ctx.date, ctx.snapshot_dir)
    if d is None:
        return None
    G = _goalies(str(d))
    if not len(G):
        return None
    G = G[(G["team_abbrev"] == team) & (G["game_date"].astype(str) == ctx.date)]
    if not len(G):
        return None
    row = G.iloc[0]
    if row["status"] not in ("Confirmed", "Likely"):
        notes.append(f"DF goalie feed: {row['goalie']} ({row['status']}), not used")
        return None
    aliases, fb = _id_refs()
    m, _ = map_ids(pd.DataFrame({"team_abbrev": [team], "name": [row["goalie"]],
                                 "jersey": pd.array([pd.NA], dtype="Int64"),
                                 "group": ["g"], "slot": ["g1"]}), ctx.rosters, aliases, fb)
    pid = m["player_id"].iloc[0]
    if pd.isna(pid):
        notes.append(f"DF goalie feed: {row['goalie']} unmatched")
        return None
    if int(pid) in exclude:
        notes.append(f"DF goalie feed names unavailable {row['goalie']}; ignored")
        return None
    return int(pid), ("DF_CONFIRMED" if row["status"] == "Confirmed" else "DF_LIKELY")


def _prev_games(ctx: _Ctx, team: str) -> list[int]:
    s = schedule()
    if not len(s):
        return []
    m = s[((s["home"] == team) | (s["away"] == team)) & (s["date"] < ctx.date)]
    return [int(x) for x in m.sort_values(["date", "game_id"])["game_id"]][::-1]


def _fallback_skaters(ctx: _Ctx, team: str, exclude: set, notes: list):
    F, D = [], []
    prev = _prev_games(ctx, team) if ctx.use_api else []
    base = None
    for gid in prev[:3]:
        box = boxscore(gid, final=True)
        if not box:
            continue
        side = "home" if (box.get("homeTeam") or {}).get("abbrev") == team else "away"
        b = box_side(box, side)
        if b:
            base = (gid, str(box.get("gameDate", "")), b)
            break
    toi = prior_toi()
    if base:
        gid, gdate, b = base
        F = [i for i in sorted(b["F"], key=lambda i: -b["toi"].get(i, 0)) if i not in exclude]
        D = [i for i in sorted(b["D"], key=lambda i: -b["toi"].get(i, 0)) if i not in exclude]
        # drop players no longer on the team, but only if the roster snapshot is newer than that game
        if ctx.rday >= gdate and ctx.rday != "2026-08":
            on = set(int(x) for x in ctx.roster_team(team)["player_id"])
            gone = [i for i in F + D if i not in on]
            if gone:
                notes.append(f"dropped {len(gone)} no longer on roster {ctx.rday}")
            F, D = [i for i in F if i in on], [i for i in D if i in on]
        notes.append(f"base: dressed in last game {gid} ({gdate})")
    else:
        if prev:
            notes.append(f"no boxscore for the last {min(3, len(prev))} game(s); using roster")
        r = ctx.roster_team(team)
        r = r[~r["player_id"].isin(exclude)].copy()
        r["toi"] = r["player_id"].map(lambda i: toi.get(int(i), 0))
        r = r.sort_values(["toi", "player_id"], ascending=[False, True])
        F = [int(i) for i in r.loc[r["pos"].isin(FWD_POS), "player_id"]][:N_F]
        D = [int(i) for i in r.loc[r["pos"].eq("D"), "player_id"]][:N_D]
        notes.append(f"base: roster {ctx.rday} skaters ranked by 2025-26 TOI")
    F, D = F[:N_F], D[:N_D]
    return _top_up(ctx, team, F, D, exclude, notes)


def _fallback_goalie(ctx: _Ctx, team: str, exclude: set, notes: list) -> int | None:
    cands = [int(i) for i in ctx.roster_team(team).query("pos == 'G'")["player_id"] if int(i) not in exclude]
    season = club_starts(team) if (ctx.use_api and _prev_games(ctx, team)) else {}
    prior = prior_starts()
    if not cands:
        notes.append("no available roster goalie")
        return None
    best = max(cands, key=lambda i: (season.get(i, 0), prior.get(i, 0), -i))
    how = ("2026-27 starts to date" if season else "2025-26 starts")
    notes.append(f"goalie by {how} ({season.get(best, prior.get(best, 0))})")
    return best


def _side(ctx: _Ctx, team: str, side: str) -> dict:
    notes: list[str] = []
    out = {"team": team, "forwards": [], "defense": [], "skaters": [], "goalie": None,
           "lineup_source": "FALLBACK", "goalie_source": "FALLBACK", "notes": notes}
    # (a) NHL boxscore
    try:
        b = box_side(ctx.box, side) if ctx.box else None
        if b and len(b["F"]) >= 10 and len(b["D"]) >= 5:
            out.update(forwards=b["F"], defense=b["D"], lineup_source="NHL_API")
            if b["starter"]:
                out.update(goalie=b["starter"], goalie_source="NHL_API")
        elif ctx.use_api:
            notes.append("boxscore lineup not published")
    except Exception as e:  # noqa: BLE001
        notes.append(f"boxscore parse failed ({type(e).__name__}: {e})")

    exclude = set(ctx.unavail)
    if out["lineup_source"] != "NHL_API" or out["goalie"] is None:
        exclude |= _df_injured(ctx, team, notes)
        exclude |= _mp_injured(ctx, team, notes)

    g_cands: list[int] = []
    # (b) DailyFaceoff lines
    if out["lineup_source"] != "NHL_API":
        try:
            r = _df_lineup(ctx, team, exclude, notes)
            if r:
                F, D, src, g_cands = r
                F, D = _top_up(ctx, team, F, D, exclude, notes)
                out.update(forwards=F, defense=D, lineup_source=src)
        except Exception as e:  # noqa: BLE001
            notes.append(f"DF lines failed ({type(e).__name__}: {e})")
    # (c) fallback
    if out["lineup_source"] == "FALLBACK":
        try:
            F, D = _fallback_skaters(ctx, team, exclude, notes)
            out.update(forwards=F, defense=D)
        except Exception as e:  # noqa: BLE001
            notes.append(f"fallback skaters failed ({type(e).__name__}: {e})")

    # goalie
    if out["goalie"] is None:
        try:
            r = _df_goalie_feed(ctx, team, exclude, notes)
            if r:
                out.update(goalie=r[0], goalie_source=r[1])
        except Exception as e:  # noqa: BLE001
            notes.append(f"DF goalie feed failed ({type(e).__name__}: {e})")
    if out["goalie"] is None and out["lineup_source"].startswith("DF_"):
        g = [i for i in g_cands if i not in exclude]
        if g:
            out.update(goalie=g[0], goalie_source="DF_LINES")
    if out["goalie"] is None:
        try:
            out["goalie"] = _fallback_goalie(ctx, team, exclude, notes)
        except Exception as e:  # noqa: BLE001
            notes.append(f"fallback goalie failed ({type(e).__name__}: {e})")
        out["goalie_source"] = "FALLBACK"

    out["skaters"] = list(out["forwards"]) + list(out["defense"])
    out["n_f"], out["n_d"] = len(out["forwards"]), len(out["defense"])
    return out


def resolve(game_id, date, home, away, snapshot_dir=None, *, rosters_date: str | None = None,
            use_api: bool = True, as_of: datetime | None = None) -> dict:
    """Lineups for both sides of one game; never raises (see module docstring)."""
    base = {"game_id": int(game_id), "date": str(date), "start_utc": None, "as_of": None,
            "roster_snapshot": None, "notes": []}
    try:
        ctx = _Ctx(game_id, date, home, away, snapshot_dir, rosters_date, use_api, as_of)
        base.update(start_utc=ctx.start.isoformat() if ctx.start else None,
                    as_of=ctx.as_of.isoformat(), roster_snapshot=ctx.rday, notes=ctx.notes_game)
    except Exception as e:  # noqa: BLE001
        base["notes"].append(f"context failed ({type(e).__name__}: {e})")
        for side, team in (("home", home), ("away", away)):
            base[side] = {"team": team, "forwards": [], "defense": [], "skaters": [], "goalie": None,
                          "lineup_source": "FALLBACK", "goalie_source": "FALLBACK", "n_f": 0, "n_d": 0,
                          "notes": ["no context; empty lineup"]}
        return base
    for side, team in (("home", home), ("away", away)):
        try:
            base[side] = _side(ctx, team, side)
        except Exception as e:  # noqa: BLE001
            base[side] = {"team": team, "forwards": [], "defense": [], "skaters": [], "goalie": None,
                          "lineup_source": "FALLBACK", "goalie_source": "FALLBACK", "n_f": 0, "n_d": 0,
                          "notes": [f"side failed ({type(e).__name__}: {e})"]}
    return base


def games_on(date: str, use_api: bool = True) -> list[tuple[int, str, str]]:
    """[(game_id, home, away)] on a date: NHL score endpoint, else the local schedule."""
    if use_api:
        gs = [g for g in score(date) if int(g.get("gameType", 2)) == 2 and g.get("gameDate", date) == date]
        if gs:
            return [(int(g["id"]), g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]) for g in gs]
    s = schedule()
    s = s[s["date"] == date] if len(s) else s
    return [(int(r.game_id), r.home, r.away) for r in s.itertuples()]


# ------------------------------------------------------------------ CLI
def _names(rosters: pd.DataFrame) -> dict[int, str]:
    out = {}
    try:
        aug = load_august()
        for fr in (aug, rosters):
            out.update({int(r.player_id): f"{r.first} {r.last}" for r in fr.itertuples()})
    except Exception:  # noqa: BLE001
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description="Resolve lineups for the games on a date.")
    ap.add_argument("--date", default=datetime.now().date().isoformat())
    ap.add_argument("--game", type=int, help="only this game id (its date is taken from the schedule)")
    ap.add_argument("--snapshot-dir", help="read DailyFaceoff files only from this directory")
    ap.add_argument("--rosters", help="roster snapshot date to use (default: latest on/before the game date)")
    ap.add_argument("--no-api", action="store_true", help="offline: no NHL API calls")
    ap.add_argument("--json", help="write the resolved dicts to this JSON file")
    ap.add_argument("--ids", action="store_true", help="also print the skater ids")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    use_api = not a.no_api
    if a.game:
        s = schedule()
        m = s[s["game_id"] == a.game]
        if not len(m):
            sys.exit(f"game {a.game} not in the 2026-27 schedule")
        date = m.iloc[0]["date"]
        games = [(a.game, m.iloc[0]["home"], m.iloc[0]["away"])]
    else:
        date = a.date
        games = games_on(date, use_api)
    if not games:
        print(f"no regular-season games on {date}")
        return
    res = [resolve(g, date, h, aw, a.snapshot_dir, rosters_date=a.rosters, use_api=use_api) for g, h, aw in games]
    names = _names(_rosters(a.rosters or date)[1])
    rows = []
    for r in res:
        for side in ("away", "home"):
            s = r[side]
            rows.append({"game_id": r["game_id"], "side": side, "team": s["team"],
                         "lineup": s["lineup_source"], "F": s["n_f"], "D": s["n_d"],
                         "goalie_src": s["goalie_source"], "goalie": s["goalie"],
                         "goalie_name": names.get(s["goalie"], "-") if s["goalie"] else "-",
                         "why": " | ".join(s["notes"])})
    t = pd.DataFrame(rows)
    print(f"{date}: {len(res)} games; roster snapshot {res[0].get('roster_snapshot')}; "
          f"as of {res[0].get('as_of')}")
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(t.drop(columns="why").to_string(index=False))
        print()
        for x in rows:
            print(f"{x['game_id']} {x['team']}: {x['why']}")
    if a.ids:
        for r in res:
            for side in ("away", "home"):
                s = r[side]
                print(f"{r['game_id']} {s['team']} F {s['forwards']} D {s['defense']} G {s['goalie']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
