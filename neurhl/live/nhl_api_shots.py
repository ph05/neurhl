"""NeurHL LIVE: MoneyPuck-compatible shot rows built from the NHL play-by-play.

Fallback source for mp_shots_S while MoneyPuck's in-season shot file
(shots_{S-1}.zip) is not published. From the gamecenter play-by-play the
ingest already downloads (data/raw/pbp[_po]/S/<gid>.json.gz) and the game's
shift chart (data/raw/shifts/S/<gid>.json.gz) it builds one row per unblocked
attempt (shot-on-goal, missed-shot, goal; no shootout) with MoneyPuck's column
names, units and conventions, so models/xg.build_features, the era covariates,
build_stream and ingest_2027.build_xg read it exactly as they read
parse_mp_shots output. The conventions were reverse-engineered on 2025-26,
where both sources exist (neurhl/live/test_nhl_api_shots.py):

  geometry      to the NEAREST net, whatever the attacking direction:
                dx = 89 - |x| (1 when |x| is exactly 89), shotDistance =
                hypot(dx, y), shotAngle = degrees(atan(y' / |dx|)) with y' = -y
                when x < 0; shotAngleAdjusted = |shotAngle|. Missing
                coordinates are 0, 0 (89 ft, 0 degrees), as in MoneyPuck
  last event    the previous play in feed order, skipping stoppages other than
                coach's challenges (CHL); lastEventTeam is the play's owner
                (for a blocked shot, the shooting team); coordinates 0 when the
                play has none. timeSinceLastEvent, distanceFromLastEvent
                (raw coordinates), speedFromLastEvent (distance / time, the
                distance itself at 0 s), timeSinceFaceoff (last faceoff in feed
                order)
  same team     MoneyPuck credits a blocked shot to the BLOCKING team, so the
                "same team" tests below treat a block by the shooting team
                (of an opponent's shot) as its own event
  last shot     lastEventShotDistance / lastEventShotAngle: geometry of the
                last event if it is a SHOT/MISS by the shooting team, else 0
  rebound       shotRebound: last event a SHOT/MISS by the shooting team within
                3 s. shotAnglePlusRebound: |shotAngle - lastEventShotAngle| when
                the last event is a SHOT/MISS/BLOCK of the shooting team (any
                time gap), else 0. shotAngleReboundRoyalRoad: shotAngle and
                lastEventShotAngle of opposite sign
  rush          last event a SHOT/MISS/BLOCK of the shooting team within 4 s
                and not in the offensive zone of the shot's (nearest) net:
                x * sign(shot x) < 25
  state         skaters and empty nets from the play's situationCode (away
                goalie, away skaters, home skaters, home goalie; the shift chart
                when a play has none); score before the play; shotOnEmptyNet =
                the net shot at is empty. Where MoneyPuck's empty-net flag or
                skater count differs on 2025-26, the shift charts side with the
                situationCode in 337 of 356 cases
  shooter       position from the game's rosterSpots; handedness from
                handedness() (MoneyPuck history, player landing files, roster
                snapshots); offWing = left shot with y' < 0 or right shot with
                y' > 0; shooterTimeOnIce = seconds since the start of the
                shooter's shift (start < t <= end, periods 1-4)

Not reproduced (left NaN, none is read by the xG model or any live builder):
the penalty clocks, the on-ice forward/defence counts, the team rest and
time-on-ice aggregates, and the post-shot outcome flags (shotGeneratedRebound,
shotGoalieFroze, shotPlay*). timeUntilNextEvent is approximate (next play kept
by the last-event rule). shotID is synthetic: (game number) * 1000 + the shot's
index in the game, so it never collides with MoneyPuck's season counter.

Library for neurhl/live/ingest_2027.py; the CLI prints per-game counts:
  python neurhl/live/nhl_api_shots.py --season 2026 [--games GID ...] [--out FILE]
"""
import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NRL = Path(__file__).resolve().parents[1]
if str(NRL) not in sys.path:
    sys.path.insert(0, str(NRL))
import common  # noqa: E402

UNBLOCKED = {"shot-on-goal": "SHOT", "missed-shot": "MISS", "goal": "GOAL"}
CATEGORY = {"faceoff": "FAC", "shot-on-goal": "SHOT", "missed-shot": "MISS",
            "blocked-shot": "BLOCK", "goal": "GOAL", "hit": "HIT", "giveaway": "GIVE",
            "takeaway": "TAKE", "penalty": "PENL", "delayed-penalty": "DELPEN",
            "period-start": "PSTR", "period-end": "PEND", "game-end": "GEND"}
SHOT_TYPE = {"wrist": "WRIST", "snap": "SNAP", "slap": "SLAP", "backhand": "BACK",
             "tip-in": "TIP", "deflected": "DEFL", "wrap-around": "WRAP"}
GOAL_LINE = 89
BLUE_LINE = 25
REBOUND_SEC = 3
RUSH_SEC = 4
# parse_mp_shots output, in its column order (MoneyPuck's CSV order)
COLUMNS = [
    "shotID", "averageRestDifference", "awayEmptyNet", "awayPenalty1Length",
    "awayPenalty1TimeLeft", "awaySkatersOnIce", "awayTeamGoals",
    "defendingTeamAverageTimeOnIce", "defendingTeamAverageTimeOnIceOfDefencemen",
    "defendingTeamAverageTimeOnIceOfDefencemenSinceFaceoff",
    "defendingTeamAverageTimeOnIceOfForwards",
    "defendingTeamAverageTimeOnIceOfForwardsSinceFaceoff",
    "defendingTeamAverageTimeOnIceSinceFaceoff", "defendingTeamDefencemenOnIce",
    "defendingTeamForwardsOnIce", "defendingTeamMaxTimeOnIce",
    "defendingTeamMaxTimeOnIceOfDefencemen",
    "defendingTeamMaxTimeOnIceOfDefencemenSinceFaceoff",
    "defendingTeamMaxTimeOnIceOfForwards", "defendingTeamMaxTimeOnIceOfForwardsSinceFaceoff",
    "defendingTeamMaxTimeOnIceSinceFaceoff", "defendingTeamMinTimeOnIce",
    "defendingTeamMinTimeOnIceOfDefencemen",
    "defendingTeamMinTimeOnIceOfDefencemenSinceFaceoff",
    "defendingTeamMinTimeOnIceOfForwards", "defendingTeamMinTimeOnIceOfForwardsSinceFaceoff",
    "defendingTeamMinTimeOnIceSinceFaceoff", "distanceFromLastEvent", "event", "game_id",
    "goal", "goalieIdForShot", "homeEmptyNet", "homePenalty1Length", "homePenalty1TimeLeft",
    "homeSkatersOnIce", "homeTeamGoals", "isHomeTeam", "isPlayoffGame", "lastEventCategory",
    "lastEventShotAngle", "lastEventShotDistance", "lastEventTeam", "lastEventxCord",
    "lastEventyCord", "offWing", "period", "playerPositionThatDidEvent", "season",
    "shooterLeftRight", "shooterPlayerId", "shooterTimeOnIce", "shooterTimeOnIceSinceFaceoff",
    "shootingTeamAverageTimeOnIce", "shootingTeamAverageTimeOnIceOfDefencemen",
    "shootingTeamAverageTimeOnIceOfDefencemenSinceFaceoff",
    "shootingTeamAverageTimeOnIceOfForwards",
    "shootingTeamAverageTimeOnIceOfForwardsSinceFaceoff",
    "shootingTeamAverageTimeOnIceSinceFaceoff", "shootingTeamDefencemenOnIce",
    "shootingTeamForwardsOnIce", "shootingTeamMaxTimeOnIce",
    "shootingTeamMaxTimeOnIceOfDefencemen",
    "shootingTeamMaxTimeOnIceOfDefencemenSinceFaceoff",
    "shootingTeamMaxTimeOnIceOfForwards", "shootingTeamMaxTimeOnIceOfForwardsSinceFaceoff",
    "shootingTeamMaxTimeOnIceSinceFaceoff", "shootingTeamMinTimeOnIce",
    "shootingTeamMinTimeOnIceOfDefencemen",
    "shootingTeamMinTimeOnIceOfDefencemenSinceFaceoff",
    "shootingTeamMinTimeOnIceOfForwards", "shootingTeamMinTimeOnIceOfForwardsSinceFaceoff",
    "shootingTeamMinTimeOnIceSinceFaceoff", "shotAngle", "shotAngleAdjusted",
    "shotAnglePlusRebound", "shotAngleReboundRoyalRoad", "shotDistance",
    "shotGeneratedRebound", "shotGoalieFroze", "shotOnEmptyNet", "shotPlayContinuedInZone",
    "shotPlayContinuedOutsideZone", "shotPlayStopped", "shotRebound", "shotRush", "shotType",
    "shotWasOnGoal", "speedFromLastEvent", "team", "teamCode", "time",
    "timeDifferenceSinceChange", "timeSinceFaceoff", "timeSinceLastEvent",
    "timeUntilNextEvent", "xCord", "yCord", "season_end"]
TEXT = ("event", "lastEventCategory", "lastEventTeam", "playerPositionThatDidEvent",
        "shooterLeftRight", "shotType", "team", "teamCode")
INTS = ("shotID", "game_id", "season", "season_end", "isPlayoffGame", "period", "time",
        "goal", "isHomeTeam", "shooterPlayerId", "goalieIdForShot", "xCord", "yCord",
        "homeSkatersOnIce", "awaySkatersOnIce", "homeEmptyNet", "awayEmptyNet",
        "shotOnEmptyNet", "homeTeamGoals", "awayTeamGoals", "shotWasOnGoal", "offWing",
        "lastEventxCord", "lastEventyCord", "timeSinceLastEvent", "timeSinceFaceoff",
        "shotRebound", "shotRush", "shotAngleReboundRoyalRoad")


# ------------------------------------------------------------------ helpers
def mmss(s: str) -> int:
    m, ss = s.split(":")
    return int(m) * 60 + int(ss)


def read_gz(p: Path):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def geometry(x, y):
    """MoneyPuck distance and signed angle, measured to the nearest net."""
    dx = GOAL_LINE - abs(x)
    dx = 1 if dx == 0 else dx
    yn = -y if x < 0 else y
    return float(np.hypot(dx, y)), float(np.degrees(np.arctan(yn / abs(dx))))


def category(p: dict):
    """MoneyPuck event category of a play; None for plays it skips."""
    t = p.get("typeDescKey")
    if t == "stoppage":
        d = p.get("details") or {}
        rs = f"{d.get('reason') or ''} {d.get('secondaryReason') or ''}"
        return "CHL" if "chlg" in rs else None
    return CATEGORY.get(t)


def shift_rows(recs) -> list:
    """Shift chart -> [(team id, player id, t0, t1)] in game seconds, periods 1-4,
    as build_onice.load_shifts reads it."""
    if isinstance(recs, dict):
        recs = recs.get("data", [])
    out = []
    for r in recs or []:
        if r.get("typeCode") != 517 or not r.get("startTime") or not r.get("endTime"):
            continue
        if r["period"] >= 5:
            continue
        t0 = (r["period"] - 1) * 1200 + mmss(r["startTime"])
        t1 = (r["period"] - 1) * 1200 + mmss(r["endTime"])
        if t1 > t0:
            out.append((r.get("teamId"), r["playerId"], t0, t1))
    return out


def state_from_shifts(sh: list, goalies: set, home_id, t: int):
    """(home skaters, away skaters, home net empty, away net empty) at t from the
    shift chart (start < t <= end); used only when a play has no situationCode."""
    on = [(tm, p) for tm, p, a, b in sh if a < t <= b]
    sk = {True: 0, False: 0}
    gk = {True: 0, False: 0}
    for tm, p in on:
        (gk if p in goalies else sk)[tm == home_id] += 1
    return sk[True], sk[False], int(not gk[True]), int(not gk[False])


# --------------------------------------------------------------- handedness
def handedness(S: int, tensors: Path = None, raw: Path = None) -> dict:
    """player id -> 'L' / 'R' from sources that exist before season S's shots:
    MoneyPuck history (mp_shots_s, s < S), the player landing files, then the
    roster snapshots (newest last, so it wins). They agree wherever they
    overlap (checked on 2025-26)."""
    tensors = Path(tensors or common.TENSORS)
    raw = Path(raw or common.RAW)
    hand: dict = {}
    for s in range(2008, S):
        p = tensors / f"mp_shots_{s}.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p, columns=["shooterPlayerId", "shooterLeftRight"])
        m = m[m.shooterLeftRight.isin(["L", "R"]) & m.shooterPlayerId.notna()]
        hand.update(zip(m.shooterPlayerId.astype("int64"), m.shooterLeftRight))
    for p in (raw / "player_landing").glob("*.json.gz"):
        try:
            sc = read_gz(p).get("shootsCatches")
        except (OSError, ValueError):
            continue
        if sc in ("L", "R"):
            hand[int(p.name.split(".")[0])] = sc
    ros = sorted(glob.glob(str(raw / "nhl_roster_*.json"))) + \
        sorted(glob.glob(str(raw / "rosters" / "*" / "nhl_roster_*.json")))
    for p in ros:
        try:
            d = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        for grp in ("forwards", "defensemen", "goalies"):
            for r in d.get(grp, []) if isinstance(d, dict) else []:
                if r.get("shootsCatches") in ("L", "R"):
                    hand[int(r["id"])] = r["shootsCatches"]
    return hand


# ------------------------------------------------------------------ one game
def game_rows(g: dict, shifts=None, hand: dict = None) -> list:
    """Rows (dicts, MoneyPuck columns) for the unblocked attempts of one game."""
    hand = hand or {}
    gid, gtype = int(g["id"]), int(g["gameType"])
    home_id, away_id = g["homeTeam"]["id"], g["awayTeam"]["id"]
    abbrev = {home_id: g["homeTeam"]["abbrev"], away_id: g["awayTeam"]["abbrev"]}
    pos, team_of = {}, {}
    for r in g.get("rosterSpots", []):
        pos[r["playerId"]] = r.get("positionCode")
        team_of[r["playerId"]] = r.get("teamId")
    goalies = {p for p, c in pos.items() if c == "G"}
    sh = shift_rows(shifts) if shifts else []
    on_ice: dict = {}                       # player -> sorted shifts, for shooterTimeOnIce
    for _, p, a, b in sh:
        on_ice.setdefault(p, []).append((a, b))
    on_ice = {p: sorted(v) for p, v in on_ice.items()}
    season = int(g.get("season", 0)) // 10000
    plays = g.get("plays", [])
    ev = []                 # kept plays: (i, cat, time, owner, x, y); no x/y -> 0, 0
    for i, p in enumerate(plays):
        c = category(p)
        if c is None:
            continue
        d = p.get("details") or {}
        per = p.get("periodDescriptor", {}).get("number", 0)
        ev.append((i, c, (per - 1) * 1200 + mmss(p.get("timeInPeriod", "0:00")),
                   d.get("eventOwnerTeamId"), d.get("xCoord") or 0, d.get("yCoord") or 0))
    ev_at = {e[0]: e for e in ev}
    nxt = {ev[k][0]: ev[k + 1] for k in range(len(ev) - 1)}
    rows, last, last_fo, hs, as_, k = [], None, None, 0, 0, 0
    for i, p in enumerate(plays):
        t = p.get("typeDescKey")
        d = p.get("details") or {}
        pd_ = p.get("periodDescriptor", {})
        per = pd_.get("number", 0)
        time_ = (per - 1) * 1200 + mmss(p.get("timeInPeriod", "0:00"))
        shootout = pd_.get("periodType") == "SO" or (gtype == 2 and per >= 5)
        if t in UNBLOCKED and not shootout:
            shooter = d.get("shootingPlayerId") or d.get("scoringPlayerId") or 0
            owner = d.get("eventOwnerTeamId") or team_of.get(shooter)
            is_home = int(owner == home_id)
            x, y = d.get("xCoord") or 0, d.get("yCoord") or 0     # MoneyPuck: missing = 0
            dist, ang = geometry(x, y)
            sit = str(p.get("situationCode") or "").zfill(4)
            if sit.isdigit() and sit != "0000":     # away goalie, away sk, home sk, home goalie
                ag, ask, hsk, hg = (int(c) for c in sit)
                h_sk, a_sk, h_en, a_en = hsk, ask, int(hg == 0), int(ag == 0)
            elif sh:
                h_sk, a_sk, h_en, a_en = state_from_shifts(sh, goalies, home_id, time_)
            else:                                   # never seen: 5 on 5, both goalies in
                h_sk, a_sk, h_en, a_en = 5, 5, 0, 0
            row = {"game_id": gid, "season": season, "season_end": season + 1,
                   "isPlayoffGame": int(gtype == 3), "period": per, "time": time_,
                   "event": UNBLOCKED[t], "goal": int(t == "goal"), "isHomeTeam": is_home,
                   "team": "HOME" if is_home else "AWAY", "teamCode": abbrev.get(owner),
                   "shooterPlayerId": shooter, "goalieIdForShot": d.get("goalieInNetId") or 0,
                   "playerPositionThatDidEvent": pos.get(shooter),
                   "xCord": x, "yCord": y, "shotDistance": dist, "shotAngle": ang,
                   "shotAngleAdjusted": abs(ang), "shotType": SHOT_TYPE.get(d.get("shotType")),
                   "shooterLeftRight": hand.get(shooter),
                   "homeSkatersOnIce": h_sk, "awaySkatersOnIce": a_sk,
                   "homeEmptyNet": h_en, "awayEmptyNet": a_en,
                   "homeTeamGoals": hs, "awayTeamGoals": as_,
                   "shotOnEmptyNet": a_en if is_home else h_en,
                   "shotWasOnGoal": int(t != "missed-shot")}
            yn = -y if x < 0 else y
            hd = row["shooterLeftRight"]
            row["offWing"] = int((hd == "L" and yn < 0) or (hd == "R" and yn > 0))
            row.update({"lastEventxCord": 0, "lastEventyCord": 0, "timeSinceLastEvent": 0,
                        "distanceFromLastEvent": 0.0, "speedFromLastEvent": 0.0,
                        "lastEventShotDistance": 0.0, "lastEventShotAngle": 0.0,
                        "shotRebound": 0, "shotAnglePlusRebound": 0.0,
                        "shotAngleReboundRoyalRoad": 0, "shotRush": 0})
            # last event (MoneyPuck credits a blocked shot to the blocker)
            if last is not None:
                _, lc, lt, lo, lx, ly = last
                own = (lo == owner) if lc != "BLOCK" else (lo is not None and lo != owner)
                shot_own = own and lc in ("SHOT", "MISS")
                tsl = time_ - lt
                dfl = float(np.hypot(x - lx, y - ly))
                ldist, lang = geometry(lx, ly) if shot_own else (0.0, 0.0)
                side = -1 if x < 0 else 1
                row.update({
                    "lastEventCategory": lc, "lastEventxCord": lx, "lastEventyCord": ly,
                    "lastEventTeam": ("HOME" if lo == home_id else "AWAY" if lo == away_id
                                      else None),
                    "timeSinceLastEvent": tsl, "distanceFromLastEvent": dfl,
                    "speedFromLastEvent": dfl / tsl if tsl > 0 else dfl,
                    "lastEventShotDistance": ldist, "lastEventShotAngle": lang,
                    "shotRebound": int(shot_own and 0 <= tsl <= REBOUND_SEC),
                    "shotAnglePlusRebound": (abs(ang - lang) if own and lc in
                                             ("SHOT", "MISS", "BLOCK") else 0.0),
                    "shotAngleReboundRoyalRoad": int(ang * lang < 0),
                    "shotRush": int(own and lc in ("SHOT", "MISS", "BLOCK")
                                    and 0 <= tsl <= RUSH_SEC and lx * side < BLUE_LINE)})
            tsf = time_ - (last_fo if last_fo is not None else (per - 1) * 1200)
            nx = nxt.get(i)
            toi = next((time_ - a for a, b in on_ice.get(shooter, []) if a < time_ <= b), None)
            row.update({"timeSinceFaceoff": tsf,
                        "timeUntilNextEvent": nx[2] - time_ if nx else np.nan,
                        "shooterTimeOnIce": np.nan if toi is None else toi,
                        "shooterTimeOnIceSinceFaceoff": np.nan if toi is None else min(toi, tsf)})
            row["shotID"] = (gid % 1_000_000) * 1000 + k
            k += 1
            rows.append(row)
        if t == "faceoff":
            last_fo = time_
        if t == "goal" and not shootout:
            hs = d.get("homeScore", hs + (d.get("eventOwnerTeamId") == home_id))
            as_ = d.get("awayScore", as_ + (d.get("eventOwnerTeamId") == away_id))
        last = ev_at.get(i, last)
    return rows


def frame(rows: list) -> pd.DataFrame:
    """Rows -> parse_mp_shots-shaped frame (every MoneyPuck column, same order)."""
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df:
            df[c] = None if c in TEXT else np.nan
    df = df[COLUMNS]
    for c in df.columns:
        if c not in TEXT:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in INTS:                          # never missing, integer as in MoneyPuck
        df[c] = df[c].astype("int64")
    return df


# --------------------------------------------------------------- season build
def pbp_file(raw: Path, S: int, gid: int) -> Path:
    src = "pbp" if (gid // 10_000) % 100 == 2 else "pbp_po"
    return raw / src / str(S) / f"{gid}.json.gz"


def build(gids, S: int, raw: Path = None, hand: dict = None) -> pd.DataFrame:
    """MoneyPuck-shaped rows for `gids` of season S from the raw files on disk.
    A game without its play-by-play contributes no rows; a game without its
    shift chart gets NaN shooterTimeOnIce."""
    raw = Path(raw or common.RAW)
    hand = handedness(S, raw=raw) if hand is None else hand
    rows = []
    for gid in sorted(int(g) for g in gids):
        pp = pbp_file(raw, S, gid)
        if not pp.exists():
            continue
        sp = raw / "shifts" / str(S) / f"{gid}.json.gz"
        rows.extend(game_rows(read_gz(pp), read_gz(sp) if sp.exists() else None, hand))
    if not rows:
        return frame([]).iloc[0:0]
    return frame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", type=int, default=2027)
    ap.add_argument("--raw", default=str(common.RAW))
    ap.add_argument("--games", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None, help="write the rows to this parquet file")
    a = ap.parse_args(argv)
    raw = Path(a.raw)
    gids = a.games or sorted(int(p.name.split(".")[0]) for p in
                             (raw / "pbp" / str(a.season)).glob("*.json.gz"))
    df = build(gids, a.season, raw)
    n = df.groupby("game_id").size()
    print(f"{len(df):,} unblocked attempts in {len(n)} games "
          f"(per game {n.mean():.1f}, min {n.min() if len(n) else 0}); "
          f"handedness unknown {df.shooterLeftRight.isna().mean():.2%}, "
          f"shooter TOI missing {df.shooterTimeOnIce.isna().mean():.2%}")
    if a.out:
        df.to_parquet(a.out, index=False)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
