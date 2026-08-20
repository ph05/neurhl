"""NeurHL EDA shared scanner — one multiprocessing pass over the PBP corpus.

Reads every game in data/raw/pbp/ (+ data/raw/pbp_po/) once and caches compact
aggregate tables under neurhl/data/tensors/_edacache/ (gitignored) for the six
EDA chapters:

  events_summary.parquet     (season_end, game_type, event_type) -> n, n_xy
  situation_codes.parquet    (season_end, situationCode) -> n
  shot_events.parquet        one row per shot-family event (coords, actors, venue)
  game_summary.parquet       one row per game (scores, roster counts, OT/SO, venue)
  player_appearances.parquet (season_end, playerId) -> dressed, event_mentions

Rerun with --force to rebuild. Standalone: uv run --no-project --python 3.12
--with numpy --with "pandas<3" --with pyarrow python neurhl/eda/scan_pbp.py
"""
import gzip
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402

CACHE = TENSORS / "_edacache"
SHOT_TYPES = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
ACTOR_KEYS = ("scoringPlayerId", "shootingPlayerId", "winningPlayerId",
              "hittingPlayerId", "playerId", "committedByPlayerId",
              "blockingPlayerId", "losingPlayerId", "hitteePlayerId",
              "assist1PlayerId", "assist2PlayerId", "drawnByPlayerId",
              "goalieInNetId")


def scan_game(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        g = json.load(f)
    season_end = int(str(g["season"])[4:])
    gtype = g["gameType"]
    home, away = g["homeTeam"], g["awayTeam"]
    ev_counter: Counter = Counter()          # (event_type,) -> [n, n_xy] flattened below
    ev_xy: Counter = Counter()
    sit: Counter = Counter()
    mentions: Counter = Counter()
    shots = []
    n_xy_total = 0
    for p in g.get("plays", []):
        d = p.get("details", {}) or {}
        et = p.get("typeDescKey", "?")
        has_xy = "xCoord" in d and "yCoord" in d
        ev_counter[et] += 1
        if has_xy:
            ev_xy[et] += 1
            n_xy_total += 1
        sc = p.get("situationCode")
        if sc is not None:
            sit[sc] += 1
        for k in ACTOR_KEYS:
            pid = d.get(k)
            if pid:
                mentions[pid] += 1
        if et in SHOT_TYPES:
            shots.append((
                season_end, gtype, g["id"], home["abbrev"],
                et, d.get("shotType", ""),
                d.get("xCoord"), d.get("yCoord"), d.get("zoneCode", ""),
                d.get("eventOwnerTeamId") == home["id"],
                p.get("homeTeamDefendingSide", ""),
                p.get("periodDescriptor", {}).get("number", 0),
                p.get("periodDescriptor", {}).get("periodType", ""),
                d.get("shootingPlayerId") or d.get("scoringPlayerId") or 0,
                d.get("goalieInNetId") or 0,
            ))
    roster = g.get("rosterSpots", [])
    pos = Counter(r.get("positionCode", "?") for r in roster)
    dressed = [r["playerId"] for r in roster if r.get("playerId")]
    outcome = g.get("gameOutcome", {}) or {}
    return {
        "season_end": season_end, "gtype": gtype,
        "ev": dict(ev_counter), "ev_xy": dict(ev_xy), "sit": dict(sit),
        "mentions": dict(mentions), "dressed": dressed, "shots": shots,
        "game_row": (g["id"], season_end, gtype, g.get("gameDate", ""),
                     home["abbrev"], away["abbrev"],
                     home.get("score"), away.get("score"),
                     len(g.get("plays", [])), n_xy_total, len(roster),
                     pos.get("C", 0) + pos.get("L", 0) + pos.get("R", 0),
                     pos.get("D", 0), pos.get("G", 0),
                     outcome.get("lastPeriodType", ""),
                     g.get("regPeriods", 3),
                     (g.get("venue", {}) or {}).get("default", "")),
    }


def main(force: bool = False):
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / "game_summary.parquet"
    if done.exists() and not force:
        print("cache exists (use --force to rebuild)")
        return
    paths = sorted((RAW / "pbp").glob("*/*.json.gz")) + \
        sorted((RAW / "pbp_po").glob("*/*.json.gz"))
    print(f"scanning {len(paths)} games ...")
    ev: Counter = Counter()
    ev_xy: Counter = Counter()
    sit: Counter = Counter()
    app_dressed: Counter = Counter()
    app_mentions: Counter = Counter()
    shots, game_rows = [], []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_game, paths, chunksize=64)):
            se, gt = r["season_end"], r["gtype"]
            for et, n in r["ev"].items():
                ev[(se, gt, et)] += n
            for et, n in r["ev_xy"].items():
                ev_xy[(se, gt, et)] += n
            for sc, n in r["sit"].items():
                sit[(se, sc)] += n
            for pid, n in r["mentions"].items():
                app_mentions[(se, pid)] += n
            for pid in r["dressed"]:
                app_dressed[(se, pid)] += 1
            shots.extend(r["shots"])
            game_rows.append(r["game_row"])
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(paths)}")

    df = pd.DataFrame([(k[0], k[1], k[2], n, ev_xy.get(k, 0))
                       for k, n in ev.items()],
                      columns=["season_end", "game_type", "event_type", "n", "n_xy"])
    df.to_parquet(CACHE / "events_summary.parquet", index=False)

    pd.DataFrame([(k[0], k[1], n) for k, n in sit.items()],
                 columns=["season_end", "situation_code", "n"]
                 ).to_parquet(CACHE / "situation_codes.parquet", index=False)

    pd.DataFrame(shots, columns=[
        "season_end", "game_type", "game_id", "venue_team", "event_type",
        "shot_type", "x", "y", "zone", "home_event", "defending_side",
        "period", "period_type", "shooter", "goalie",
    ]).to_parquet(CACHE / "shot_events.parquet", index=False)

    keys = set(app_dressed) | set(app_mentions)
    pd.DataFrame([(k[0], k[1], app_dressed.get(k, 0), app_mentions.get(k, 0))
                  for k in keys],
                 columns=["season_end", "player_id", "dressed", "event_mentions"]
                 ).to_parquet(CACHE / "player_appearances.parquet", index=False)

    pd.DataFrame(game_rows, columns=[
        "game_id", "season_end", "game_type", "date", "home", "away",
        "home_score", "away_score", "n_plays", "n_xy", "n_roster",
        "n_f", "n_d", "n_g", "last_period_type", "reg_periods", "venue",
    ]).to_parquet(CACHE / "game_summary.parquet", index=False)
    print(f"cached to {CACHE}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
