"""Player absence spells from PBP rosterSpots (PLAN_V6 D8). No new fetch.

Per (playerId, season_end, team): games dressed, games missed inside the player's
first-to-last-dressed window on that team, split into injury-like spells
(consecutive missed runs >= 3 team games) and scratch-like (runs 1-2). Trades are
handled by computing windows per team. Writes data/processed/player_absences.csv.
"""
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from players import MP_FRAN

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw" / "pbp"
PROC = PROJ / "data" / "processed"


def parse_season(sdir: Path) -> list[dict]:
    end = int(sdir.name)
    games = []                      # (date, gameId, {team: {playerIds}})
    for f in sorted(sdir.glob("*.json.gz")):
        with gzip.open(f, "rt") as fh:
            g = json.load(fh)
        id2abbr = {g["homeTeam"]["id"]: g["homeTeam"]["abbrev"],
                   g["awayTeam"]["id"]: g["awayTeam"]["abbrev"]}
        dressed: dict = {}
        for rs in g.get("rosterSpots", []):
            t = MP_FRAN.get(id2abbr.get(rs["teamId"], ""), id2abbr.get(rs["teamId"]))
            if t:
                dressed.setdefault(t, set()).add(rs["playerId"])
        games.append((g.get("gameDate", ""), g["id"], dressed))
    games.sort()
    # per-team ordered schedule + dressed sets
    sched: dict = {}
    for _, gid, dressed in games:
        for t, pids in dressed.items():
            sched.setdefault(t, []).append(pids)
    rows = []
    for t, glist in sched.items():
        players = set().union(*glist)
        for p in players:
            mask = [p in pids for pids in glist]
            first, last = mask.index(True), len(mask) - 1 - mask[::-1].index(True)
            window = mask[first:last + 1]
            inj = scr = 0
            run = 0
            for m in window + [True]:
                if not m:
                    run += 1
                else:
                    if run >= 3:
                        inj += run
                    else:
                        scr += run
                    run = 0
            rows.append({"season_end": end, "team": t, "playerId": p,
                         "dressed": sum(window), "window_games": len(window),
                         "injury_spell_games": inj, "scratch_games": scr})
    return rows


def main():
    sdirs = sorted(d for d in RAW.iterdir() if d.is_dir())
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for rr in ex.map(parse_season, sdirs):
            rows += rr
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "player_absences.csv", index=False)
    print(f"player_absences: {len(df)} rows, seasons "
          f"{df.season_end.min()}-{df.season_end.max()}; "
          f"injury-spell share of missed: "
          f"{df.injury_spell_games.sum() / max(df.injury_spell_games.sum() + df.scratch_games.sum(), 1):.3f}")


if __name__ == "__main__":
    main()
