"""Fetch NHL playoff play-by-play (PLAN_V6 D5), season_end 2012-2026.

Same shape as fetch_pbp.py with gameType=3, stored under data/raw/pbp_po/
(gitignored). Report-only input this plan.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_pbp import SEASONS, UA, fetch_one

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "pbp_po"


def game_ids(season: str) -> list[int]:
    url = (f"https://api.nhle.com/stats/rest/en/game?limit=-1&"
           f"cayenneExp=season={season}%20and%20gameType=3")
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return sorted(g["id"] for g in r.json()["data"])


def main():
    t0 = time.time()
    for season in SEASONS:
        end = int(season[4:])
        sdir = DEST / str(end)
        sdir.mkdir(parents=True, exist_ok=True)
        ids = game_ids(season)
        counts: dict = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in ex.map(fetch_one,
                              [(gid, sdir / f"{gid}.json.gz") for gid in ids]):
                counts[res] = counts.get(res, 0) + 1
        print(f"{end}: {len(ids)} playoff games -> {counts}  [{time.time() - t0:.0f}s]")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
