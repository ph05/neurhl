"""Fetch NHL EDGE skater-detail (PLAN_V6 D6), seasons 2022-2026, GP>=20 skaters.

Report-only + snapshot archive (EDGE history is not guaranteed to stay served).
data/raw/edge_players/<playerId>_<season_end>.json.gz (gitignored); derived
edge_players.csv committed by build_v6.py.
"""
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
DEST = RAW / "edge_players"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def targets() -> list[tuple[int, int]]:
    out = []
    for y in range(2021, 2026):          # start-year -> season_end y+1
        f = RAW / f"mp_skaters_{y}.csv"
        if not f.exists():
            continue
        sk = pd.read_csv(f, usecols=["playerId", "situation", "games_played"])
        sk = sk[(sk.situation == "all") & (sk.games_played >= 20)]
        out += [(int(p), y + 1) for p in sk.playerId.unique()]
    return sorted(set(out))


def fetch_one(job) -> str:
    pid, end = job
    dest = DEST / f"{pid}_{end}.json.gz"
    if dest.exists() and dest.stat().st_size > 100:
        return "cached"
    season = f"{end - 1}{end}"
    url = f"https://api-web.nhle.com/v1/edge/skater-detail/{pid}/{season}/2"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                tmp = dest.with_suffix(".part")
                with gzip.open(tmp, "wt") as f:
                    json.dump(r.json(), f, separators=(",", ":"))
                tmp.rename(dest)
                return "ok"
            if r.status_code == 404:
                return "404"
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return "fail"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    jobs = targets()
    print(f"{len(jobs)} player-seasons")
    counts: dict = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(fetch_one, jobs):
            counts[res] = counts.get(res, 0) + 1
    print(f"{counts}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
