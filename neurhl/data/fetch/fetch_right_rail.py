"""Fetch NHL gamecenter right-rail records for regular-season games.

api-web.nhle.com/v1/gamecenter/{gid}/right-rail carries, per game: referees
and linesmen, scratches, head coaches, and official team game stats (shots,
power-play opportunities as "goals/opportunities", PIM, hits, blocks,
giveaways, takeaways, faceoffs). NeurHL-G uses the official power-play
opportunity counts as a target and the referees for a walk-forward penalty-rate
feature (PLAN_NeurHL4, section D).

Raw JSON goes to data/raw/right_rail/<season>/<gid>.json.gz (gitignored).
Resumable: existing files are skipped. Two workers, 0.3 s politeness each.

Usage: ... python neurhl/data/fetch/fetch_right_rail.py [--seasons 2012-2026]
"""
import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import RAW, TENSORS, UA  # noqa: E402

URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/right-rail"
OUT = RAW / "right_rail"


def fetch(job):
    season, gid = job
    p = OUT / str(season) / f"{gid}.json.gz"
    if p.exists():
        return "skip"
    for attempt in range(4):
        try:
            r = requests.get(URL.format(gid=gid), headers=UA, timeout=30)
            if r.status_code == 200:
                p.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(p, "wt") as f:
                    json.dump(r.json(), f)
                time.sleep(0.3)
                return "ok"
            if r.status_code == 404:
                return "404"
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2012-2026")
    a, b = map(int, ap.parse_args().seasons.split("-"))
    jobs = []
    for s in range(a, b + 1):
        gc = pd.read_parquet(TENSORS / f"games_ctx_{s}.parquet",
                             columns=["game_id", "game_type"])
        jobs += [(s, int(g)) for g in gc.loc[gc.game_type == 2, "game_id"]]
    counts = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, res in enumerate(ex.map(fetch, jobs), 1):
            counts[res] = counts.get(res, 0) + 1
            if i % 500 == 0:
                print(f"{i:,}/{len(jobs):,} {counts}", flush=True)
    print(f"done {counts}", flush=True)


if __name__ == "__main__":
    main()
