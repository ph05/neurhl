"""Fetch NHL shift charts (PLAN_V6 D1), regular season, season_end 2010-2026.

One request per game (api.nhle.com stats/rest shiftcharts), gzipped to
data/raw/shifts/<season_end>/<gameId>.json.gz (gitignored; derived aggregates are
committed). Resumable; seasons the endpoint lacks are tolerated (empty -> skip).
"""
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "shifts"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
SEASONS = [f"{y}{y + 1}" for y in range(2009, 2026)]   # season_end 2010..2026


def game_ids(season: str) -> list[int]:
    url = (f"https://api.nhle.com/stats/rest/en/game?limit=-1&"
           f"cayenneExp=season={season}%20and%20gameType=2")
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return sorted(g["id"] for g in r.json()["data"])


def fetch_one(args) -> str:
    gid, dest = args
    if dest.exists() and dest.stat().st_size > 200:
        return "cached"
    url = (f"https://api.nhle.com/stats/rest/en/shiftcharts?"
           f"cayenneExp=gameId={gid}")
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if not data:
                    return "empty"
                tmp = dest.with_suffix(".part")
                with gzip.open(tmp, "wt") as f:
                    json.dump(data, f, separators=(",", ":"))
                tmp.rename(dest)
                return "ok"
            if r.status_code == 404:
                return "404"
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return "fail"


def main():
    t0 = time.time()
    for season in SEASONS:
        end = int(season[4:])
        sdir = DEST / str(end)
        sdir.mkdir(parents=True, exist_ok=True)
        ids = game_ids(season)
        jobs = [(gid, sdir / f"{gid}.json.gz") for gid in ids]
        counts: dict = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in ex.map(fetch_one, jobs):
                counts[res] = counts.get(res, 0) + 1
        print(f"{end}: {len(ids)} games -> {counts}  [{time.time() - t0:.0f}s]")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
