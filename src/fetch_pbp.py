"""Fetch NHL play-by-play JSON (PLAN_V5 D2), regular season, season_end 2012-2026.

Game IDs come from the official game index (api.nhle.com/stats/rest); each game's
play-by-play is fetched from api-web.nhle.com and stored gzipped at
data/raw/pbp/<season_end>/<gameId>.json.gz (gitignored — derived aggregates are what
gets committed). Resumable: existing files are skipped. 8 worker threads, retries
with backoff; a run summary prints per season.
"""
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "pbp"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
SEASONS = [f"{y}{y + 1}" for y in range(2011, 2026)]   # season_end 2012..2026


def game_ids(season: str) -> list[int]:
    url = (f"https://api.nhle.com/stats/rest/en/game?limit=-1&"
           f"cayenneExp=season={season}%20and%20gameType=2")
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return sorted(g["id"] for g in r.json()["data"])


def fetch_one(args) -> str:
    gid, dest = args
    if dest.exists() and dest.stat().st_size > 500:
        return "cached"
    url = f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                blob = r.json()
                if not blob.get("plays"):
                    return "empty"
                tmp = dest.with_suffix(".part")
                with gzip.open(tmp, "wt") as f:
                    json.dump(blob, f, separators=(",", ":"))
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
        print(f"{end}: {len(ids)} games -> {counts}  "
              f"[{time.time() - t0:.0f}s elapsed]")
        sys.stdout.flush()
        if counts.get("fail"):
            print(f"  WARNING {end}: {counts['fail']} failures (rerun to resume)")
    print("done")


if __name__ == "__main__":
    main()
