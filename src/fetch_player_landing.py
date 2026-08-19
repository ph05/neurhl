"""Fetch player-landing career stats (PLAN_V6 D2) for joined draft picks.

api-web player/{id}/landing includes seasonTotals across ALL leagues (junior,
AHL, European, NHL) — the raw material for production-based prospect ramps.
Players = every playerId in data/raw/nhl_draft_records.json (records.nhl.com —
covers ALL picks, including players who never reached the NHL, so ramps are not
built on the survivor subset). Gzipped to
data/raw/player_landing/<playerId>.json.gz (gitignored).
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
DEST = PROJ / "data" / "raw" / "player_landing"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch_one(pid: int) -> str:
    dest = DEST / f"{pid}.json.gz"
    if dest.exists() and dest.stat().st_size > 200:
        return "cached"
    url = f"https://api-web.nhle.com/v1/player/{pid}/landing"
    for attempt in range(4):
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
    rec = json.loads((PROJ / "data/raw/nhl_draft_records.json").read_text())
    pids = sorted({int(r["playerId"]) for r in rec if r.get("playerId")})
    dj = pd.read_csv(PROJ / "data/processed/draft_join.csv")
    pids = sorted(set(pids) | set(dj.playerId.dropna().astype(int)))
    print(f"{len(pids)} draft-pick playerIds")
    counts: dict = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(fetch_one, pids):
            counts[res] = counts.get(res, 0) + 1
    print(f"{counts}  [{time.time() - t0:.0f}s]")
    if counts.get("fail"):
        print(f"WARNING: {counts['fail']} failures (rerun to resume)")


if __name__ == "__main__":
    main()
