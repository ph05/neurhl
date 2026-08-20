"""NeurHL A6 — player_landing careers for EVERY dressed player (gap-fill).

EDA-05 finding: data/raw/player_landing/ covers draft picks only (66% of
dressed player-seasons, 58% of rookies) because it was fetched from
draft_join.csv. The api-web player/{id}/landing endpoint works for any
playerId, so this fetcher diffs the set of players ever dressed in the PBP
corpus (scan_pbp player_appearances cache) against existing files and fetches
the missing careers into the same directory. Closes the career-encoder
cold-start gap to ~100%.
"""
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import RAW, TENSORS, UA  # noqa: E402

DEST = RAW / "player_landing"
CACHE = TENSORS / "_edacache"


def dressed_ids() -> set:
    """Union of all corpus-dressed players = vocab_2027 (JSON + HTM eras)."""
    v = TENSORS / "vocab_2027.json"
    if v.exists():
        return {int(k) for k in json.loads(v.read_text())}
    import pandas as pd
    app = pd.read_parquet(CACHE / "player_appearances.parquet")
    return set(app[app.dressed > 0].player_id.unique().tolist())


def fetch_one(pid: int) -> str:
    dest = DEST / f"{pid}.json.gz"
    if dest.exists() and dest.stat().st_size > 100:
        return "cached"
    url = f"https://api-web.nhle.com/v1/player/{pid}/landing"
    for attempt in range(4):
        try:
            time.sleep(0.2)
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
    t0 = time.time()
    ids = sorted(dressed_ids())
    have = {int(p.name.split(".")[0]) for p in DEST.glob("*.json.gz")}
    missing = [i for i in ids if i not in have]
    print(f"dressed players: {len(ids)}; already on disk: {len(ids) - len(missing)}; "
          f"fetching {len(missing)}")
    counts: dict = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(fetch_one, missing):
            counts[res] = counts.get(res, 0) + 1
    print(f"{counts}  [{time.time() - t0:.0f}s]")
    if counts.get("fail"):
        print(f"  WARNING: {counts['fail']} failures (rerun to resume)")
    print("done")


if __name__ == "__main__":
    main()
