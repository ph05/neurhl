"""Download the per-game box scores HatTrick adds to the repository's data.

Source: the sportsdataverse/fastRhockey-data mirror of the NHL API
(github.com/sportsdataverse/fastRhockey-data), files nhl/{team_box,
player_box, schedules}/parquet/*_<season>.parquet, into data/raw/fastrhockey/
(gitignored). Each file is checked against the SHA-256 the 2026-27 freeze
recorded in output/freeze_2027/manifest_2027.json; a mismatch is reported,
not silently accepted.

Run: python3 -m hattrick.fetch_boxes
"""
from __future__ import annotations

import hashlib
import json

import requests

from hattrick import config as C

BASE = "https://raw.githubusercontent.com/sportsdataverse/fastRhockey-data/main/nhl"
DEST = C.ROOT / "data" / "raw" / "fastrhockey"
KINDS = {"team_box": "team_box_{y}.parquet", "player_box": "player_box_{y}.parquet",
         "schedules": "nhl_schedule_{y}.parquet"}


def recorded() -> dict:
    m = json.loads((C.OUT / f"freeze_{C.TARGET_SEASON}" / f"manifest_{C.TARGET_SEASON}.json").read_text())
    return next(v for k, v in m["added_inputs"].items() if k.startswith("data/raw/fastrhockey"))


def main():
    want = recorded()
    DEST.mkdir(parents=True, exist_ok=True)
    bad = []
    for kind, pat in KINDS.items():
        for y in range(2008, 2027):
            name = pat.format(y=y)
            if name not in want:
                continue
            f = DEST / name
            if not f.exists():
                r = requests.get(f"{BASE}/{kind}/parquet/{name}", timeout=120)
                r.raise_for_status()
                f.write_bytes(r.content)
            ok = hashlib.sha256(f.read_bytes()).hexdigest() == want[name]
            print(f"{'ok ' if ok else 'BAD'} {name}")
            bad += [] if ok else [name]
    if bad:
        raise SystemExit(f"{len(bad)} files differ from the freeze manifest: {bad}")


if __name__ == "__main__":
    main()
