"""Fetch sportsdataverse fastRhockey-data NHL bulk files (PLAN_V5 D6).

Pre-processed NHL data (parquet) from github.com/sportsdataverse/fastRhockey-data:
team_box, player_box, schedules, and processed play-by-play, seasons as published
(team_box 2010-2024 at fetch time). Used as a processed cross-check against the raw
api-web PBP corpus (D2) and for game-level box aggregates without JSON parsing.
Cached to data/raw/fastrhockey/ (pbp gitignored via size; small files committed).
"""
import sys
import time
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "fastrhockey"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
API = "https://api.github.com/repos/sportsdataverse/fastRhockey-data/contents/nhl/{d}/parquet"
RAWBASE = ("https://raw.githubusercontent.com/sportsdataverse/fastRhockey-data/main/"
           "nhl/{d}/parquet/{name}")
DIRS = ("team_box", "player_box", "schedules", "pbp")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        r = requests.get(API.format(d=d), headers=UA, timeout=60)
        r.raise_for_status()
        files = [(x["name"], x.get("size", 0)) for x in r.json()]
        for name, size in sorted(files):
            dest = DEST / name
            if dest.exists() and dest.stat().st_size > 0.9 * size:
                continue
            url = RAWBASE.format(d=d, name=name)
            for attempt in range(3):
                try:
                    with requests.get(url, headers=UA, timeout=300, stream=True) as g:
                        if g.status_code != 200:
                            print(f"{name}: HTTP {g.status_code}")
                            continue
                        tmp = dest.with_suffix(".part")
                        with open(tmp, "wb") as f:
                            for chunk in g.iter_content(1 << 20):
                                f.write(chunk)
                        tmp.rename(dest)
                        print(f"{name}: {dest.stat().st_size // 1000}KB")
                        break
                except requests.RequestException as e:
                    print(f"{name}: {e!r}")
                    time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"failed {name}")
            time.sleep(0.3)
        print(f"{d}: done ({len(files)} files)")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
