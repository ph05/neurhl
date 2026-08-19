"""Fetch MoneyPuck shot-level data (PLAN_V5 D1), seasons 2007-2025 (start-year keys).

Zips cached to data/raw/mp_shots/shots_<year>.zip (gitignored — derived aggregates
are what gets committed). Data (c) MoneyPuck.com, free for non-commercial use with
attribution.
"""
import sys
import time
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "mp_shots"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
URLS = ("https://moneypuck.com/moneypuck/playerData/shots/shots_{y}.zip",
        "https://peter-tanner.com/moneypuck/downloads/shots_{y}.zip")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for y in range(2007, 2026):
        dest = DEST / f"shots_{y}.zip"
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"{y}: cached ({dest.stat().st_size // 1_000_000}MB)")
            continue
        ok = False
        for url in URLS:
            try:
                with requests.get(url.format(y=y), headers=UA, timeout=120,
                                  stream=True) as r:
                    if r.status_code != 200:
                        print(f"{y}: HTTP {r.status_code} at {url.format(y=y)}")
                        continue
                    tmp = dest.with_suffix(".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    tmp.rename(dest)
                    print(f"{y}: {dest.stat().st_size // 1_000_000}MB")
                    ok = True
                    break
            except requests.RequestException as e:
                print(f"{y}: {e!r}")
        if not ok:
            raise RuntimeError(f"failed {y}")
        sys.stdout.flush()
        time.sleep(2.0)
    print("done")


if __name__ == "__main__":
    main()
