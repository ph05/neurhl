"""Fetch hockey-reference coaches pages (PLAN_V6 D3), 2006-2026.

Polite (>= 4.5s between requests), cached to data/raw/hr_html/NHL_<yyyy>_coaches.html
(gitignored; parsed coaches.csv is committed by build_v6.py).
"""
import sys
import time
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "hr_html"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for y in range(2006, 2027):
        dest = DEST / f"NHL_{y}_coaches.html"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"{y}: cached")
            continue
        url = f"https://www.hockey-reference.com/leagues/NHL_{y}_coaches.html"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=60)
                if r.status_code == 200:
                    dest.write_text(r.text)
                    print(f"{y}: {len(r.text) // 1000}KB")
                    break
                print(f"{y}: HTTP {r.status_code} (attempt {attempt})")
                time.sleep(20 * (attempt + 1))
            except requests.RequestException as e:
                print(f"{y}: {e!r}")
                time.sleep(10)
        else:
            raise RuntimeError(f"failed {y}")
        sys.stdout.flush()
        time.sleep(4.5)
    print("done")


if __name__ == "__main__":
    main()
