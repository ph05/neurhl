"""Fetch hockey-reference team season pages (PLAN_V6 D3), 2006-2026.

Team-season URLs are discovered from the cached league pages (handles era codes
MDA/PHX/ARI/ATL/VEG/... without guessing). Head-coach lines are parsed by
build_v6.py. Polite: >=4.5s between requests; ~650 pages, resumable, cached to
data/raw/hr_html/team_<code>_<yyyy>.html (gitignored).
"""
import re
import sys
import time
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "hr_html"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def team_links() -> list[tuple[str, int]]:
    out = set()
    for y in range(2006, 2027):
        p = DEST / f"NHL_{y}.html"
        if not p.exists():
            continue
        for code, yy in re.findall(r'href="/teams/([A-Z]{3})/(\d{4})\.html"',
                                   p.read_text()):
            if int(yy) == y:
                out.add((code, y))
    return sorted(out)


def main():
    links = team_links()
    print(f"{len(links)} team-season pages")
    for code, y in links:
        dest = DEST / f"team_{code}_{y}.html"
        if dest.exists() and dest.stat().st_size > 20_000:
            continue
        url = f"https://www.hockey-reference.com/teams/{code}/{y}.html"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=60)
                if r.status_code == 200:
                    dest.write_text(r.text)
                    break
                print(f"{code} {y}: HTTP {r.status_code}")
                time.sleep(30 * (attempt + 1))
            except requests.RequestException as e:
                print(f"{code} {y}: {e!r}")
                time.sleep(10)
        else:
            raise RuntimeError(f"failed {code} {y}")
        time.sleep(4.5)
        if (links.index((code, y)) + 1) % 50 == 0:
            print(f"...{links.index((code, y)) + 1}/{len(links)}")
            sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
