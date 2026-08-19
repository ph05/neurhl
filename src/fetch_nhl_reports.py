"""Fetch NHL stats-rest team aggregate reports (PLAN_V5 D3), season_end 2006-2026.

Official per-season team aggregates (faceoffs, penalties, real-time events, special
teams) — the independent cross-check for PBP- and MoneyPuck-derived tables, plus
pre-2008 coverage where MoneyPuck is absent. One JSON per (report, season) at
data/raw/nhl_reports/<report>_<season_end>.json (gitignored; derived tables are
committed).
"""
import json
import sys
import time
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "nhl_reports"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
REPORTS = ("summary", "faceoffpercentages", "penalties", "realtime",
           "summaryshooting", "powerplay", "penaltykill")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for y in range(2005, 2026):          # season 20052006 .. 20252026
        season, end = f"{y}{y + 1}", y + 1
        for rep in REPORTS:
            dest = DEST / f"{rep}_{end}.json"
            if dest.exists() and dest.stat().st_size > 500:
                continue
            url = (f"https://api.nhle.com/stats/rest/en/team/{rep}?limit=-1&"
                   f"cayenneExp=gameTypeId=2%20and%20seasonId={season}")
            for attempt in range(3):
                try:
                    r = requests.get(url, headers=UA, timeout=60)
                    if r.status_code == 200 and r.json().get("data"):
                        dest.write_text(json.dumps(r.json()["data"]))
                        break
                    print(f"{end} {rep}: HTTP {r.status_code}")
                except requests.RequestException as e:
                    print(f"{end} {rep}: {e!r}")
                time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"failed {end} {rep}")
            time.sleep(0.4)
        print(f"{end}: ok")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
