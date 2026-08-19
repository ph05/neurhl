"""Fetch NHL EDGE team tracking data (PLAN_V5 D4), seasons 2021-22 .. 2025-26.

EDGE exists only from 2021-22, far too short for the 2012-2017 train window, so this
is REPORT-ONLY/EDA input by construction (prereg'd in PLAN_V5 D4). Per team+season:
team-detail (skating speed/distance, shot speed, shot location, zone time) and
team-zone-time-details. Cached to data/raw/edge/<endpoint>_<abbr>_<season_end>.json
(gitignored; derived tables are committed).
"""
import json
import sys
import time
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "edge"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ENDPOINTS = ("team-detail", "team-zone-time-details")


def team_ids() -> dict[str, int]:
    r = requests.get("https://api.nhle.com/stats/rest/en/team", headers=UA, timeout=60)
    r.raise_for_status()
    return {t["triCode"]: t["id"] for t in r.json()["data"]}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    ids = team_ids()
    active = sorted(pdteams())
    for y in range(2021, 2026):          # 20212022 .. 20252026
        season, end = f"{y}{y + 1}", y + 1
        for abbr in active:
            if abbr == "UTA" and end <= 2024:
                continue                  # Utah exists from 2024-25
            if abbr == "ARI" and end >= 2025:
                continue
            for ep in ENDPOINTS:
                dest = DEST / f"{ep}_{abbr}_{end}.json"
                if dest.exists() and dest.stat().st_size > 200:
                    continue
                url = f"https://api-web.nhle.com/v1/edge/{ep}/{ids[abbr]}/{season}/2"
                for attempt in range(3):
                    try:
                        r = requests.get(url, headers=UA, timeout=30)
                        if r.status_code == 200:
                            dest.write_text(json.dumps(r.json()))
                            break
                        if r.status_code == 404:
                            print(f"{end} {abbr} {ep}: 404 (no data)")
                            dest.write_text("{}")
                            break
                    except requests.RequestException as e:
                        print(f"{end} {abbr} {ep}: {e!r}")
                    time.sleep(2 * (attempt + 1))
                time.sleep(0.25)
        print(f"{end}: ok")
        sys.stdout.flush()
    print("done")


def pdteams() -> list[str]:
    import pandas as pd
    ts = pd.read_csv(PROJ / "data/processed/team_seasons.csv")
    cur = set(ts[ts.season_end >= 2022].team) | {"UTA"}
    return sorted(cur)


if __name__ == "__main__":
    main()
