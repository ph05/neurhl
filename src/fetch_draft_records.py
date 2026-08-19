"""Fetch the full NHL draft table from records.nhl.com (PLAN_V6 D2 support).

Unlike the api-web draft JSONs, records.nhl.com carries a playerId for EVERY
pick (including players who never reached the NHL) — required so prospect
production ramps are built on the full pick population, not the NHL-survivor
subset. One request per draft year 2006-2025 -> data/raw/nhl_draft_records.json
(committed; ~1MB).
"""
import json
import time
from pathlib import Path

import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "raw" / "nhl_draft_records.json"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
KEEP = ("playerId", "playerName", "draftYear", "overallPickNumber", "roundNumber",
        "triCode", "position", "amateurClubName", "amateurLeague", "birthDate",
        "countryCode")


def main():
    rows = []
    for y in range(2006, 2026):
        url = (f"https://records.nhl.com/site/api/draft?"
               f"cayenneExp=draftYear={y}")
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        rows += [{k: d.get(k) for k in KEEP} for d in data]
        print(f"{y}: {len(data)} picks")
        time.sleep(0.5)
    DEST.write_text(json.dumps(rows))
    print(f"total {len(rows)} picks -> {DEST.name}")


if __name__ == "__main__":
    main()
