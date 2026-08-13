"""Fetch MoneyPuck team season summaries (xG etc.), 2007-08 .. 2025-26.

Season key = starting year (2007 => 2007-08). Cached to data/raw/mp_teams_<year>.csv.
Data (c) MoneyPuck.com, free for non-commercial use with attribution.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for season in range(2007, 2026):
        dest = RAW / f"mp_teams_{season}.csv"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"{season}: cached")
            continue
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=30)
                if r.status_code == 200 and r.text.startswith("team"):
                    dest.write_text(r.text)
                    df = pd.read_csv(dest)
                    print(f"{season}: {df['team'].nunique()} teams, {len(df)} rows")
                    break
                print(f"{season}: attempt {attempt}: HTTP {r.status_code}")
            except requests.RequestException as e:
                print(f"{season}: attempt {attempt}: {e!r}")
            time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {season}")
        sys.stdout.flush()
        time.sleep(1.5)
    print("done")


if __name__ == "__main__":
    main()
