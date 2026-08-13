"""Fetch MoneyPuck player-level season summaries (skaters + goalies) and the player bios lookup.

Seasons 2008..2025 by start year (= 2008-09 .. 2025-26). Cached in data/raw/.
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
BASE = "https://moneypuck.com/moneypuck/playerData"


def get(url: str) -> str:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200 and r.text.startswith("playerId"):
                return r.text
            print(f"  attempt {attempt}: HTTP {r.status_code} head={r.text[:40]!r}")
        except requests.RequestException as e:
            print(f"  attempt {attempt}: {e!r}")
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"failed {url}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    # bios lookup
    dest = RAW / "mp_lookup.csv"
    if not (dest.exists() and dest.stat().st_size > 10000):
        dest.write_text(get(f"{BASE}/playerBios/allPlayersLookup.csv"))
        df = pd.read_csv(dest)
        assert {"playerId", "birthDate", "position"} <= set(df.columns)
        print(f"lookup: {len(df)} players, {df.birthDate.notna().mean():.1%} with birthDate")
    else:
        print("lookup: cached")

    for kind in ("skaters", "goalies"):
        for season in range(2008, 2026):
            dest = RAW / f"mp_{kind}_{season}.csv"
            if dest.exists() and dest.stat().st_size > 10000:
                print(f"{kind} {season}: cached")
                continue
            dest.write_text(get(f"{BASE}/seasonSummary/{season}/regular/{kind}.csv"))
            df = pd.read_csv(dest)
            req = {"playerId", "season", "team", "situation", "games_played", "icetime"}
            req |= {"I_F_points", "OnIce_F_xGoals"} if kind == "skaters" else {"xGoals", "goals", "ongoal"}
            missing = req - set(df.columns)
            assert not missing, f"{kind} {season} missing {missing}"
            print(f"{kind} {season}: {df.playerId.nunique()} players, {len(df)} rows")
            sys.stdout.flush()
            time.sleep(1.5)
    print("done")


if __name__ == "__main__":
    main()
