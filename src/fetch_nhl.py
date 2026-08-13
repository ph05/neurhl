"""Fetch from the NHL API: current 2026-27 rosters (post-July-FA snapshots), draft picks
2009-2026, and the REAL 2026-27 schedule. Cached in data/raw/.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
API = "https://api-web.nhle.com/v1"
TEAMS = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA",
         "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
         "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"]
SEASON = "20262027"


def get_json(url: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            print(f"  attempt {attempt}: HTTP {r.status_code} {url}")
        except requests.RequestException as e:
            print(f"  attempt {attempt}: {e!r}")
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed {url}")


def fetch_rosters():
    rows = []
    for t in TEAMS:
        dest = RAW / f"nhl_roster_{t}_{SEASON}.json"
        if dest.exists() and dest.stat().st_size > 500:
            data = json.loads(dest.read_text())
        else:
            data = get_json(f"{API}/roster/{t}/{SEASON}")
            dest.write_text(json.dumps(data))
            time.sleep(0.7)
        n = 0
        for bucket in ("forwards", "defensemen", "goalies"):
            for p in data.get(bucket, []):
                rows.append({
                    "playerId": p["id"], "team": t, "position": p.get("positionCode", "?"),
                    "birthDate": p.get("birthDate"),
                    "first": p.get("firstName", {}).get("default", ""),
                    "last": p.get("lastName", {}).get("default", ""),
                })
                n += 1
        print(f"roster {t}: {n} players")
        sys.stdout.flush()
    df = pd.DataFrame(rows)
    df.to_csv(RAW / f"nhl_rosters_{SEASON}.csv", index=False)
    print(f"rosters: {len(df)} players, {df.team.nunique()} teams")


def fetch_drafts():
    rows = []
    for y in range(2009, 2027):
        dest = RAW / f"nhl_draft_{y}.json"
        if dest.exists() and dest.stat().st_size > 500:
            data = json.loads(dest.read_text())
        else:
            data = get_json(f"{API}/draft/picks/{y}/all")
            dest.write_text(json.dumps(data))
            time.sleep(0.7)
        for p in data.get("picks", []):
            rows.append({"draft_year": y, "round": p.get("round"),
                         "overall": p.get("overallPick"), "team": p.get("teamAbbrev")})
        print(f"draft {y}: {len(data.get('picks', []))} picks")
        sys.stdout.flush()
    df = pd.DataFrame(rows).dropna(subset=["overall"])
    df.to_csv(RAW / "nhl_drafts.csv", index=False)
    print(f"drafts: {len(df)} picks {df.draft_year.min()}-{df.draft_year.max()}")


def fetch_schedule():
    games = {}
    for t in TEAMS:
        dest = RAW / f"nhl_sched_{t}_{SEASON}.json"
        if dest.exists() and dest.stat().st_size > 500:
            data = json.loads(dest.read_text())
        else:
            data = get_json(f"{API}/club-schedule-season/{t}/{SEASON}")
            dest.write_text(json.dumps(data))
            time.sleep(0.7)
        n = 0
        for gm in data.get("games", []):
            if gm.get("gameType") != 2:  # regular season only
                continue
            gid = gm["id"]
            games[gid] = {"game_id": gid, "date": gm.get("gameDate"),
                          "home": gm["homeTeam"]["abbrev"], "away": gm["awayTeam"]["abbrev"]}
            n += 1
        print(f"sched {t}: {n} regular-season games")
        sys.stdout.flush()
    df = pd.DataFrame(games.values()).sort_values(["date", "game_id"])
    counts = pd.concat([df.home, df.away]).value_counts()
    print(f"schedule: {len(df)} unique games; per-team min/max {counts.min()}/{counts.max()}; "
          f"home min/max {df.home.value_counts().min()}/{df.home.value_counts().max()}")
    assert len(df) == 1344 and (counts == 84).all(), "2026-27 schedule shape unexpected"
    assert set(counts.index) == set(TEAMS), "unexpected team codes in schedule"
    df.to_csv(RAW / f"nhl_schedule_{SEASON}.csv", index=False)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_rosters()
    fetch_drafts()
    fetch_schedule()
    print("done")
