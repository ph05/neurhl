"""Travel/timezone game table (PLAN_V6 D4). No fetch — games.csv + arenas.csv.

Per regular-season game and side: great-circle km from that team's previous game
venue, km traveled over the prior 3 days, timezone offset change in the prior 48
hours, and rest days. Writes data/processed/travel_games.csv keyed
(date, home, away). Input for the T1 gate measurement in backtest6.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
PROC = PROJ / "data" / "processed"


def haversine_km(a, b):
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    x = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))


def arena_of(team, season_end, table):
    r = table[(table.team == team) & (table.start_end <= season_end)
              & (table.end_end >= season_end)]
    if not len(r):
        raise KeyError(f"no arena for {team} {season_end}")
    r = r.iloc[0]
    return (r.lat, r.lon), int(r.utc_std)


def main():
    ar = pd.read_csv(RAW / "arenas.csv")
    g = pd.read_csv(PROC / "games.csv", keep_default_na=False,
                    parse_dates=["date"])
    g = g[g.game_type == "R"].sort_values("date").reset_index(drop=True)
    state: dict = {}   # team -> (last_date, (lat,lon), utc, [(date, km), ...])
    rows = []
    for gm in g.itertuples():
        venue, vutc = arena_of(gm.home, gm.season_end, ar)
        rec = {"date": gm.date, "home": gm.home, "away": gm.away,
               "season_end": gm.season_end}
        for side, team in (("home", gm.home), ("away", gm.away)):
            last = state.get(team)
            if last is None or (gm.date - last[0]).days > 30:
                km, dtz, rest, km3 = 0.0, 0, 9, 0.0
            else:
                km = float(haversine_km(last[1], venue))
                dtz = vutc - last[2]
                rest = min((gm.date - last[0]).days, 9)
                legs = [(d, k) for d, k in last[3] if (gm.date - d).days <= 3]
                km3 = sum(k for _, k in legs) + km
            rec[f"{side}_km"] = round(km, 1)
            rec[f"{side}_km3d"] = round(km3, 1)
            rec[f"{side}_dtz"] = dtz
            rec[f"{side}_rest"] = rest
            legs = [(d, k) for d, k in (last[3] if last else [])
                    if (gm.date - d).days <= 3] + [(gm.date, km)]
            state[team] = (gm.date, venue, vutc, legs)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "travel_games.csv", index=False)
    print(f"travel_games: {len(df)} rows; away km/game mean "
          f"{df.away_km.mean():.0f}, |dtz|>0 share "
          f"{(df.away_dtz != 0).mean():.3f}")


if __name__ == "__main__":
    main()
