"""Schedule context for an upcoming season, in the games_ctx schema.

games_ctx_<season>.parquet exists only for completed seasons, because
data/tensorize_games.py builds it from play-by-play. A pre-season projection
needs the same per-game context for a schedule that has not been played: team
indices, days into the season, rest, kilometres travelled over three days and
the timezone change. Everything here is knowable from the published schedule.

Rest and travel follow src/build_travel.py exactly (rest capped at 9, a gap of
more than 30 days resets the state, three-day travel window), so pre-season
rows match the historical table's definitions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from build_travel import arena_of, haversine_km  # noqa: E402


def arenas_for(season_end: int) -> pd.DataFrame:
    """Arena table valid for `season_end`. Rows current through the last
    completed season carry forward: no franchise changes arena for 2026-27."""
    ar = pd.read_csv(RAW / "arenas.csv")
    last = ar.end_end.max()
    if season_end > last:
        ar.loc[ar.end_end == last, "end_end"] = season_end
    return ar


def build(sched: pd.DataFrame, season_end: int) -> pd.DataFrame:
    """`sched` has game_id, date, home, away (NHL abbreviations)."""
    team_idx = json.loads((TENSORS / "maps.json").read_text())["team"]
    ar = arenas_for(season_end)
    s = sched.assign(date=pd.to_datetime(sched.date)).sort_values(
        ["date", "game_id"]).reset_index(drop=True)
    state: dict = {}
    rows = []
    for gm in s.itertuples():
        venue, vutc = arena_of(gm.home, season_end, ar)
        rec = {"game_id": int(gm.game_id), "game_type": 2,
               "date": gm.date.strftime("%Y-%m-%d"),
               "home_idx": team_idx[gm.home], "away_idx": team_idx[gm.away]}
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
            rec[f"{side}_rest"] = float(rest)
            rec[f"{side}_km3d"] = round(km3, 1)
            rec[f"{side}_dtz"] = float(dtz)
            legs = [(d, k) for d, k in (last[3] if last else [])
                    if (gm.date - d).days <= 3] + [(gm.date, km)]
            state[team] = (gm.date, venue, vutc, legs)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["days_in"] = (pd.to_datetime(out.date)
                      - pd.to_datetime(out.date).min()).dt.days
    return out
