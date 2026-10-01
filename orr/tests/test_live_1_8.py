"""Checks for ORR 1.8: forecast diff and goalie rest-of-season lines.

Run: python3 -m orr.tests.test_live_1_8
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from orr import inseason as IS


def _games(p, goalie_home=np.nan, adj=0.0):
    return pd.DataFrame({"game_id": [2026020009], "date": ["2026-10-01"], "home": ["NJD"], "away": ["PHI"],
                         "p_home_win": [p], "goalie_home": [goalie_home], "goalie_away": [np.nan],
                         "lineup_adj_h": [adj], "lineup_adj_a": [0.0], "created_utc": ["2026-10-01T15:00:00+00:00"]})


def test_forecast_diff_against_earlier_run_and_preseason():
    with tempfile.TemporaryDirectory() as d:
        prev = Path(d) / "games.csv"
        first = IS.forecast_diff(_games(0.60), prev)                 # no earlier run: preseason file
        assert first.prev_source.iloc[0] == "preseason file"
        _games(0.60).to_csv(prev, index=False)
        later = IS.forecast_diff(_games(0.63, goalie_home=8471679.0, adj=0.01), prev)
        assert np.isclose(later.d_p.iloc[0], 0.03)
        assert "home starter" in later.change_reason.iloc[0] and "home lineup" in later.change_reason.iloc[0]
        same = IS.forecast_diff(_games(0.61), prev)
        assert same.change_reason.iloc[0] == "ratings"


def test_goalie_ros_lines():
    fz = IS.load_freeze()
    g = IS.goalies_ros(fz, pd.Timestamp("2026-10-01"), pd.DataFrame(columns=["home", "away"]), sims=300)
    team_starts = g.groupby("team").starts_ros.sum()
    assert (team_starts > 70).all() and (team_starts <= 84.5).all()       # most of 84 games, share < 1 (call-ups)
    assert ((g.sv_season_p10 <= g.sv_season) & (g.sv_season <= g.sv_season_p90)).all()
    assert g.sv_season.between(0.86, 0.94).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
