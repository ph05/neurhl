"""Checks for in-season skater rate updating (ORR 1.2).

Run: python3 -m orr.tests.test_player_update
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from orr import player_update as PU


def test_posterior_is_conjugate_mean():
    prior = pd.DataFrame({"player_id": [1, 2], "g_pg": [0.5, 0.2], "a_pg": [0.5, 0.3], "sog_pg": [3.0, 2.0]})
    obs = pd.DataFrame({"player_id": [1], "n": [10], "g": [10], "a": [0], "sog": [40]})
    r = PU.posterior(prior, obs, {"g": 40.0, "a": 40.0, "sog": 20.0}).set_index("player_id")
    assert np.isclose(r.loc[1, "g_pg"], (40 * 0.5 + 10) / 50)
    assert np.isclose(r.loc[1, "a_pg"], (40 * 0.5) / 50)
    assert np.isclose(r.loc[1, "sog_pg"], (20 * 3 + 40) / 30)
    assert np.isclose(r.loc[2, "g_pg"], 0.2) and r.loc[2, "n"] == 0      # no games: prior


def test_live_rates_ignore_goalies_and_future_games():
    sk = pd.DataFrame({"player_id": [1], "gp": [80], "g": [40], "a": [40], "sog": [240]})
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "boxes.csv"
        pd.DataFrame({"game_id": [1, 2, 3], "date": ["2026-10-01", "2026-10-02", "2026-10-09"],
                      "player_id": [1, 99, 1], "pos": ["F", "G", "F"], "toi": [18.0, 60.0, 18.0],
                      "g": [3, 0, 5], "a": [0, 0, 0], "sog": [6, 0, 9]}).to_csv(p, index=False)
        r = PU.live_rates(sk, {"g": 40.0, "a": 40.0, "sog": 20.0}, path=p, before=pd.Timestamp("2026-10-05"))
        r = r.set_index("player_id")
        assert 99 not in r.index                                   # goalie rows dropped
        assert r.loc[1, "n"] == 1                                  # the 10-09 game is after the date
        assert np.isclose(r.loc[1, "g_pg"], (40 * 0.5 + 3) / 41)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
