"""Checks for ORR 1.4: long-term absence offsets and in-season start shares.

Run: python3 -m orr.tests.test_live_1_4
"""
import numpy as np
import pandas as pd

from orr import lineups as LU
from orr.backtest.start_share_bt import update_shares


def _season(absent_star: bool):
    """Team A plays 30 games; player 1 (a strong forward) dresses in the first
    20 (all 30 if not absent); player 9 is traded to B after game 15."""
    rows = []
    for g in range(30):
        for p in (2, 3):
            rows.append((g, "A", p, "F"))
        if g < 20 or not absent_star:
            rows.append((g, "A", 1, "F"))
        rows.append((g, "A" if g < 15 else "B", 9, "D"))
    d = pd.DataFrame(rows, columns=["game_id", "team", "player_id", "pos"])
    vals = pd.DataFrame({"player_id": [1, 2, 3, 9], "v_xgf": [0.10, 0.0, 0.0, 0.05],
                         "v_xga": [-0.05, 0.0, 0.0, 0.0], "toi": [20.0, 12.0, 12.0, 20.0]})
    return d, {"A": list(range(30)), "B": list(range(15, 30))}, vals


def test_absent_star_lowers_scoring_and_raises_opponent_scoring():
    d, tg, vals = _season(True)
    off = LU.absence_offsets(d, tg, vals, xg_pg=3.0, n_missed=5).set_index("team")
    assert off.loc["A", "n_out"] == 1                 # player 1; the traded player 9 is not counted
    assert off.loc["A", "s"] < 0 and off.loc["A", "c"] > 0
    d2, tg2, _ = _season(False)
    off2 = LU.absence_offsets(d2, tg2, vals, xg_pg=3.0, n_missed=5).set_index("team")
    assert off2.loc["A", "n_out"] == 0 and off2.loc["A", "s"] == 0


def test_offsets_enter_home_and_away_games():
    off = pd.DataFrame({"team": ["A"], "s": [-0.02], "c": [0.01], "n_out": [1]})
    sch = pd.DataFrame({"game_id": [1, 2], "home": ["A", "B"], "away": ["B", "A"]})
    adj = pd.DataFrame({"game_id": [1, 2], "adj_h": [0.0, 0.0], "adj_a": [0.0, 0.0]})
    out = LU.apply_team_offsets(adj, sch, off).set_index("game_id")
    assert np.isclose(out.loc[1, "adj_h"], -0.02) and np.isclose(out.loc[1, "adj_a"], 0.01)
    assert np.isclose(out.loc[2, "adj_h"], 0.01) and np.isclose(out.loc[2, "adj_a"], -0.02)


def test_start_shares_move_toward_actual_starts():
    pre = pd.DataFrame({"team": ["A", "A"], "player_id": [1, 2], "share": [0.7, 0.3]})
    starts = pd.DataFrame({"team": ["A", "A"], "player_id": [2, 3], "n": [15, 5]})
    sh = update_shares(pre, starts, pd.Series({"A": 20}), alpha=20.0).set_index("player_id").share
    assert np.isclose(sh.sum(), 1.0)
    assert sh[2] > 0.3 and sh[1] < 0.7 and sh[3] > 0      # a new goalie gets a share
    same = update_shares(pre, starts, pd.Series({"A": 20}), alpha=np.inf).set_index("player_id").share
    assert np.isclose(same[1], 0.7) and np.isclose(same.get(3, 0.0), 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
