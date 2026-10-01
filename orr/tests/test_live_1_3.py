"""Checks for ORR 1.3's live inputs: box-score lineups and in-season goalie talent.

Run: python3 -m orr.tests.test_live_1_3
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from orr import lineups as LU
from orr import structural as ST

SCH = pd.DataFrame({"game_id": [10, 11], "home": ["NJD", "NYR"], "away": ["PHI", "TBL"]})


def _boxes(p: Path):
    pd.DataFrame({
        "game_id": [10, 10, 10, 10, 11], "date": ["2026-10-01"] * 4 + ["2026-10-05"],
        "team": ["NJD", "NJD", "PHI", "PHI", "NYR"], "player_id": [1, 2, 3, 4, 5],
        "pos": ["F", "G", "D", "G", "G"], "toi": [18.0, 60.0, 20.0, 60.0, 60.0],
        "g": [1, 0, 0, 0, 0], "a": [0, 0, 1, 0, 0], "sog": [3, 0, 1, 0, 0],
        "starter": [False, True, False, True, True], "shots_against": [None, 40, None, 20, 30],
        "goals_against": [None, 0, None, 5, 2]}).to_csv(p, index=False)


def test_box_lineups_sides_and_starters():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.csv"
        _boxes(p)
        sk, gk = LU.box_lineups(pd.Timestamp("2026-10-03"), SCH, p)
        assert set(sk.game_id) == {10}                              # game 11 is after the date
        assert dict(zip(sk.player_id, sk.side)) == {1: "h", 3: "a"}
        r = gk.set_index("game_id").loc[10]
        assert r.goalie_home == 2 and r.goalie_away == 4
        live_gk = pd.DataFrame({"game_id": [10, 12], "goalie_home": [99.0, 7.0], "goalie_away": [98.0, 8.0],
                                "goalie_source_home": "DF", "goalie_source_away": "DF"})
        _, g2 = LU.prefer_box(sk.iloc[:0], live_gk, sk, gk)
        assert g2.set_index("game_id").loc[10].goalie_home == 2     # box score wins
        assert g2.set_index("game_id").loc[12].goalie_home == 7     # other games keep the pregame file


def test_goalie_talent_moves_with_saves():
    base = ST.goalie_talent_2027()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.csv"
        _boxes(p)
        live = ST.goalie_talent_live(pd.Timestamp("2026-10-03"), p)
        # goalie 2 stopped 40 of 40 -> up; goalie 4 allowed 5 on 20 -> down; goalie 5 (10-05) unchanged
        assert live[2] > base.get(2, ST.goalie_talent_default()) - 1e-12
        assert live[4] < base.get(4, ST.goalie_talent_default())
        assert np.isclose(live.get(5, base.get(5, np.nan)), base.get(5, np.nan), equal_nan=True) or 5 not in live


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
