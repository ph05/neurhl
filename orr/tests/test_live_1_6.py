"""Checks for ORR 1.6: remaining strength of schedule and rest-of-season player lines.

Run: python3 -m orr.tests.test_live_1_6
"""
import numpy as np
import pandas as pd

from orr import inseason as IS


def test_remaining_sos_is_mean_opponent_net():
    sch = pd.DataFrame({"game_id": [1, 2, 3, 4], "home": ["A", "B", "A", "C"], "away": ["B", "C", "C", "A"]})
    done = pd.DataFrame({"game_id": [1]})
    cur = pd.DataFrame({"team": ["A", "B", "C"], "o": [0.1, 0.0, -0.1], "d": [0.0, 0.05, 0.0]})
    s = IS.remaining_sos(sch, done, cur).set_index("team")
    net = {"A": 0.1, "B": -0.05, "C": -0.1}
    assert np.isclose(s.loc["A", "sos_remaining"], np.mean([net["C"], net["C"]])) and s.loc["A", "games_left"] == 2
    assert np.isclose(s.loc["B", "sos_remaining"], net["C"]) and s.loc["B", "games_left"] == 1
    assert np.isclose(s.loc["C", "sos_remaining"], np.mean([net["B"], net["A"], net["A"]]))


def test_players_ros_without_box_scores_is_the_prior():
    fz = IS.load_freeze()
    r = IS.players_ros(fz, pd.Timestamp("2026-10-01"), pd.DataFrame(columns=["home", "away"]),
                       n0={"g": 40.0, "a": 40.0, "sog": 20.0})
    sk = pd.read_csv(IS.FREEZE / "skaters_2027.csv").set_index("player_id")
    x = r.set_index("player_id").loc[sk.index]
    if (x.gp_td == 0).all():                      # no live box scores in the checkout
        assert np.allclose(x.p_ros, (sk.g + sk.a) * np.minimum(sk.gp, 84) / sk.gp.clip(lower=1), atol=1e-9)
    assert (x.p_ros_p10 <= x.p_ros + 1e-9).all() and (x.p_ros <= x.p_ros_p90 + 1).all()


def test_start_shares_untouched_without_starts():
    import tempfile
    from pathlib import Path
    g = pd.DataFrame({"player_id": [1, 2, 3, 4], "team": ["A", "A", "B", "B"], "start_share": [0.6, 0.3, 0.5, 0.4],
                      "p_present": 1.0})
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.csv"
        pd.DataFrame({"game_id": [10], "date": ["2026-10-01"], "team": ["A"], "player_id": [2], "pos": ["G"],
                      "toi": [60.0], "starter": [True]}).to_csv(p, index=False)
        out = IS.update_start_shares_live(g, pd.Timestamp("2026-10-03"), path=p).set_index("player_id")
    assert np.isclose(out.loc[3, "start_share"], 0.5) and np.isclose(out.loc[4, "start_share"], 0.4)   # team B untouched
    assert out.loc[2, "start_share"] > 0.3                                                             # team A moved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
