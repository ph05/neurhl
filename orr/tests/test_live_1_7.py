"""Checks for ORR 1.7: rest-of-season intervals and the version history.

Run: python3 -m orr.tests.test_live_1_7
"""
import numpy as np
import pandas as pd

from orr import inseason as IS
from orr.site import build_changelog as BC


def test_games_and_rate_variance_widen_intervals():
    fz = IS.load_freeze()
    res = pd.DataFrame(columns=["home", "away"])
    base = IS.players_ros(fz, pd.Timestamp("2026-10-01"), res, interval_cal=False).set_index("player_id")
    orig = IS.ros_params
    try:
        IS.ros_params = lambda: {"v": 2.0, "games_var": True}
        wide = IS.players_ros(fz, pd.Timestamp("2026-10-01"), res, interval_cal=True).set_index("player_id")
    finally:
        IS.ros_params = orig
    top = base.p_ros.nlargest(50).index
    w0 = (base.p_ros_p90 - base.p_ros_p10).loc[top].mean()
    w1 = (wide.p_ros_p90 - wide.p_ros_p10).loc[top].mean()
    assert w1 > w0                                         # more variance -> wider intervals
    assert np.allclose(base.p_ros, wide.p_ros)             # the mean does not change


def test_version_history_lists_every_release():
    html = BC.releases_table()
    for v in ("1.2", "1.3", "1.4", "1.5", "1.6"):
        assert f">{v}</a>" in html


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
