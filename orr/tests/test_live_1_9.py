"""Checks for ORR 1.9: paired live comparison and clinch flags.

Run: python3 -m orr.tests.test_live_1_9
"""
import numpy as np
import pandas as pd

from orr import inseason as IS
from orr import score as SC


def test_paired_comparison_and_verdicts():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    good = np.where(y == 1, 0.65, 0.35)
    ep = pd.concat([pd.DataFrame({"model": "orr_inseason", "game_id": range(n), "date": "2026-10-01", "p": good, "y": y}),
                    pd.DataFrame({"model": "neurhl_1.1", "game_id": range(n), "date": "2026-10-01", "p": 0.5, "y": y}),
                    pd.DataFrame({"model": "neurhl_1.0", "game_id": range(10), "date": "2026-10-01", "p": 0.5, "y": y[:10]})])
    out = SC.paired_vs_neurhl(ep)
    assert out["neurhl_1.1"]["verdict"] == "ORR ahead" and out["neurhl_1.1"]["diff"] < 0
    assert out["neurhl_1.0"]["n"] == 10 and out["neurhl_1.0"]["verdict"] == "too few games"


def test_clinch_flags_and_magic_number():
    teams = [f"T{i}" for i in range(10)]
    st = pd.DataFrame({"team": teams, "conf": "E", "playoff_pct": [100.0] + [50.0] * 8 + [0.0],
                       "division_pct": 10.0, "presidents_pct": 5.0})
    res = pd.DataFrame({"home": ["T0", "T1"], "away": ["T9", "T8"], "home_g": [3, 1], "away_g": [1, 2],
                        "extra": ["REG", "OT"]})
    c = IS.clinch_table(st, pd.DataFrame(), res).set_index("team")
    assert c.loc["T0", "playoff_flag"] == "clinched" and c.loc["T9", "playoff_flag"] == "eliminated"
    assert c.loc["T0", "pts_now"] == 2 and c.loc["T1", "pts_now"] == 1 and c.loc["T8", "pts_now"] == 2
    # T0 (2 pts, 83 left) vs the ninth-placed team's maximum points
    ninth_max = c.sort_values("pts_now", ascending=False).iloc[8]
    assert np.isfinite(c.loc["T0", "magic_number"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
