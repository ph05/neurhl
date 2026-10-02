"""Checks for ORR 1.5: shot distributions and the running accuracy table.

Run: python3 -m orr.tests.test_live_1_5
"""
import numpy as np
import pandas as pd

from orr import player_update as PU
from orr import score as SC


def test_shot_probabilities():
    mu = np.array([1.0, 2.5, 4.0])
    for k in (2, 3, 4):
        pp, pn = PU.p_shots_ge(mu, k), PU.p_shots_ge(mu, k, r=16.6)
        assert np.all(np.diff(pp) > 0) and np.all((pn > 0) & (pn < 1))
    # overdispersion fattens the far tail: P(>=4) for a low-rate shooter
    assert PU.p_shots_ge(np.array([1.0]), 4, r=2.0)[0] > PU.p_shots_ge(np.array([1.0]), 4)[0]
    assert np.allclose(PU.p_shots_ge(mu, 3, r=1e9), PU.p_shots_ge(mu, 3), atol=1e-6)   # r -> inf: Poisson


def test_running_log_loss_is_cumulative():
    ep = pd.DataFrame({"model": ["m"] * 3, "game_id": [1, 2, 3], "date": ["2026-10-01", "2026-10-01", "2026-10-02"],
                       "p": [0.6, 0.4, 0.7], "y": [1, 1, 0]})
    r = SC.running_by_date(ep)["m"]
    ll = -np.log([0.6, 0.4, 0.3])
    assert r[0]["n"] == 2 and np.isclose(r[0]["log_loss"], ll[:2].mean())
    assert r[1]["n"] == 3 and np.isclose(r[1]["log_loss"], ll.mean())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
