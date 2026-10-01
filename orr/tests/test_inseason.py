"""In-season loop checks: ratings move toward the evidence, played games stay
played, and nothing dated on or after the forecast date is read.

Run (after the freeze): python3 -m orr.tests.test_inseason
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from orr import inseason as IS


def _results(rows):
    d = Path(tempfile.mkdtemp())
    p = d / "results.csv"
    pd.DataFrame(rows, columns=["game_id", "date", "home", "away", "home_g", "away_g",
                                "last_period"]).to_csv(p, index=False)
    return p


def test_filter_moves_toward_winner_and_ignores_future():
    sch = pd.read_csv(IS.FREEZE / "schedule_2027.csv")
    g = sch.iloc[0]
    blowout = _results([(g.game_id, g.date, g.home, g.away, 8, 0, "REG"),
                        # a result dated ON the forecast date must be ignored
                        (sch.iloc[5].game_id, "2026-09-30", sch.iloc[5].home, sch.iloc[5].away, 9, 0, "REG")])
    res = IS.load_results(blowout, pd.Timestamp("2026-09-30"))
    assert len(res) == 1
    from orr import ratings as R
    fz = IS.load_freeze()
    fp = R.load_filter_params()
    fp["P"] = {**fp["P"], "mu": fz["league_level"]["mu_used"]}
    pre = fz["ratings"][["team", "o", "d", "o_sd", "d_sd"]]
    filt = R.InSeasonFilter(pre, fp, start_date="2026-09-29")
    before = filt.state().set_index("team")
    filt.update_day(res)
    after = filt.state().set_index("team")
    assert after.o[g.home] > before.o[g.home]          # scored 8
    assert after.d[g.away] > before.d[g.away]          # allowed 8
    assert after.o_sd[g.home] < before.o_sd[g.home]    # learned something


def test_run_end_to_end():
    sch = pd.read_csv(IS.FREEZE / "schedule_2027.csv")
    first = sch[sch.date == sch.date.min()]
    rows = [(r.game_id, r.date, r.home, r.away, 3, 2, "REG") for r in first.itertuples()]
    p = _results(rows)
    nxt = sorted(sch.date.unique())[1]
    games, standings = IS.run(nxt, str(p), None, sims=2000, seed=1)
    assert len(games) == (sch.date == nxt).sum()
    assert np.allclose(games.p_home_reg + games.p_away_reg + games.p_ot, 1, atol=1e-6)
    assert abs(standings.playoff_pct.sum() - 1600) < 0.5
    # the opening-night winners keep their 2 points in every simulation
    for r in first.itertuples():
        row = standings.set_index("team").loc[r.home]
        assert row.points >= 2
    import shutil
    shutil.rmtree(IS.LIVE / nxt, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
