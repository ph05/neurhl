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


def _filter():
    from orr import ratings as R
    fz = IS.load_freeze()
    fp = R.load_filter_params()
    fp["P"] = {**fp["P"], "mu": fz["league_level"]["mu_used"]}
    return R.InSeasonFilter(fz["ratings"][["team", "o", "d", "o_sd", "d_sd"]], fp, start_date="2026-09-29")


def test_lineup_and_starter_offsets_enter_the_update():
    """ORR 1.1 (X1): a lineup offset that already expects the home team to
    score more makes the same 5-goal night move its offence LESS; known
    starters change the update; zero offsets change nothing."""
    sch = pd.read_csv(IS.FREEZE / "schedule_2027.csv")
    g = sch.iloc[0]
    row = {"game_id": g.game_id, "date": g.date, "home": g.home, "away": g.away,
           "home_g": 5, "away_g": 1, "extra": "REG"}
    base, zero, lineup, gk = _filter(), _filter(), _filter(), _filter()
    base.update_day(pd.DataFrame([row]))
    zero.update_day(pd.DataFrame([{**row, "lo_h": 0.0, "lo_a": 0.0}]))
    lineup.update_day(pd.DataFrame([{**row, "lo_h": 0.15, "lo_a": 0.0}]))
    gk.update_day(pd.DataFrame([{**row, "gdiff_h": 0.0, "gdiff_a": -0.005}]))   # weak away starter
    b, z, l, k = (f.state().set_index("team") for f in (base, zero, lineup, gk))
    assert np.allclose(b.o, z.o) and np.allclose(b.d, z.d)
    assert l.o[g.home] < b.o[g.home]
    assert k.o[g.home] < b.o[g.home]        # the weak starter explains part of the 5 goals
    eh, ea = lineup.predict_eta(pd.DataFrame([{**row, "lo_h": 0.15}]))
    eh0, _ = lineup.predict_eta(pd.DataFrame([row]))
    assert np.isclose(eh[0] - eh0[0], 0.15)


def test_live_lineups_and_offsets():
    """NeurHL's pregame lineup files parse into 18 skaters a side, and a
    team's first known lineup carries no signal."""
    from orr import lineups as LU
    if not LU.NEURHL_LIVE.exists():
        return
    sk, gk, files = LU.live_lineups("2026-09-29")
    assert len(files) and set(sk.game_id) <= set(gk.game_id)
    assert (sk.groupby(["game_id", "side"]).size() <= 20).all()
    lo = LU.live_offsets(sk[pd.to_datetime(sk.date) == pd.Timestamp("2026-09-29")])
    assert np.allclose(lo.lo_h, 0) and np.allclose(lo.lo_a, 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
