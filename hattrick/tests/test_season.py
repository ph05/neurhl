"""Checks for the season simulator and the rating calibration.

Run: python3 -m hattrick.tests.test_season
"""
import numpy as np
import pandas as pd

from hattrick import calibrate as K
from hattrick import config as C
from hattrick import data as D
from hattrick import season as S


def _flat(sd=0.0):
    return pd.DataFrame({"team": C.TEAMS_2027, "o": 0.0, "d": 0.0,
                         "o_sd": sd, "d_sd": sd})


def test_points_identity_and_brackets():
    sch = D.schedule_2027()
    res = S.simulate(sch, _flat(0.05), S.ScoringModel(), n_sims=2000, seed=1)
    n_games = len(sch)
    # every game hands out 2 points, plus 1 more when it goes past regulation
    total = res.points.sum(axis=1)
    extra = res.otl.sum(axis=1)
    assert np.all(total == 2 * n_games + extra)
    assert np.all(res.gp == 84)
    assert np.all(res.wins.sum(axis=1) == n_games)
    # exactly 16 playoff teams, 8 per conference, seeds 1..8 once each
    seeds = res.playoff_seed
    assert np.all((seeds > 0).sum(axis=1) == 16)
    for conf, divs in C.CONFERENCES.items():
        ids = [res.teams.index(t) for d in divs for t in C.DIVISIONS[d]]
        s = np.sort(seeds[:, ids], axis=1)[:, -8:]
        assert np.all(s == np.arange(1, 9))
    # the three division leaders per division are all in
    # one champion per simulation
    assert np.all((res.rounds == 4).sum(axis=1) == 1)
    assert np.all((res.rounds >= 3).sum(axis=1) == 2)
    summ = S.summarise(res)
    assert abs(summ.playoff_pct.sum() - 1600) < 1e-6
    assert abs(summ.cup_pct.sum() - 100) < 1e-6


def test_division_winners_qualify():
    sch = D.schedule_2027()
    res = S.simulate(sch, _flat(0.1), S.ScoringModel(), n_sims=500, seed=3)
    for dv, teams in C.DIVISIONS.items():
        ids = [res.teams.index(t) for t in teams]
        best = np.array(ids)[np.argmax(res.key[:, ids], axis=1)]
        assert np.all(res.playoff_seed[np.arange(500), best] > 0)


def test_completed_games_are_fixed():
    sch = D.schedule_2027()
    done = sch.head(5).assign(home_g=[5, 0, 2, 3, 1], away_g=[1, 4, 1, 2, 2],
                              extra=["REG", "REG", "OT", "SO", "REG"])
    res = S.simulate(sch, _flat(0.05), S.ScoringModel(), n_sims=300, seed=2,
                     completed=done, playoffs=False)
    assert np.allclose(res.game_home_win[:5], [1, 0, 1, 1, 0])


def test_calibration_hits_targets():
    sch = D.schedule_2027()
    rng = np.random.default_rng(0)
    targets = pd.Series(92 + rng.normal(0, 9, 32), index=C.TEAMS_2027)
    style = pd.DataFrame({"team": C.TEAMS_2027, "o_m": rng.normal(0, .05, 32),
                          "d_m": rng.normal(0, .05, 32)})
    m = S.ScoringModel()
    r = K.solve_ratings(sch, targets, style, m, max_iter=200, tol=0.05)
    assert (r.target - r.expected).abs().max() < 0.05
    # relative order of targets is preserved
    assert np.corrcoef(r.target, r.o - r.d)[0, 1] > 0.98


def test_probs_match_sampler():
    m = S.ScoringModel()
    lam_h, lam_a = np.full(200000, 3.2), np.full(200000, 2.8)
    rng = np.random.default_rng(5)
    gh, ga, ext, hw = m.sample(lam_h, lam_a, rng)
    p = m.probs(np.array([3.2]), np.array([2.8]))
    assert abs(hw.mean() - p["p_home"][0]) < 0.005
    assert abs((ext > 0).mean() - p["tie"][0]) < 0.005


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
