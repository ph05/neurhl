"""Tests for orr.gamemodel and orr.ratings (plain asserts).

Run: cd /home/user/neurhl && python3 -m orr.tests.test_gamemodel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orr import gamemodel as GM


def _P(**ot):
    P = GM.load_params()
    P["ot"] = {**P["ot"], **ot}
    return P


def test_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    lh = rng.uniform(1.5, 4.5, 500)
    la = rng.uniform(1.5, 4.5, 500)
    P = GM.load_params()
    p = GM.outcome_probs(lh, la, P)
    reg = p["p_home_reg"] + p["p_tie"] + p["p_away_reg"]
    assert np.allclose(reg, 1, atol=1e-9), np.abs(reg - 1).max()
    extra = p["p_home_ot"] + p["p_away_ot"] + p["p_home_so"] + p["p_away_so"]
    assert np.allclose(extra, p["p_tie"], atol=1e-12)
    p_away_win = p["p_away_reg"] + p["p_away_ot"] + p["p_away_so"]
    assert np.allclose(p["p_home_win"] + p_away_win, 1, atol=1e-9)
    M = GM.margin_pmf(lh, la, P)
    assert np.all(M >= -1e-12) and np.allclose(M.sum(1), 1, atol=1e-9)
    J = GM.reg_joint(lh[:50], la[:50], P)
    assert np.allclose(J.sum((1, 2)), 1, atol=1e-4)     # K=16 truncation
    assert np.all(J >= -1e-12)


def test_mean_preserving_layer():
    """E[regulation goals] = lambda exactly (ratings = expected reg goals)."""
    lh = np.array([2.0, 2.8, 3.1, 3.9])
    la = np.array([3.5, 2.8, 2.6, 2.0])
    J = GM.reg_joint(lh, la, GM.load_params(), K=20)
    x = np.arange(20)
    assert np.allclose((J.sum(2) * x).sum(1), lh, atol=2e-3)
    assert np.allclose((J.sum(1) * x).sum(1), la, atol=2e-3)


def test_symmetry_equal_teams_no_home_ice():
    P = _P(a_ot=0.0, a_so=0.0)
    lam = np.array([2.0, 2.9, 3.6])
    p = GM.outcome_probs(lam, lam, P)
    assert np.allclose(p["p_home_win"], 0.5, atol=1e-9)
    assert np.allclose(p["p_home_reg"], p["p_away_reg"], atol=1e-12)
    M = GM.margin_pmf(lam, lam, P)
    assert np.allclose(M, M[:, ::-1], atol=1e-12)
    # rates(): equal o, d and h = 0 give equal lambdas
    lh, la = GM.rates({**P, "h": 0.0}, 0.05, -0.02, 0.05, -0.02)
    assert np.isclose(lh, la)


def test_monotone_in_strength():
    P = GM.load_params()
    o = np.linspace(-0.4, 0.4, 41)
    z = np.zeros_like(o)
    lh, la = GM.rates(P, o, z, z, z)
    p = GM.outcome_probs(lh, la, P)["p_home_win"]
    assert np.all(np.diff(p) > 0)
    lh, la = GM.rates(P, z, o, z, z)          # home allows more -> worse
    p = GM.outcome_probs(lh, la, P)["p_home_win"]
    assert np.all(np.diff(p) < 0)
    # expected points of a stronger team are higher
    pts = GM.expected_points_vs_average(np.linspace(-0.5, 0.5, 21), P)
    assert np.all(np.diff(pts) > 0)


def test_tie_rate_in_nhl_range():
    """League-average matchup: regulation ties 0.20-0.26 (NHL 2006-2026)."""
    P = GM.load_params()
    p = GM.outcome_probs(np.array([3.05]), np.array([2.85]), P)
    assert 0.20 < p["p_tie"][0] < 0.26, p["p_tie"]


def test_sampling_matches_probabilities():
    P = GM.load_params()
    rng = np.random.default_rng(1)
    n = 400_000
    lh = np.full(n, 3.1)
    la = np.full(n, 2.7)
    s = GM.sample(lh, la, P, rng)
    p = GM.outcome_probs(lh[:1], la[:1], P)
    assert abs(s["home_win"].mean() - p["p_home_win"][0]) < 0.004
    assert abs((s["reg_h"] == s["reg_a"]).mean() - p["p_tie"][0]) < 0.004
    assert abs((s["extra"] == 2).mean() - p["p_so"][0]) < 0.003
    assert abs(s["reg_h"].mean() - 3.1) < 0.02 and abs(s["reg_a"].mean() - 2.7) < 0.02
    # standings goals: winner of a tied game gets exactly one more goal
    tie = s["extra"] > 0
    assert np.all(np.abs(s["gf_h"][tie] - s["gf_a"][tie]) == 1)
    assert np.all((s["gf_h"] > s["gf_a"]) == (s["home_win"] == 1))


def test_schedule_features_keep_order():
    from orr import data as D
    sch = D.schedule_2027().sample(frac=1.0, random_state=3)
    f = GM.schedule_features(sch, 2027)
    assert (f.index == sch.index).all()
    assert (f.game_id.to_numpy() == sch.game_id.to_numpy()).all()
    assert set(f.rest_h.unique()) <= set(range(1, 10))


def test_known_starter_offsets():
    """Unknown starters reproduce ctx_offsets exactly. A known home starter
    switches the HOME team's rest/travel terms to ctx_gk and adds
    goalie_offset(diff) to the AWAY team's goals only."""
    from orr import data as D
    P = GM.load_params()
    sch = GM.schedule_features(D.schedule_2027().head(60), 2027)
    n = len(sch)
    nan = np.full(n, np.nan)
    h0, a0 = GM.ctx_offsets(P, sch)
    h1, a1 = GM.known_starter_offsets(P, sch, nan, nan)
    assert np.allclose(h0, h1) and np.allclose(a0, a1)
    d = np.full(n, 0.004)
    h2, a2 = GM.known_starter_offsets(P, sch, d, nan)
    fh = GM.ctx_features(sch.rest_h, sch.km_h, sch.dtz_h)
    c0, c1 = P["ctx"], P["ctx_gk"]
    dh_own = sum((c1[f"{k}_o"] - c0[f"{k}_o"]) * fh[k] for k in GM.CTX_FEATURES)
    dh_opp = sum((c1[f"{k}_d"] - c0[f"{k}_d"]) * fh[k] for k in GM.CTX_FEATURES)
    assert np.allclose(h2, h0 + dh_own)
    assert np.allclose(a2, a0 + dh_opp + GM.goalie_offset(P, d))
    assert np.all(GM.goalie_offset(P, d) < 0)          # better starter, fewer goals against


def test_goalie_units():
    """Talents and gaps are in the fitted units: league-typical gaps are of
    the order of the historical avg_gap, not GSAx/FA x 1.4."""
    from orr import structural as S
    t = S.goalie_talent_2027()
    assert len(t) > 80 and abs(t.median()) < 0.005 and t.std() < 0.005
    g = pd.DataFrame({"player_id": t.index[:4], "team": ["AAA", "AAA", "BBB", "BBB"],
                      "start_share": [0.7, 0.3, 0.6, 0.4]})
    gap = S.team_gap_2027(g, t)
    assert np.isclose(gap["AAA"], t.iloc[0] - t.iloc[1])


# ---------------------------------------------------------------------------
# Walk-forward guarantees
# ---------------------------------------------------------------------------
def _corrupted_frame(cut: pd.Timestamp, seed: int = 5) -> pd.DataFrame:
    from orr import structural as S
    g = S.game_frame.__wrapped__().copy()
    rng = np.random.default_rng(seed)
    m = g.date >= cut
    n = int(m.sum())
    g.loc[m, "reg_h"] = rng.integers(0, 9, n)
    g.loc[m, "reg_a"] = rng.integers(0, 9, n)
    g.loc[m, "sh_h"] = rng.integers(10, 60, n).astype(float)
    g.loc[m, "sh_a"] = rng.integers(10, 60, n).astype(float)
    g.loc[m, "margin"] = g.reg_h - g.reg_a
    return g


def _clear_caches():
    from orr import ratings as R
    from orr import structural as S
    for f in (S.game_frame, S.structural, S._glm_all_fe, S.goalie_game_talent,
              R._preseason_fit, R.team_features):
        if hasattr(f, "cache_clear"):
            f.cache_clear()


def test_walk_forward_filter_never_reads_future():
    """Corrupt every result on/after a cut date: pregame predictions for games
    BEFORE the cut and ON the cut date must be bit-identical, and so must the
    preseason ratings of the cut's season."""
    from orr import ratings as R
    from orr import structural as S
    hp = R.HP()
    cut = pd.Timestamp("2015-01-15")
    orig = S.game_frame
    bad = _corrupted_frame(cut)
    try:
        # Both arms are computed fresh in this process. The disk cache may have been
        # built on another machine (CI restores it), whose BLAS kernels differ in the
        # last bits, and that would break the bit-identical comparison below.
        S._DISK_CACHE = False
        _clear_caches()
        base = R.run_filter(hp, [2015], first=2014)
        pre_base = R.preseason_table(2015)
        S.game_frame = lambda: bad             # module-level lookups see this
        R.S.game_frame = S.game_frame
        _clear_caches()
        alt = R.run_filter(hp, [2015], first=2014)
        pre_alt = R.preseason_table(2015)
    finally:
        S._DISK_CACHE = True
        S.game_frame = orig
        R.S.game_frame = orig
        _clear_caches()
    dates = orig()[["gid", "date"]]
    b = base.merge(dates, on="gid")
    a = alt.merge(dates, on="gid")
    keep = b.date <= cut
    cols = ["eta_h", "eta_a", "v_h", "v_a", "feta_h", "feta_a"]
    assert keep.sum() > 500
    assert np.array_equal(b.loc[keep, cols].to_numpy(), a.loc[keep, cols].to_numpy())
    # after the cut the in-season predictions must differ (the test has teeth)
    later = b.date > cut + pd.Timedelta(days=3)
    assert not np.allclose(b.loc[later, "eta_h"], a.loc[later, "eta_h"])
    # frozen predictions never depend on in-season results
    assert np.array_equal(b.feta_h.to_numpy(), a.feta_h.to_numpy())
    assert np.array_equal(pre_base[["o", "d"]].to_numpy(), pre_alt[["o", "d"]].to_numpy())


def test_preseason_reads_only_earlier_seasons():
    """Corrupting the whole target season (and later) leaves preseason(V)
    unchanged."""
    from orr import ratings as R
    from orr import structural as S
    V = 2019
    base = R.preseason_table(V)
    orig = S.game_frame
    bad = _corrupted_frame(pd.Timestamp("2018-09-01"), seed=9)
    try:
        S._DISK_CACHE = False
        S.game_frame = lambda: bad
        R.S.game_frame = S.game_frame
        _clear_caches()
        alt = R.preseason_table(V)
    finally:
        S._DISK_CACHE = True
        S.game_frame = orig
        R.S.game_frame = orig
        _clear_caches()
    assert np.allclose(base[["o", "d", "o_sd"]].to_numpy(), alt[["o", "d", "o_sd"]].to_numpy())


def test_inseason_filter_api():
    from orr import ratings as R
    pre = pd.DataFrame({"team": ["BOS", "TOR"], "o": [0.05, 0.0], "d": [-0.05, 0.0],
                        "o_sd": [0.06, 0.06], "d_sd": [0.06, 0.06]})
    f = R.InSeasonFilter(pre, season_end=2027, start_date="2026-10-01")
    s0 = f.state().set_index("team")
    f.update({"date": pd.Timestamp("2026-10-02"), "home": "TOR", "away": "BOS",
              "home_g": 7, "away_g": 1, "extra": "REG", "sh_h": 45, "sh_a": 18})
    s1 = f.state().set_index("team")
    assert s1.loc["TOR", "o"] > s0.loc["TOR", "o"]
    assert s1.loc["BOS", "d"] > s0.loc["BOS", "d"]
    assert s1.loc["TOR", "o_sd"] < s0.loc["TOR", "o_sd"]
    assert set(s1.columns) >= {"o", "d", "o_sd", "d_sd"}


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
