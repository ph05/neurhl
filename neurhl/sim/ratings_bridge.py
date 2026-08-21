"""NeurHL Tier-3a — bridge NN game probabilities into the engine's Elo space.

Given per-game NN home-win probabilities over a schedule, invert the engine's
outcome model (game_probs is monotone in d) to an implied per-game rating diff,
then least-squares team ratings (sum-to-mean constraint) with the residual kept
as a per-game d_adj offset — the schedule-specific context (roster/rest/travel)
the NN sees beyond team strength. simulate_season consumes both unchanged, so
every downstream tool (reports, market) works verbatim.

Sigma policy: the strength-draw sigma is v1's train-frozen sigma1=30 Elo
(house-tuned on <=2017; not retuned here — any change is a ledgered config).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJ  # noqa: E402

import engine as E  # noqa: E402

MEAN = 1505.0
SIGMA_V1 = 30.0


def p_home_overall(d: np.ndarray, om: dict) -> np.ndarray:
    p_ot, p_reg, p_otw = E.game_probs(d, om)
    return (1 - p_ot) * p_reg + p_ot * p_otw


def invert_phome(p: np.ndarray, om: dict, lo=-600.0, hi=600.0,
                 iters: int = 60) -> np.ndarray:
    """Vectorized bisection: d such that p_home_overall(d) = p."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    a = np.full_like(p, lo)
    b = np.full_like(p, hi)
    for _ in range(iters):
        m = (a + b) / 2
        below = p_home_overall(m, om) < p
        a = np.where(below, m, a)
        b = np.where(below, b, m)
    return (a + b) / 2


def bridge(games: pd.DataFrame, om: dict) -> tuple[dict, np.ndarray]:
    """games: columns home, away, p_home -> ({team: elo_rating}, d_adj[N]).

    Least squares r over teams with mean(r)=0 (ridge 1e-6 for numerical rank),
    d_imp ~= r_home - r_away; d_adj = residual.
    """
    teams = sorted(set(games.home) | set(games.away))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(games), len(teams)
    X = np.zeros((n, k))
    X[np.arange(n), games.home.map(idx)] = 1.0
    X[np.arange(n), games.away.map(idx)] = -1.0
    d_imp = invert_phome(games.p_home.to_numpy(), om)
    A = X.T @ X + 1e-6 * np.eye(k) + np.ones((k, k))   # +1s enforces sum(r)=0
    r = np.linalg.solve(A, X.T @ d_imp)
    d_adj = d_imp - X @ r
    ratings = {t: MEAN + r[idx[t]] for t in teams}
    return ratings, d_adj


def simulate(games: pd.DataFrame, om: dict, n_sims: int, seed: int,
             season_end: int, playoffs: bool = False,
             sigma: float = SIGMA_V1) -> dict:
    """NN per-game p_home over a full schedule -> howe.rebuild_sim-style dict."""
    ratings, d_adj = bridge(games, om)
    sched = games[["home", "away"]].copy()
    sched["d_adj"] = d_adj
    return E.simulate_season(ratings, sigma, sched, om,
                             E.divisions_for(season_end), n_sims,
                             np.random.default_rng(seed), playoffs=playoffs)
