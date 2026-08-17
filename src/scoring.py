"""I2: proper scoring + market comparison machinery (PLAN_V4 — report-only axes).

CRPS: empirical CRPS from simulation draws via the sorted-sample identity
  CRPS = E|X - y| - 0.5 E|X - X'|,   E|X-X'| = (2/m^2) * sum_i x_(i) * (2i - m - 1)
(verified against the closed-form Gaussian CRPS in review_tests.py).

Market: loaders for hand-recorded odds boards under data/market/ (no synthetic lines;
the 2026-08-17 Cup board was hand-recorded from a sportsbook screenshot).
Devig: proportional and power methods; edges reported under both — a bet thesis that
does not survive both devigs is not an edge. Kelly fractions vs the ACTUAL price.
All of this is evaluation/reporting; nothing here feeds back into model parameters.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
MARKET = PROJ / "data" / "market"


# ------------------------------------------------------------------ CRPS
def crps_draws(draws: np.ndarray, y: float) -> float:
    """Empirical CRPS of sample `draws` against outcome y. O(m log m)."""
    x = np.sort(np.asarray(draws, dtype=float))
    m = len(x)
    e_xy = np.abs(x - y).mean()
    i = np.arange(1, m + 1)
    e_xx = 2.0 / (m * m) * float((x * (2 * i - m - 1)).sum())
    return float(e_xy - 0.5 * e_xx)


def crps_matrix(pts: np.ndarray, teams: list[str], actual: pd.Series) -> pd.Series:
    """Per-team CRPS from a sim points matrix (n_sims, n_teams)."""
    out = {}
    for j, t in enumerate(teams):
        if t in actual.index:
            out[t] = crps_draws(pts[:, j], float(actual[t]))
    return pd.Series(out, name="crps")


# ------------------------------------------------------------------ market
def american_to_prob(american: np.ndarray) -> np.ndarray:
    a = np.asarray(american, dtype=float)
    return np.where(a >= 0, 100.0 / (a + 100.0), -a / (-a + 100.0))


def american_to_decimal(american: np.ndarray) -> np.ndarray:
    a = np.asarray(american, dtype=float)
    return np.where(a >= 0, 1.0 + a / 100.0, 1.0 + 100.0 / (-a))


def devig(p_imp: np.ndarray) -> pd.DataFrame:
    """Proportional and power devigs for a mutually-exclusive board."""
    p = np.asarray(p_imp, dtype=float)
    prop = p / p.sum()
    lo, hi = 1.0, 5.0
    for _ in range(80):  # bisection on the power exponent
        k = 0.5 * (lo + hi)
        s = (p ** k).sum()
        if s > 1.0:
            lo = k
        else:
            hi = k
    power = p ** k
    return pd.DataFrame({"p_prop": prop, "p_pow": power})


def load_cup_board(name: str = "nhl_cup_2027.csv") -> pd.DataFrame | None:
    f = MARKET / name
    if not f.exists():
        return None
    mkt = pd.read_csv(f)
    mkt["p_imp"] = american_to_prob(mkt.american.to_numpy())
    mkt["dec"] = american_to_decimal(mkt.american.to_numpy())
    dv = devig(mkt.p_imp.to_numpy())
    return pd.concat([mkt.reset_index(drop=True), dv], axis=1)


def cup_market_sheet(model_cup: pd.Series, board: pd.DataFrame,
                     kelly_frac: float = 0.25, bankroll: float = 100.0) -> pd.DataFrame:
    """Model-vs-market Cup table: edges under both devigs, Kelly stakes at actual
    prices. Report-only; fair odds are break-even prices, not commands to bet."""
    m = board.copy()
    m["model_p"] = m.team.map(model_cup)
    m["edge_prop"] = m.model_p - m.p_prop
    m["edge_pow"] = m.model_p - m.p_pow
    m["ev_per_dollar"] = m.model_p * m.dec - 1.0
    m["kelly_full"] = ((m.model_p * m.dec - 1.0) / (m.dec - 1.0)).clip(lower=0.0)
    both_pos = (m.edge_prop > 0) & (m.edge_pow > 0) & (m.ev_per_dollar > 0)
    m["stake"] = np.where(both_pos, (kelly_frac * m.kelly_full * bankroll), 0.0)
    m = m.sort_values("ev_per_dollar", ascending=False)
    m.attrs["overround"] = float(m.p_imp.sum())
    return m
