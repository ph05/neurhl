"""M2: walk-forward logistic stack of ORR's in-season logit, ORR's preseason
logit and the Elo logit.

    P(home win) = sigmoid(a + b_in x_in + b_pre x_pre + b_elo x_elo)

fitted by penalised maximum likelihood (intercept unpenalised):
    mean log loss + lam * |b - b0|^2,
with b0 = (1, 0, 0) ("to_base": shrink to the in-season probability alone)
or b0 = (0, 0, 0) ("to_zero": ordinary ridge).
Stage-2 families (amendment 1 in prereg.json, added after stage-1 tuning):
"to_base_noint": intercept fixed at 0, slopes shrunk to (1, 0, 0);
"to_base_int": as to_base with the intercept also penalised, lam * a^2.

The stack for season V is fitted on the seasons 2012 .. V-1 (2013 excluded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FIRST = 2012
EXCLUDED = {2013}
COLS = ["p_in", "p_pre", "p_elo"]
TARGETS = {"to_base": np.array([1.0, 0.0, 0.0]), "to_zero": np.zeros(3),
           "to_base_noint": np.array([1.0, 0.0, 0.0]), "to_base_int": np.array([1.0, 0.0, 0.0])}


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(f: pd.DataFrame) -> np.ndarray:
    return np.column_stack([logit(f[c]) for c in COLS])


def fit(X: np.ndarray, y: np.ndarray, lam: float, family: str) -> np.ndarray:
    """Returns w = (a, b_in, b_pre, b_elo)."""
    b0 = TARGETS[family]
    y = np.asarray(y, float)
    n = len(y)
    free_a = family != "to_base_noint"
    pen_a = lam if family == "to_base_int" else 0.0

    def obj(v):
        w = v if free_a else np.r_[0.0, v]
        z = w[0] + X @ w[1:]
        ll = np.mean(np.logaddexp(0.0, z) - y * z)
        r = 1.0 / (1.0 + np.exp(-z)) - y
        d = w[1:] - b0
        g = np.r_[r.mean() + 2 * pen_a * w[0], X.T @ r / n + 2 * lam * d]
        f = ll + lam * d @ d + pen_a * w[0] ** 2
        return (f, g) if free_a else (f, g[1:])

    v0 = np.r_[0.0, 1.0, 0.0, 0.0] if free_a else np.r_[1.0, 0.0, 0.0]
    res = minimize(obj, v0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "gtol": 1e-10, "ftol": 1e-14})
    return res.x if free_a else np.r_[0.0, res.x]


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(w[0] + X @ w[1:])))


def train_seasons(V: int) -> list[int]:
    return [s for s in range(FIRST, V) if s not in EXCLUDED]


def walk_forward(feat: pd.DataFrame, variant: str, seasons: list[int], lam: float,
                 family: str, train_src: str = "ht", apply_src: str = "ht",
                 train_ship_where_available: bool = False) -> tuple[pd.DataFrame, dict]:
    """Stack predictions for every game of ``seasons`` (one fit per season V on
    seasons before V). Returns (frame with p_stack, {V: weights})."""
    fv = feat[feat.variant == variant]
    outs, ws = [], {}
    for V in seasons:
        tr_s = train_seasons(V)
        if train_ship_where_available:
            ship_s = set(fv[fv.src == "ship"].season_end.unique())
            tr = pd.concat([fv[(fv.src == "ship") & fv.season_end.isin(set(tr_s) & ship_s)],
                            fv[(fv.src == "ht") & fv.season_end.isin(set(tr_s) - ship_s)]])
        else:
            tr = fv[(fv.src == train_src) & fv.season_end.isin(tr_s)]
        te = fv[(fv.src == apply_src) & (fv.season_end == V)].copy()
        if len(tr) == 0 or len(te) == 0:
            raise ValueError(f"no data for V={V} ({len(tr)} train, {len(te)} apply)")
        w = fit(design(tr), tr.home_win.to_numpy(), lam, family)
        te["p_stack"] = predict(w, design(te))
        outs.append(te)
        ws[int(V)] = {"a": float(w[0]), "b_in": float(w[1]), "b_pre": float(w[2]),
                      "b_elo": float(w[3]), "n_train": int(len(tr)),
                      "train_seasons": sorted(map(int, tr.season_end.unique()))}
    return pd.concat(outs, ignore_index=True), ws
