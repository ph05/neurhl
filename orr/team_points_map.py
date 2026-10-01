"""Map bottom-up roster components to standings points, walk-forward.

orr.team_components produces, per team-season, the summed projections of
the players on the opening roster (TOI-weighted on-ice xG impacts, power-play
and penalty-kill quality, finishing, goaltending, projected goals). This module
learns, on seasons before the target only, how those components translate
into points per 82 games above the league mean, and returns that prediction
as the bottom-up view `bu_rel82` for the team-layer blend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orr import config as C
from orr import teams as T

# Projected (preseason) components only. The historical table also carries the
# season's ACTUAL outcomes (act_*) for evaluation; they must never be features.
COMPONENTS = ["ev_xgf_impact", "ev_xga_impact", "pp_xgf60", "pk_xga60",
              "finishing_pg", "gf_pg_skaters", "goalie_gsax60"]


def component_cols(hist: pd.DataFrame) -> list[str]:
    cols = [c for c in COMPONENTS if c in hist.columns]
    assert not any(c.startswith("act_") for c in cols)
    return cols


def _within_season(hist, cols):
    """Components relative to their season's league mean (rosters are compared
    with their contemporaries, as standings points are)."""
    h = hist.copy()
    for c in cols:
        h[c] = h[c] - h.groupby("season_end")[c].transform("mean")
    return h


def fit(hist: pd.DataFrame, target_season: int, alpha: float = 8.0, cols=None) -> dict:
    cols = cols or component_cols(hist)
    h = _within_season(hist, cols)
    st = T.standings_all()
    st = st.assign(act=st.pts82 - st.groupby("season_end").pts82.transform("mean"))
    tr = h[(h.season_end < target_season) & ~h.season_end.isin(C.BROKEN_SEASONS)].merge(
        st[["season_end", "team", "act"]], on=["season_end", "team"]).dropna(subset=cols + ["act"])
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + alpha * len(Z) / 100.0 * np.eye(len(cols))
    beta = np.linalg.solve(A, Z.T @ tr.act.to_numpy())
    resid = tr.act.to_numpy() - Z @ beta
    return {"cols": cols, "mu": mu, "sd": sd, "beta": beta, "n": len(tr),
            "resid_sd": float(resid.std())}


def predict(f: dict, hist: pd.DataFrame, season_end: int) -> pd.DataFrame:
    h = _within_season(hist, f["cols"])
    h = h[h.season_end == season_end]
    Z = (h[f["cols"]].to_numpy(float) - f["mu"]) / f["sd"]
    return pd.DataFrame({"team": h.team.to_numpy(), "season_end": season_end,
                         "bu_rel82": Z @ f["beta"]})


def walk_forward_points(hist: pd.DataFrame, first: int = 2012) -> pd.DataFrame:
    """bu_rel82 for every season from `first`, each fitted on earlier seasons."""
    out = []
    for V in sorted(hist.season_end.unique()):
        if V < first:
            continue
        f = fit(hist, V)
        if f["n"] < 60:
            continue
        out.append(predict(f, hist, V))
    return pd.concat(out, ignore_index=True)
