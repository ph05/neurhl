"""Turn team points targets into scoring-model ratings.

The team layer (orr.teams) produces a points target per team: the
blend of the market line, team history and the roster model.
The season simulator works on offence/defence log-rates. This module solves
for ratings whose expected points over the ACTUAL schedule (opponents, home
games, rest) equal the targets, while keeping each team's offence/defence
style from the model. It then converts the target's uncertainty (in points)
into a rating SD, so the simulated spread of outcomes is set by measured
backtest error rather than by an assumed constant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_points(schedule: pd.DataFrame, ratings: pd.DataFrame, model,
                    game_adj: pd.DataFrame | None = None) -> pd.Series:
    """Analytic expected standings points per team over `schedule`."""
    r = ratings.set_index("team")
    o_h, d_h = r.o.reindex(schedule.home).to_numpy(), r.d.reindex(schedule.home).to_numpy()
    o_a, d_a = r.o.reindex(schedule.away).to_numpy(), r.d.reindex(schedule.away).to_numpy()
    ah = aa = 0.0
    if game_adj is not None:
        ga = game_adj.set_index("game_id").reindex(schedule.game_id)
        ah, aa = ga.adj_h.fillna(0).to_numpy(), ga.adj_a.fillna(0).to_numpy()
    lam_h, lam_a = model.rates(o_h, d_h, o_a, d_a, ah, aa)
    p = model.probs(lam_h, lam_a)
    pts = pd.concat([pd.Series(p["exp_pts_home"], index=schedule.home.to_numpy()),
                     pd.Series(p["exp_pts_away"], index=schedule.away.to_numpy())])
    return pts.groupby(level=0).sum()


def solve_ratings(schedule: pd.DataFrame, targets: pd.Series, style: pd.DataFrame,
                  model, game_adj=None, tol: float = 0.02, max_iter: int = 60) -> pd.DataFrame:
    """Find (o, d) per team whose expected points equal `targets`.

    style: team, o_m, d_m -- the model's offence/defence log-rates. The solver
    moves each team by a net amount delta split evenly (o = o_m + delta/2,
    d = d_m - delta/2), so the team's GF/GA style is preserved.
    """
    teams = sorted(targets.index)
    st = style.set_index("team").reindex(teams).fillna(0.0)
    # centre style so the league average is exactly the model's mu
    o_m = st.o_m - st.o_m.mean()
    d_m = st.d_m - st.d_m.mean()
    # The league's total points are fixed by the model's extra-time rate, so
    # targets are shifted (not scaled) to that total before solving.
    flat = pd.DataFrame({"team": teams, "o": 0.0, "d": 0.0})
    total = expected_points(schedule, flat, model, game_adj).sum()
    targets = targets.reindex(teams) + (total - targets.sum()) / len(teams)
    delta = pd.Series(0.0, index=teams)

    def ratings_for(dl):
        return pd.DataFrame({"team": teams, "o": (o_m + dl / 2).to_numpy(),
                             "d": (d_m - dl / 2).to_numpy()})

    for it in range(max_iter):
        r = ratings_for(delta)
        e = expected_points(schedule, r, model, game_adj).reindex(teams)
        # League total points depend on how unequal the teams are (fewer ties
        # between mismatched teams), so only RELATIVE points are matched and
        # the total is whatever the scoring model implies.
        err = (targets - targets.mean()) - (e - e.mean())
        if err.abs().max() < tol:
            break
        # Newton step with a numerical Jacobian (every team's points depend on
        # its opponents' ratings too); the system is singular along the
        # all-teams-equal direction, so solve in the least-norm sense.
        if it < 4 or it % 4 == 0:
            J = np.zeros((len(teams), len(teams)))
            h = 0.01
            for j, t in enumerate(teams):
                d2 = delta.copy()
                d2[t] += h
                J[:, j] = (expected_points(schedule, ratings_for(d2), model, game_adj)
                           .reindex(teams) - e).to_numpy() / h
        step = np.linalg.lstsq(J, err.to_numpy(), rcond=None)[0]
        delta = delta + step
        delta -= delta.mean()
    r["target"] = (targets - targets.mean() + e.mean()).to_numpy()
    r["expected"] = e.to_numpy()
    return r


def points_per_rating(schedule, ratings, model, game_adj=None) -> pd.Series:
    """d(expected season points)/d(net rating) per team (one team moved alone)."""
    base = expected_points(schedule, ratings, model, game_adj)
    out = {}
    for t in ratings.team:
        r2 = ratings.copy()
        m = r2.team == t
        r2.loc[m, "o"] += 0.01
        r2.loc[m, "d"] -= 0.01
        out[t] = (expected_points(schedule, r2, model, game_adj)[t] - base[t]) / 0.02
    return pd.Series(out)
