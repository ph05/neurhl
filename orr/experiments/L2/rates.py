"""L2: component-specific shrinkage for skater scoring rates.

A copy of the rate step of orr.players (rates_raw, the aging fit that reads
it, and the per-60 columns of project) with one change: the empirical-Bayes
prior weight of each scoring component carries its own multiplier,

    K[c, pos] = kappa * m[comp, sit, pos] * phi * prior / (league_rate * tau2)

where comp is one of

    g     goals (the directly-shrunk goal rate, g_<sit>)
    a1    primary assists (a1_<sit>)
    a2    secondary assists (a2_<sit>)
    shot  shots: sog_<sit> and ixg_<sit> share one multiplier. ixG (the
          individual expected goals of a player's shots) is the shot
          component that enters points, through the goals-via-xG path;
          shots on goal do not enter points.

sit is 'ev' or 'pp', pos 'F' or 'D' (16 multipliers). With every m = 1 this is
exactly orr.players (checked by run.py --verify). Everything else stays as
shipped: kappa (0.35) for SH / other-situation columns and peripherals, the
finishing multiplier's prior weight, usage, prospects, on-ice impact.

The age curves are fitted walk-forward from the residuals of the model's own
un-aged projections (players._aging_cached). Changing the shrinkage changes
those projections, so the curves are refitted here from the modified rates,
as they would be if the multipliers were Params fields in players.py. The TOI
and peripheral curves do not read the scoring rates and are unchanged.
"""
from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd

from orr import players as PL

SITS = PL.SITS
COMPS = ("g", "a1", "a2", "shot")
KSITS = ("ev", "pp")
POSS = ("F", "D")
KNOBS = tuple(f"{c}_{s}_{p}" for p in POSS for s in KSITS for c in ("a1", "a2", "g", "shot"))
COMP_COLS = {"g": ("g",), "a1": ("a1",), "a2": ("a2",), "shot": ("sog", "ixg")}


def ones() -> dict:
    return {k: 1.0 for k in KNOBS}


def mkey(m: dict) -> str:
    return json.dumps({k: round(float(m[k]), 6) for k in KNOBS}, sort_keys=True)


def col_mult(m: dict, c: str, pos: str) -> float:
    """Multiplier on K for stat column c (e.g. 'a1_ev') and position."""
    st, sit = c.split("_", 1)
    if sit not in KSITS:
        return 1.0
    for comp, sts in COMP_COLS.items():
        if st in sts:
            return float(m[f"{comp}_{sit}_{pos}"])
    return 1.0


# ---------------------------------------------------------------------------
# Copy of players.rates_raw with per-component multipliers on K
# ---------------------------------------------------------------------------
def rates_raw_l2(P: PL.Panel, V: int, prm: PL.Params, usage: dict, m: dict,
                 pr_prosp: np.ndarray | None = None) -> dict:
    lv = PL.league_levels()
    idx, w = PL._weights(P, V, prm.rate_decay)
    pri = PL.rate_priors(V, prm.prior_window, prm.eb_window)
    X = PL._design(usage["tpg_ev"], usage["tpg_pp"], usage["tpg_sh"])
    pmult = np.ones(len(P.ids))
    if pr_prosp is not None:
        pu = PL.usage_points_prior(V, prm, P.pos, X)
        ok = np.isfinite(pr_prosp)
        pmult[ok] = (prm.prosp_alpha * pr_prosp[ok]
                     + (1 - prm.prosp_alpha) * pu[ok]) / pu[ok]
        pmult = np.clip(pmult, 0.3, 4.0)
    rel, var, T = {}, {}, {}
    cols = [f"{s}_{k}" for s in PL.SIT_STATS for k in SITS] + [f"{s}_all" for s in PL.PERIPH]
    tgtV = PL.project_league(V)
    for c in cols:
        tcol = PL.stat_toi(c)
        ell = lv[c].to_numpy()[idx]
        t = (P.x[tcol][:, idx] * w).sum(1)
        n = (P.x[c][:, idx] / np.maximum(ell, 1e-12) * w).sum(1)
        prior = np.zeros(len(P.ids))
        K = np.zeros(len(P.ids))
        tau2 = np.zeros(len(P.ids))
        for pos in ("F", "D"):
            mk = P.pos == pos
            e = pri[(c, pos)]
            pr = np.clip(X[mk] @ e["beta"], 1e-3, None)
            if c.split("_")[0] in PL.SCORING_PRIOR_STATS:
                pr = pr * pmult[mk]
            prior[mk] = pr
            ellV = float(tgtV[c])
            K[mk] = prm.kappa * col_mult(m, c, pos) * e["phi"] * pr / (max(ellV, 1e-12) * e["tau2"])
            tau2[mk] = e["tau2"]
        rel[c] = (n + K * prior) / (t + K)
        var[c] = tau2 * K / (K + t)
        T[c] = t
    fp = PL.finishing_prior(V, prm.eb_window)
    rho = lv[[f"rho_{k}" for k in SITS]].to_numpy()[idx]
    g = sum((P.x[f"g_{k}"][:, idx] * w).sum(1) for k in SITS)
    xg = sum((P.x[f"ixg_{k}"][:, idx] * rho[:, i] * w).sum(1) for i, k in enumerate(SITS))
    f0 = np.where(P.pos == "D", fp["D"]["f0"], fp["F"]["f0"])
    kf = prm.kappa * np.where(P.pos == "D", fp["D"]["k_fin"], fp["F"]["k_fin"])
    fin = (g + kf * f0) / (xg + kf)
    for k in SITS:
        c = f"g_{k}"
        via_xg = rel[f"ixg_{k}"] * tgtV[f"ixg_{k}"] * tgtV[f"rho_{k}"] * fin / tgtV[c]
        rel[c] = prm.fin_weight * via_xg + (1 - prm.fin_weight) * rel[c]
    rel["fin"] = fin
    return {"rel": rel, "var": var, "T": T}


_RAW: dict = {}


def raw_l2(V: int, prm: PL.Params, m: dict) -> dict:
    """players.raw_projection with the L2 rates (usage and prospect priors do
    not depend on the rates and are taken from the shipped function)."""
    key = (V, prm.key(), mkey(m))
    if key not in _RAW:
        P = PL.panel()
        base = PL.raw_projection(V, prm)
        r = rates_raw_l2(P, V, prm, base["usage"], m, pr_prosp=base["pr_prosp"])
        _RAW[key] = {"usage": base["usage"], **r, "has_hist": base["has_hist"]}
    return _RAW[key]


# ---------------------------------------------------------------------------
# Copy of players._aging_cached reading the L2 rates
# ---------------------------------------------------------------------------
_AGE_ROWS: dict = {}


def _age_rows(s: int, prm: PL.Params, m: dict) -> dict:
    """(age, actual, projected) per aging group and position in season s."""
    key = (s, prm.key(), mkey(m))
    if key in _AGE_ROWS:
        return _AGE_ROWS[key]
    P = PL.panel()
    lv = PL.league_levels()
    raw = raw_l2(s, prm, m)
    js = P.j(s)
    tgt = PL.project_league(s)
    has = raw["has_hist"] & (P.x["gp"][:, js] > 0)
    age = PL.ages(P, s)
    out = {}
    for grp, stats in PL.AGE_GROUPS.items():
        act = np.zeros(len(P.ids))
        pred = np.zeros(len(P.ids))
        for st in stats:
            cols = [f"{st}_{k}" for k in SITS] if st in PL.SIT_STATS else [f"{st}_all"]
            for c in cols:
                act += P.x[c][:, js]
                pred += raw["rel"][c] * lv.loc[s, c] * P.x[PL.stat_toi(c)][:, js]
        for pos in "FD":
            mk = has & (P.pos == pos)
            out[(grp, pos)] = (age[mk], act[mk], pred[mk])
    act = sum(P.x[f"toi_{k}"][:, js] for k in SITS)
    pred = sum(raw["usage"][f"tpg_{k}"] * lv.loc[s, f"team_min_{k}"] / tgt[f"team_min_{k}"]
               for k in SITS) * P.x["gp"][:, js]
    for pos in "FD":
        mk = has & (P.pos == pos)
        out[("toi", pos)] = (age[mk], act[mk], pred[mk])
    _AGE_ROWS[key] = out
    return out


_AGE: dict = {}


def aging_l2(V: int, prm: PL.Params, m: dict) -> dict:
    key = (V, prm.key(), mkey(m))
    if key in _AGE:
        return _AGE[key]
    P = PL.panel()
    seasons = [s for s in range(PL.FIRST_SEASON + 1, V) if s <= P.seasons[-1]]
    out = {}
    for grp in (*PL.AGE_GROUPS, "toi"):
        for pos in "FD":
            if not seasons:
                out[(grp, pos)] = np.zeros(2 + len(PL.AGE_KNOTS))
                continue
            rows = [_age_rows(s, prm, m)[(grp, pos)] for s in seasons]
            a, y, e = (np.concatenate([r[i] for r in rows]) for i in range(3))
            out[(grp, pos)] = PL._fit_age_curve(a, y, e)
    _AGE[key] = out
    return out


def age_multiplier_l2(V, prm, m, group, pos, age) -> np.ndarray:
    if not prm.age_curve:
        return np.ones(len(age))
    coef = aging_l2(V, prm, m)
    X = PL._age_basis(np.asarray(age, float))
    out = np.ones(len(age))
    for p in "FD":
        mk = pos == p
        out[mk] = np.exp(X[mk] @ coef[(group, p)])
    return np.clip(out, 0.5, 1.6)


# ---------------------------------------------------------------------------
# players.project with the L2 per-60 scoring columns
# ---------------------------------------------------------------------------
def project_l2(V: int, prm: PL.Params, m: dict, ids) -> pd.DataFrame:
    """PL.project(V, prm, ids=ids) with every r60_/sd60_ column of the
    situational stats (g, a1, a2, sog, ixg) and 'fin' recomputed from the L2
    rates and L2 age curves. Usage, peripherals and on-ice columns are the
    shipped ones (they do not read the scoring rates)."""
    base = PL.project(V, prm, ids=ids)
    P = PL.panel()
    raw = raw_l2(V, prm, m)
    tgt = PL.project_league(V)
    age = PL.ages(P, V)
    has = raw["has_hist"]
    mult = {g: np.where(has, age_multiplier_l2(V, prm, m, g, P.pos, age), 1.0)
            for g in ("g", "a", "shot")}
    new = {}
    for s in PL.SIT_STATS:
        for k in SITS:
            c = f"{s}_{k}"
            mm = mult[PL.stat_group(s)]
            new[f"r60_{c}"] = raw["rel"][c] * tgt[c] * 60.0 * mm
            new[f"sd60_{c}"] = np.sqrt(raw["var"][c]) * tgt[c] * 60.0 * mm
    new["fin"] = raw["rel"]["fin"]
    row = base.player_id.map(P.row).to_numpy()
    out = base.copy()
    for c, v in new.items():
        out[c] = v[row]
    return out
