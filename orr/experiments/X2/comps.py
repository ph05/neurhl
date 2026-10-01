"""X2: comparables (PECOTA-style) skater projections.

For a target season V, every skater with recent NHL minutes is matched to
historical player-seasons ("comparables") on

  * age (in the season being projected),
  * position (hard match: forwards with forwards, defence with defence),
  * the last three seasons' per-60 scoring rates (goals, assists), shot
    rates (shots on goal, individual xG) and usage (share of the team's EV
    and PP minutes per game, share of the schedule played).

Rates are expressed relative to league expectation for the player's own
situational minutes in that season (count / sum_k toi_k x league per-minute
rate_k), so they are era- and usage-mix-free. Each per-season rate is lightly
shrunk toward the positional mean (K_FEAT_MIN minutes) so that a 30-minute
season does not look like a 4-goals-per-60 season.

A comparable is a player-season pair (t, t+1) with t+1 <= V-1: every
trajectory comes from seasons strictly before V (walk-forward). The
projection for player i is his own recency-weighted baseline plus the
distance-weighted mean of his comparables' next-season changes:

    proj_i = base_i + sum_j w_ij r_j (y_j - base_j) / sum_j w_ij r_j

with y_j = the comparable's next-season ratio, base_j = his baseline built
exactly as base_i, r_j = toi_{t+1} / (toi_{t+1} + K_OUT_MIN) (reliability
of the outcome) and w_ij a tricube kernel on the standardised, weighted
distance over the k nearest comparables.

Features of seasons outside the panel (before 2008-09) are treated as
missing, and the distance is computed over the dimensions both players
observe (rescaled to the full weight).

Two baselines for the "change" (CompParams.baseline):
  own  base = the player's own recency-weighted (1/0.7/0.5), lightly shrunk
       3-season ratio -- the classic PECOTA construction (search s1);
  orr  base = ORR's walk-forward projected ratio for that season (for a
       comparable: ORR's projection of season t+1 made from seasons <= t),
       so the comparables carry ORR's residuals for players like the target
       (search s2; outcome seasons from 2011, since ORR's 2010 projection
       rests on a single season).
The result is blended with ORR's per-60 goal and assist rates by one weight
w: rate x ((1 - w) + w x comp / ORR), see blend_proj.
"""
from __future__ import annotations

import functools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from orr import players as PL

SITS = PL.SITS
LAG_W = (1.0, 0.7, 0.5)        # recency weights of t, t-1, t-2 (features and baseline)
K_FEAT_MIN = 150.0             # minutes of positional-mean prior on per-season feature rates
K_BASE_MIN = 150.0             # minutes of positional-mean prior on the baseline
K_OUT_MIN = 300.0              # minutes: outcome reliability r = toi/(toi + K_OUT_MIN)
MIN_TOI_HIST = 100.0           # minutes over t..t-2 to be projected / used as a comparable
MIN_TOI_OUT = 20.0             # minutes in t+1 for a comparable's outcome to count
MASK_TOI = 50.0                # a season's rate features are missing below this many minutes
Y_CLIP = 5.0                   # outcome ratio winsorised at 5x league expectation
TARGETS = ("g", "a")           # projected components
STATS = {"g": ("g",), "a": ("a1", "a2"), "sog": ("sog",), "ixg": ("ixg",)}
GROUPS = ("age", "rate", "shot", "use", "exp")


@dataclass(frozen=True)
class CompParams:
    k: int = 100                # nearest comparables
    w_age: float = 2.0          # feature-group weights in the distance
    w_rate: float = 1.0         # g/60, a/60 ratios (3 lags)
    w_shot: float = 0.5         # sog/60, ixg/60 ratios (3 lags)
    w_use: float = 1.0          # EV and PP share of team minutes per game (3 lags)
    w_exp: float = 0.5          # share of the schedule played (3 lags)
    # what a comparable's next-season change is measured from:
    #   "own": his own recency-weighted, lightly shrunk 3-season ratio (PECOTA)
    #   "orr": ORR's walk-forward projection of him for that season, so the
    #          comparables carry ORR's residuals for players like the target
    baseline: str = "own"

    def key(self) -> str:
        return str(sorted(asdict(self).items()))


# ---------------------------------------------------------------------------
# Per-season arrays (players x seasons)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def season_arrays() -> dict:
    P = PL.panel()
    lv = PL.league_levels()
    toi = P.x["toi_all"]
    out = {"toi": toi}
    for name, parts in STATS.items():
        n = sum(P.x[f"{p}_{k}"] for p in parts for k in SITS)
        e = sum(P.x[f"toi_{k}"] * lv[f"{p}_{k}"].to_numpy()[None, :] for p in parts for k in SITS)
        out[f"n_{name}"], out[f"e_{name}"] = n, e
        out[f"lpm_{name}"] = e.sum(0) / np.maximum(toi.sum(0), 1e-9)     # expected per skater-minute
        for pos in "FD":
            m = P.pos == pos
            out[f"mu_{name}_{pos}"] = n[m].sum(0) / np.maximum(e[m].sum(0), 1e-9)
    gp = P.x["gp"]
    for k in ("ev", "pp"):
        tm = lv[f"team_min_{k}"].to_numpy()[None, :]
        out[f"{k}_share"] = np.where(gp > 0, P.x[f"toi_{k}"] / np.maximum(gp, 1) / tm, 0.0)
    out["gp_share"] = gp / np.maximum(P.x["sched"], 1.0)
    return out


def _mu(A: dict, name: str, j: int, pos: np.ndarray) -> np.ndarray:
    return np.where(pos == "D", A[f"mu_{name}_D"][j], A[f"mu_{name}_F"][j])


def _shrunk_ratio(A, name, j, pos, kmin):
    ke = kmin * A[f"lpm_{name}"][j]
    return (A[f"n_{name}"][:, j] + ke * _mu(A, name, j, pos)) / (A[f"e_{name}"][:, j] + ke)


@functools.lru_cache(maxsize=None)
def base_block(t: int) -> dict:
    """Features, masks and baselines of every panel player at base season t
    (projecting t+1). Feature order: age, then per lag (t, t-1, t-2):
    g, a | sog, ixg | ev_share, pp_share | gp_share."""
    P = PL.panel()
    A = season_arrays()
    pos = P.pos
    npl = len(P.ids)
    feats, masks, groups, lagw = [], [], [], []
    # age in the projected season
    feats.append(PL.ages(P, t + 1))
    masks.append(np.ones(npl, bool))
    groups.append("age")
    lagw.append(1.0)
    hist_toi = np.zeros(npl)
    num = {s: np.zeros(npl) for s in TARGETS}
    den = {s: np.zeros(npl) for s in TARGETS}
    for lag, lw in enumerate(LAG_W):
        s = t - lag
        cens = s < P.seasons[0]
        if cens:
            for grp, nf in (("rate", 2), ("shot", 2), ("use", 2), ("exp", 1)):
                for _ in range(nf):
                    feats.append(np.zeros(npl))
                    masks.append(np.zeros(npl, bool))
                    groups.append(grp)
                    lagw.append(lw)
            continue
        j = P.j(s)
        toi = A["toi"][:, j]
        hist_toi += toi
        okr = toi >= MASK_TOI
        for grp, names in (("rate", ("g", "a")), ("shot", ("sog", "ixg"))):
            for nm in names:
                feats.append(_shrunk_ratio(A, nm, j, pos, K_FEAT_MIN))
                masks.append(okr)
                groups.append(grp)
                lagw.append(lw)
        played = P.x["gp"][:, j] > 0
        for nm in ("ev_share", "pp_share"):
            feats.append(A[nm][:, j])
            masks.append(played)
            groups.append("use")
            lagw.append(lw)
        feats.append(A["gp_share"][:, j])
        masks.append(np.ones(npl, bool))
        groups.append("exp")
        lagw.append(lw)
        for s_ in TARGETS:
            num[s_] += lw * A[f"n_{s_}"][:, j]
            den[s_] += lw * A[f"e_{s_}"][:, j]
    jt = P.j(t)
    base = {}
    for s_ in TARGETS:
        ke = K_BASE_MIN * A[f"lpm_{s_}"][jt]
        base[s_] = (num[s_] + ke * _mu(A, s_, jt, pos)) / (den[s_] + ke)
    return {"X": np.column_stack(feats), "M": np.column_stack(masks),
            "groups": np.array(groups), "lagw": np.array(lagw),
            "hist_toi": hist_toi, "base": base}


@functools.lru_cache(maxsize=None)
def pool(V: int) -> dict:
    """Comparable player-seasons: base t in [first panel season, V-2],
    outcome t+1 <= V-1, by position."""
    P = PL.panel()
    A = season_arrays()
    rows = {pos: {"X": [], "M": [], "y_g": [], "y_a": [], "b_g": [], "b_a": [], "r": [],
                  "pid": [], "t": []} for pos in "FD"}
    for t in range(int(P.seasons[0]), V - 1):
        if t + 1 > P.seasons[-1]:
            break
        bb = base_block(t)
        j1 = P.j(t + 1)
        toi1 = A["toi"][:, j1]
        ok = (bb["hist_toi"] >= MIN_TOI_HIST) & (toi1 >= MIN_TOI_OUT)
        for pos in "FD":
            m = ok & (P.pos == pos)
            R = rows[pos]
            R["X"].append(bb["X"][m])
            R["M"].append(bb["M"][m])
            for s_ in TARGETS:
                y = A[f"n_{s_}"][m, j1] / np.maximum(A[f"e_{s_}"][m, j1], 1e-9)
                R[f"y_{s_}"].append(np.clip(y, 0.0, Y_CLIP))
                R[f"b_{s_}"].append(bb["base"][s_][m])
            R["r"].append(toi1[m] / (toi1[m] + K_OUT_MIN))
            R["pid"].append(P.ids[m])
            R["t"].append(np.full(m.sum(), t))
    out = {}
    for pos, R in rows.items():
        if not R["X"]:
            out[pos] = None
            continue
        d = {k: np.concatenate(v) for k, v in R.items()}
        # standardisation over the pool (observed values only)
        X, M = d["X"], d["M"]
        mu = np.array([X[M[:, f], f].mean() if M[:, f].any() else 0.0 for f in range(X.shape[1])])
        sd = np.array([X[M[:, f], f].std() if M[:, f].sum() > 1 else 1.0 for f in range(X.shape[1])])
        d["mu"], d["sd"] = mu, np.where(sd > 1e-9, sd, 1.0)
        d["Z"] = np.where(M, (X - d["mu"]) / d["sd"], 0.0)
        out[pos] = d
    return out


ORR_MIN_OUT_SEASON = 2011     # "orr" baseline: ORR's projection of 2010 rests on one season


@functools.lru_cache(maxsize=None)
def orr_ratio_season(s: int) -> dict:
    """ORR's walk-forward projected ratio (g, a) for season s, every panel
    player with NHL history before s (NaN otherwise), with the tuned knobs."""
    P = PL.panel()
    proj = PL.project(s, PL.load_params())
    o = orr_ratio(proj, s)
    rows = proj.player_id.map(P.row).to_numpy()
    out = {}
    for s_ in TARGETS:
        a = np.full(len(P.ids), np.nan)
        a[rows] = o[s_]
        out[s_] = a
    return out


@functools.lru_cache(maxsize=None)
def pool_orr(V: int) -> dict:
    """ORR's projected ratio of each pool row's outcome season (t+1)."""
    P = PL.panel()
    pl = pool(V)
    out = {}
    for pos, d in pl.items():
        if d is None:
            out[pos] = None
            continue
        o = {s_: np.full(len(d["t"]), np.nan) for s_ in TARGETS}
        for t in np.unique(d["t"]):
            if t + 1 < ORR_MIN_OUT_SEASON:
                continue
            m = d["t"] == t
            rr = np.array([P.row[p] for p in d["pid"][m]])
            ors = orr_ratio_season(int(t + 1))
            for s_ in TARGETS:
                o[s_][m] = ors[s_][rr]
        out[pos] = o
    return out


def _weights_vec(cp: CompParams, groups: np.ndarray, lagw: np.ndarray) -> np.ndarray:
    g = {"age": cp.w_age, "rate": cp.w_rate, "shot": cp.w_shot, "use": cp.w_use, "exp": cp.w_exp}
    return np.array([g[x] for x in groups]) * lagw


def project_comps(V: int, cp: CompParams, chunk: int = 128) -> pd.DataFrame:
    """Comparables projection (relative ratios for g and a) for every panel
    player with >= MIN_TOI_HIST minutes over V-1..V-3."""
    P = PL.panel()
    bb = base_block(V - 1)
    pl = pool(V)
    W = _weights_vec(cp, bb["groups"], bb["lagw"])
    wsum = W.sum()
    tgt = bb["hist_toi"] >= MIN_TOI_HIST
    res = []
    for pos in "FD":
        d = pl.get(pos)
        idx = np.where(tgt & (P.pos == pos))[0]
        if d is None or not len(idx):
            continue
        if cp.baseline == "orr":
            ot = orr_ratio_season(V)
            po = pool_orr(V)[pos]
            tb = {s_: ot[s_][idx] for s_ in TARGETS}
            pb = {s_: po[s_] for s_ in TARGETS}
            okp = np.isfinite(pb["g"]) & np.isfinite(pb["a"])
        else:
            tb = {s_: bb["base"][s_][idx] for s_ in TARGETS}
            pb = {s_: d[f"b_{s_}"] for s_ in TARGETS}
            okp = np.ones(len(d["t"]), bool)
        rsel = np.where(okp, d["r"], 0.0)             # rows without a baseline get no weight
        Zt = np.where(bb["M"][idx], (bb["X"][idx] - d["mu"]) / d["sd"], 0.0)
        Mt = bb["M"][idx].astype(float)
        Zp, Mp = d["Z"], d["M"].astype(float)
        k = min(cp.k, len(Zp) - 1)
        comp = {s_: np.zeros(len(idx)) for s_ in TARGETS}
        neff = np.zeros(len(idx))
        dmed = np.zeros(len(idx))
        Zp2 = Zp ** 2
        for c0 in range(0, len(idx), chunk):
            sl = slice(c0, c0 + chunk)
            zt, mt = Zt[sl], Mt[sl]
            # masked weighted squared distance over the dimensions both observe
            # (Z is 0 where masked), rescaled to the full weight:
            # sum_f W m_t m_p (z_t - z_p)^2 = (W m_t z_t^2) Mp' - 2 (W z_t) Zp' + (W m_t) Zp2'
            wm = W[None, :] * mt
            d2 = (wm * zt ** 2) @ Mp.T - 2.0 * (W[None, :] * zt) @ Zp.T + wm @ Zp2.T
            frac = (wm @ Mp.T) / wsum
            d2 = np.maximum(d2, 0.0) / np.maximum(frac, 1e-3)
            dist = np.sqrt(d2)
            nn = np.argpartition(dist, k, axis=1)[:, :k + 1]
            dn = np.take_along_axis(dist, nn, 1)
            order = np.argsort(dn, 1)
            nn = np.take_along_axis(nn, order, 1)
            dn = np.take_along_axis(dn, order, 1)
            dmax = dn[:, -1:] * 1.0001 + 1e-9                             # (k+1)-th distance
            kern = (1.0 - (dn[:, :k] / dmax) ** 3) ** 3
            nn = nn[:, :k]
            wr = kern * rsel[nn]
            ws = wr.sum(1)
            for s_ in TARGETS:
                delta = np.nan_to_num(d[f"y_{s_}"][nn] - pb[s_][nn])
                comp[s_][sl] = tb[s_][sl] + (wr * delta).sum(1) / np.maximum(ws, 1e-12)
            neff[sl] = ws ** 2 / np.maximum((wr ** 2).sum(1), 1e-12)
            dmed[sl] = np.median(dn[:, :k], 1)
        res.append(pd.DataFrame({"player_id": P.ids[idx], "pos": pos,
                                 "comp_g": np.maximum(comp["g"], 0.01),
                                 "comp_a": np.maximum(comp["a"], 0.01),
                                 "base_g": bb["base"]["g"][idx], "base_a": bb["base"]["a"][idx],
                                 "n_eff": neff, "d_med": dmed}))
    return pd.concat(res, ignore_index=True)


# ---------------------------------------------------------------------------
# Blend with ORR's rates
# ---------------------------------------------------------------------------
F_CLIP = (1 / 3.0, 3.0)
COMP_COLS = {"g": ("g",), "a": ("a1", "a2")}


def orr_ratio(proj: pd.DataFrame, V: int) -> dict:
    """ORR's projected rate relative to league expectation at its own
    projected situational usage (same units as the comparables' ratios)."""
    tgt = PL.project_league(V)
    out = {}
    for s_, parts in COMP_COLS.items():
        num = sum(proj[f"tpg_{k}"] * proj[f"r60_{p}_{k}"] / 60.0 for p in parts for k in SITS)
        den = sum(proj[f"tpg_{k}"] * float(tgt[f"{p}_{k}"]) for p in parts for k in SITS)
        out[s_] = (num / den.clip(lower=1e-12)).to_numpy()
    return out


def blend_proj(proj: pd.DataFrame, V: int, comps: pd.DataFrame, w: float) -> pd.DataFrame:
    """Multiply ORR's per-60 goal and assist rates (every situation, and
    their posterior SDs) by (1 - w) + w * comp/ORR. Players without a
    comparables projection keep ORR's rates."""
    if w <= 0 or comps is None or not len(comps):
        return proj
    proj = proj.copy()
    o = orr_ratio(proj, V)
    c = comps.set_index("player_id")
    for s_, parts in COMP_COLS.items():
        cv = proj.player_id.map(c[f"comp_{s_}"]).to_numpy(float)
        f = np.clip(cv / np.maximum(o[s_], 1e-9), *F_CLIP)
        m = np.where(np.isfinite(f), (1.0 - w) + w * f, 1.0)
        for p in parts:
            for k in SITS:
                proj[f"r60_{p}_{k}"] = proj[f"r60_{p}_{k}"] * m
                proj[f"sd60_{p}_{k}"] = proj[f"sd60_{p}_{k}"] * m
    return proj
