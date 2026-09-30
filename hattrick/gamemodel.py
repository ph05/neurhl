"""HatTrick game model: one NHL game from two teams' log-rate strengths.

Scoring model
-------------
Team strength lives on the LOG-RATE scale. For a game

    log lam_home = mu + h + o_home + d_away + ctx_home
    log lam_away = mu     + o_away + d_home + ctx_away

where ``lam`` is the EXPECTED NUMBER OF REGULATION GOALS (empty-net goals
included), ``mu`` the league log scoring level of the season, ``h`` home ice,
``o`` a team's offence and ``d`` its defence (a positive ``d`` means it allows
more), and ``ctx`` rest / travel / starting-goalie offsets (``rates``).

Regulation score = Poisson base + end-game layer
    NHL regulation ties run 21-25%, against 16-18% under independent Poisson,
    and one-goal regulation margins are far rarer than Poisson says (19-23% vs
    30%) while three-goal margins are more common (19-23% vs 15%). Both are the
    signature of the pulled goalie: a team trailing by one or two late either
    scores (tie / closer) or concedes into the empty net. Each team's goals
    are also UNDERDISPERSED (var/mean ~0.92 given its rate: a leading team
    sits on its lead, a trailing team pushes), which makes results more
    predictable from strength than Poisson says. The model therefore draws
    base goals X, Y independently from a Conway-Maxwell-Poisson distribution
    p(x) ~ theta^x / (x!)^kappa (kappa = 1 is Poisson, fitted kappa ~1.1-1.2)
    and then applies one end-game transition to the base margin D = X - Y:

      |D| = 1: trailing team ties with prob a1*exp(g*s),
               leader scores an empty-netter with prob b1*exp(-g*s)
      |D| = 2: trailing team scores with prob a2*exp(g*s),
               leader scores an empty-netter with prob b2*exp(-g*s)

    with s = log(lam_trailing / lam_leading). The base means (bh, ba) are
    solved so that the layer is MEAN-PRESERVING: E[regulation goals] = lam
    exactly, so ratings keep the interpretation "expected regulation goals".
    kappa and the layer are fitted jointly by maximum likelihood on the full
    regulation score (``fit_layer``) and validated out of sample against
    independent Poisson and a diagonal (Dixon-Coles-style) tie inflation in
    ``hattrick.backtest.games_bt`` (margin distribution, tie rate,
    total-goals mean and variance, win-probability calibration).

Overtime / shootout
    P(decided in OT | tie)       = sigmoid(c0 + c1*log((lh+la)/LREF))
    P(home wins OT | OT decided) = sigmoid(a_ot + b_ot*log(lh/la))
    P(home wins SO)              = sigmoid(a_so + b_so*log(lh/la))
    Parameters are era-specific (3-on-3 overtime from 2015-16) and fitted on
    walk-forward ratings (``fit_ot``).

Goals convention
    ``e_gf_*`` counts regulation goals + the overtime winner + one goal for a
    shootout win (the NHL standings convention); ``e_gf_*_noso`` excludes the
    shootout goal.

Calling the model for a schedule (e.g. the 2026-27 season simulator)
--------------------------------------------------------------------
    from hattrick import gamemodel as GM, ratings as R
    P = GM.load_params()                       # hattrick/output/params/gamemodel.json
    pre = R.preseason(2027)                    # team, o, d, o_sd, d_sd, od_cov
    sch = GM.schedule_features(D.schedule_2027(), 2027)   # rest/travel columns
    o = pre.set_index("team").o; d = pre.set_index("team").d
    lh, la = GM.rates(P, o[sch.home].to_numpy(), d[sch.home].to_numpy(),
                      o[sch.away].to_numpy(), d[sch.away].to_numpy(), sch)
    probs = GM.outcome_probs(lh, la, P)        # dict of arrays, one per game
    # Monte Carlo: tile lh/la over simulations (or draw o, d per simulation
    # from N(o, o_sd) etc. and recompute rates), then
    s = GM.sample(lh_tiled, la_tiled, P, rng)  # reg_h, reg_a, extra, home_win,
                                               # gf_h, gf_a (standings convention)
Everything is vectorised numpy; ``sample`` handles ~27M games in chunks.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, ive

from hattrick import config as C

PARAMS_FILE = C.PARAMS / "gamemodel.json"
LREF = 5.8          # reference total regulation goals for the OT-decided logit
KMAX = 12           # margin support |D| <= KMAX for probability sums


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DEFAULT = {
    # end-game layer (pulled goalie) -- fitted by fit_layer
    "layer": {"a1": 0.20, "b1": 0.20, "a2": 0.06, "b2": 0.33, "g": 0.2},
    # base goal dispersion (Conway-Maxwell-Poisson; 1 = Poisson) -- fit_layer
    "kappa": 1.15,
    # overtime / shootout (3-on-3 era defaults) -- fitted by fit_ot
    "ot": {"c0": 0.62, "c1": 0.0, "a_ot": 0.08, "b_ot": 1.0, "a_so": 0.0,
           "b_so": 0.3},
    # league level and home ice for the target season (log scale)
    "mu": float(np.log(2.95)), "h": 0.07,
    # context coefficients on the log-rate scale: for team X in a game, its
    # own offence gets beta_o[f]*f_X and the goals it allows beta_d[f]*f_X
    "ctx": {"b2b_o": -0.03, "b2b_d": 0.04, "rest3_o": 0.0, "rest3_d": 0.0,
            "km_o": 0.0, "km_d": 0.0, "tz_o": 0.0, "tz_d": 0.0},
    # starting goalie: log multiplier on goals AGAINST per unit of save
    # talent (goals saved above average per shot on goal, a fraction) above
    # the team's usual starter. coef is fitted (structural goalie GLM); the
    # "1:1" hypothesis (talent converts fully into goals) is -1/(1 - sv_lg).
    "goalie": {"coef": -10.5, "sv_lg": 0.905},
}


def load_params(path=PARAMS_FILE) -> dict:
    """Fitted parameters (falls back to DEFAULT if never fitted)."""
    try:
        P = json.loads(path.read_text())
    except FileNotFoundError:
        P = {}
    out = json.loads(json.dumps(DEFAULT))
    for k, v in P.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def save_params(P: dict, path=PARAMS_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(P, indent=1, default=float))


# ---------------------------------------------------------------------------
# Context -> rates
# ---------------------------------------------------------------------------
CTX_FEATURES = ("b2b", "rest3", "km", "tz")


def ctx_features(rest, km, dtz) -> dict:
    """Per-team context features from rest days, km travelled and time-zone
    change (hours) before the game.
      b2b   played the previous day (rest == 1)
      rest3 two or more full days off (rest >= 3), season openers excluded
      km    log1p(km / 1000)  (distance from the previous venue)
      tz    |time-zone change| in hours, capped at 3
    """
    rest = np.asarray(rest, float)
    return {"b2b": (rest == 1).astype(float),
            "rest3": ((rest >= 3) & (rest < 9)).astype(float),
            "km": np.log1p(np.asarray(km, float) / 1000.0),
            "tz": np.minimum(np.abs(np.asarray(dtz, float)), 3.0)}


def ctx_offsets(P: dict, sch) -> tuple[np.ndarray, np.ndarray]:
    """Log-rate offsets (home goals, away goals) from rest/travel columns of a
    schedule frame (rest_h, rest_a, km_h, km_a, dtz_h, dtz_a)."""
    c = P["ctx"]
    fh = ctx_features(sch["rest_h"], sch["km_h"], sch["dtz_h"])
    fa = ctx_features(sch["rest_a"], sch["km_a"], sch["dtz_a"])
    off_h = sum(c[f"{f}_o"] * fh[f] + c[f"{f}_d"] * fa[f] for f in CTX_FEATURES)
    off_a = sum(c[f"{f}_o"] * fa[f] + c[f"{f}_d"] * fh[f] for f in CTX_FEATURES)
    return np.asarray(off_h, float), np.asarray(off_a, float)


def goalie_offset(P: dict, talent_diff) -> np.ndarray:
    """Log multiplier on goals AGAINST for a starter whose save talent is
    ``talent_diff`` (goals saved above average per shot on goal, e.g. +0.005
    = half a save-percentage point) above the team's usual starter.
    NaN (unknown starter) -> 0. Talent units are those of
    ``structural.goalie_game_talent`` (MoneyPuck GSAx per shot, Marcel-
    weighted and regressed, plus in-season saves above league average)."""
    x = np.nan_to_num(np.asarray(talent_diff, float))
    return P["goalie"]["coef"] * x


def b2b_goalie_offset(P: dict, gap) -> np.ndarray:
    """Team-specific backup-on-back-to-back adjustment when starters are NOT
    known (season simulation): extra log multiplier on goals AGAINST in a
    team's back-to-back games, for a primary-minus-backup save-talent gap
    ``gap`` (goals saved per shot), beyond the league-average backup effect
    that ctx['b2b_d'] already contains:
        coef * -(p_backup_b2b - p_backup_other) * (gap - avg_gap)
    (usage shares and avg_gap from ``structural.backup_usage``). Add it to
    the goals of the team's OPPONENT in the team's b2b games."""
    g = P["goalie"]
    x = -(g.get("p_backup_b2b", 0.52) - g.get("p_backup_other", 0.34)) * (
        np.asarray(gap, float) - g.get("avg_gap", 0.004))
    return g["coef"] * x


def rates(P, o_h, d_h, o_a, d_a, sch=None, mu=None, h=None, goalie_h=None,
          goalie_a=None):
    """Expected regulation goals (lam_home, lam_away) for arrays of games.

    ``sch`` (optional) supplies rest/travel columns; ``goalie_h`` / ``goalie_a``
    are the home / away starters' save-talent differences vs their teams'
    usual starters (optional)."""
    mu = P["mu"] if mu is None else mu
    h = P["h"] if h is None else h
    eh = mu + h + np.asarray(o_h, float) + np.asarray(d_a, float)
    ea = mu + np.asarray(o_a, float) + np.asarray(d_h, float)
    if sch is not None:
        ch, ca = ctx_offsets(P, sch)
        eh, ea = eh + ch, ea + ca
    if goalie_a is not None:
        eh = eh + goalie_offset(P, goalie_a)
    if goalie_h is not None:
        ea = ea + goalie_offset(P, goalie_h)
    return np.exp(eh), np.exp(ea)


def schedule_features(sched, season_end: int):
    """Add rest/travel columns to a future schedule (date, home, away)."""
    from hattrick.gametable import schedule_context
    return schedule_context(sched, season_end)


# ---------------------------------------------------------------------------
# Regulation: base Poisson margins + end-game layer
# ---------------------------------------------------------------------------
def _skellam(bh, ba, ks):
    """P(X - Y = k) for X~Pois(bh), Y~Pois(ba); ks array of ints. Shape (N, len(ks))."""
    bh = np.asarray(bh, float)[:, None]
    ba = np.asarray(ba, float)[:, None]
    ks = np.asarray(ks)[None, :]
    z = 2.0 * np.sqrt(bh * ba)
    return (np.exp(-(np.sqrt(bh) - np.sqrt(ba)) ** 2) * (bh / ba) ** (ks / 2.0)
            * ive(np.abs(ks), z))


def _transitions(lh, la, L):
    """End-game transition probabilities (home leading / away leading).

    Returns dict with, for |D|=1 and 2:
      th1: P(away ties | home leads by 1), eh1: P(home EN | home leads by 1)
      ta1: P(home ties | away leads by 1), ea1: P(away EN | away leads by 1)
      th2, eh2, ta2, ea2 likewise for 2-goal leads.
    """
    s = np.log(np.asarray(la, float) / np.asarray(lh, float))   # away trailing: log(la/lh)
    g = L["g"]
    out = {}
    for k, (a, b) in ((1, (L["a1"], L["b1"])), (2, (L["a2"], L["b2"]))):
        # home leads: trailing = away (s), leader = home
        t_h = a * np.exp(g * s)
        e_h = b * np.exp(-g * s)
        # away leads: trailing = home (-s)
        t_a = a * np.exp(-g * s)
        e_a = b * np.exp(g * s)
        sc = np.minimum(1.0, 0.97 / (t_h + e_h))        # keep t + e <= 0.97
        t_h, e_h = t_h * sc, e_h * sc
        sc = np.minimum(1.0, 0.97 / (t_a + e_a))
        t_a, e_a = t_a * sc, e_a * sc
        out[f"th{k}"], out[f"eh{k}"], out[f"ta{k}"], out[f"ea{k}"] = t_h, e_h, t_a, e_a
    return out


NX = 25            # goal support 0..24 per team for the base distribution


@functools.lru_cache(maxsize=64)
def _cmp_table(kappa: float):
    """Mean of the Conway-Maxwell-Poisson pmf  p(x) ~ theta^x / (x!)^kappa
    on a grid of log(theta), for inverting mean -> theta."""
    x = np.arange(NX)
    lt = np.linspace(-4.0, 4.5 * max(kappa, 1.0), 2001)
    lp = lt[:, None] * x[None, :] - kappa * gammaln(x + 1)[None, :]
    pm = np.exp(lp - lp.max(1, keepdims=True))
    pm /= pm.sum(1, keepdims=True)
    return lt, (pm * x).sum(1)


def base_pmf(mean, kappa: float) -> np.ndarray:
    """Base goal pmf over 0..NX-1 with the given means, shape (N, NX).
    kappa = 1 is Poisson; kappa > 1 is underdispersed (var/mean ~ 1/kappa)."""
    mean = np.asarray(mean, float)
    x = np.arange(NX)
    if abs(kappa - 1.0) < 1e-12:
        lp = x[None, :] * np.log(mean)[:, None] - gammaln(x + 1)[None, :] - mean[:, None]
        pm = np.exp(lp)
        return pm / pm.sum(1, keepdims=True)
    lt, mu = _cmp_table(round(float(kappa), 6))
    t = np.interp(mean, mu, lt)
    lp = t[:, None] * x[None, :] - kappa * gammaln(x + 1)[None, :]
    pm = np.exp(lp - lp.max(1, keepdims=True))
    return pm / pm.sum(1, keepdims=True)


def _diff_pmf(ph, pa, ks):
    """P(X - Y = k) for independent X~ph, Y~pa (rows), for each k in ks."""
    out = np.empty((ph.shape[0], len(ks)))
    for j, k in enumerate(ks):
        if k >= 0:
            out[:, j] = (ph[:, k:] * pa[:, :NX - k]).sum(1)
        else:
            out[:, j] = (ph[:, :NX + k] * pa[:, -k:]).sum(1)
    return out


def base_rates(lh, la, P, iters: int = 4):
    """Base means (bh, ba) and pmfs that make the end-game layer
    MEAN-PRESERVING: E[regulation goals] = (lh, la) after the layer.
    Returns bh, ba, T (transition probabilities), ph, pa."""
    lh = np.asarray(lh, float)
    la = np.asarray(la, float)
    kappa = P.get("kappa", 1.0)
    T = _transitions(lh, la, P["layer"])
    bh, ba = lh.copy(), la.copy()
    for _ in range(iters):
        ph, pa = base_pmf(bh, kappa), base_pmf(ba, kappa)
        m = _diff_pmf(ph, pa, [-2, -1, 1, 2])
        add_h = T["eh1"] * m[:, 2] + T["eh2"] * m[:, 3] + T["ta1"] * m[:, 1] + T["ta2"] * m[:, 0]
        add_a = T["ea1"] * m[:, 1] + T["ea2"] * m[:, 0] + T["th1"] * m[:, 2] + T["th2"] * m[:, 3]
        bh = np.maximum(lh - add_h, 0.05)
        ba = np.maximum(la - add_a, 0.05)
    return bh, ba, T, base_pmf(bh, kappa), base_pmf(ba, kappa)


def margin_pmf(lh, la, P) -> np.ndarray:
    """Final regulation margin pmf, shape (N, 2*KMAX+1), index k+KMAX."""
    bh, ba, T, ph, pa = base_rates(lh, la, P)
    ks = np.arange(-KMAX, KMAX + 1)
    M = _diff_pmf(ph, pa, ks)
    M = M / M.sum(1, keepdims=True)
    o = KMAX
    Mp = M.copy()
    for k, tk, ek, ta, ea in ((1, "th1", "eh1", "ta1", "ea1"), (2, "th2", "eh2", "ta2", "ea2")):
        # home leads by k: -> k-1 (away scores) or k+1 (home empty-netter)
        mv_t = T[tk] * M[:, o + k]
        mv_e = T[ek] * M[:, o + k]
        Mp[:, o + k] -= mv_t + mv_e
        Mp[:, o + k - 1] += mv_t
        Mp[:, o + k + 1] += mv_e
        mv_t = T[ta] * M[:, o - k]
        mv_e = T[ea] * M[:, o - k]
        Mp[:, o - k] -= mv_t + mv_e
        Mp[:, o - k + 1] += mv_t
        Mp[:, o - k - 1] += mv_e
    return Mp


def reg_probs(lh, la, P):
    """(P(home regulation win), P(regulation tie), P(away regulation win))."""
    M = margin_pmf(lh, la, P)
    o = KMAX
    return M[:, o + 1:].sum(1), M[:, o], M[:, :o].sum(1)


def reg_joint(lh, la, P, K: int = 16) -> np.ndarray:
    """Joint pmf of regulation goals (home, away), shape (N, K, K)."""
    bh, ba, T, ph, pa = base_rates(lh, la, P)
    ph, pa = ph[:, :K], pa[:, :K]
    J = ph[:, :, None] * pa[:, None, :]
    x = np.arange(K)
    D = x[:, None] - x[None, :]
    Jp = J.copy()
    for k, tk, ek, ta, ea in ((1, "th1", "eh1", "ta1", "ea1"), (2, "th2", "eh2", "ta2", "ea2")):
        m = (D == k)[None]
        # home leads by k: away scores (y+1) w.p. t, home scores (x+1) w.p. e
        cell = J * m
        t = T[tk][:, None, None] * cell
        e = T[ek][:, None, None] * cell
        Jp -= t + e
        Jp[:, :, 1:] += t[:, :, :-1]
        Jp[:, 1:, :] += e[:, :-1, :]
        m = (D == -k)[None]
        cell = J * m
        t = T[ta][:, None, None] * cell
        e = T[ea][:, None, None] * cell
        Jp -= t + e
        Jp[:, 1:, :] += t[:, :-1, :]
        Jp[:, :, 1:] += e[:, :, :-1]
    return Jp


def reg_score_prob(lh, la, x, y, P) -> np.ndarray:
    """Probability of the observed regulation score (x, y) for each game
    (O(N) after the base pmfs; used for fitting and goal log-likelihood)."""
    bh, ba, T, ph, pa = base_rates(lh, la, P)
    n = len(ph)
    r = np.arange(n)
    x = np.clip(np.asarray(x, int), 0, NX - 1)
    y = np.clip(np.asarray(y, int), 0, NX - 1)

    def base(xx, yy):
        ok = (xx >= 0) & (yy >= 0) & (xx < NX) & (yy < NX)
        v = ph[r, np.clip(xx, 0, NX - 1)] * pa[r, np.clip(yy, 0, NX - 1)]
        return np.where(ok, v, 0.0)

    d = x - y
    out = base(x, y)
    # outflow from the observed cell itself
    for k, tk, ek, ta, ea in ((1, "th1", "eh1", "ta1", "ea1"), (2, "th2", "eh2", "ta2", "ea2")):
        out = out - np.where(d == k, (T[tk] + T[ek]) * base(x, y), 0.0)
        out = out - np.where(d == -k, (T[ta] + T[ea]) * base(x, y), 0.0)
        # inflow: away scored from (x, y-1) where home led by k (d+1 == k)
        out = out + np.where(d + 1 == k, T[tk] * base(x, y - 1), 0.0)
        # home empty-netter from (x-1, y) where home led by k (d-1 == k)
        out = out + np.where(d - 1 == k, T[ek] * base(x - 1, y), 0.0)
        # home scored from (x-1, y) where away led by k (d-1 == -k)
        out = out + np.where(d - 1 == -k, T[ta] * base(x - 1, y), 0.0)
        # away empty-netter from (x, y-1) where away led by k (d+1 == -k)
        out = out + np.where(d + 1 == -k, T[ea] * base(x, y - 1), 0.0)
    return np.maximum(out, 1e-15)


# ---------------------------------------------------------------------------
# Overtime and shootout
# ---------------------------------------------------------------------------
def ot_probs(lh, la, P):
    """(P(OT decided | tie), P(home wins OT | decided), P(home wins SO))."""
    o = P["ot"]
    lh = np.asarray(lh, float)
    la = np.asarray(la, float)
    r = np.log(lh / la)
    p_dec = expit(o["c0"] + o["c1"] * np.log((lh + la) / LREF))
    p_ot = expit(o["a_ot"] + o["b_ot"] * r)
    p_so = expit(o["a_so"] + o["b_so"] * r)
    return p_dec, p_ot, p_so


def outcome_probs(lh, la, P) -> dict:
    """All outcome probabilities and expected goals for arrays of games."""
    lh = np.asarray(lh, float)
    la = np.asarray(la, float)
    hw, tie, aw = reg_probs(lh, la, P)
    p_dec, p_ot, p_so = ot_probs(lh, la, P)
    out = {
        "p_home_reg": hw, "p_tie": tie, "p_away_reg": aw,
        "p_home_ot": tie * p_dec * p_ot, "p_away_ot": tie * p_dec * (1 - p_ot),
        "p_home_so": tie * (1 - p_dec) * p_so, "p_away_so": tie * (1 - p_dec) * (1 - p_so),
    }
    out["p_home_win"] = hw + out["p_home_ot"] + out["p_home_so"]
    out["p_so"] = tie * (1 - p_dec)
    out["e_reg_h"], out["e_reg_a"] = lh, la
    out["e_gf_h_noso"] = lh + out["p_home_ot"]
    out["e_gf_a_noso"] = la + out["p_away_ot"]
    out["e_gf_h"] = out["e_gf_h_noso"] + out["p_home_so"]
    out["e_gf_a"] = out["e_gf_a_noso"] + out["p_away_so"]
    # standings points (2 win, 1 OT/SO loss)
    out["e_pts_h"] = 2 * out["p_home_win"] + out["p_away_ot"] + out["p_away_so"]
    out["e_pts_a"] = 2 * (1 - out["p_home_win"]) + out["p_home_ot"] + out["p_home_so"]
    return out


def p_home_win(lh, la, P) -> np.ndarray:
    return outcome_probs(lh, la, P)["p_home_win"]


# ---------------------------------------------------------------------------
# Sampling (season simulation)
# ---------------------------------------------------------------------------
def _draw(pm, u):
    """Inverse-CDF draws from row pmfs (N, NX) with uniforms u (N,)."""
    c = np.cumsum(pm, axis=1)
    return (u[:, None] > c[:, :-1]).sum(1)


_GRID_LO, _GRID_HI, _GRID_N = np.log(0.4), np.log(9.0), 161
_CDF_LO, _CDF_HI, _CDF_N = np.log(0.03), np.log(10.0), 6001


def _pkey(P):
    L = P["layer"]
    return (round(float(P.get("kappa", 1.0)), 8),) + tuple(
        round(float(L[k]), 8) for k in ("a1", "b1", "a2", "b2", "g"))


@functools.lru_cache(maxsize=16)
def _base_grid(key):
    """Exact mean-preserving base means on a (log lh, log la) grid."""
    kappa, a1, b1, a2, b2, g = key
    P = {"kappa": kappa, "layer": {"a1": a1, "b1": b1, "a2": a2, "b2": b2, "g": g}}
    gr = np.linspace(_GRID_LO, _GRID_HI, _GRID_N)
    LH, LA = np.meshgrid(np.exp(gr), np.exp(gr), indexing="ij")
    bh, ba, _, _, _ = base_rates(LH.ravel(), LA.ravel(), P, iters=6)
    return bh.reshape(LH.shape), ba.reshape(LH.shape)


@functools.lru_cache(maxsize=16)
def _cdf_table(kappa):
    """CDF rows of the base distribution on a fine grid of log means."""
    m = np.exp(np.linspace(_CDF_LO, _CDF_HI, _CDF_N))
    return np.cumsum(base_pmf(m, kappa), axis=1)[:, :-1].astype(np.float32)


def _fast_base(lh, la, P):
    """Bilinear interpolation (in log rates) of the exact base means."""
    bhg, bag = _base_grid(_pkey(P))
    step = (_GRID_HI - _GRID_LO) / (_GRID_N - 1)
    fx = np.clip((np.log(lh) - _GRID_LO) / step, 0, _GRID_N - 1 - 1e-9)
    fy = np.clip((np.log(la) - _GRID_LO) / step, 0, _GRID_N - 1 - 1e-9)
    i, j = fx.astype(int), fy.astype(int)
    u, v = fx - i, fy - j

    def bil(G):
        return ((1 - u) * (1 - v) * G[i, j] + u * (1 - v) * G[i + 1, j]
                + (1 - u) * v * G[i, j + 1] + u * v * G[i + 1, j + 1])
    return bil(bhg), bil(bag)


def _fast_draw(mean, kappa, u):
    c = _cdf_table(round(float(kappa), 8))
    step = (_CDF_HI - _CDF_LO) / (_CDF_N - 1)
    k = np.clip(np.rint((np.log(mean) - _CDF_LO) / step).astype(int), 0, _CDF_N - 1)
    return (u[:, None] > c[k]).sum(1)


def sample(lh, la, P, rng: np.random.Generator, chunk: int = 1_000_000,
           exact: bool = False) -> dict:
    """Draw complete game results. Returns int16 arrays:
      reg_h, reg_a  regulation goals (incl. empty-netters)
      extra         0 regulation, 1 overtime, 2 shootout
      home_win      1/0
      gf_h, gf_a    final goals, standings convention (OT goal and +1 for a
                    shootout win count)
    The default fast path interpolates the exact mean-preserving base means
    on a 161x161 grid of log rates and draws from tabulated CDFs (6,001 log
    means): about 0.3 s per million games, indistinguishable from
    ``exact=True`` (tested in hattrick/tests/test_gamemodel.py)."""
    lh = np.asarray(lh, float).ravel()
    la = np.asarray(la, float).ravel()
    n = len(lh)
    out = {k: np.empty(n, np.int16) for k in ("reg_h", "reg_a", "extra", "home_win",
                                              "gf_h", "gf_a")}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = lh[s:e], la[s:e]
        m = e - s
        if exact:
            bh, ba, T, ph, pa = base_rates(a, b, P)
            x = _draw(ph, rng.random(m))
            y = _draw(pa, rng.random(m))
        else:
            T = _transitions(a, b, P["layer"])
            bh, ba = _fast_base(a, b, P)
            kap = P.get("kappa", 1.0)
            x = _fast_draw(bh, kap, rng.random(m))
            y = _fast_draw(ba, kap, rng.random(m))
        d = x - y
        u = rng.random(m)
        for k in (1, 2):
            t, en = T[f"th{k}"], T[f"eh{k}"]
            mk = d == k
            y = y + (mk & (u < t))
            x = x + (mk & (u >= t) & (u < t + en))
            t, en = T[f"ta{k}"], T[f"ea{k}"]
            mk = d == -k
            x = x + (mk & (u < t))
            y = y + (mk & (u >= t) & (u < t + en))
        p_dec, p_ot, p_so = ot_probs(a, b, P)
        tie = x == y
        u1 = rng.random(m)
        u2 = rng.random(m)
        dec = tie & (u1 < p_dec)
        so = tie & ~dec
        hw_ot = dec & (u2 < p_ot)
        hw_so = so & (u2 < p_so)
        hw = (x > y) | hw_ot | hw_so
        out["reg_h"][s:e], out["reg_a"][s:e] = x, y
        out["extra"][s:e] = np.where(dec, 1, np.where(so, 2, 0))
        out["home_win"][s:e] = hw
        out["gf_h"][s:e] = x + (tie & hw)
        out["gf_a"][s:e] = y + (tie & ~hw)
    return out


# ---------------------------------------------------------------------------
# Alternative regulation models (for validation only)
# ---------------------------------------------------------------------------
def margin_pmf_poisson(lh, la):
    ks = np.arange(-KMAX, KMAX + 1)
    M = _skellam(lh, la, ks)
    return M / M.sum(1, keepdims=True)


def margin_pmf_diag(lh, la, theta):
    """Independent Poisson with the tie cell inflated by exp(theta) (a
    Dixon-Coles-style diagonal inflation), renormalised."""
    M = margin_pmf_poisson(lh, la)
    M[:, KMAX] *= np.exp(theta)
    return M / M.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
MARGIN_BINS = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])   # clipped at +-4


def _binned(M):
    """Collapse a margin pmf to 9 bins (<=-4, -3..3, >=4)."""
    o = KMAX
    return np.column_stack([M[:, :o - 3].sum(1)] + [M[:, o + k] for k in range(-3, 4)]
                           + [M[:, o + 4:].sum(1)])


def margin_loglik(M, margin, w=None):
    B = _binned(M)
    idx = np.clip(np.asarray(margin), -4, 4) + 4
    p = np.clip(B[np.arange(len(idx)), idx], 1e-12, None)
    w = np.ones(len(idx)) if w is None else np.asarray(w, float)
    return float(np.sum(w * np.log(p)) / w.sum())


def _aggregate(lh, la, margin, w, step=0.02):
    """Collapse games onto a grid of (log lh, log la) cells with per-cell
    weighted counts of the 9 margin bins (exact up to the grid rounding)."""
    kh = np.round(np.log(lh) / step).astype(np.int64)
    ka = np.round(np.log(la) / step).astype(np.int64)
    key = (kh - kh.min()) * 100000 + (ka - ka.min())
    uk, inv = np.unique(key, return_inverse=True)
    idx = np.clip(np.asarray(margin), -4, 4) + 4
    cnt = np.zeros((len(uk), 9))
    np.add.at(cnt, (inv, idx), w)
    lh_c = np.exp(np.bincount(inv, np.log(lh) * w) / np.bincount(inv, w))
    la_c = np.exp(np.bincount(inv, np.log(la) * w) / np.bincount(inv, w))
    return lh_c, la_c, cnt


def fit_layer(lh, la, reg_h, reg_a, w=None, x0=None, prior_sd: float = 1.5,
              fit_kappa: bool = True, max_games: int = 12000, seed: int = 0) -> dict:
    """Joint ML fit of the base dispersion kappa and the end-game layer on
    observed regulation scores given expected regulation goals.

    Maximises sum_i w_i log P(reg_h_i, reg_a_i | lh_i, la_i) (the full joint,
    so kappa is identified by each team's goal dispersion and the layer by
    the margins). Weak N(logit default, prior_sd) priors on the four
    transition probabilities keep them identified. Uses at most
    ``max_games`` games (a seeded subsample, weights kept).
    Returns {'layer': {...}, 'kappa': float}."""
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    reg_h, reg_a = np.asarray(reg_h), np.asarray(reg_a)
    w = np.ones(len(lh)) if w is None else np.asarray(w, float)
    if len(lh) > max_games:
        idx = np.random.default_rng(seed).choice(len(lh), max_games, replace=False)
        lh, la, reg_h, reg_a, w = lh[idx], la[idx], reg_h[idx], reg_a[idx], w[idx]
    L0 = DEFAULT["layer"] if x0 is None else x0
    lg = lambda p: np.log(p / (1 - p))
    z_prior = np.array([lg(DEFAULT["layer"][k]) for k in ("a1", "b1", "a2", "b2")])

    def unpack(z):
        L = {"a1": float(expit(z[0])), "b1": float(expit(z[1])),
             "a2": float(expit(z[2])), "b2": float(expit(z[3])), "g": float(z[4])}
        return L, (float(1.0 + np.exp(z[5])) if fit_kappa else 1.0)

    def f(z):
        L, kap = unpack(z)
        p = reg_score_prob(lh, la, reg_h, reg_a, {"layer": L, "kappa": kap})
        pen = 0.5 * np.sum(((z[:4] - z_prior) / prior_sd) ** 2) + 0.5 * z[4] ** 2
        return -(np.sum(w * np.log(p)) - pen) / w.sum()

    z0 = np.array([lg(L0["a1"]), lg(L0["b1"]), lg(L0["a2"]), lg(L0["b2"]), L0["g"],
                   np.log(max(DEFAULT["kappa"] - 1.0, 1e-3))])
    r = minimize(f, z0, method="L-BFGS-B")
    L, kap = unpack(r.x)
    return {"layer": L, "kappa": kap, "nll": float(r.fun)}


def fit_ot(lh, la, extra, home_win, w=None, prior_sd=None) -> dict:
    """Fit the OT/SO layer on regulation-tied games.

    extra: 'OT' or 'SO' per tied game; home_win 1/0. Weak Gaussian priors keep
    the shootout slope and home terms from wandering on small samples."""
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    extra = np.asarray(extra)
    y = np.asarray(home_win, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    r = np.log(lh / la)
    lt = np.log((lh + la) / LREF)
    dec = (extra == "OT").astype(float)
    ps = prior_sd or {"c1": 1.0, "a_ot": 0.5, "b_ot": 1.0, "a_so": 0.3, "b_so": 0.5}
    pm = {"c1": 0.0, "a_ot": 0.0, "b_ot": 1.0, "a_so": 0.0, "b_so": 0.0}

    def nll_dec(z):
        p = np.clip(expit(z[0] + z[1] * lt), 1e-9, 1 - 1e-9)
        return (-np.sum(w * (dec * np.log(p) + (1 - dec) * np.log(1 - p)))
                + 0.5 * ((z[1] - pm["c1"]) / ps["c1"]) ** 2)

    def nll_win(z, mask, keys):
        p = np.clip(expit(z[0] + z[1] * r[mask]), 1e-9, 1 - 1e-9)
        yy, ww = y[mask], w[mask]
        pen = sum(0.5 * ((z[i] - pm[k]) / ps[k]) ** 2 for i, k in enumerate(keys))
        return -np.sum(ww * (yy * np.log(p) + (1 - yy) * np.log(1 - p))) + pen

    zd = minimize(nll_dec, np.array([0.5, 0.0]), method="BFGS").x
    ot_m = dec == 1
    zo = minimize(nll_win, np.array([0.0, 1.0]), args=(ot_m, ("a_ot", "b_ot")),
                  method="BFGS").x
    zs = minimize(nll_win, np.array([0.0, 0.0]), args=(~ot_m, ("a_so", "b_so")),
                  method="BFGS").x
    return {"c0": float(zd[0]), "c1": float(zd[1]), "a_ot": float(zo[0]),
            "b_ot": float(zo[1]), "a_so": float(zs[0]), "b_so": float(zs[1])}


@dataclass
class Strengths:
    """Convenience container: per-game log-rate inputs."""
    o_h: np.ndarray
    d_h: np.ndarray
    o_a: np.ndarray
    d_a: np.ndarray


def expected_points_vs_average(strength, P, home_share: float = 0.5) -> np.ndarray:
    """Expected standings points per game for a team whose overall log-rate
    strength is ``strength`` (o = strength/2, d = -strength/2) against a
    league-average opponent (o = d = 0), home half the time. Used to map a
    season point total to a strength (``ratings.market_strength``)."""
    s = np.asarray(strength, float)
    o, d = s / 2.0, -s / 2.0
    z = np.zeros_like(s)
    lh, la = rates(P, o, d, z, z)
    ph = outcome_probs(lh, la, P)["e_pts_h"]
    lh, la = rates(P, z, z, o, d)
    pa = outcome_probs(lh, la, P)["e_pts_a"]
    return home_share * ph + (1 - home_share) * pa
