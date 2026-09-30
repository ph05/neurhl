"""HatTrick skater projections: usage, true-talent rates and their uncertainty.

For a target season V (season_end convention) every number here is computed
from seasons strictly before V -- the walk-forward rule. The model is a
Marcel-family estimator built from explicit hockey quantities rather than a
black box, because each piece can then be checked against what we know about
how the sport works:

1. **Era control.** League scoring per minute moves (2009-17 ~0.54 G/60 all
   situations, 2018-26 ~0.60-0.63). Every past count is restated at the league
   level projected for V ("relative" counts), and V's level is the last intact
   season's (tuned on 2011-17; 2021, the division-only no-crowd season, is
   skipped as unrepresentative).
2. **Recency-weighted history.** Season V-j gets weight decay**(j-1) (up to
   four seasons). Weights multiply ice time, so a 20-game season counts as
   20 games of evidence, not as a full season.
3. **Empirical-Bayes shrinkage toward a usage prior.** A player's rate is
   pulled toward what players of his position AND deployment produce: coaches
   see practice and video, so ice time per game (EV, PP, SH) is itself a strong
   talent signal. The pull strength K (minutes of prior) is estimated by the
   method of moments per stat/situation/position -- talent variance from the
   year-to-year covariance of residuals, noise from the Poisson scale with an
   overdispersion factor -- then scaled by one global multiplier tuned by CV.
4. **Finishing is regressed hard.** Goals are a blend of the directly-shrunk
   goal rate and (shrunk ixG rate) x (shrunk finishing multiplier G/ixG),
   whose own prior weight is ~hundreds of expected goals.
5. **Aging from residuals.** The age curve is fitted walk-forward as the ratio
   of realised to projected (un-aged) production by age. Because the
   projection already conditions on each player's own past, this is the delta
   method without its usual survivor bias (players are not compared with
   themselves at their luckiest).
6. **Rookies and thin NHL records.** Pre-NHL production is translated with
   league equivalency factors (NHLe) estimated walk-forward, turned into a
   prior on NHL scoring rate (with draft position as a weak extra signal), and
   blended with the usage prior; NHL minutes then take over as they accrue.

Ice time per game is projected here per situation and later normalised within
each team by hattrick.deploy (conservation of ice time and dressed games).

Run the 2026-27 freeze:  python3 -m hattrick.players --season 2027
"""
from __future__ import annotations

import argparse
import functools
import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import data as D

SITS = ("ev", "pp", "sh", "oth")
SIT_STATS = ("g", "a1", "a2", "sog", "ixg")            # projected by situation
PERIPH = ("hits", "blk", "pim", "fow", "fol", "tk", "gv", "pen", "pend")
ONICE = ("oi_xgf", "oi_xga", "off_xgf", "off_xga")     # 5v5 on/off-ice xG
MAX_LAGS = 4
FIRST_SEASON = 2009                                    # MoneyPuck coverage
ERA_SKIP = {2021}          # never the reference level for a later season
AGE_LO, AGE_HI = 19.0, 38.0
# Aging groups: which counting stats share one age multiplier.
AGE_GROUPS = {"g": ("g",), "a": ("a1", "a2"), "shot": ("sog", "ixg"),
              "periph": ("hits", "blk", "tk", "gv", "pim", "pen", "pend")}
PARAMS_PATH = C.PARAMS / "players.json"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Tunable knobs (tuned only on V<=2017 by hattrick.backtest.players_bt).

    Everything else (regression constants, priors, aging, NHLe factors, league
    levels) is FITTED walk-forward inside each target season.
    """
    rate_decay: float = 0.6        # season weight ratio for scoring rates
    toi_decay: float = 0.35        # season weight ratio for TOI per game
    kappa: float = 1.0             # multiplier on the EB prior weights K
    fin_weight: float = 0.5        # share of goals from ixG x finishing
    k_toi_games: float = 12.0      # games of prior on TOI per game
    prosp_alpha: float = 0.6       # weight of the NHLe prior vs usage prior
    age_curve: bool = True
    eb_window: int = 8             # seasons used to estimate EB constants
    prior_window: int = 5          # seasons used to fit the usage prior

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def load_params() -> Params:
    """Tuned parameters if the tuner has written them, else the defaults."""
    if PARAMS_PATH.exists():
        d = json.loads(PARAMS_PATH.read_text()).get("hyper", {})
        known = {k: v for k, v in d.items() if k in Params.__dataclass_fields__}
        return Params(**known)
    return Params()


# ---------------------------------------------------------------------------
# Panel: players x seasons arrays (fast walk-forward slicing)
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    ids: np.ndarray                 # (P,) player ids
    seasons: np.ndarray             # (S,) season_end
    pos: np.ndarray                 # (P,) 'F' | 'D' (latest observed)
    name: np.ndarray                # (P,)
    birth: np.ndarray               # (P,) datetime64[ns] (NaT if unknown)
    x: dict = field(default_factory=dict)     # col -> (P, S) float, 0 if absent
    team: np.ndarray | None = None  # (P, S) object: main team that season
    row: dict = field(default_factory=dict)   # player_id -> row index

    def j(self, season: int) -> int:
        return int(season - self.seasons[0])

    def lags(self, V: int, n: int = MAX_LAGS) -> list[int]:
        """Column indices of seasons V-1 .. V-n that exist in the panel."""
        return [self.j(V - k) for k in range(1, n + 1)
                if self.seasons[0] <= V - k <= self.seasons[-1]]


def _panel_cols() -> list[str]:
    cols = ["gp", "toi_all", "bench_ev"]
    cols += [f"toi_{k}" for k in SITS]
    cols += [f"{s}_{k}" for s in SIT_STATS for k in SITS]
    cols += [f"{s}_all" for s in PERIPH]
    cols += [f"{s}_ev" for s in ONICE]
    return cols


@functools.lru_cache(maxsize=None)
def panel() -> Panel:
    """Every skater-season 2009..2026 as dense arrays."""
    s = D.skater_seasons().copy()
    ids = np.sort(s.player_id.unique())
    seasons = np.arange(FIRST_SEASON, int(s.season_end.max()) + 1)
    row = {p: i for i, p in enumerate(ids)}
    ri = s.player_id.map(row).to_numpy()
    ci = (s.season_end - FIRST_SEASON).to_numpy()
    x = {}
    for c in _panel_cols():
        a = np.zeros((len(ids), len(seasons)))
        a[ri, ci] = s[c].fillna(0.0).to_numpy(float)
        x[c] = a
    last = s.sort_values("season_end").drop_duplicates("player_id", keep="last")
    last = last.set_index("player_id").reindex(ids)
    b = D.bios().drop_duplicates("player_id").set_index("player_id").birth
    birth = pd.to_datetime(pd.Series(ids).map(b)).to_numpy()
    # main team per season = most ice time (trades split in skater_team_seasons)
    st = D.skater_team_seasons()
    st = st.sort_values("toi", ascending=False).drop_duplicates(
        ["player_id", "season_end"])
    team = np.full((len(ids), len(seasons)), None, dtype=object)
    st = st[st.player_id.isin(row) & st.season_end.between(seasons[0], seasons[-1])]
    team[st.player_id.map(row).to_numpy(), (st.season_end - FIRST_SEASON).to_numpy()] = \
        st.team.to_numpy()
    # team games per season (2020 and 2021 differ by team; 2013 = 48)
    tg = D.team_seasons().set_index(["team", "season_end"]).gp
    sched = np.zeros((len(ids), len(seasons)))
    for j, y in enumerate(seasons):
        med = float(tg.xs(y, level="season_end").median())
        t = pd.Series(team[:, j]).map(lambda z, y=y: tg.get((z, y), np.nan))
        sched[:, j] = t.fillna(med).to_numpy()
    x["sched"] = sched
    return Panel(ids=ids, seasons=seasons, pos=last.pos.to_numpy(),
                 name=last.name.to_numpy(), birth=birth, x=x, team=team, row=row)


# ---------------------------------------------------------------------------
# League environment
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def league_levels() -> pd.DataFrame:
    """Per season: league per-MINUTE rates of every stat by situation, team
    minutes per game by situation, and skater-minutes per team-game by
    position and situation (the ice-time budgets deploy.py conserves)."""
    P = panel()
    out = {}
    for s in SIT_STATS:
        for k in SITS:
            out[f"{s}_{k}"] = P.x[f"{s}_{k}"].sum(0) / P.x[f"toi_{k}"].sum(0)
    for s in PERIPH:
        out[f"{s}_all"] = P.x[f"{s}_all"].sum(0) / P.x["toi_all"].sum(0)
    # league G/ixG by situation (xG-model calibration drifts by season)
    for k in SITS:
        out[f"rho_{k}"] = P.x[f"g_{k}"].sum(0) / P.x[f"ixg_{k}"].sum(0)
    lv = pd.DataFrame(out, index=P.seasons)
    t = D.team_seasons()
    t = t[t.season_end.isin(P.seasons)]
    ng = t.groupby("season_end").gp.sum()
    for k in SITS:
        lv[f"team_min_{k}"] = t.groupby("season_end")[f"toi_{k}"].sum() / ng
        for pos in ("F", "D"):
            m = P.pos == pos
            lv[f"sk_min_{pos}_{k}"] = P.x[f"toi_{k}"][m].sum(0) / ng.reindex(P.seasons).to_numpy()
    lv["games"] = t.groupby("season_end").gp.median()
    return lv


def project_league(V: int) -> pd.Series:
    """League environment expected in V: the last intact season before V.

    Choice validated on 2011-17 (log-error of league G/60, A/60 by situation):
    last season 0.020, mean of two 0.022, of three 0.025, 3/2/1 0.023.
    """
    lv = league_levels()
    prev = [y for y in range(V - 1, FIRST_SEASON - 1, -1)
            if y in lv.index and y not in ERA_SKIP]
    if not prev:
        raise ValueError(f"no league history before {V}")
    return lv.loc[prev[0]]


# ---------------------------------------------------------------------------
# Ages
# ---------------------------------------------------------------------------
def ages(P: Panel, V: int) -> np.ndarray:
    """Age on Feb 1 of season V; unknown births get the league median (27)."""
    ref = np.datetime64(f"{V}-02-01")
    birth = np.asarray(P.birth, dtype="datetime64[ns]")
    a = (ref - birth).astype("timedelta64[D]").astype(float) / 365.25
    return np.where(np.isnat(birth), 27.0, a)


# ---------------------------------------------------------------------------
# Usage (TOI per game) projection
# ---------------------------------------------------------------------------
def _weights(P: Panel, V: int, decay: float) -> tuple[list[int], np.ndarray]:
    idx = P.lags(V)
    w = np.array([decay ** k for k in range(len(idx))])
    return idx, w


def usage_raw(P: Panel, V: int, prm: Params, tpg_mult: np.ndarray | None = None) -> dict:
    """Un-aged, un-normalised TOI per game played by situation.

    Past per-game minutes are restated to V's team minutes per game in that
    situation (PP time per game fell from 6.2 in 2009 to 4.5 in 2026), then
    recency-weighted by games and shrunk toward a low-usage positional prior
    with k_toi_games of pseudo-games (a 10-game call-up is not yet a
    20-minute player).
    """
    lv = league_levels()
    tgt = project_league(V)
    idx, w = _weights(P, V, prm.toi_decay)
    gp = P.x["gp"][:, idx]
    wg = (gp * w).sum(1)
    out = {"w_gp": wg, "gp_hist": gp.sum(1)}
    prior = usage_prior_tpg(V)
    for k in SITS:
        scale = tgt[f"team_min_{k}"] / lv[f"team_min_{k}"].to_numpy()[idx]
        num = (P.x[f"toi_{k}"][:, idx] * scale * w).sum(1)
        pr = np.where(P.pos == "D", prior[("D", k)], prior[("F", k)])
        if tpg_mult is not None:
            pr = pr * np.where(np.isfinite(tpg_mult), tpg_mult, 1.0)
        out[f"tpg_{k}"] = (num + prm.k_toi_games * pr) / (wg + prm.k_toi_games)
    return out


@functools.lru_cache(maxsize=None)
def usage_prior_tpg(V: int) -> dict:
    """Minutes per game of fringe NHL skaters (players with <25 GP in a
    season, seasons V-5..V-1): the usage a thin record regresses toward."""
    P = panel()
    out = {}
    js = [P.j(y) for y in range(V - 5, V) if P.seasons[0] <= y <= P.seasons[-1]]
    for pos in ("F", "D"):
        m = P.pos == pos
        gp = P.x["gp"][m][:, js]
        sel = (gp > 0) & (gp < 25)
        for k in SITS:
            t = P.x[f"toi_{k}"][m][:, js]
            out[(pos, k)] = float(t[sel].sum() / gp[sel].sum())
    return out


# ---------------------------------------------------------------------------
# Usage-dependent priors for rates, and EB regression constants
# ---------------------------------------------------------------------------
def _design(tpg_ev, tpg_pp, tpg_sh):
    return np.column_stack([np.ones_like(tpg_ev), tpg_ev, tpg_pp, tpg_sh])


def _season_tpg(P: Panel, js) -> dict:
    gp = np.maximum(P.x["gp"][:, js], 1.0)
    return {k: P.x[f"toi_{k}"][:, js] / gp for k in ("ev", "pp", "sh")}


def stat_toi(stat_col: str) -> str:
    """Exposure column for a stat: its own situation, or all-situation TOI."""
    k = stat_col.rsplit("_", 1)[1]
    return f"toi_{k}" if k in SITS else "toi_all"


@functools.lru_cache(maxsize=None)
def rate_priors(V: int, prior_window: int, eb_window: int) -> dict:
    """For every stat column and position: WLS coefficients of relative rate
    on usage (TOI/GP at EV, PP, SH), and the EB constants (talent variance
    tau2, Poisson overdispersion phi) of residuals around that prior.

    Relative rate = count / (minutes x league per-minute rate of that season).
    tau2 is the TOI-weighted covariance of a player's residuals in consecutive
    seasons (persistent talent); phi scales Poisson noise so that
    E[(resid^2 - tau2) * minutes * league_rate] = phi * prior.
    """
    P = panel()
    lv = league_levels()
    js = [P.j(y) for y in range(V - prior_window, V) if P.seasons[0] <= y <= P.seasons[-1]]
    je = [P.j(y) for y in range(V - eb_window, V) if P.seasons[0] <= y <= P.seasons[-1]]
    out = {}
    cols = [f"{s}_{k}" for s in SIT_STATS for k in SITS] + [f"{s}_all" for s in PERIPH]
    for c in cols:
        tcol = stat_toi(c)
        ell = lv[c].to_numpy()                      # per-minute league rate
        for pos in ("F", "D"):
            m = (P.pos == pos)
            # --- usage prior (WLS)
            tp = _season_tpg(P, js)
            t = P.x[tcol][m][:, js]
            y = P.x[c][m][:, js] / np.maximum(t * ell[js], 1e-9)
            ok = (t > 1.0) & (P.x["gp"][m][:, js] >= 5)
            X = _design(tp["ev"][m][ok], tp["pp"][m][ok], tp["sh"][m][ok])
            wt = t[ok]
            beta = np.linalg.lstsq(X * np.sqrt(wt)[:, None],
                                   y[ok] * np.sqrt(wt), rcond=None)[0]
            # --- EB constants on the eb window
            tpe = _season_tpg(P, je)
            te = P.x[tcol][m][:, je]
            ye = P.x[c][m][:, je] / np.maximum(te * ell[je], 1e-9)
            pr = np.clip(_design(tpe["ev"][m].ravel(), tpe["pp"][m].ravel(),
                                 tpe["sh"][m].ravel()) @ beta, 1e-3, None)
            pr = pr.reshape(te.shape)
            d = ye - pr
            okd = te > 20.0
            a, b = d[:, :-1], d[:, 1:]
            wpair = np.where(okd[:, :-1] & okd[:, 1:],
                             2.0 / (1.0 / np.maximum(te[:, :-1], 1e-9)
                                    + 1.0 / np.maximum(te[:, 1:], 1e-9)), 0.0)
            tau2 = float((wpair * a * b).sum() / max(wpair.sum(), 1e-9))
            tau2 = max(tau2, 1e-4)
            ellm = np.broadcast_to(ell[je], te.shape)
            sel = te > 100.0
            num = ((d[sel] ** 2 - tau2) * te[sel] * ellm[sel]).sum()
            phi = float(np.clip(num / max(pr[sel].sum(), 1e-9), 0.1, 25.0))
            out[(c, pos)] = {"beta": beta, "tau2": tau2, "phi": phi}
    return out


@functools.lru_cache(maxsize=None)
def finishing_prior(V: int, eb_window: int) -> dict:
    """Finishing multiplier (goals / league-calibrated ixG) prior and EB
    constant, per position. Noise of G given ixG is Poisson (var = G), so
    K_fin = f0 / tau2 expected goals."""
    P = panel()
    lv = league_levels()
    je = [P.j(y) for y in range(V - eb_window, V) if P.seasons[0] <= y <= P.seasons[-1]]
    out = {}
    for pos in ("F", "D"):
        m = P.pos == pos
        g = sum(P.x[f"g_{k}"][m][:, je] for k in SITS)
        xg = sum(P.x[f"ixg_{k}"][m][:, je] * lv[f"rho_{k}"].to_numpy()[je] for k in SITS)
        f0 = float(g.sum() / xg.sum())
        f = np.where(xg > 0, g / np.maximum(xg, 1e-9), f0) - f0
        ok = xg > 3.0
        w = np.where(ok[:, :-1] & ok[:, 1:],
                     2.0 / (1 / np.maximum(xg[:, :-1], 1e-9) + 1 / np.maximum(xg[:, 1:], 1e-9)),
                     0.0)
        tau2 = max(float((w * f[:, :-1] * f[:, 1:]).sum() / max(w.sum(), 1e-9)), 1e-4)
        out[pos] = {"f0": f0, "tau2": tau2, "k_fin": f0 / tau2}
    return out


# ---------------------------------------------------------------------------
# Raw (un-aged) rates
# ---------------------------------------------------------------------------
SCORING_PRIOR_STATS = ("g", "a1", "a2", "sog", "ixg")


def usage_points_prior(V: int, prm: Params, pos: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Relative points rate implied by the usage prior at usage X: the
    league-rate-weighted mean of the g/a1/a2 priors over the situations the
    player's minutes fall in."""
    pri = rate_priors(V, prm.prior_window, prm.eb_window)
    tgt = project_league(V)
    num = np.zeros(len(pos))
    den = np.zeros(len(pos))
    tpg = {"ev": X[:, 1], "pp": X[:, 2], "sh": X[:, 3]}
    for st in ("g", "a1", "a2"):
        for k in ("ev", "pp", "sh"):
            c = f"{st}_{k}"
            pr = np.zeros(len(pos))
            for p_ in ("F", "D"):
                m = pos == p_
                pr[m] = np.clip(X[m] @ pri[(c, p_)]["beta"], 1e-3, None)
            e = tgt[c] * tpg[k]
            num += pr * e
            den += e
    return num / np.maximum(den, 1e-12)


def rates_raw(P: Panel, V: int, prm: Params, usage: dict,
              prior_override: dict | None = None,
              pr_prosp: np.ndarray | None = None) -> dict:
    """Shrunk relative rates for every stat column (un-aged).

    Returns rel[c] (relative to V's league per-minute level), their posterior
    variance var[c], and the weighted exposure T[c]. `prior_override` maps a
    column to per-player prior relative rates (the NHLe blend for players
    with thin NHL records).
    """
    lv = league_levels()
    idx, w = _weights(P, V, prm.rate_decay)
    pri = rate_priors(V, prm.prior_window, prm.eb_window)
    X = _design(usage["tpg_ev"], usage["tpg_pp"], usage["tpg_sh"])
    pmult = np.ones(len(P.ids))
    if pr_prosp is not None:
        pu = usage_points_prior(V, prm, P.pos, X)
        ok = np.isfinite(pr_prosp)
        pmult[ok] = (prm.prosp_alpha * pr_prosp[ok]
                     + (1 - prm.prosp_alpha) * pu[ok]) / pu[ok]
        pmult = np.clip(pmult, 0.3, 4.0)
    rel, var, T = {}, {}, {}
    cols = [f"{s}_{k}" for s in SIT_STATS for k in SITS] + [f"{s}_all" for s in PERIPH]
    for c in cols:
        tcol = stat_toi(c)
        ell = lv[c].to_numpy()[idx]
        t = (P.x[tcol][:, idx] * w).sum(1)
        n = (P.x[c][:, idx] / np.maximum(ell, 1e-12) * w).sum(1)   # league-min units
        prior = np.zeros(len(P.ids))
        K = np.zeros(len(P.ids))
        tau2 = np.zeros(len(P.ids))
        for pos in ("F", "D"):
            m = P.pos == pos
            e = pri[(c, pos)]
            pr = np.clip(X[m] @ e["beta"], 1e-3, None)
            if c.split("_")[0] in SCORING_PRIOR_STATS:
                pr = pr * pmult[m]
            if prior_override and c in prior_override:
                o = prior_override[c][m]
                pr = np.where(np.isfinite(o), o, pr)
            prior[m] = pr
            ellV = float(project_league(V)[c])
            K[m] = prm.kappa * e["phi"] * pr / (max(ellV, 1e-12) * e["tau2"])
            tau2[m] = e["tau2"]
        rel[c] = (n + K * prior) / (t + K)
        var[c] = tau2 * K / (K + t)
        T[c] = t
    # --- finishing: goals = ixG x league G/ixG x finishing multiplier
    fp = finishing_prior(V, prm.eb_window)
    rho = lv[[f"rho_{k}" for k in SITS]].to_numpy()[idx]           # (L, 4)
    g = sum((P.x[f"g_{k}"][:, idx] * w).sum(1) for k in SITS)
    xg = sum((P.x[f"ixg_{k}"][:, idx] * rho[:, i] * w).sum(1) for i, k in enumerate(SITS))
    f0 = np.where(P.pos == "D", fp["D"]["f0"], fp["F"]["f0"])
    kf = prm.kappa * np.where(P.pos == "D", fp["D"]["k_fin"], fp["F"]["k_fin"])
    fin = (g + kf * f0) / (xg + kf)
    tgt = project_league(V)
    for k in SITS:
        c = f"g_{k}"
        via_xg = rel[f"ixg_{k}"] * tgt[f"ixg_{k}"] * tgt[f"rho_{k}"] * fin / tgt[c]
        rel[c] = prm.fin_weight * via_xg + (1 - prm.fin_weight) * rel[c]
    rel["fin"] = fin
    return {"rel": rel, "var": var, "T": T}


# ---------------------------------------------------------------------------
# On-ice 5v5 impact relative to team (off-ice)
# ---------------------------------------------------------------------------
def onice_rel(P: Panel, V: int, prm: Params) -> dict:
    """Shrunk 5v5 xGF/60 and xGA/60 relative to the team without him.

    rel = on-ice rate - off-ice rate (same games). Noise in xG rates is far
    larger than the talent spread, so the prior (0 for everyone) carries a
    weight of K_ONICE minutes, estimated by MoM like the counting rates.
    """
    lv = league_levels()
    tgt = project_league(V)
    idx, w = _weights(P, V, prm.rate_decay)
    out = {}
    for side in ("xgf", "xga"):
        ell = lv[f"ixg_ev"].to_numpy()[idx]         # scale by league xG level
        on = P.x[f"oi_{side}_ev"][:, idx] / np.maximum(ell, 1e-12)
        off = P.x[f"off_{side}_ev"][:, idx] / np.maximum(ell, 1e-12)
        ton = P.x["toi_ev"][:, idx]
        toff = P.x["bench_ev"][:, idx]
        r = np.where(ton > 0, on / np.maximum(ton, 1e-9), 0) - \
            np.where(toff > 0, off / np.maximum(toff, 1e-9), 0)
        t = (ton * w).sum(1)
        num = (r * ton * w).sum(1)
        k = _onice_k(V, side, prm.eb_window) * prm.kappa
        out[f"rel_{side}60"] = num / (t + k) * float(tgt["ixg_ev"]) * 60.0
        out[f"rel_{side}60_sd"] = np.sqrt(_onice_tau2(V, side, prm.eb_window) * k / (k + t)) \
            * float(tgt["ixg_ev"]) * 60.0
    return out


@functools.lru_cache(maxsize=None)
def _onice_stats(V: int, side: str, eb_window: int) -> tuple:
    P = panel()
    lv = league_levels()
    je = [P.j(y) for y in range(V - eb_window, V) if P.seasons[0] <= y <= P.seasons[-1]]
    ell = lv["ixg_ev"].to_numpy()[je]
    ton = P.x["toi_ev"][:, je]
    toff = P.x["bench_ev"][:, je]
    r = (np.where(ton > 0, P.x[f"oi_{side}_ev"][:, je] / np.maximum(ton, 1e-9), 0)
         - np.where(toff > 0, P.x[f"off_{side}_ev"][:, je] / np.maximum(toff, 1e-9), 0)) / ell
    ok = ton > 100
    w = np.where(ok[:, :-1] & ok[:, 1:],
                 2 / (1 / np.maximum(ton[:, :-1], 1e-9) + 1 / np.maximum(ton[:, 1:], 1e-9)), 0)
    tau2 = max(float((w * r[:, :-1] * r[:, 1:]).sum() / max(w.sum(), 1e-9)), 1e-6)
    sig2 = float(np.mean((r[ok] ** 2 - tau2) * ton[ok]))
    return tau2, max(sig2, 1e-6)


def _onice_k(V, side, eb_window):
    tau2, sig2 = _onice_stats(V, side, eb_window)
    return sig2 / tau2


def _onice_tau2(V, side, eb_window):
    return _onice_stats(V, side, eb_window)[0]


# ---------------------------------------------------------------------------
# Raw projection for one season (cached per hyperparameter set)
# ---------------------------------------------------------------------------
_RAW_CACHE: dict = {}


def raw_projection(V: int, prm: Params) -> dict:
    """Un-aged usage and rates for every panel player with NHL history < V."""
    key = (V, prm.key())
    if key not in _RAW_CACHE:
        P = panel()
        rk = prospect_block(V, P.ids, P.pos, ages(P, V), thin_gp=_hist_gp(P, V))
        u = usage_raw(P, V, prm, tpg_mult=rk["tpg_mult"])
        r = rates_raw(P, V, prm, u, pr_prosp=rk["pr"])
        _RAW_CACHE[key] = {"usage": u, **r, "has_hist": u["gp_hist"] > 0,
                           "pr_prosp": rk["pr"], "tpg_mult": rk["tpg_mult"]}
    return _RAW_CACHE[key]


def _hist_gp(P: Panel, V: int) -> np.ndarray:
    js = [P.j(y) for y in range(P.seasons[0], V) if y <= P.seasons[-1]]
    return P.x["gp"][:, js].sum(1) if js else np.zeros(len(P.ids))


# ---------------------------------------------------------------------------
# Aging: walk-forward residual curves
# ---------------------------------------------------------------------------
AGE_KNOTS = (23.0, 27.0, 31.0)


def _age_basis(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, AGE_LO, AGE_HI)
    cols = [np.ones_like(a), a - 27.0] + [np.maximum(a - k, 0.0) for k in AGE_KNOTS]
    return np.column_stack(cols)


def _fit_age_curve(age, actual, pred, ridge=2.0) -> np.ndarray:
    """Piecewise-linear log multiplier in age, fitted to per-age totals.

    log(sum actual / sum pred) per integer age, weighted by sum pred (the
    Poisson information), with a light ridge on the slope changes so thin
    ages (19, 37+) borrow from their neighbours.
    """
    a = np.clip(np.floor(age), AGE_LO, AGE_HI)
    df = pd.DataFrame({"a": a, "y": actual, "e": pred}).groupby("a").sum()
    df = df[(df.e > 5) & (df.y > 0)]
    X = _age_basis(df.index.to_numpy(float) + 0.5)
    yv = np.log(df.y / df.e).to_numpy()
    w = df.e.to_numpy()
    pen = np.diag([0.0, 0.0] + [ridge] * len(AGE_KNOTS))
    A = X.T @ (X * w[:, None]) + pen * w.mean()
    return np.linalg.solve(A, X.T @ (w * yv))


@functools.lru_cache(maxsize=64)
def _aging_cached(V: int, pkey: str) -> dict:
    prm = Params(**json.loads(pkey))
    P = panel()
    lv = league_levels()
    rows = {g: {pos: ([], [], []) for pos in "FD"} for g in (*AGE_GROUPS, "toi")}
    for s in range(FIRST_SEASON + 1, V):
        if s > P.seasons[-1]:
            break
        raw = raw_projection(s, prm)
        js = P.j(s)
        tgt = project_league(s)
        has = raw["has_hist"] & (P.x["gp"][:, js] > 0)
        age = ages(P, s)
        for grp, stats in AGE_GROUPS.items():
            act = np.zeros(len(P.ids))
            pred = np.zeros(len(P.ids))
            for st in stats:
                cols = [f"{st}_{k}" for k in SITS] if st in SIT_STATS else [f"{st}_all"]
                for c in cols:
                    act += P.x[c][:, js]
                    pred += raw["rel"][c] * lv.loc[s, c] * P.x[stat_toi(c)][:, js]
            for pos in "FD":
                m = has & (P.pos == pos)
                rows[grp][pos][0].append(age[m])
                rows[grp][pos][1].append(act[m])
                rows[grp][pos][2].append(pred[m])
        # TOI per game (all situations), era-restated to s's team minutes
        act = sum(P.x[f"toi_{k}"][:, js] for k in SITS)
        pred = sum(raw["usage"][f"tpg_{k}"] * lv.loc[s, f"team_min_{k}"] / tgt[f"team_min_{k}"]
                   for k in SITS) * P.x["gp"][:, js]
        for pos in "FD":
            m = has & (P.pos == pos)
            rows["toi"][pos][0].append(age[m])
            rows["toi"][pos][1].append(act[m])
            rows["toi"][pos][2].append(pred[m])
    out = {}
    for grp, d in rows.items():
        for pos in "FD":
            a, y, e = (np.concatenate(v) for v in d[pos])
            out[(grp, pos)] = _fit_age_curve(a, y, e)
    return out


def age_multiplier(V: int, prm: Params, group: str, pos: np.ndarray,
                   age: np.ndarray) -> np.ndarray:
    """Multiplier on the un-aged projection for a player of this age in V."""
    if not prm.age_curve:
        return np.ones(len(age))
    coef = _aging_cached(V, prm.key())
    X = _age_basis(np.asarray(age, float))
    out = np.ones(len(age))
    for p in "FD":
        m = pos == p
        out[m] = np.exp(X[m] @ coef[(group, p)])
    return np.clip(out, 0.5, 1.6)


def stat_group(stat: str) -> str | None:
    for g, sts in AGE_GROUPS.items():
        if stat in sts:
            return g
    return None


# ---------------------------------------------------------------------------
# Final projection table
# ---------------------------------------------------------------------------
def project(V: int, prm: Params | None = None, ids=None,
            extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-player projection for season V from seasons < V only.

    ids: restrict to these player ids (default: everyone with NHL history
    before V). extra: rows (player_id, pos, birth, name) for players absent
    from the MoneyPuck panel (true rookies in the forecast season).

    Columns: player_id, name, pos, age, has_hist, nhl_gp, score (depth-chart
    score = projected min/game), tpg_<sit> (un-normalised min/game),
    r60_<stat>_<sit> and r60_<periph> (per-60 at V's league level),
    sd60_<...> (posterior SD of the per-60 talent), fin (finishing
    multiplier), rel_xgf60 / rel_xga60 (+ _sd): 5v5 on-ice impact vs team.
    """
    prm = prm or load_params()
    P = panel()
    raw = raw_projection(V, prm)
    tgt = project_league(V)
    age = ages(P, V)
    has = raw["has_hist"]
    rel, var = raw["rel"], raw["var"]
    usage = raw["usage"]
    df = pd.DataFrame({"player_id": P.ids, "name": P.name, "pos": P.pos,
                       "age": age, "has_hist": has, "nhl_gp": usage["gp_hist"]})
    mult = {g: age_multiplier(V, prm, g, P.pos, age) for g in (*AGE_GROUPS, "toi")}
    for g in mult:                                   # no aging without a past
        mult[g] = np.where(has, mult[g], 1.0)
    for k in SITS:
        df[f"tpg_{k}"] = usage[f"tpg_{k}"] * mult["toi"]
    for s in SIT_STATS:
        for k in SITS:
            c = f"{s}_{k}"
            m = mult[stat_group(s)]
            df[f"r60_{c}"] = rel[c] * tgt[c] * 60.0 * m
            df[f"sd60_{c}"] = np.sqrt(var[c]) * tgt[c] * 60.0 * m
    for s in PERIPH:
        c = f"{s}_all"
        g = stat_group(s)
        m = mult[g] if g else 1.0
        df[f"r60_{s}"] = rel[c] * tgt[c] * 60.0 * m
        df[f"sd60_{s}"] = np.sqrt(var[c]) * tgt[c] * 60.0 * m
    df["fin"] = rel["fin"]
    oi = onice_rel(P, V, prm)
    for c, v in oi.items():
        df[c] = v
    df["score"] = sum(df[f"tpg_{k}"] for k in SITS)
    df["pr_prosp"] = raw["pr_prosp"]
    if extra is not None and len(extra):
        ex = extra[~extra.player_id.isin(P.row)]
        if len(ex):
            df = pd.concat([df, project_extra(V, prm, ex)], ignore_index=True)
    if ids is not None:
        df = df[df.player_id.isin(set(ids))]
    else:
        df = df[df.has_hist]
    return df.reset_index(drop=True)




# ---------------------------------------------------------------------------
# Rookies and thin NHL records: NHLe translation + draft position
# ---------------------------------------------------------------------------
THIN_GP = 82              # NHL games before V below which prospect data count
MIN_LEAGUE_GP = 10        # league seasons shorter than this are ignored
NHLE_SHRINK = 30.0        # harmonic-GP units of pseudo-data per league factor
UNDRAFTED_PICK = 250.0


def points_ratio(P: Panel, js: int) -> tuple[np.ndarray, np.ndarray]:
    """(actual points, points expected at league-average rates for the same
    minutes by situation) in panel season column js."""
    lv = league_levels()
    y = P.seasons[js]
    act = np.zeros(len(P.ids))
    exp = np.zeros(len(P.ids))
    for st in ("g", "a1", "a2"):
        for k in SITS:
            act += P.x[f"{st}_{k}"][:, js]
            exp += P.x[f"toi_{k}"][:, js] * lv.loc[y, f"{st}_{k}"]
    return act, exp


@functools.lru_cache(maxsize=None)
def _league_rows() -> pd.DataFrame:
    p = D.prospects()
    p = p[p.gp >= MIN_LEAGUE_GP].copy()
    p["ppg"] = p.pts / p.gp
    return p


@functools.lru_cache(maxsize=None)
def nhle_factors(V: int) -> dict:
    """League -> factor mapping league points/game to NHL relative points
    rate (1.0 = league-average NHL skater), from transitions into NHL seasons
    V-10..V-1. Every NHL minute counts (no games-played threshold, which
    would select on the outcome); pairs are weighted by the harmonic mean of
    league games and NHL games, and each factor is shrunk toward the pooled
    factor with NHLE_SHRINK pseudo-weight."""
    P = panel()
    L = _league_rows()
    rows = []
    for y in range(max(V - 10, P.seasons[0]), V):
        if y > P.seasons[-1]:
            break
        act, exp = points_ratio(P, P.j(y))
        gp = P.x["gp"][:, P.j(y)]
        nhl = pd.DataFrame({"player_id": P.ids, "y": act / np.maximum(exp, 1e-9),
                            "gp_nhl": gp, "exp": exp})
        nhl = nhl[nhl.gp_nhl > 0]
        lg = L[L.season_end == y - 1]
        m = lg.merge(nhl, on="player_id")
        m["w"] = 1.0 / (1.0 / m.gp + 1.0 / m.gp_nhl)
        rows.append(m)
    m = pd.concat(rows)
    pooled = float((m.w * m.y).sum() / (m.w * m.ppg).sum())
    g = m.groupby("league").apply(
        lambda d: pd.Series({"n": d.w.sum(), "num": (d.w * d.y).sum(),
                             "den": (d.w * d.ppg).sum()}), include_groups=False)
    f = (g.num + NHLE_SHRINK * pooled * (g.den / g.n)) / (g.den + NHLE_SHRINK * (g.den / g.n))
    return {"pooled": pooled, **f.to_dict()}


def prospect_score(V: int, ids) -> pd.Series:
    """Translated pre-V production: NHLe-weighted points/game over the two
    most recent seasons before V (weights: games x 1.0 / 0.5 by recency)."""
    f = nhle_factors(V)
    L = _league_rows()
    L = L[(L.season_end < V) & (L.season_end >= V - 2) & L.player_id.isin(set(ids))]
    L = L[L.league.isin(f.keys())]
    if not len(L):
        return pd.Series(dtype=float)
    w = L.gp * np.where(L.season_end == V - 1, 1.0, 0.5)
    x = L.league.map(f) * L.ppg
    return (x * w).groupby(L.player_id).sum() / w.groupby(L.player_id).sum()


@functools.lru_cache(maxsize=None)
def _pick_table() -> pd.Series:
    p = D.prospects().dropna(subset=["overall"]).drop_duplicates("player_id")
    a = p.set_index("player_id").overall
    d = D.draft().dropna(subset=["player_id"]).drop_duplicates("player_id")
    b = d.set_index(d.player_id.astype(int)).overall
    return pd.concat([a, b[~b.index.isin(a.index)]]).astype(float)


def draft_pick(ids, names=None) -> np.ndarray:
    """Overall draft pick (UNDRAFTED_PICK if none): by id, else by name."""
    t = _pick_table()
    out = pd.Series(list(ids)).map(t).to_numpy(float)
    if names is not None:
        import unicodedata
        def norm(z):
            z = unicodedata.normalize("NFKD", str(z)).encode("ascii", "ignore").decode()
            return z.lower().replace(".", "").replace("'", "").strip()
        d = D.draft()
        by = dict(zip(d.name.map(norm), d.overall.astype(float)))
        miss = ~np.isfinite(out)
        out[miss] = [by.get(norm(n), np.nan) for n in np.asarray(names)[miss]]
    return np.where(np.isfinite(out), out, UNDRAFTED_PICK)


def _prosp_design(logx, has_x, pick, is_d, age):
    return np.column_stack([np.ones(len(logx)), logx * has_x, has_x,
                            np.log(pick), is_d, np.clip(age, 18, 28) - 21.0])


@functools.lru_cache(maxsize=None)
def prospect_model(V: int) -> dict:
    """Walk-forward regressions for players with thin NHL records (<THIN_GP
    NHL games before the season), fitted on NHL seasons V-10..V-1:

      points ratio (actual / league-rate expectation)  ~ features, weights =
                                                       expected points
      log(TOI per game / fringe TOI per game)           ~ features, weights = GP

    features: log NHLe score (0 if none) and a has-score flag, log draft
    pick, defence flag, age. Every player who dressed counts."""
    P = panel()
    Xs, ys, ws, Ts, wt = [], [], [], [], []
    for y in range(max(V - 10, P.seasons[0] + 1), V):
        if y > P.seasons[-1]:
            break
        js = P.j(y)
        gp = P.x["gp"][:, js]
        thin = _hist_gp(P, y) < THIN_GP
        m = thin & (gp > 0)
        ids = P.ids[m]
        xs = prospect_score(y, ids).reindex(ids)
        has_x = np.isfinite(xs.to_numpy(float)).astype(float)
        logx = np.log(np.clip(xs.fillna(1.0).to_numpy(float), 0.05, None))
        X = _prosp_design(logx, has_x, draft_pick(ids), (P.pos[m] == "D").astype(float),
                          ages(P, y)[m])
        act, exp = points_ratio(P, js)
        Xs.append(X)
        ys.append(act[m] / np.maximum(exp[m], 1e-9))
        ws.append(exp[m])
        fr = usage_prior_tpg(y)
        fr_all = np.where(P.pos[m] == "D", sum(fr[("D", k)] for k in SITS),
                          sum(fr[("F", k)] for k in SITS))
        Ts.append(np.log(np.maximum(P.x["toi_all"][:, js][m] / gp[m], 1.0) / fr_all))
        wt.append(gp[m])
    X = np.vstack(Xs)
    y, w, T, wg = map(np.concatenate, (ys, ws, Ts, wt))
    ridge = np.diag([0, 1, 1, 1, 1, 1]) * 1e-3
    b_pr = np.linalg.solve(X.T @ (X * w[:, None]) + ridge * w.sum(), X.T @ (w * y))
    b_t = np.linalg.solve(X.T @ (X * wg[:, None]) + ridge * wg.sum(), X.T @ (wg * T))
    return {"b_pr": b_pr, "b_toi": b_t, "n": int(len(y))}


def prospect_block(V: int, ids, pos, age, thin_gp=None, names=None) -> dict:
    """Prospect priors for players with thin NHL records.

    Returns pr (predicted NHL points ratio) and tpg_mult (multiplier on the
    fringe TOI prior), NaN for players with >= THIN_GP NHL games or with
    neither a translated score nor a draft pick.
    """
    ids = np.asarray(ids)
    n = len(ids)
    out = {"pr": np.full(n, np.nan), "tpg_mult": np.full(n, np.nan)}
    thin = np.ones(n, bool) if thin_gp is None else np.asarray(thin_gp) < THIN_GP
    if V - 1 < FIRST_SEASON + 1 or not thin.any():
        return out
    mdl = prospect_model(V)
    xs = prospect_score(V, ids[thin]).reindex(ids[thin])
    has_x = np.isfinite(xs.to_numpy(float)).astype(float)
    pick = draft_pick(ids[thin], None if names is None else np.asarray(names)[thin])
    info = (has_x > 0) | (pick < UNDRAFTED_PICK)
    logx = np.log(np.clip(xs.fillna(1.0).to_numpy(float), 0.05, None))
    X = _prosp_design(logx, has_x, pick, (np.asarray(pos)[thin] == "D").astype(float),
                      np.asarray(age)[thin])
    pr = np.clip(X @ mdl["b_pr"], 0.05, 3.0)
    tm = np.exp(np.clip(X @ mdl["b_toi"], -0.5, 0.8))
    idx = np.where(thin)[0]
    out["pr"][idx[info]] = pr[info]
    out["tpg_mult"][idx[info]] = tm[info]
    return out


def project_extra(V: int, prm: Params, ex: pd.DataFrame) -> pd.DataFrame:
    """Rows for skaters with no MoneyPuck record at all (true rookies of the
    forecast season): prior usage and prior rates, prior-width SDs."""
    pos = np.where(ex.pos.to_numpy() == "D", "D", "F")
    age = ((pd.Timestamp(f"{V}-02-01") - pd.to_datetime(ex.birth)).dt.days / 365.25) \
        .fillna(21.0).to_numpy()
    rk = prospect_block(V, ex.player_id.to_numpy(), pos, age,
                        names=ex.name.to_numpy() if "name" in ex else None)
    fr = usage_prior_tpg(V)
    tmult = np.where(np.isfinite(rk["tpg_mult"]), rk["tpg_mult"], 1.0)
    df = pd.DataFrame({"player_id": ex.player_id.to_numpy(),
                       "name": ex.get("name", pd.Series([""] * len(ex))).to_numpy(),
                       "pos": pos, "age": age, "has_hist": False, "nhl_gp": 0.0})
    for k in SITS:
        df[f"tpg_{k}"] = np.array([fr[(p, k)] for p in pos]) * tmult
    X = _design(df.tpg_ev.to_numpy(), df.tpg_pp.to_numpy(), df.tpg_sh.to_numpy())
    pu = usage_points_prior(V, prm, pos, X)
    ok = np.isfinite(rk["pr"])
    pm = np.ones(len(df))
    pm[ok] = (prm.prosp_alpha * rk["pr"][ok] + (1 - prm.prosp_alpha) * pu[ok]) / pu[ok]
    pm = np.clip(pm, 0.3, 4.0)
    pri = rate_priors(V, prm.prior_window, prm.eb_window)
    tgt = project_league(V)
    cols = [f"{s}_{k}" for s in SIT_STATS for k in SITS] + [f"{s}_all" for s in PERIPH]
    for c in cols:
        r = np.zeros(len(df))
        sd = np.zeros(len(df))
        for p_ in ("F", "D"):
            m = pos == p_
            e = pri[(c, p_)]
            r[m] = np.clip(X[m] @ e["beta"], 1e-3, None)
            sd[m] = np.sqrt(e["tau2"])
        if c.split("_")[0] in SCORING_PRIOR_STATS:
            r = r * pm
        name = f"r60_{c}" if c.split("_")[0] in SIT_STATS else f"r60_{c[:-4]}"
        sdn = name.replace("r60_", "sd60_")
        df[name] = r * tgt[c] * 60.0
        df[sdn] = sd * tgt[c] * 60.0
    # goals via the finishing path at the positional finishing prior
    fp = finishing_prior(V, prm.eb_window)
    df["fin"] = np.where(pos == "D", fp["D"]["f0"], fp["F"]["f0"])
    for side in ("xgf", "xga"):
        df[f"rel_{side}60"] = 0.0
        df[f"rel_{side}60_sd"] = np.sqrt(_onice_tau2(V, side, prm.eb_window)) \
            * float(tgt["ixg_ev"]) * 60.0
    df["score"] = sum(df[f"tpg_{k}"] for k in SITS)
    df["pr_prosp"] = rk["pr"]
    return df
