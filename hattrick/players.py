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
LEAGUE_RULE = "last_intact"   # or "mean2_intact" (round-2 test, see ledger)
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
    # --- deployment (hattrick.deploy)
    dress_noise: float = 2.0       # SD (min/game) of coaches' depth-chart noise
    q_scale: float = 1.0           # calibration of absence-spell injury rates
    # --- Monte Carlo intervals (calibrated on 2011-17 coverage)
    spell_len: float = 5.2         # mean injury spell (games): matches the
                                   # games-weighted spell length 9.4 of 2015-19
    usage_sd: float = 0.10         # season-level log-SD of min/game vs projection
    sd_scale: float = 1.0          # multiplier on posterior talent SDs
    team_sd: float = 0.04          # team-season shooting-luck log-SD
    rho_ga: float = 0.6            # goal/assist talent correlation
    # --- thin NHL records (fitted on the 2011-17 strict window, unconditional)
    # rookies (no NHL games), by draft class (top-10 pick, 11-32, later,
    # undrafted): multipliers on expected games and on scoring rates
    rk_gp_mult: tuple = (1.0, 1.0, 1.0, 1.0)
    rk_rate_mult: tuple = (1.0, 1.0, 1.0, 1.0)
    thin_gp_mult: float = 1.0      # 1-81 NHL games: expected games
    thin_rate_mult: float = 1.0    # 1-81 NHL games: scoring rates
    # --- forecast averaging of season totals (weights; HatTrick gets the rest)
    blend_marcel: float = 0.0      # Marcel-style 5/4/3 baseline
    blend_lastgp: float = 0.0      # HatTrick per-game x last season's games

    RATE_FIELDS = ("rate_decay", "toi_decay", "kappa", "fin_weight", "k_toi_games",
                   "prosp_alpha", "age_curve", "eb_window", "prior_window")

    def key(self) -> str:
        """Cache key of everything the RATE/usage projection depends on."""
        return json.dumps({k: getattr(self, k) for k in self.RATE_FIELDS}, sort_keys=True)

    def mc(self) -> dict:
        return {"noise_sd": self.dress_noise, "spell_len": self.spell_len,
                "usage_sd": self.usage_sd, "sd_scale": self.sd_scale,
                "team_sd": self.team_sd, "rho_ga": self.rho_ga}


def load_params() -> Params:
    """Tuned parameters if the tuner has written them, else the defaults."""
    if PARAMS_PATH.exists():
        d = json.loads(PARAMS_PATH.read_text()).get("hyper", {})
        known = {k: (tuple(v) if isinstance(v, list) else v)
                 for k, v in d.items() if k in Params.__dataclass_fields__}
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
    cols += ["oi_xga_sh", "oi_xgf_pp"]
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
    if LEAGUE_RULE == "mean2_intact" and len(prev) >= 2:
        out = (lv.loc[prev[0]] + lv.loc[prev[1]]) / 2.0
        out.name = prev[0]
        return out
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


def special_teams_onice(P: Panel, V: int, prm: Params) -> dict:
    """Shrunk on-ice xGF/60 on the power play and xGA/60 on the penalty
    kill, relative to the league's PP/PK rate (the same for everyone on the
    ice, so the prior is the league rate and K comes from the MoM)."""
    lv_rate = {}
    idx, w = _weights(P, V, prm.rate_decay)
    out = {}
    for col, sit in (("oi_xgf_pp", "pp"), ("oi_xga_sh", "sh")):
        js = [P.j(y) for y in range(V - prm.eb_window, V) if P.seasons[0] <= y <= P.seasons[-1]]
        num_l = P.x[col][:, js].sum(0)
        den_l = P.x[f"toi_{sit}"][:, js].sum(0)
        ell = num_l / np.maximum(den_l, 1e-9)                   # per minute by season
        t = P.x[f"toi_{sit}"][:, js]
        r = np.where(t > 0, P.x[col][:, js] / np.maximum(t, 1e-9), 0) / ell - 1.0
        ok = t > 30
        wp = np.where(ok[:, :-1] & ok[:, 1:], 2 / (1 / np.maximum(t[:, :-1], 1e-9) + 1 / np.maximum(t[:, 1:], 1e-9)), 0)
        tau2 = max(float((wp * r[:, :-1] * r[:, 1:]).sum() / max(wp.sum(), 1e-9)), 1e-5)
        sig2 = max(float(np.mean((r[ok] ** 2 - tau2) * t[ok])), 1e-5)
        k = prm.kappa * sig2 / tau2
        lvl = {y: v for y, v in zip([P.seasons[j] for j in js], ell)}
        ellV = lvl[max(lvl)]
        ti = P.x[f"toi_{sit}"][:, idx]
        ri = np.where(ti > 0, P.x[col][:, idx] / np.maximum(ti, 1e-9), 0)
        ells = np.array([lvl.get(P.seasons[j], ellV) for j in idx])
        rel = ((ri / ells - 1.0) * ti * w).sum(1) / ((ti * w).sum(1) + k)
        out[f"{sit}_onice_rel"] = rel
        out[f"{sit}_onice_60"] = (1 + rel) * ellV * 60.0
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
            if not d[pos][0]:                   # no earlier season: no aging
                out[(grp, pos)] = np.zeros(2 + len(AGE_KNOTS))
                continue
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
    for c, v in special_teams_onice(P, V, prm).items():
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
    out = pd.Series(list(ids)).map(t).to_numpy(float).copy()
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
    df["pp_onice_rel"] = 0.0
    df["sh_onice_rel"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Season pipeline (backtests, team components and the freeze share it)
# ---------------------------------------------------------------------------
SCORING = ("g", "a1", "a2")


RK_CLASSES = ("rk_top10", "rk_r1", "rk_later", "rk_undrafted")


def thin_group(proj: pd.DataFrame) -> pd.Series:
    """Rookies (no NHL games before V) by draft class -- 'rk_top10',
    'rk_r1' (picks 11-32), 'rk_later', 'rk_undrafted' -- then 'thin'
    (1-81 NHL games) and 'est'."""
    g = proj.nhl_gp.fillna(0.0).to_numpy()
    pick = draft_pick(proj.player_id.to_numpy(),
                      proj.name.to_numpy() if "name" in proj else None)
    rk = np.select([pick <= 10, pick <= 32, pick < UNDRAFTED_PICK],
                   list(RK_CLASSES[:3]), RK_CLASSES[3])
    return pd.Series(np.where(g <= 0, rk, np.where(g < THIN_GP, "thin", "est")),
                     index=proj.index)


def group_mults(prm: Params, kind: str) -> dict:
    rk = prm.rk_gp_mult if kind == "gp" else prm.rk_rate_mult
    out = dict(zip(RK_CLASSES, rk))
    out["thin"] = prm.thin_gp_mult if kind == "gp" else prm.thin_rate_mult
    out["est"] = 1.0
    return out


def calibrate_thin_rates(proj: pd.DataFrame, prm: Params) -> pd.DataFrame:
    """Multiply rookies' / thin-record players' scoring rates by the factors
    fitted on the tuning window (their priors were biased there)."""
    proj = proj.copy()
    grp = thin_group(proj)
    m = group_mults(prm, "rate")
    m["thin"] = 1.0              # thin records are calibrated after blending
    mult = grp.map(m).to_numpy(float)
    for c in [c for c in proj.columns if c.startswith(("r60_", "sd60_"))
              and c.split("_")[1] in SCORING]:
        proj[c] = proj[c] * mult
    return proj


def pipeline(V: int, prm: Params, roster: pd.DataFrame, games: int, kind: str,
             games_out=None, games_out_range=None, extra=None, eval_ids=(),
             sims: int = 0, stack: bool = True, neurhl_budget: bool = False):
    """Projection -> deployment -> totals (-> intervals) -> forecast averaging.

    roster: player_id, team (skaters). kind: 'opening' (depth-chart games
    with call-up coverage) or 'expost' (team-agnostic games). eval_ids:
    extra players to project (not deployed). Returns (totals, projection).
    """
    from hattrick import deploy as DP
    ids = set(roster.player_id) | set(eval_ids)
    proj = calibrate_thin_rates(project(V, prm, ids=ids, extra=extra), prm)
    ros = roster[roster.player_id.isin(proj.player_id)]
    gmap = DP.league_gp_map(V, prm) if kind == "expost" else None
    cover = DP.coverage(V, "opening") if kind == "opening" else None
    kw = dict(games_out=games_out, games_out_range=games_out_range)
    dep = DP.deploy(proj, ros, V, games, prm.dress_noise, prm.q_scale, gp_map=gmap,
                    cover=cover, **kw)
    gs = DP.apply_stack(dep, V, prm, kind, games) if stack else \
        dep.set_index("player_id").gp.to_dict()
    grp = dict(zip(proj.player_id, thin_group(proj)))
    gm = group_mults(prm, "gp")
    cap = dict(zip(dep.player_id, games - dep.games_out))
    gs = {k: min(v * gm[grp.get(k, "est")], cap.get(k, games)) for k, v in gs.items()}
    dep = DP.deploy(proj, ros, V, games, prm.dress_noise, prm.q_scale, gp_override=gs,
                    cover=cover, **kw)
    if neurhl_budget:
        g = DP.neurhl_gp_budget(dep.set_index("player_id").gp, dep.team.nunique(), games)
        dep = DP.deploy(proj, ros, V, games, prm.dress_noise, prm.q_scale,
                        gp_override=g.to_dict(), **kw)
    tot = DP.add_totals(dep, proj)
    if sims:
        q = DP.simulate(tot, proj, V, games, prm.mc(), S=sims)
        tot = tot.merge(q, on="player_id", how="left")
    tot = blend_paths(tot, V, prm, games)
    # thin NHL records (1-81 games): calibrate the FINAL (blended) scoring,
    # since Marcel and the last-season-games path both pull them down
    thin = tot.player_id.map(dict(zip(proj.player_id, thin_group(proj)))) == "thin"
    if prm.thin_rate_mult != 1.0 and thin.any():
        tot = tot.copy()
        cols = [c for c in tot.columns if c in ("g", "a", "p", "a1", "a2", "ppg", "ppa")
                or c.startswith(("g_p", "a_p", "p_p")) or c in ("g_sd", "a_sd", "p_sd")]
        tot.loc[thin, cols] = tot.loc[thin, cols] * prm.thin_rate_mult
    return tot.copy(), proj


MARCEL_W = (5.0, 4.0, 3.0)
MARCEL_REG_GP = 30.0        # games of positional league PPG added


@functools.lru_cache(maxsize=None)
def marcel_baseline(V: int) -> pd.DataFrame:
    """Marcel-style baseline (82-game basis): 5/4/3-weighted goals and
    assists per game over V-1..V-3, each season restated to the scoring level
    of the last intact season before V; regressed with 30 games of the
    positional league average; age factor +1%/yr below 27, -1.5%/yr above
    29; games = 0.5 GP(V-1) + 0.1 GP(V-2) + 25 (per 82, capped at 82)."""
    P = panel()
    lv = league_levels()
    ptot = lambda y: sum(lv.loc[y, f"{s}_{k}"] for s in ("g", "a1", "a2") for k in SITS)
    ref = project_league(V).name
    num = {s: 0.0 for s in "ga"}
    den = 0.0
    for w, lag in zip(MARCEL_W, (1, 2, 3)):
        y = V - lag
        if y < P.seasons[0]:
            continue
        j = P.j(y)
        era = ptot(ref) / ptot(y)
        num["g"] = num["g"] + w * sum(P.x[f"g_{k}"][:, j] for k in SITS) * era
        num["a"] = num["a"] + w * sum(P.x[f"a1_{k}"][:, j] + P.x[f"a2_{k}"][:, j]
                                      for k in SITS) * era
        den = den + w * P.x["gp"][:, j]
    out = pd.DataFrame({"player_id": P.ids, "pos": P.pos})
    age = ages(P, V)
    af = np.where(age < 27, 1 + 0.01 * (27 - age), np.where(age > 29, 1 - 0.015 * (age - 29), 1.0))
    for s in "ga":
        lg = {p: num[s][P.pos == p].sum() / max(den[P.pos == p].sum(), 1) for p in "FD"}
        prior = np.where(P.pos == "D", lg["D"], lg["F"])
        out[f"{s}_pg"] = (num[s] + MARCEL_REG_GP * prior) / (den + MARCEL_REG_GP) * af
    gp = 0.0
    for lag, w in ((1, 0.5), (2, 0.1)):
        y = V - lag
        gp = gp + w * P.x["gp"][:, P.j(y)] * 82.0 / float(lv.loc[y, "games"])
    out["gp"] = np.minimum(gp + 25.0, 82.0)
    out["g"] = out.g_pg * out.gp
    out["a"] = out.a_pg * out.gp
    out["p"] = out.g + out.a
    out["gp1"] = P.x["gp"][:, P.j(V - 1)] * 82.0 / float(lv.loc[V - 1, "games"])
    out = out[np.asarray(den) > 0]
    return out[["player_id", "gp", "g", "a", "p", "gp1"]].reset_index(drop=True)


def blend_paths(tot: pd.DataFrame, V: int, prm: Params, games: int,
                w_marcel: float | None = None, w_last: float | None = None) -> pd.DataFrame:
    """Forecast averaging of season goals and assists (points follow):

      final = (1 - wm - wl) * HatTrick + wm * Marcel + wl * LastGP

    Marcel = the 5/4/3 baseline scaled to `games`; LastGP = HatTrick's
    per-game goals/assists x last season's games (per 82, scaled). Players
    a path cannot price (no history in the last three seasons / no games
    last season) keep the weights of the paths that can. Games, ice time
    and every other stat stay HatTrick's (they carry the conservation
    laws); interval columns of g/a/p are rescaled with their mean."""
    wm = prm.blend_marcel if w_marcel is None else w_marcel
    wl = prm.blend_lastgp if w_last is None else w_last
    if wm <= 0 and wl <= 0:
        return tot
    t = tot.copy()
    m = marcel_baseline(V).set_index("player_id")
    f = games / 82.0
    ids = t.player_id
    mg, ma = ids.map(m.g * f), ids.map(m.a * f)
    gp1 = ids.map(m.gp1).fillna(0.0) * f
    per = t.gp.clip(lower=1e-6)
    lg, la = t.g / per * gp1, t.a / per * gp1
    has_m, has_l = mg.notna(), gp1 > 0
    wm_i = np.where(has_m, wm, 0.0)
    wl_i = np.where(has_l, wl, 0.0)
    wh = 1.0 - wm_i - wl_i
    new_g = wh * t.g + wm_i * mg.fillna(0) + wl_i * lg
    new_a = wh * t.a + wm_i * ma.fillna(0) + wl_i * la
    rg = np.where(t.g > 1e-9, new_g / t.g.clip(lower=1e-9), 1.0)
    ra = np.where(t.a > 1e-9, new_a / t.a.clip(lower=1e-9), 1.0)
    rp = np.where(t.p > 1e-9, (new_g + new_a) / t.p.clip(lower=1e-9), 1.0)
    for s, r in (("g", rg), ("a", ra), ("p", rp)):
        for c in [c for c in t.columns if c.startswith(f"{s}_p") or c == f"{s}_sd"]:
            t[c] = t[c] * r
    for c, r in (("a1", ra), ("a2", ra), ("ppg", rg), ("ppa", ra)):
        if c in t:
            t[c] = t[c] * r
    t["g_hattrick"], t["a_hattrick"] = t.g, t.a
    t["g"], t["a"] = new_g, new_a
    t["p"] = t.g + t.a
    return t


# ---------------------------------------------------------------------------
# 2026-27 freeze
# ---------------------------------------------------------------------------
def roster_2027() -> pd.DataFrame:
    """Opening rosters (2026-09-29) plus injured players the clubs carried
    off the 23-man roster (non-roster / injured reserve): they return during
    the season, so they belong in the depth chart with games out.

    Sources for the additions: the raw availability table (MoneyPuck injured
    flag, DailyFaceoff IR statuses, NeurHL's researched overrides) and the
    injury research file. A player's team is the roster file's when listed,
    else the research file's (it records moves up to the cutoff, e.g.
    Merzlikins CBJ -> TOR on 9/28), else the availability table's.
    """
    from hattrick import deploy as DP
    r = D.rosters_2027()[["player_id", "team", "name", "grp", "birth", "pos_raw"]].copy()
    r["on_opening_roster"] = True
    av = D.availability_raw_2027()
    flagged = av[(av.injured.astype(str) == "True")
                 | av.df_status.isin(["ir:out", "ir:ir"])
                 | (av.override_games_out.fillna(0) > 0)]
    flagged = flagged[flagged.team.notna() & ~flagged.player_id.isin(r.player_id)]
    add = flagged[["player_id", "team", "name"]].drop_duplicates("player_id").copy()
    res = DP.research_rows()
    if len(res):
        res["nn"] = res.name.map(DP._norm_name)
        avn = av.dropna(subset=["name"]).assign(nn=lambda d: d.name.map(DP._norm_name))
        rr = res.merge(avn[["nn", "player_id"]].drop_duplicates("nn"), on="nn")
        rr = rr[~rr.player_id.isin(r.player_id)]
        tm = dict(zip(rr.player_id, rr.team))
        add = pd.concat([add, rr[["player_id", "team", "name"]]]).drop_duplicates("player_id")
        add["team"] = add.player_id.map(tm).fillna(add.team)
    # positions and births for the additions
    s = D.skater_seasons().sort_values("season_end").drop_duplicates("player_id", keep="last")
    pos = dict(zip(s.player_id, s.pos))
    g = D.goalie_seasons()
    gids = set(g.player_id)
    b = D.bios().drop_duplicates("player_id").set_index("player_id")
    add["grp"] = [("G" if p in gids else pos.get(p, "F")) for p in add.player_id]
    add["birth"] = add.player_id.map(b.birth)
    add["pos_raw"] = add.player_id.map(b.position)
    add["on_opening_roster"] = False
    out = pd.concat([r, add], ignore_index=True)
    out = out[out.team.isin(C.TEAMS_2027)]
    return out.reset_index(drop=True)


def run_freeze(V: int = C.TARGET_SEASON, sims: int = 2000) -> pd.DataFrame:
    """Write freeze_<V>/player_rates_<V>.csv and goalie_rates_<V>.csv."""
    from hattrick import deploy as DP
    from hattrick import goalies as GL
    assert V == C.TARGET_SEASON, "the freeze uses the 2026-27 rosters and injuries"
    games = C.GAMES_PER_TEAM.get(V, C.DEFAULT_GAMES)
    prm = load_params()
    out_dir = C.OUT / f"freeze_{V}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ros = roster_2027()
    sk = ros[ros.grp != "G"].copy()
    extra = sk.assign(pos=np.where(sk.grp == "D", "D", "F"))[["player_id", "pos", "birth", "name"]]
    go, rng_, src = DP.games_out_2027(sk, games, with_range=True)
    tot, proj = pipeline(V, prm, sk[["player_id", "team"]], games, "opening",
                         games_out=go, games_out_range=rng_, extra=extra, sims=sims)
    keep = [c for c in proj.columns if c not in tot.columns or c == "player_id"]
    tot = tot.merge(proj[keep], on="player_id")
    tot["games_out_source"] = tot.player_id.map(src)
    tot = tot.merge(ros[["player_id", "name", "on_opening_roster"]].rename(
        columns={"name": "roster_name"}), on="player_id", how="left")
    tot["name"] = tot.roster_name.fillna(tot.name)
    # last season (for the sanity print and the regression-slope check)
    P = panel()
    j = P.j(V - 1)
    last = pd.DataFrame({"player_id": P.ids, "gp_last": P.x["gp"][:, j],
                         "p_last": sum(P.x[f"{s}_{k}"][:, j] for s in ("g", "a1", "a2") for k in SITS)})
    tot = tot.merge(last, on="player_id", how="left")
    lead = ["player_id", "name", "team", "pos", "age", "on_opening_roster", "has_hist",
            "nhl_gp", "q", "games_out", "games_out_min", "games_out_max", "games_out_source",
            "score", "gp", "gp_p10", "gp_p90", "toi", "toi_p10", "toi_p90",
            *[f"toi_{k}" for k in SITS], *[f"tpg_{k}_n" for k in SITS],
            "g", "g_p10", "g_p90", "g_sd", "a", "a_p10", "a_p90", "a_sd",
            "p", "p_p10", "p_p50", "p_p90", "p_sd", "g_hattrick", "a_hattrick",
            "a1", "a2", "ppg", "ppa",
            "sog", "sog_p10", "sog_p90", "ixg", "hits", "blk", "pim", "fow", "fol",
            "tk", "gv", "fin", "rel_xgf60", "rel_xgf60_sd", "rel_xga60", "rel_xga60_sd",
            "pr_prosp", "gp_last", "p_last"]
    rest = [c for c in tot.columns if c.startswith(("r60_", "sd60_", "tpg_")) and c not in lead]
    tot = tot[lead + rest].sort_values("p", ascending=False)
    tot.round(4).to_csv(out_dir / f"player_rates_{V}.csv", index=False)
    gl = GL.run_2027(out_dir, games=games)
    export_params(V, prm)
    return tot, gl


def export_params(V: int, prm: Params) -> dict:
    """Write every constant fitted for season V next to the tuned knobs in
    hattrick/output/params/players.json (key 'fitted_<V>')."""
    from hattrick import deploy as DP
    pri = rate_priors(V, prm.prior_window, prm.eb_window)
    tgt = project_league(V)
    eb = {}
    for c in ("g_ev", "g_pp", "a1_ev", "a2_ev", "a1_pp", "a2_pp", "sog_ev", "ixg_ev",
              "hits_all", "blk_all", "pim_all", "fow_all", "tk_all", "gv_all"):
        for pos in ("F", "D"):
            e = pri[(c, pos)]
            # K at the positional mean usage, minutes of prior (before kappa)
            X = np.array([1.0, 15.5 if pos == "F" else 19.0, 1.5, 1.0])
            prr = float(max(X @ e["beta"], 1e-3))
            eb[f"{c}_{pos}"] = {"prior_beta_[1,ev,pp,sh]": [round(float(b), 4) for b in e["beta"]],
                                "tau2": e["tau2"], "phi": e["phi"],
                                "K_minutes_at_mean_usage": prm.kappa * e["phi"] * prr / (float(tgt[c]) * e["tau2"])}
    ages_ = np.arange(19, 39)
    aging = {f"{g}_{pos}": dict(zip(ages_.tolist(), np.round(age_multiplier(
        V, prm, g, np.array([pos] * len(ages_)), ages_ + 0.5), 3).tolist()))
        for g in (*AGE_GROUPS, "toi") for pos in "FD"}
    f = nhle_factors(V)
    top = {k: round(v, 3) for k, v in sorted(f.items(), key=lambda kv: -kv[1])
           if k in ("pooled", "AHL", "KHL", "SHL", "Liiga", "NL", "NCAA", "OHL", "WHL",
                    "QMJHL", "USHL", "Czechia", "DEL", "HockeyAllsvenskan", "VHL", "MHL")}
    pm = prospect_model(V)
    out = {
        "league_level_used": {"season": int(tgt.name),
                              **{k: round(float(tgt[k]) * 60, 4) for k in ("g_ev", "g_pp", "a1_ev", "a2_ev", "g_oth")},
                              **{k: round(float(tgt[k]), 3) for k in tgt.index if k.startswith("sk_min_")}},
        "eb_constants": eb,
        "finishing": finishing_prior(V, prm.eb_window),
        "aging_multipliers": aging,
        "nhle_factors_selected": top,
        "prospect_model": {"features": ["1", "log NHLe score", "has score", "log pick", "is D", "age-21"],
                           "points_ratio_coef": [round(float(b), 4) for b in pm["b_pr"]],
                           "log_toi_coef": [round(float(b), 4) for b in pm["b_toi"]], "n": pm["n"]},
        "fringe_tpg": {f"{p}_{k}": round(v, 3) for (p, k), v in usage_prior_tpg(V).items()},
        "health_prior": DP.health_prior(V),
        "coverage_opening": DP.coverage(V, "opening"),
        "gp_stack_opening_[1,struct,gp1,gp2]": [round(float(b), 4) for b in DP.gp_stack(V, prm, "opening")],
    }
    prev = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    prev[f"fitted_{V}"] = out
    D.write_json(prev, PARAMS_PATH)
    return out


def _sanity_print(tot: pd.DataFrame, gl: pd.DataFrame, V: int) -> None:
    print(f"\nTop 30 projected point scorers, {V} ({C.GAMES_PER_TEAM.get(V, 82)} games)")
    print(f"{'#':>3} {'name':<24}{'tm':>4}{'pos':>4}{'age':>5}{'GP':>6}{'TOI/g':>6}"
          f"{'G':>6}{'A':>6}{'P':>7}{'p10':>6}{'p90':>6}{'last P':>7}{'last GP':>8}")
    for i, r in enumerate(tot.head(30).itertuples(), 1):
        print(f"{i:>3} {str(r.name)[:23]:<24}{r.team:>4}{r.pos:>4}{r.age:>5.1f}{r.gp:>6.1f}"
              f"{r.toi / max(r.gp, 1e-9):>6.1f}{r.g:>6.1f}{r.a:>6.1f}{r.p:>7.1f}{r.p_p10:>6.0f}"
              f"{r.p_p90:>6.0f}{(r.p_last if r.p_last == r.p_last else float('nan')):>7.0f}"
              f"{(r.gp_last if r.gp_last == r.gp_last else float('nan')):>8.0f}")
    t = tot.groupby("team").agg(toi=("toi", "sum"), g=("g", "sum"), gp=("gp", "sum"),
                                ppg=("ppg", "sum"))
    lg = project_league(V)
    games = C.GAMES_PER_TEAM.get(V, 82)
    budget = games * sum(lg[f"sk_min_{p}_{k}"] for p in ("F", "D") for k in SITS)
    print(f"\nPer-team skater TOI (budget {budget:,.0f} min incl. call-ups) and goals")
    print(t.assign(toi_share=t.toi / budget, g_per_game=t.g / games).round(2).to_string())
    reg = tot[(tot.gp_last >= 60) & (tot.gp > 1)]
    b = np.polyfit(reg.p_last / reg.gp_last, reg.p / reg.gp, 1)[0]
    print(f"\nslope of projected P/GP on last season's P/GP (regulars, n={len(reg)}): {b:.3f}")
    print(f"goalies: {len(gl)} rows; top starts:")
    print(gl.sort_values("starts", ascending=False).head(12)[
        ["name", "team", "starts", "starts_p10", "starts_p90", "gsax60", "sv_pct"]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=C.TARGET_SEASON)
    ap.add_argument("--sims", type=int, default=2000)
    a = ap.parse_args()
    tot, gl = run_freeze(a.season, a.sims)
    _sanity_print(tot, gl, a.season)
    print("->", C.OUT / f"freeze_{a.season}")


if __name__ == "__main__":
    main()
