"""ORR team ratings: preseason prior, in-season Kalman filter, Elo baseline.

All ratings live on the log-rate scale of ``orr.gamemodel``: team offence
``o`` and defence ``d`` (positive d = allows more), so that expected
regulation goals are exp(mu + h + o_home + d_away + ...).

(a) PRESEASON  ``preseason(V)``
    Top-down: a pooled ridge/OLS regression, fitted on seasons < V only, of
    each team-season's realised strength (per-season Poisson GLM fixed
    effects, ``structural.glm_all_fe``) on its previous two seasons' goals,
    score-and-venue-adjusted xG and shots (MoneyPuck, log ratio to league).
    The regression coefficients ARE the lag weights and the regression to
    the mean; they are re-estimated for every target season (expanding
    window). Uncertainty: the regression's residual covariance minus the
    targets' own sampling noise ("true-talent" prior SD), per component.
    Hooks: a bottom-up roster component (``ROSTER_FILE``, enters as extra
    regressors when present) and a market prior (``market_strength``,
    ``blend_market``; the blend weight is left to the caller).

(b) IN-SEASON  ``run_filter`` (backtests) / ``InSeasonFilter`` (live)
    State per team: shot-rate offence/defence (so, sd) and finishing
    (fo, fd) with o = so + fo, d = sd + fd; plus the league goal level mu,
    home ice h, and their shot counterparts. An extended Kalman filter with
    Poisson pseudo-observations updates the state after every game DAY from
    regulation goals and (where known) shots on goal, with random-walk process
    noise per day and the preseason prior at every season start. Games on the
    same date are always predicted from the state at the end of the previous
    date. Starting goalies (when known) enter as offsets on goals against.

(c) ELO  ``elo_run``: MOV Elo with home ice and season carryover, tuned on
    2012-2017 like everything else.

Hyperparameters (``HP``) were tuned walk-forward on 2011-12, 2013-14 ..
2016-17 (see orr/backtest/tune_games.py and its ledger) and frozen in
orr/output/params/ratings_hp.json; nothing was tuned on 2017-18 or later.
"""
from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from orr import config as C
from orr import data as D
from orr import gamemodel as GM
from orr import structural as S

HP_FILE = C.PARAMS / "ratings_hp.json"
ROSTER_FILE = C.OUT / "backtest" / "team_components_hist.csv"
MARKET_HIST = C.PKG / "data_market_history.csv"     # tracked copy (provenance in teams.py)

# franchise slots (all codes that ever appear, normalised)
SLOTS = sorted(set(S.game_frame().home) | set(C.TEAMS_2027))
SLOT = {t: i for i, t in enumerate(SLOTS)}
NG = 4                                  # global states: mu_g, h_g, mu_s, h_s
NT = 4                                  # per-team: so, sd, fo, fd
NSTATE = NG + NT * len(SLOTS)


def _ix(team: str, comp: int) -> int:
    return NG + NT * SLOT[team] + comp


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HP:
    prior_scale: float = 1.0      # x preseason (true-talent) prior covariance
    q_s: float = 1.0e-5           # per-day random-walk variance, shot components
    q_f: float = 0.5e-5           # per-day random-walk variance, finishing
    phi_g: float = 1.0            # x Pearson dispersion of goals (measurement)
    phi_s: float = 1.0            # x Pearson dispersion of shots
    use_shots: bool = True
    mu_sd: float = 0.03           # prior SD of the season's league goal level
    h_sd_scale: float = 1.0       # x historical SD of home ice around its mean
    q_mu: float = 2e-6            # per-day variance of league levels
    integrate: bool = True        # average win prob over rating uncertainty
    ridge: float = 0.05           # preseason regression ridge (features are log ratios, SD ~0.1)
    goalie: tuple = (3000.0, -0.002, 1.0, 12.0)   # n0, m0, w_in, ref half-life
    gk_scale: float = 1.0         # x fitted goalie coefficient
    window: int = 8               # seasons of history for structural fits
    h_halflife: float = 3.0       # recency half-life (seasons) for home ice
    use_roster: bool = False      # add the bottom-up roster second stage
    pre_sp: bool = False          # preseason: separate net (o-d) / pace (o+d) regressions
    pace_prior: float = 1.0       # x prior variance along pace (o + d) directions
    pace_q: float = 1.0           # x process variance along pace directions

    def key(self):
        return tuple(sorted(asdict(self).items()))


def load_hp() -> HP:
    try:
        d = json.loads(HP_FILE.read_text())["hp"]
        d["goalie"] = tuple(d["goalie"])
        return HP(**d)
    except FileNotFoundError:
        return HP()


def load_filter_params() -> dict:
    """Everything InSeasonFilter needs, for the live CLI: hyperparameters and
    the 2027 structural parameters (dispersions, shot-model coefficients)."""
    hp = load_hp()
    return {"hp": hp, "struct": S.structural(C.TARGET_SEASON, hp.window, hp.h_halflife,
                                             goalie_key=hp.goalie),
            "P": GM.load_params()}


# ---------------------------------------------------------------------------
# (a) Preseason
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def team_features() -> pd.DataFrame:
    """Per team-season log ratios to league per-game means (MoneyPuck, all
    situations): x_gf, x_ga, x_xgf, x_xga (score/venue-adjusted xG), x_sf,
    x_sa, x_cf, x_ca. Seasons 2009..2026."""
    ts = D.team_seasons()
    ts = ts[ts.gp > 0].copy()
    out = ts[["team", "season_end", "gp"]].copy()
    for a, b in (("gf", "gf_all"), ("ga", "ga_all"), ("xgf", "axgf_all"),
                 ("xga", "axga_all"), ("sf", "sf_all"), ("sa", "sa_all"),
                 ("cf", "cf_all"), ("ca", "ca_all")):
        rate = ts[b] / ts.gp
        lg = ts.groupby("season_end")[b].transform("sum") / ts.groupby("season_end").gp.transform("sum")
        out[f"x_{a}"] = np.log(rate / lg)
    return out.reset_index(drop=True)


OFF_FEATS = ["gf", "xgf", "sf"]
DEF_FEATS = ["ga", "xga", "sa"]


def _design(V: int, teams: list[str]) -> pd.DataFrame:
    """Lag-1 and lag-2 features for the teams of season V. A missing lag 2
    (expansion, relocation) copies lag 1; a missing lag 1 (expansion team)
    is zero with the flag ``new`` set."""
    f = team_features().set_index(["team", "season_end"])
    rows = []
    for t in teams:
        r = {"team": t, "season_end": V}
        l1 = f.loc[(t, V - 1)] if (t, V - 1) in f.index else None
        l2 = f.loc[(t, V - 2)] if (t, V - 2) in f.index else None
        r["new"] = float(l1 is None)
        for k in OFF_FEATS + DEF_FEATS + ["cf", "ca"]:
            v1 = 0.0 if l1 is None else float(l1[f"x_{k}"])
            v2 = v1 if l2 is None else float(l2[f"x_{k}"])
            r[f"{k}1"], r[f"{k}2"] = v1, v2
        rows.append(r)
    return pd.DataFrame(rows)


def _teams_of(V: int) -> list[str]:
    if V == C.TARGET_SEASON:
        return list(C.TEAMS_2027)
    g = S.game_frame()
    g = g[g.season_end == V]
    return sorted(set(g.home) | set(g.away))


ROSTER_O = ["gf_pg_total", "ev_xgf_impact", "pp_xgf60"]
ROSTER_D = ["ev_xga_impact", "pk_xga60", "goalie_gsax60"]


def _roster_components() -> pd.DataFrame | None:
    """Bottom-up hook: orr/output/backtest/team_components_hist.csv from
    the player-model agent (per team-season V, summed projections of the
    players on the opening-roster proxy, built from seasons < V). Only the
    PROJECTED components are read (never the act_* outcome columns); each is
    centred on its season mean (gf_pg_total as a log ratio). Returns None
    when the file is absent."""
    if not ROSTER_FILE.exists():
        return None
    r = pd.read_csv(ROSTER_FILE)
    need = {"team", "season_end", *ROSTER_O, *ROSTER_D}
    if not need <= set(r.columns):
        return None
    r["team"] = D.norm_team(r.team)
    r["gf_pg_total"] = np.log(r.gf_pg_total)
    for c in ROSTER_O + ROSTER_D:
        r[c] = r[c] - r.groupby("season_end")[c].transform("mean")
    return r[["team", "season_end", *ROSTER_O, *ROSTER_D]]


@functools.lru_cache(maxsize=None)
def _roster_stage(V: int, ridge: float, lam_r: float = 1.0, min_seasons: int = 3):
    """Second-stage roster regression for season V, fitted on seasons
    2011 .. V-1: residual (realised FE - top-down preseason) on the centred
    roster components, separately for offence and defence, ridge lam_r on
    standardised features. None if fewer than ``min_seasons`` seasons."""
    rc = _roster_components()
    if rc is None:
        return None
    fe = S.glm_all_fe()
    rows = []
    for Vp in range(2011, V):
        if Vp not in set(rc.season_end):
            continue
        top = preseason_table(Vp, ridge, use_roster=False)
        x = top[["team", "o", "d"]].assign(season_end=Vp).merge(
            fe[["team", "season_end", "o_fe", "d_fe", "gp"]], on=["team", "season_end"]).merge(
            rc, on=["team", "season_end"])
        rows.append(x)
    if len(rows) < min_seasons:
        return None
    tr = pd.concat(rows, ignore_index=True)
    w = (tr.gp / 82.0).to_numpy()
    out = {}
    for side, cols in (("o", ROSTER_O), ("d", ROSTER_D)):
        X = tr[cols].to_numpy(float)
        sd = X.std(0) + 1e-9
        Z = X / sd
        y = (tr[f"{side}_fe"] - tr[side]).to_numpy()
        A = Z.T @ (Z * w[:, None]) + lam_r * len(Z) / 100.0 * np.eye(len(cols))
        b = np.linalg.solve(A, Z.T @ (w * y))
        out[side] = {"cols": cols, "beta": (b / sd).tolist(),
                     "resid_var_reduction": float(np.var(y) - np.var(y - Z @ b))}
    out["n"] = len(tr)
    return out


def _ridge(X, y, w, lam):
    lam = max(lam, 1e-4)             # keeps collinear early windows solvable
    Xc = np.column_stack([np.ones(len(X)), X])
    W = np.diag(w)
    A = Xc.T @ W @ Xc + lam * np.diag([0.0] + [1.0] * X.shape[1])
    return np.linalg.solve(A, Xc.T @ W @ y)


@functools.lru_cache(maxsize=None)
def _preseason_fit(V: int, ridge: float, use_roster: bool) -> dict:
    """Fit the top-down regressions on target seasons 2010 .. V-1."""
    fe = S.glm_all_fe()
    rows = []
    for Vp in range(2010, V):
        des = _design(Vp, _teams_of(Vp))
        rows.append(des.merge(fe[fe.season_end == Vp], on=["team", "season_end"]))
    if not rows:
        return {}
    tr = pd.concat(rows, ignore_index=True)
    w = (tr.gp / 82.0).to_numpy()
    out = {"n": len(tr), "seasons": [int(tr.season_end.min()), int(tr.season_end.max())]}
    # goals: pooled offence & defence, shared lag/feature weights
    xo = tr[[f"{k}{l}" for l in (1, 2) for k in OFF_FEATS]].to_numpy()
    xd = tr[[f"{k}{l}" for l in (1, 2) for k in DEF_FEATS]].to_numpy()
    Xo, Xd = xo, xd
    if use_roster == "sp":
        # net strength and pace regress at their own rates
        bs_ = _ridge(Xo - Xd, (tr.o_fe - tr.d_fe).to_numpy(), w, ridge)
        bp_ = _ridge(Xo + Xd, (tr.o_fe + tr.d_fe).to_numpy(), w, ridge)
        s_hat = np.column_stack([np.ones(len(Xo)), Xo - Xd]) @ bs_
        p_hat = np.column_stack([np.ones(len(Xo)), Xo + Xd]) @ bp_
        res_o = tr.o_fe.to_numpy() - (s_hat + p_hat) / 2
        res_d = tr.d_fe.to_numpy() - (p_hat - s_hat) / 2
        out["beta_net"], out["beta_pace"] = bs_.tolist(), bp_.tolist()
        b = None
    else:
        X = np.r_[Xo, Xd]
        y = np.r_[tr.o_fe, tr.d_fe]
        b = _ridge(X, y, np.r_[w, w], ridge)
        res_o = tr.o_fe.to_numpy() - np.column_stack([np.ones(len(Xo)), Xo]) @ b
        res_d = tr.d_fe.to_numpy() - np.column_stack([np.ones(len(Xd)), Xd]) @ b
    out["beta_goals"] = None if b is None else b.tolist()
    out["feat_goals"] = ["const"] + [f"{k}{l}" for l in (1, 2) for k in OFF_FEATS]
    # shots (seasons with shot targets)
    ts = tr[tr.so_fe.notna()]
    if len(ts) >= 30:
        so_X = ts[["sf1", "cf1", "sf2"]].to_numpy()
        sd_X = ts[["sa1", "ca1", "sa2"]].to_numpy()
        ws = (ts.gp / 82.0).to_numpy()
        if use_roster == "sp":
            b1 = _ridge(so_X - sd_X, (ts.so_fe - ts.sd_fe).to_numpy(), ws, ridge)
            b2 = _ridge(so_X + sd_X, (ts.so_fe + ts.sd_fe).to_numpy(), ws, ridge)
            sn = np.column_stack([np.ones(len(ts)), so_X - sd_X]) @ b1
            sp_ = np.column_stack([np.ones(len(ts)), so_X + sd_X]) @ b2
            r_so = ts.so_fe.to_numpy() - (sn + sp_) / 2
            r_sd = ts.sd_fe.to_numpy() - (sp_ - sn) / 2
            out["beta_shots_net"], out["beta_shots_pace"] = b1.tolist(), b2.tolist()
        else:
            bs = _ridge(np.r_[so_X, sd_X], np.r_[ts.so_fe, ts.sd_fe], np.r_[ws, ws], ridge)
            out["beta_shots"] = bs.tolist()
            r_so = ts.so_fe.to_numpy() - np.column_stack([np.ones(len(ts)), so_X]) @ bs
            r_sd = ts.sd_fe.to_numpy() - np.column_stack([np.ones(len(ts)), sd_X]) @ bs
        m = tr.so_fe.notna().to_numpy()
        r_fo = res_o[m] - r_so
        r_fd = res_d[m] - r_sd
        R = np.column_stack([r_so, r_sd, r_fo, r_fd])
        cov = np.cov(R.T, aweights=ws)
        noise = np.diag([np.mean(ts.so_fe_se ** 2), np.mean(ts.sd_fe_se ** 2),
                         np.mean(ts.o_fe_se ** 2), np.mean(ts.d_fe_se ** 2)])
        # goals noise also sits in fo/fd; the shot noise is small
        out["cov4"] = _psd(cov - noise, floor=1e-4).tolist()
    # goal-level covariance (o, d)
    cov2 = np.cov(np.column_stack([res_o, res_d]).T, aweights=w)
    noise2 = np.diag([np.mean(tr.o_fe_se ** 2), np.mean(tr.d_fe_se ** 2)])
    out["cov2"] = _psd(cov2 - noise2, floor=4e-4).tolist()
    out["resid_sd_goals"] = [float(np.sqrt(cov2[0, 0])), float(np.sqrt(cov2[1, 1]))]
    return out


def _psd(M, floor):
    M = (M + M.T) / 2
    w, v = np.linalg.eigh(M)
    return v @ np.diag(np.maximum(w, floor)) @ v.T


def preseason_table(V: int, ridge: float = 0.05, use_roster: bool = False,
                    teams: list[str] | None = None, sp: bool = False) -> pd.DataFrame:
    """Top-down preseason ratings for season V from seasons < V only.

    Columns: team, o, d, so, sd, fo, fd, o_sd, d_sd, od_cov (true-talent
    prior covariance of o and d), and cov4 (4x4 covariance of so, sd, fo, fd
    as a nested list) for the filter."""
    teams = teams or _teams_of(V)
    fit = _preseason_fit(V, ridge, "sp" if sp else False)
    des = _design(V, teams)
    if not fit:                       # before the regression has any data
        fe = S.glm_all_fe().set_index(["team", "season_end"])
        o = np.array([0.5 * fe.o_fe.get((t, V - 1), 0.0) for t in teams])
        d = np.array([0.5 * fe.d_fe.get((t, V - 1), 0.0) for t in teams])
        cov2 = np.diag([0.08 ** 2, 0.08 ** 2])
        so, sd = 0.6 * o, 0.6 * d
        cov4 = np.diag([0.05 ** 2, 0.05 ** 2, 0.06 ** 2, 0.06 ** 2])
    else:
        xo = des[[f"{k}{l}" for l in (1, 2) for k in OFF_FEATS]].to_numpy()
        xd = des[[f"{k}{l}" for l in (1, 2) for k in DEF_FEATS]].to_numpy()
        if fit.get("beta_goals") is not None:
            b = np.array(fit["beta_goals"])
            o = b[0] + xo @ b[1:]
            d = b[0] + xd @ b[1:]
        else:
            bn, bp = np.array(fit["beta_net"]), np.array(fit["beta_pace"])
            s_ = bn[0] + (xo - xd) @ bn[1:]
            p_ = bp[0] + (xo + xd) @ bp[1:]
            o, d = (s_ + p_) / 2, (p_ - s_) / 2
        # centre (ratings are relative to the league level mu)
        o, d = o - o.mean(), d - d.mean()
        cov2 = np.array(fit["cov2"])
        if "beta_shots" in fit or "beta_shots_net" in fit:
            fo_X = des[["sf1", "cf1", "sf2"]].to_numpy()
            fd_X = des[["sa1", "ca1", "sa2"]].to_numpy()
            if "beta_shots" in fit:
                bs = np.array(fit["beta_shots"])
                so = bs[0] + fo_X @ bs[1:]
                sd = bs[0] + fd_X @ bs[1:]
            else:
                b1, b2 = np.array(fit["beta_shots_net"]), np.array(fit["beta_shots_pace"])
                sn = b1[0] + (fo_X - fd_X) @ b1[1:]
                spc = b2[0] + (fo_X + fd_X) @ b2[1:]
                so, sd = (sn + spc) / 2, (spc - sn) / 2
            so, sd = so - so.mean(), sd - sd.mean()
            cov4 = np.array(fit["cov4"])
        else:
            so, sd = 0.6 * o, 0.6 * d
            cov4 = np.diag([0.05 ** 2, 0.05 ** 2, 0.06 ** 2, 0.06 ** 2])
    new = des.new.to_numpy() == 1
    roster_used = False
    if use_roster:
        rs = _roster_stage(V, ridge)
        rc = _roster_components()
        if rs is not None and V in set(rc.season_end):
            x = pd.DataFrame({"team": teams}).merge(rc[rc.season_end == V], on="team", how="left")
            for side in ("o", "d"):
                adj = x[rs[side]["cols"]].fillna(0.0).to_numpy() @ np.array(rs[side]["beta"])
                if side == "o":
                    o = o + adj
                else:
                    d = d + adj
            o, d = o - o.mean(), d - d.mean()
            roster_used = True
    out = pd.DataFrame({"team": teams, "o": o, "d": d, "so": so, "sd": sd,
                        "fo": o - so, "fd": d - sd})
    out.attrs["roster_used"] = roster_used
    # an expansion team: league-average mean, prior SD inflated by 1.5
    infl = np.where(new, 1.5, 1.0)
    out["o_sd"] = np.sqrt(cov2[0, 0]) * infl
    out["d_sd"] = np.sqrt(cov2[1, 1]) * infl
    out["od_cov"] = cov2[0, 1] * infl ** 2
    out["cov4"] = [(cov4 * f ** 2).tolist() for f in infl]
    out["new"] = new
    return out


def preseason(V: int, market: pd.DataFrame | None = None, market_weight: float = 0.0,
              P: dict | None = None) -> pd.DataFrame:
    """PUBLIC API for the season simulator: preseason ratings for season V.

    Returns DataFrame [team, o, d, o_sd, d_sd, od_cov] on the log-rate scale
    of gamemodel (o_sd/d_sd/od_cov: prior uncertainty of the TRUE season
    strength, from walk-forward residuals). With ``market`` (columns team,
    line [, games]) and ``market_weight`` w in [0, 1], the overall strength
    o - d is moved a fraction w toward the market-implied strength
    (``blend_market``); the uncertainty is then left unchanged (the caller
    owns the blend weight and any variance reduction)."""
    hp = load_hp()
    pre = preseason_table(V, hp.ridge, use_roster=hp.use_roster, sp=hp.pre_sp)
    if market is not None and market_weight > 0:
        P = P or GM.load_params()
        pre = blend_market(pre, market_strength(market, P), market_weight)
    return pre[["team", "o", "d", "o_sd", "d_sd", "od_cov"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Market hook
# ---------------------------------------------------------------------------
def market_strength(lines: pd.DataFrame, P: dict | None = None,
                    games: int | None = None, center: bool = True) -> pd.DataFrame:
    """Map preseason point totals to an overall log-rate strength s = o - d.

    lines: team, line (season points O/U), optional games (default 82, or
    ``games``). s solves games * E[points per game | s vs a league-average
    opponent, half at home] = line under the game model. With ``center`` the
    strengths are shifted to mean zero (market totals carry an inflated sum).
    Returns team, line, ppg, s_mkt."""
    P = P or GM.load_params()
    grid = np.linspace(-1.2, 1.2, 961)
    ppg = GM.expected_points_vs_average(grid, P)
    m = lines.copy()
    m["team"] = D.norm_team(m.team)
    n = m["games"] if "games" in m else (games or C.DEFAULT_GAMES)
    m["ppg"] = m.line / n
    m["s_mkt"] = np.interp(m.ppg, ppg, grid)
    if center:
        m["s_mkt"] = m.s_mkt - m.s_mkt.mean()
    return m[["team", "line", "ppg", "s_mkt"]]


def blend_market(pre: pd.DataFrame, mkt: pd.DataFrame, w: float) -> pd.DataFrame:
    """Move each team's strength s = o - d a fraction w toward the market's,
    split equally between offence (+) and defence (-)."""
    out = pre.merge(mkt[["team", "s_mkt"]], on="team", how="left")
    s_pre = out.o - out.d
    shift = w * (out.s_mkt.fillna(s_pre) - s_pre)
    out["o"] = out.o + shift / 2
    out["d"] = out.d - shift / 2
    for c in ("so", "sd"):
        if c in out:
            out[c] = out[c] + (shift / 2 if c == "so" else -shift / 2)
    return out.drop(columns="s_mkt")


def market_history() -> pd.DataFrame | None:
    """Historical preseason point totals (2019, 2020, 2022-2026; 82-game
    lines). Comparison / optional prior only: never used for tuning."""
    if not MARKET_HIST.exists():
        return None
    m = pd.read_csv(MARKET_HIST)
    m["team"] = D.norm_team(m.team.replace({"ARI": "UTA"}))
    return m[["season_end", "team", "line"]]


# ---------------------------------------------------------------------------
# (b) In-season Kalman filter
# ---------------------------------------------------------------------------
def _season_setup(V: int, hp: HP) -> dict:
    """Priors and structural parameters for season V (from seasons < V)."""
    st = S.structural(V, hp.window, hp.h_halflife, goalie_key=hp.goalie)
    pre = preseason_table(V, hp.ridge, use_roster=hp.use_roster, sp=hp.pre_sp)
    return {"st": st, "pre": pre}


# rotation of a team block [so, sd, fo, fd] into [net_s, pace_s, net_f, pace_f]
_ROT = np.array([[1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]]) / np.sqrt(2)


def _pace_scaled(C4: np.ndarray, scale: float) -> np.ndarray:
    """Scale a team block's covariance along the two pace directions
    (so + sd, fo + fd) by ``scale`` (net-strength directions unchanged)."""
    S_ = np.diag([1.0, np.sqrt(scale), 1.0, np.sqrt(scale)])
    M = _ROT.T @ S_ @ _ROT
    return M @ C4 @ M.T


def _prior(V: int, hp: HP, setup: dict):
    """Initial state mean and covariance for season V."""
    st, pre = setup["st"], setup["pre"]
    x = np.zeros(NSTATE)
    P = np.zeros((NSTATE, NSTATE))
    x[0], P[0, 0] = st["mu0"], hp.mu_sd ** 2
    x[1], P[1, 1] = st["h0"], (st["h_sd"] * hp.h_sd_scale) ** 2
    sh = st.get("shots")
    x[2], P[2, 2] = (sh["mu0"] if sh else 3.4), hp.mu_sd ** 2
    x[3], P[3, 3] = (sh["h0"] if sh else 0.05), (st["h_sd"] * hp.h_sd_scale) ** 2
    for r in pre.itertuples(index=False):
        i = NG + NT * SLOT[r.team]
        x[i:i + 4] = [r.so, r.sd, r.fo, r.fd]
        P[i:i + 4, i:i + 4] = _pace_scaled(np.array(r.cov4) * hp.prior_scale, hp.pace_prior)
    # teams not in the league this season keep a tiny variance
    for t in SLOTS:
        if t not in set(pre.team):
            i = NG + NT * SLOT[t]
            P[i:i + 4, i:i + 4] = np.eye(4) * 1e-6
    return x, P


def _proc(hp: HP) -> np.ndarray:
    """Per-day process-noise covariance (NSTATE x NSTATE, block diagonal)."""
    Q = np.zeros((NSTATE, NSTATE))
    Q[0, 0] = Q[2, 2] = hp.q_mu
    blk = _pace_scaled(np.diag([hp.q_s, hp.q_s, hp.q_f, hp.q_f]), hp.pace_q)
    for j in range(len(SLOTS)):
        i = NG + NT * j
        Q[i:i + 4, i:i + 4] = blk
    return Q


def _ctx(beta: dict, g: pd.DataFrame, side: str) -> np.ndarray:
    """Context log offset for the goals (or shots) of `side` ('h' or 'a')."""
    other = "a" if side == "h" else "h"
    return sum(beta.get(f"{f}_o", 0.0) * g[f"f_{f}_{side}"].to_numpy()
               + beta.get(f"{f}_d", 0.0) * g[f"f_{f}_{other}"].to_numpy()
               for f in GM.CTX_FEATURES)


GH_T, GH_W = np.polynomial.hermite_e.hermegauss(7)
GH_W = GH_W / GH_W.sum()


def run_filter(hp: HP, seasons: range | list, use_goalie: bool = False,
               first: int = 2008, pre_override: dict | None = None) -> pd.DataFrame:
    """Walk-forward filter over seasons ``first`` .. max(seasons).

    Returns one row per regular-season game of ``seasons`` (gid) with the
    pregame in-season linear predictors and their uncertainty
        eta_h, eta_a, v_h, v_a, c_ha
    and the preseason-frozen counterparts (no in-season updates)
        feta_h, feta_a, fv_h, fv_a, fc_ha
    Only the state at the end of the previous game day is ever read.
    With ``use_goalie`` the known starters' offsets enter both the pregame
    prediction and the update (seasons with starters), and the rest/travel
    coefficients of the goalie-aware GLM are used for those games.
    ``pre_override``: {V: (ratings, mu)} replaces season V's starting team
    strengths (team, o, d, o_sd, d_sd; e.g. the freeze pipeline's market-
    anchored ratings) and league level, exactly as ``InSeasonFilter`` does live.
    """
    g = S.game_frame()
    gt = S.goalie_game_talent(*hp.goalie) if use_goalie else None
    seasons = sorted(seasons)
    last = max(seasons)
    Q = _proc(hp)
    rows = []
    for V in range(first, last + 1):
        gv = g[g.season_end == V].reset_index(drop=True)
        setup = _season_setup(V, hp)
        st = setup["st"]
        if pre_override and V in pre_override:
            r_V, mu_V = pre_override[V]
            setup["pre"] = split_prior(V, hp, r_V)
        x, P = _prior(V, hp, setup)
        if pre_override and V in pre_override:
            x[0] = mu_V
        x0, P0 = x.copy(), P.copy()
        # per-game offsets
        cg_h, cg_a = _ctx(st["ctx"], gv, "h"), _ctx(st["ctx"], gv, "a")
        if use_goalie and st.get("ctx_gk") is not None:
            gm = gv[["gid"]].merge(gt, on="gid", how="left")
            known = (gm.gdiff_h.notna() & gm.gdiff_a.notna()).to_numpy()
            b = st["beta_gk"] * hp.gk_scale
            kh = _ctx(st["ctx_gk"], gv, "h") + b * np.nan_to_num(gm.gdiff_a.to_numpy())
            ka = _ctx(st["ctx_gk"], gv, "a") + b * np.nan_to_num(gm.gdiff_h.to_numpy())
            cg_h = np.where(known, kh, cg_h)
            cg_a = np.where(known, ka, cg_a)
        sh = st.get("shots") if hp.use_shots else None
        if sh is not None:
            cs_h = _ctx(sh["beta"], gv, "h") + sh["beta"]["margin"] * np.clip(gv.margin, -3, 3) \
                + sh["beta"]["extra"] * gv.went_extra
            cs_a = _ctx(sh["beta"], gv, "a") + sh["beta"]["margin"] * np.clip(-gv.margin, -3, 3) \
                + sh["beta"]["extra"] * gv.went_extra
        hi = gv.home.map(SLOT).to_numpy()
        ai = gv.away.map(SLOT).to_numpy()
        disp_g = st["disp_g"] * hp.phi_g
        disp_s = (sh["dispersion"] * hp.phi_s) if sh is not None else None
        prev = None
        d0 = gv.date.iloc[0]
        for d, idx in gv.groupby("date").indices.items():
            if prev is not None:
                P += Q * (d - prev).days
            prev = d
            k = len(idx)
            h_, a_ = hi[idx], ai[idx]
            # goal rows: home goals, away goals
            Hg = np.zeros((2 * k, NSTATE))
            r = np.arange(k)
            Hg[r, 0] = 1
            Hg[r, 1] = 1
            Hg[r, NG + NT * h_] = 1          # so home
            Hg[r, NG + NT * h_ + 2] = 1      # fo home
            Hg[r, NG + NT * a_ + 1] = 1      # sd away
            Hg[r, NG + NT * a_ + 3] = 1      # fd away
            Hg[k + r, 0] = 1
            Hg[k + r, NG + NT * a_] = 1
            Hg[k + r, NG + NT * a_ + 2] = 1
            Hg[k + r, NG + NT * h_ + 1] = 1
            Hg[k + r, NG + NT * h_ + 3] = 1
            cg = np.r_[cg_h[idx], cg_a[idx]]
            eta = Hg @ x + cg
            HP_ = Hg @ P
            V2 = np.einsum("ij,ij->i", HP_, Hg)
            cha = np.einsum("ij,ij->i", HP_[:k], Hg[k:])
            # frozen: prior mean + prior covariance grown by process noise
            days = (d - d0).days
            feta = Hg @ x0 + cg
            HP0 = Hg @ P0
            HQ = Hg @ Q
            fV = np.einsum("ij,ij->i", HP0 + HQ * days, Hg)
            fc = np.einsum("ij,ij->i", HP0[:k] + HQ[:k] * days, Hg[k:])
            rows.append(np.column_stack([
                gv.gid.to_numpy()[idx], eta[:k], eta[k:], V2[:k], V2[k:], cha,
                feta[:k], feta[k:], fV[:k], fV[k:], fc]))
            # ---- measurement update
            y = np.r_[gv.reg_h.to_numpy()[idx], gv.reg_a.to_numpy()[idx]].astype(float)
            m = np.exp(eta)
            H, z, R = [Hg], [(y - m) / m], [disp_g / m]
            if sh is not None:
                sy = np.r_[gv.sh_h.to_numpy()[idx], gv.sh_a.to_numpy()[idx]]
                ok = ~np.isnan(sy)
                if ok.any():
                    Hs = np.zeros((2 * k, NSTATE))
                    Hs[r, 2] = 1
                    Hs[r, 3] = 1
                    Hs[r, NG + NT * h_] = 1
                    Hs[r, NG + NT * a_ + 1] = 1
                    Hs[k + r, 2] = 1
                    Hs[k + r, NG + NT * a_] = 1
                    Hs[k + r, NG + NT * h_ + 1] = 1
                    es = Hs @ x + np.r_[cs_h[idx], cs_a[idx]]
                    ms = np.exp(es)
                    H.append(Hs[ok])
                    z.append(((sy - ms) / ms)[ok])
                    R.append((disp_s / ms)[ok])
            H = np.vstack(H)
            z = np.concatenate(z)
            R = np.concatenate(R)
            PHt = P @ H.T
            Sm = H @ PHt
            Sm[np.diag_indices(len(z))] += R
            K = np.linalg.solve(Sm, PHt.T).T
            x = x + K @ z
            P = P - K @ PHt.T
            P = (P + P.T) / 2
    out = pd.DataFrame(np.vstack(rows), columns=[
        "gid", "eta_h", "eta_a", "v_h", "v_a", "c_ha",
        "feta_h", "feta_a", "fv_h", "fv_a", "fc_ha"])
    out["gid"] = out.gid.astype(int)
    out = out.merge(g[["gid", "season_end"]], on="gid")
    return out          # all seasons first .. max(seasons) (earlier ones feed OT fits)


def fit_ot_walkforward(pred: pd.DataFrame, V: int, lookback: int = 6) -> dict:
    """OT/SO parameters for season V from the filter's pregame expected goals
    in regulation-tied games of earlier seasons (same overtime era when at
    least one earlier season of that era exists)."""
    g = S.game_frame()
    m = pred.merge(g[["gid", "season_end", "extra", "home_win"]], on=["gid", "season_end"])
    era3 = V >= S.THREE_ON_THREE
    lo = V - lookback
    if era3 and V - 1 >= S.THREE_ON_THREE:
        lo = max(lo, S.THREE_ON_THREE)
    m = m[(m.season_end >= lo) & (m.season_end < V) & (m.extra != "REG")]
    if len(m) < 100:
        return dict(GM.DEFAULT["ot"])
    w = 0.5 ** ((V - 1 - m.season_end.to_numpy()) / 3.0)
    return GM.fit_ot(np.exp(m.eta_h), np.exp(m.eta_a), m.extra.to_numpy(),
                     m.home_win.to_numpy(), w)


def predict_probs(pred: pd.DataFrame, hp: HP, frozen: bool = False,
                  ot_params: dict | None = None) -> pd.DataFrame:
    """Outcome probabilities for the rows of ``run_filter`` output, using for
    each season V the end-game layer of structural(V) and OT/SO parameters
    fitted walk-forward on the same run (seasons < V)."""
    pre = "f" if frozen else ""
    outs = []
    if ot_params is None:
        ot_params = {}
    for V, pv in pred.groupby("season_end"):
        st = S.structural(V, hp.window, hp.h_halflife, goalie_key=hp.goalie)
        ot = ot_params.get(V) or fit_ot_walkforward(pred, V)
        ot_params[V] = ot
        P = {"layer": st["layer"], "kappa": st["kappa"], "ot": ot}
        eh, ea = pv[f"{pre}eta_h"].to_numpy(), pv[f"{pre}eta_a"].to_numpy()
        vd = np.maximum(pv[f"{pre}v_h"] + pv[f"{pre}v_a"] - 2 * pv[f"{pre}c_ha"], 0).to_numpy()
        if hp.integrate:
            acc = None
            for t, w in zip(GH_T, GH_W):
                u = t * np.sqrt(vd)
                p = GM.outcome_probs(np.exp(eh + u / 2), np.exp(ea - u / 2), P)
                acc = {k: w * v for k, v in p.items()} if acc is None else \
                    {k: acc[k] + w * v for k, v in p.items()}
            p = acc
            p["e_reg_h"], p["e_reg_a"] = np.exp(eh), np.exp(ea)
        else:
            p = GM.outcome_probs(np.exp(eh), np.exp(ea), P)
        o = pd.DataFrame({k: p[k] for k in ("p_home_win", "p_home_reg", "p_tie",
                                            "p_away_reg", "p_so", "e_reg_h", "e_reg_a",
                                            "e_pts_h", "e_pts_a")})
        o["gid"] = pv.gid.to_numpy()
        o["season_end"] = V
        o["lam_h"], o["lam_a"] = np.exp(eh), np.exp(ea)
        o["v_diff"] = vd
        outs.append(o)
    return pd.concat(outs, ignore_index=True)


# ---------------------------------------------------------------------------
# Live in-season API
# ---------------------------------------------------------------------------
def split_prior(V: int, hp: HP, pre: pd.DataFrame) -> pd.DataFrame:
    """An o/d prior (team, o, d, o_sd, d_sd) as the filter's four-component
    state: o (d) split between shot-rate and finishing parts in the
    proportion of season V's walk-forward prior covariance, which is rescaled
    to the given o_sd/d_sd. Tables that already carry so/sd/fo/fd/cov4 pass
    through."""
    need = {"so", "sd", "fo", "fd", "cov4"}
    if need <= set(pre.columns):
        return pre
    full = preseason_table(V, hp.ridge, teams=list(pre.team), sp=hp.pre_sp)
    c4 = np.array(full.cov4.iloc[0])
    out = pre.copy()
    # split o (d) between shot and finishing parts in the prior's proportion
    vso, vfo = c4[0, 0] + c4[0, 2], c4[2, 2] + c4[0, 2]
    vsd, vfd = c4[1, 1] + c4[1, 3], c4[3, 3] + c4[1, 3]
    so_share = vso / (vso + vfo)
    sd_share = vsd / (vsd + vfd)
    out["so"], out["fo"] = out.o * so_share, out.o * (1 - so_share)
    out["sd"], out["fd"] = out.d * sd_share, out.d * (1 - sd_share)
    scale_o = (out.o_sd ** 2) / max(c4[0, 0] + c4[2, 2] + 2 * c4[0, 2], 1e-9)
    scale_d = (out.d_sd ** 2) / max(c4[1, 1] + c4[3, 3] + 2 * c4[1, 3], 1e-9)
    s = np.sqrt((scale_o + scale_d) / 2)
    out["cov4"] = [(c4 * v ** 2).tolist() for v in s]
    return out


class InSeasonFilter:
    """Game-by-game in-season updating from a preseason ratings table.

        filt = InSeasonFilter(pre, load_filter_params())
        filt.update(row)        # row: date, home, away, home_g, away_g, extra
                                #      [, reg_h, reg_a, sh_h, sh_a, gadj_h, gadj_a]
        filt.state()            # team, o, d, o_sd, d_sd, od_cov

    ``pre``: a table like ``preseason(V)`` (team, o, d, o_sd, d_sd[, od_cov]);
    the richer ``preseason_table`` (with so/sd/fo/fd and cov4) is used when
    given, otherwise the o/d prior is split between shot-rate and finishing
    components by the walk-forward covariance of season 2027.
    ``gadj_h`` / ``gadj_a``: known starters' offsets (gamemodel.goalie_offset
    of the starter's talent minus the team's usual mix) -- the home starter's
    offset multiplies AWAY goals, as in the backtests.
    ``state()`` folds the filter's league-level drift into o and d (half each)
    relative to params['P']['mu'], so gamemodel.rates(P, o_h, d_h, o_a, d_a)
    reproduces the filter's expected goals; ``levels()`` gives mu and h.
    Games of the same day should be passed with ``update_day`` (or in any
    order through ``update``; the difference is negligible).
    """

    def __init__(self, pre: pd.DataFrame, params: dict | None = None,
                 season_end: int = C.TARGET_SEASON, start_date=None):
        params = params or load_filter_params()
        self.hp: HP = params["hp"]
        self.st = params["struct"]
        self.P = params["P"]
        self.V = season_end
        setup = {"st": self.st, "pre": self._full_prior(pre)}
        self.x, self.C = _prior(season_end, self.hp, setup)
        self.x[0], self.x[1] = self.P["mu"], self.P["h"]
        self.Q = _proc(self.hp)
        self.last = pd.Timestamp(start_date) if start_date is not None else None
        self.teams = list(setup["pre"].team)
        self.n_games = 0

    def _full_prior(self, pre: pd.DataFrame) -> pd.DataFrame:
        return split_prior(self.V, self.hp, pre)


    def _advance(self, date):
        date = pd.Timestamp(date)
        if self.last is not None and date > self.last:
            self.C += self.Q * (date - self.last).days
        self.last = date if self.last is None else max(self.last, date)

    def _rows(self, games: pd.DataFrame):
        k = len(games)
        h_ = games.home.map(SLOT).to_numpy()
        a_ = games.away.map(SLOT).to_numpy()
        r = np.arange(k)
        Hg = np.zeros((2 * k, NSTATE))
        Hg[r, 0] = Hg[r, 1] = 1
        Hg[r, NG + NT * h_] = Hg[r, NG + NT * h_ + 2] = 1
        Hg[r, NG + NT * a_ + 1] = Hg[r, NG + NT * a_ + 3] = 1
        Hg[k + r, 0] = 1
        Hg[k + r, NG + NT * a_] = Hg[k + r, NG + NT * a_ + 2] = 1
        Hg[k + r, NG + NT * h_ + 1] = Hg[k + r, NG + NT * h_ + 3] = 1
        Hs = np.zeros((2 * k, NSTATE))
        Hs[r, 2] = Hs[r, 3] = 1
        Hs[r, NG + NT * h_] = Hs[r, NG + NT * a_ + 1] = 1
        Hs[k + r, 2] = 1
        Hs[k + r, NG + NT * a_] = Hs[k + r, NG + NT * h_ + 1] = 1
        return Hg, Hs

    def _prep(self, games: pd.DataFrame) -> pd.DataFrame:
        g = games.copy()
        if "reg_h" not in g:
            ex = g.get("extra", pd.Series("REG", index=g.index)).fillna("REG")
            lo = np.minimum(g.home_g, g.away_g)
            g["reg_h"] = np.where(ex == "REG", g.home_g, lo)
            g["reg_a"] = np.where(ex == "REG", g.away_g, lo)
        if not {"rest_h", "rest_a"} <= set(g.columns):
            for c in ("rest_h", "rest_a", "km_h", "km_a", "dtz_h", "dtz_a"):
                g[c] = 2.0 if c.startswith("rest") else 0.0
        fh = GM.ctx_features(g.rest_h, g.km_h, g.dtz_h)
        fa = GM.ctx_features(g.rest_a, g.km_a, g.dtz_a)
        for f in GM.CTX_FEATURES:
            g[f"f_{f}_h"], g[f"f_{f}_a"] = fh[f], fa[f]
        g["margin"] = g.reg_h - g.reg_a
        g["went_extra"] = (g.get("extra", "REG") != "REG").astype(float)
        for c in ("gadj_h", "gadj_a", "sh_h", "sh_a"):
            if c not in g:
                g[c] = 0.0 if c.startswith("gadj") else np.nan
        return g

    def update(self, row) -> None:
        """Update with ONE finished game (namedtuple, Series or dict)."""
        if not isinstance(row, dict):
            row = row._asdict() if hasattr(row, "_asdict") else dict(row)
        self.update_day(pd.DataFrame([row]))

    def update_day(self, games: pd.DataFrame) -> None:
        """Update with all finished games of one date."""
        g = self._prep(games.reset_index(drop=True))
        self._advance(g.date.max() if "date" in g else self.last)
        k = len(g)
        Hg, Hs = self._rows(g)
        cg = np.r_[_ctx(self.st["ctx"], g, "h") + g.gadj_a.to_numpy(),
                   _ctx(self.st["ctx"], g, "a") + g.gadj_h.to_numpy()]
        eta = Hg @ self.x + cg
        m = np.exp(eta)
        y = np.r_[g.reg_h.to_numpy(float), g.reg_a.to_numpy(float)]
        H, z, R = [Hg], [(y - m) / m], [self.st["disp_g"] * self.hp.phi_g / m]
        sh = self.st.get("shots") if self.hp.use_shots else None
        sy = np.r_[g.sh_h.to_numpy(float), g.sh_a.to_numpy(float)]
        ok = ~np.isnan(sy)
        if sh is not None and ok.any():
            b = sh["beta"]
            cs = np.r_[_ctx(b, g, "h") + b["margin"] * np.clip(g.margin, -3, 3) + b["extra"] * g.went_extra,
                       _ctx(b, g, "a") + b["margin"] * np.clip(-g.margin, -3, 3) + b["extra"] * g.went_extra]
            ms = np.exp(Hs @ self.x + cs)
            H.append(Hs[ok])
            z.append(((sy - ms) / ms)[ok])
            R.append((sh["dispersion"] * self.hp.phi_s / ms)[ok])
        H, z, R = np.vstack(H), np.concatenate(z), np.concatenate(R)
        PHt = self.C @ H.T
        Sm = H @ PHt
        Sm[np.diag_indices(len(z))] += R
        K = np.linalg.solve(Sm, PHt.T).T
        self.x = self.x + K @ z
        self.C = self.C - K @ PHt.T
        self.C = (self.C + self.C.T) / 2
        self.n_games += k

    def levels(self) -> dict:
        return {"mu": float(self.x[0]), "h": float(self.x[1]),
                "mu_sd": float(np.sqrt(self.C[0, 0])), "h_sd": float(np.sqrt(self.C[1, 1]))}

    def state(self, fold_levels: bool = True) -> pd.DataFrame:
        """team, o, d, o_sd, d_sd, od_cov (o = so + fo, d = sd + fd)."""
        drift = (self.x[0] - self.P["mu"]) / 2 if fold_levels else 0.0
        rows = []
        for t in self.teams:
            i = NG + NT * SLOT[t]
            a_o = np.zeros(NSTATE)
            a_o[[i, i + 2]] = 1
            a_d = np.zeros(NSTATE)
            a_d[[i + 1, i + 3]] = 1
            vo = a_o @ self.C @ a_o
            vd = a_d @ self.C @ a_d
            rows.append((t, self.x[i] + self.x[i + 2] + drift,
                         self.x[i + 1] + self.x[i + 3] + drift,
                         np.sqrt(vo), np.sqrt(vd), a_o @ self.C @ a_d))
        return pd.DataFrame(rows, columns=["team", "o", "d", "o_sd", "d_sd", "od_cov"])


# ---------------------------------------------------------------------------
# (c) Elo baseline
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EloHP:
    K: float = 8.0
    H: float = 30.0
    carry: float = 0.7
    mov: bool = True
    playoffs: bool = True
    init_new: float = 1470.0


def elo_run(hp: EloHP = EloHP()) -> pd.DataFrame:
    """MOV Elo over every game 2005-06 .. 2025-26 (playoffs optional; the
    2020 bubble playoffs neutral). Returns per regular-season game (gid):
    p_elo (in-season, pregame) and p_elo_frozen (ratings as of the season's
    first game, i.e. after carryover, no in-season updates)."""
    g = D.games().sort_values(["date", "home"]).reset_index(drop=True)
    if not hp.playoffs:
        g = g[g.game_type == "R"].reset_index(drop=True)
    tab = S.game_frame()[["gid", "date", "home", "away"]]
    r: dict = {}
    frozen: dict = {}
    cur = None
    out = []
    MEAN = 1505.0
    for gm in g.itertuples(index=False):
        s = gm.season_end
        if cur is not None and s != cur:
            for t in r:
                r[t] = MEAN + hp.carry * (r[t] - MEAN)
            frozen = {}
        if s != cur:
            cur = s
        for t in (gm.home, gm.away):
            if t not in r:
                r[t] = MEAN if s <= 2006 else hp.init_new
        if not frozen:
            frozen = dict(r)
        for t in (gm.home, gm.away):
            frozen.setdefault(t, r[t])
        rh, ra = r[gm.home], r[gm.away]
        neutral = s == 2020 and gm.game_type == "P"
        dd = rh + (0 if neutral else hp.H) - ra
        e = 1 / (1 + 10 ** (-dd / 400))
        if gm.game_type == "R":
            fd = frozen[gm.home] + hp.H - frozen[gm.away]
            out.append((gm.date, gm.home, gm.away, e, 1 / (1 + 10 ** (-fd / 400))))
        hw = gm.home_g > gm.away_g
        gd = abs(gm.home_g - gm.away_g)
        wd = dd if hw else -dd
        mult = np.log(gd + 1) * (2.2 / (2.2 + 0.001 * wd)) if hp.mov else 1.0
        delta = hp.K * mult * (float(hw) - e)
        r[gm.home], r[gm.away] = rh + delta, ra - delta
    o = pd.DataFrame(out, columns=["date", "home", "away", "p_elo", "p_elo_frozen"])
    return tab.merge(o, on=["date", "home", "away"], how="left")[
        ["gid", "p_elo", "p_elo_frozen"]]


# ---------------------------------------------------------------------------
# Production parameters
# ---------------------------------------------------------------------------
def fit_gamemodel_params(V: int = C.TARGET_SEASON, hp: HP | None = None,
                         write: bool = True) -> dict:
    """Fit every data-estimated game-model parameter for season V from
    seasons < V with the frozen hyperparameters, and write
    orr/output/params/gamemodel.json (V = 2027 for the freeze)."""
    from datetime import datetime, timezone
    hp = hp or load_hp()
    st = S.structural(V, hp.window, hp.h_halflife, goalie_key=hp.goalie)
    pred = run_filter(hp, [V - 1])
    ot = fit_ot_walkforward(pred, V)
    _, sv = S.goalie_prior(V)
    coef = (st["beta_gk"] * hp.gk_scale) if st["beta_gk"] is not None else GM.DEFAULT["goalie"]["coef"]
    P = {"season_end": V,
         "mu": st["mu0"], "h": st["h0"], "h_sd": st["h_sd"],
         "kappa": st["kappa"], "layer": st["layer"], "ot": ot,
         "ctx": st["ctx"], "ctx_gk": st["ctx_gk"],
         "goalie": {"coef": coef, "sv_lg": sv,
                    **{k: v for k, v in S.backup_usage(V - 1, goalie_key=hp.goalie).items()
                       if k != "seasons"}},
         "disp_g": st["disp_g"],
         "meta": {"fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "seasons_used": [int(V - hp.window), int(V - 1)],
                  "hp": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(hp).items()},
                  "notes": "mu, h: log scale (expected regulation goals exp(mu+h+o+d)); "
                           "ctx: no-goalie rest/travel coefficients (use when starters are "
                           "unknown); ctx_gk: the same when both starters are known, to be "
                           "combined with goalie_offset; goalie.coef multiplies the starter's "
                           "save talent above the team's usual starter (goals saved per shot)."}}
    if write:
        GM.save_params(P)
    return P


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


if __name__ == "__main__":
    P = fit_gamemodel_params()
    print(json.dumps({k: v for k, v in P.items() if k != "meta"}, indent=1, default=float))
    pre = preseason(C.TARGET_SEASON)
    pre["s"] = pre.o - pre.d
    print(pre.sort_values("s", ascending=False).round(4).to_string(index=False))
