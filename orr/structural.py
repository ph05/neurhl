"""Structural (data-estimated, not tuned) pieces of the ORR game model.

Everything here is a function of a target season V and reads only seasons
< V (and, for goalie talent, only games dated before each game), so it can be
refit walk-forward for every backtest season and, with V = 2027, for the
production forecast.

  game_frame()             the game table plus per-team context features
  goalie_game_talent(...)  walk-forward starting-goalie save talent vs the
                           team's usual starter, per game
  fit_glm(...)             Poisson GLM with team-season fixed effects:
                           home ice by season, league level by season, rest /
                           travel / goalie coefficients (goals or shots)
  structural(V, ...)       all of the above for target season V, cached
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from orr import config as C
from orr import data as D
from orr import gamemodel as GM
from orr import gametable as T

FIRST_SEASON = 2006
SHOT_SEASONS = range(2011, 2027)
THREE_ON_THREE = 2016          # first season with 3-on-3 overtime


# ---------------------------------------------------------------------------
# Game frame
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def game_frame() -> pd.DataFrame:
    """gametable plus context features f_<feat>_h / f_<feat>_a, date index."""
    g = T.table().copy()
    fh = GM.ctx_features(g.rest_h, g.km_h, g.dtz_h)
    fa = GM.ctx_features(g.rest_a, g.km_a, g.dtz_a)
    for f in GM.CTX_FEATURES:
        g[f"f_{f}_h"] = fh[f]
        g[f"f_{f}_a"] = fa[f]
    g["margin"] = g.reg_h - g.reg_a
    g["went_extra"] = (g.extra != "REG").astype(float)
    return g


# ---------------------------------------------------------------------------
# Goalie talent (walk-forward)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _goalie_seasons_adj() -> pd.DataFrame:
    """MoneyPuck goalie seasons with GSAx normalised by the league-season
    goals/xG ratio (so league GSAx sums to 0 each season)."""
    gs = D.goalie_seasons()[["player_id", "season_end", "sa_all", "ga_all", "xga_all"]].copy()
    ratio = gs.groupby("season_end").apply(
        lambda d: d.ga_all.sum() / d.xga_all.sum(), include_groups=False)
    gs["gsax"] = gs.xga_all * gs.season_end.map(ratio) - gs.ga_all
    gs = gs.groupby(["player_id", "season_end"], as_index=False)[["sa_all", "gsax"]].sum()
    lg_sv = D.goalie_seasons().groupby("season_end").apply(
        lambda d: 1 - d.ga_all.sum() / d.sa_all.sum(), include_groups=False)
    return gs, lg_sv


def goalie_prior(V: int, n0: float = 3000.0, m0: float = -0.002,
                 weights=(3.0, 2.0, 1.0)) -> tuple[dict, float]:
    """Preseason save talent (goals saved above average per shot) for every
    goalie with MoneyPuck history before season V: shot-weighted Marcel over
    V-1, V-2, V-3, regressed toward m0 with n0 pseudo-shots.
    Returns ({player_id: (numerator, denominator)}, league sv of V-1)."""
    gs, lg_sv = _goalie_seasons_adj()
    num: dict = {}
    den: dict = {}
    for k, w in enumerate(weights, start=1):
        x = gs[gs.season_end == V - k]
        for pid, sa, gx in zip(x.player_id, x.sa_all, x.gsax):
            num[pid] = num.get(pid, 0.0) + w * gx
            den[pid] = den.get(pid, 0.0) + w * sa
    out = {pid: (num[pid] + n0 * m0, den[pid] + n0) for pid in num}
    sv = float(lg_sv.get(V - 1, lg_sv.iloc[-1]))
    return out, sv


@functools.lru_cache(maxsize=None)
def goalie_game_talent(n0: float = 3000.0, m0: float = -0.002, w_in: float = 1.0,
                       halflife: float = 12.0) -> pd.DataFrame:
    """Per game (gid): starters' talent and talent MINUS the team's usual
    starter (an exponentially weighted mean of its recent starters' talents,
    half-life ``halflife`` team games), all computed from information dated
    strictly before the game. Unknown starter -> NaN.

    In-season evidence: (saves - league_sv * shots) over the goalie's earlier
    games this season (all appearances, starter or relief), weight ``w_in``.
    """
    g = game_frame()
    log = T.goalie_game_log()
    ids = g[["gid", "game_id", "date", "season_end", "home", "away", "gk_h", "gk_a"]]
    log = log.merge(ids[["game_id", "gid", "date", "season_end"]], on="game_id")
    log = log.sort_values("date")
    by_date = {d: x for d, x in log.groupby("date")}
    out = np.full((len(g), 4), np.nan)          # t_h, t_a, diff_h, diff_a
    ref: dict = {}
    alpha = 1 - 0.5 ** (1.0 / halflife)
    for V, gv in ids[ids.gk_h.notna() | ids.gk_a.notna()].groupby("season_end"):
        prior, sv = goalie_prior(V, n0, m0)
        ins_num: dict = {}
        ins_den: dict = {}

        def talent(pid):
            a, b = prior.get(pid, (n0 * m0, n0))
            return (a + w_in * ins_num.get(pid, 0.0)) / (b + w_in * ins_den.get(pid, 0.0))

        dates = sorted(set(gv.date))
        season_log = log[log.season_end == V]
        by_date = {d: x for d, x in season_log.groupby("date")}
        for d in dates:
            day = gv[gv.date == d]
            for r in day.itertuples(index=False):
                for side, pid, team in ((0, r.gk_h, r.home), (1, r.gk_a, r.away)):
                    if np.isnan(pid):
                        continue
                    t = talent(int(pid))
                    rf = ref.get(team, t)
                    out[r.gid, side] = t
                    out[r.gid, 2 + side] = t - rf
            # after the day's games: update references and in-season evidence
            for r in day.itertuples(index=False):
                for side, pid, team in ((0, r.gk_h, r.home), (1, r.gk_a, r.away)):
                    if not np.isnan(pid):
                        rf = ref.get(team, out[r.gid, side])
                        ref[team] = rf + alpha * (out[r.gid, side] - rf)
            x = by_date.get(d)
            if x is not None:
                for pid, sh, svs in zip(x.goalie_id, x.shots, x.saves):
                    ins_num[pid] = ins_num.get(pid, 0.0) + (svs - sv * sh)
                    ins_den[pid] = ins_den.get(pid, 0.0) + sh
    return pd.DataFrame({"gid": g.gid, "gt_h": out[:, 0], "gt_a": out[:, 1],
                         "gdiff_h": out[:, 2], "gdiff_a": out[:, 3]})


# ---------------------------------------------------------------------------
# Poisson GLM with team-season fixed effects
# ---------------------------------------------------------------------------
def _long(g: pd.DataFrame, target: str, goalie: pd.DataFrame | None = None):
    """Two rows per game: (home attacking), (away attacking)."""
    n = len(g)
    y = np.r_[g[f"{target}_h"].to_numpy(float), g[f"{target}_a"].to_numpy(float)]
    home = np.r_[np.ones(n), np.zeros(n)]
    X = {}
    for f in GM.CTX_FEATURES:
        X[f"{f}_o"] = np.r_[g[f"f_{f}_h"].to_numpy(), g[f"f_{f}_a"].to_numpy()]
        X[f"{f}_d"] = np.r_[g[f"f_{f}_a"].to_numpy(), g[f"f_{f}_h"].to_numpy()]
    if goalie is not None:
        gm = g[["gid"]].merge(goalie, on="gid", how="left")
        # opponent starter's talent above its usual starter, scaled to a
        # log goals multiplier under beta = 1 (see gamemodel.goalie_offset)
        X["goalie"] = np.r_[np.nan_to_num(gm.gdiff_a.to_numpy()),
                            np.nan_to_num(gm.gdiff_h.to_numpy())]
    if target == "sh":
        m = g.margin.to_numpy(float)
        X["margin"] = np.r_[np.clip(m, -3, 3), np.clip(-m, -3, 3)]
        X["extra"] = np.r_[g.went_extra.to_numpy(), g.went_extra.to_numpy()]
    return y, home, X


def fit_glm(g: pd.DataFrame, target: str = "reg", season_w: dict | None = None,
            goalie: pd.DataFrame | None = None, ridge_sd: float = 1.0) -> dict:
    """Poisson GLM: log E[y] = mu_s + h_s*home + o_ts + d_ts' + sum_k beta_k x_k.

    target 'reg' (regulation goals) or 'sh' (shots on goal).
    Team-season effects get a N(0, ridge_sd^2) prior, only for identification
    (ridge_sd = 1 shrinks an 82-game effect by ~0.4%). A tight ridge would
    shrink the realised-strength targets of the preseason regression and so
    compress every preseason rating (0.2 compressed them by ~9%).
    Returns dict: beta, mu (by season), h (by season), fe (DataFrame
    team, season_end, o, d), lam (fitted per game, home/away), dispersion.
    """
    g = g.reset_index(drop=True)
    n = len(g)
    y, home, X = _long(g, target, goalie)
    seasons = np.r_[g.season_end.to_numpy(), g.season_end.to_numpy()]
    s_levels = np.unique(seasons)
    s_idx = np.searchsorted(s_levels, seasons)
    ts_h = g.home + "_" + g.season_end.astype(str)
    ts_a = g.away + "_" + g.season_end.astype(str)
    ts_levels = np.unique(np.r_[ts_h, ts_a])
    att = np.searchsorted(ts_levels, np.r_[ts_h, ts_a])
    dfn = np.searchsorted(ts_levels, np.r_[ts_a, ts_h])
    w = np.ones(2 * n) if season_w is None else np.array(
        [season_w.get(s, 0.0) for s in seasons], float)
    keep = ~np.isnan(y) & (w > 0)
    y, home, s_idx, att, dfn, w = y[keep], home[keep], s_idx[keep], att[keep], dfn[keep], w[keep]
    names = list(X)
    Xm = np.column_stack([X[k][keep] for k in names]) if names else np.zeros((len(y), 0))
    S, TS, K = len(s_levels), len(ts_levels), len(names)
    # parameter vector: mu[S], h[S], o[TS], d[TS], beta[K]
    iS, iH, iO, iD, iB = 0, S, 2 * S, 2 * S + TS, 2 * S + 2 * TS
    npar = iB + K
    rows = np.arange(len(y))
    A = sparse.csr_matrix(
        (np.r_[np.ones(len(y)), home, np.ones(len(y)), np.ones(len(y))],
         (np.r_[rows, rows, rows, rows],
          np.r_[iS + s_idx, iH + s_idx, iO + att, iD + dfn])), shape=(len(y), iB))
    prec = 1.0 / ridge_sd ** 2

    def f(p):
        eta = A @ p[:iB] + Xm @ p[iB:]
        lam = np.exp(eta)
        r = w * (lam - y)
        val = np.sum(w * (lam - y * eta)) + 0.5 * prec * np.sum(p[iO:iB] ** 2)
        grad = np.r_[A.T @ r, Xm.T @ r]
        grad[iO:iB] += prec * p[iO:iB]
        return val, grad

    p0 = np.zeros(npar)
    p0[iS:iS + S] = np.log(np.average(y, weights=w))
    res = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    p = res.x
    eta = A @ p[:iB] + Xm @ p[iB:]
    lam = np.exp(eta)
    disp = float(np.sum(w * (y - lam) ** 2 / lam) / (np.sum(w) - npar * np.sum(w) / len(w)))
    fe = pd.DataFrame({"ts": ts_levels, "o": p[iO:iD], "d": p[iD:iB]})
    fe[["team", "season_end"]] = fe.ts.str.split("_", expand=True)
    fe["season_end"] = fe.season_end.astype(int)
    # standard errors of the FE (diagonal of the Poisson information)
    info_o = np.bincount(att, w * lam, TS) + prec
    info_d = np.bincount(dfn, w * lam, TS) + prec
    fe["o_se"], fe["d_se"] = 1 / np.sqrt(info_o), 1 / np.sqrt(info_d)
    lam_full = np.full(2 * n, np.nan)
    lam_full[np.flatnonzero(keep)] = lam
    return {"beta": dict(zip(names, map(float, p[iB:]))),
            "mu": dict(zip(map(int, s_levels), map(float, p[iS:iS + S]))),
            "h": dict(zip(map(int, s_levels), map(float, p[iH:iH + S]))),
            "fe": fe.drop(columns="ts"), "lam_h": lam_full[:n], "lam_a": lam_full[n:],
            "dispersion": disp, "converged": bool(res.success)}


# ---------------------------------------------------------------------------
# Structural fit for a target season
# ---------------------------------------------------------------------------
def recency_weights(V: int, first: int, halflife: float) -> dict:
    """Season weights 0.5**((V-1-s)/halflife) for s in [first, V-1], with the
    shortened / bubble seasons (2013, 2020, 2021) kept at full weight (their
    games are real games) but 2020 playoffs never included (regular season
    only)."""
    return {s: 0.5 ** ((V - 1 - s) / halflife) for s in range(first, V)}


STRUCT_VERSION = "s4"            # bump when the structural fits change
STRUCT_CACHE = C.CACHE / "structural"


@functools.lru_cache(maxsize=None)
def structural(V: int, window: int = 8, h_halflife: float = 3.0,
               ctx_halflife: float = 6.0, layer_halflife: float = 3.0,
               goalie_key: tuple = (3000.0, -0.002, 1.0, 12.0)) -> dict:
    """Disk-cached wrapper of ``_structural`` (orr/cache/structural/).
    The goalie GLM (ctx_gk, beta_gk) is cached separately per goalie_key, so
    goalie-parameter searches do not refit everything else."""
    goalie_key = tuple(float(x) for x in goalie_key)
    base = _disk(("base", V, window, h_halflife, ctx_halflife, layer_halflife,
                  DEFAULT_GOALIE_KEY),
                 lambda: _structural(V, window, h_halflife, ctx_halflife,
                                     layer_halflife, DEFAULT_GOALIE_KEY))
    if goalie_key == DEFAULT_GOALIE_KEY:
        return base
    gk = _disk(("goalie", V, window, ctx_halflife, goalie_key),
               lambda: _goalie_glm(V, window, ctx_halflife, goalie_key))
    return {**base, **gk}


DEFAULT_GOALIE_KEY = (3000.0, -0.002, 1.0, 12.0)


def _disk(key: tuple, compute):
    import hashlib
    import pickle
    k = repr((STRUCT_VERSION,) + tuple(key[1:])) if key[0] == "base" else repr((STRUCT_VERSION,) + key)
    f = STRUCT_CACHE / (hashlib.sha1(k.encode()).hexdigest()[:16] + ".pkl")
    if f.exists() and _DISK_CACHE:
        return pickle.loads(f.read_bytes())
    out = compute()
    if _DISK_CACHE:
        STRUCT_CACHE.mkdir(parents=True, exist_ok=True)
        f.write_bytes(pickle.dumps(out))
    return out


def _goalie_glm(V: int, window: int, ctx_halflife: float, goalie_key: tuple) -> dict:
    """Goalie-aware goals GLM for target season V (seasons with starters)."""
    g = game_frame()
    lo = max(FIRST_SEASON, V - window)
    gw = g[(g.season_end >= lo) & (g.season_end < V)]
    w_ctx = recency_weights(V, lo, ctx_halflife)
    gt = goalie_game_talent(*goalie_key)
    gk_seasons = sorted(set(g.season_end[g.gk_h.notna()]))
    gk_use = [s for s in gk_seasons if lo <= s < V and s != 2024]
    if len(gk_use) < 2:
        return {"beta_gk": None, "ctx_gk": None}
    gg = gw[gw.season_end.isin(gk_use) & gw.gk_h.notna() & gw.gk_a.notna()]
    b = fit_glm(gg, "reg", w_ctx, goalie=gt)["beta"]
    return {"beta_gk": b.pop("goalie"), "ctx_gk": b}


_DISK_CACHE = True      # tests that corrupt data switch this off


def _structural(V: int, window: int, h_halflife: float, ctx_halflife: float,
                layer_halflife: float, goalie_key: tuple) -> dict:
    """Data-estimated parameters for target season V from seasons < V.

      ctx      rest/travel coefficients, no-goalie variant (goals GLM)
      ctx_gk   the same with the starting-goalie term (seasons with starters)
      beta_gk  fitted goalie coefficient (1 = the talent maps 1:1 to goals)
      h0, mu0  prior mean home ice and league level for V (recency weighted)
      h_sd     season-to-season SD of home ice around its recency mean
      shots    shots GLM: beta (ctx, margin, extra), mu0/h0 priors, dispersion
      layer    end-game layer and base dispersion kappa, fitted jointly (ML on
               the regulation score) on the GLM's in-sample expected goals
      disp_g   goals dispersion (Pearson)
    """
    g = game_frame()
    lo = max(FIRST_SEASON, V - window)
    gw = g[(g.season_end >= lo) & (g.season_end < V)]
    w_ctx = recency_weights(V, lo, ctx_halflife)
    glm = fit_glm(gw, "reg", w_ctx)
    out = {"V": V, "ctx": {k: v for k, v in glm["beta"].items()}, "disp_g": glm["dispersion"]}
    # goalie variant: seasons with known starters only
    gt = goalie_game_talent(*goalie_key)
    gk_seasons = sorted(set(g.season_end[g.gk_h.notna()]))
    gk_use = [s for s in gk_seasons if lo <= s < V and s != 2024]
    if len(gk_use) >= 2:
        gg = gw[gw.season_end.isin(gk_use) & gw.gk_h.notna() & gw.gk_a.notna()]
        glm_gk = fit_glm(gg, "reg", w_ctx, goalie=gt)
        b = glm_gk["beta"]
        out["beta_gk"] = b.pop("goalie")
        out["ctx_gk"] = b
    else:
        out["beta_gk"] = None
        out["ctx_gk"] = None
    # home ice / league level: recency-weighted over all seasons < V
    hs = pd.Series(glm["h"])
    ms = pd.Series(glm["mu"])
    wh = np.array([0.5 ** ((V - 1 - s) / h_halflife) for s in hs.index])
    out["h0"] = float(np.average(hs, weights=wh))
    out["h_sd"] = float(np.sqrt(np.average((hs - out["h0"]) ** 2, weights=wh)))
    out["mu0"] = float(ms.iloc[-1])       # last season's level
    out["mu_hist"] = {int(k): float(v) for k, v in ms.items()}
    out["h_hist"] = {int(k): float(v) for k, v in hs.items()}
    # shots
    gs = gw[gw.sh_h.notna() & (gw.season_end >= 2011)]
    if gs.season_end.nunique() >= 1:
        sg = fit_glm(gs, "sh", {s: w_ctx[s] for s in set(gs.season_end)})
        hs_s = pd.Series(sg["h"])
        wh = np.array([0.5 ** ((V - 1 - s) / h_halflife) for s in hs_s.index])
        out["shots"] = {"beta": sg["beta"], "h0": float(np.average(hs_s, weights=wh)),
                        "mu0": float(pd.Series(sg["mu"]).iloc[-1]),
                        "mu_hist": {int(k): float(v) for k, v in sg["mu"].items()},
                        "dispersion": sg["dispersion"]}
    else:
        out["shots"] = None
    # end-game layer on in-sample expected goals, recency weighted, same era
    era_lo = THREE_ON_THREE if V > THREE_ON_THREE + 1 else lo
    m = (gw.season_end >= era_lo).to_numpy()
    wl = np.array([0.5 ** ((V - 1 - s) / layer_halflife) for s in gw.season_end[m]])
    fl = GM.fit_layer(glm["lam_h"][m], glm["lam_a"][m], gw.reg_h.to_numpy()[m],
                      gw.reg_a.to_numpy()[m], wl)
    out["layer"], out["kappa"] = fl["layer"], fl["kappa"]
    return out


def glm_all_fe() -> pd.DataFrame:
    """In-sample team-season (o, d) for every season (targets for the
    preseason regression): one goals GLM per season (no pooling across
    seasons), with the context coefficients of a pooled fit held fixed."""
    return _glm_all_fe().copy()


@functools.lru_cache(maxsize=None)
def _glm_all_fe() -> pd.DataFrame:
    """One GLM per season (that season's games only), so no target reads any
    other season."""
    g = game_frame()
    frames = []
    for s, gs in g.groupby("season_end"):
        fe = fit_glm(gs, "reg")["fe"].rename(columns={
            "o": "o_fe", "d": "d_fe", "o_se": "o_fe_se", "d_se": "d_fe_se"})
        if gs.sh_h.notna().all():
            fs = fit_glm(gs, "sh")["fe"].rename(columns={
                "o": "so_fe", "d": "sd_fe", "o_se": "so_fe_se", "d_se": "sd_fe_se"})
            fe = fe.merge(fs, on=["team", "season_end"], how="left")
        frames.append(fe)
    fe = pd.concat(frames, ignore_index=True)
    gp = pd.concat([g[["season_end", "home"]].rename(columns={"home": "team"}),
                    g[["season_end", "away"]].rename(columns={"away": "team"})])
    gp = gp.groupby(["team", "season_end"]).size().rename("gp").reset_index()
    return fe.merge(gp, on=["team", "season_end"])


def backup_usage(max_season: int, min_season: int = 2011,
                 goalie_key: tuple = (3000.0, -0.002, 1.0, 12.0)) -> dict:
    """Back-to-back goalie usage from seasons [min_season, max_season]:
    share of starts by a team's non-primary goalie (primary = most starts
    that season) on the second night of a back-to-back vs other games, and
    the mean talent gap primary - most-used backup (goals saved per shot,
    pregame talents)."""
    g = game_frame()
    g = g[g.gk_h.notna() & g.gk_a.notna() & (g.season_end >= min_season)
          & (g.season_end <= max_season)]
    gt = goalie_game_talent(*goalie_key)
    g = g.merge(gt[["gid", "gt_h", "gt_a"]], on="gid")
    t = pd.concat([g[["season_end", "home", "gk_h", "rest_h", "gt_h"]].set_axis(
        ["season_end", "team", "gk", "rest", "talent"], axis=1),
        g[["season_end", "away", "gk_a", "rest_a", "gt_a"]].set_axis(
        ["season_end", "team", "gk", "rest", "talent"], axis=1)])
    n = t.groupby(["season_end", "team", "gk"]).agg(n=("rest", "size"),
                                                     talent=("talent", "mean")).reset_index()
    n = n.sort_values("n", ascending=False)
    prim = n.drop_duplicates(["season_end", "team"])
    back = n[~n.set_index(["season_end", "team", "gk"]).index.isin(
        prim.set_index(["season_end", "team", "gk"]).index)].drop_duplicates(["season_end", "team"])
    t = t.merge(prim[["season_end", "team", "gk"]].rename(columns={"gk": "prim"}),
                on=["season_end", "team"])
    t["backup"] = t.gk != t.prim
    gap = prim.merge(back, on=["season_end", "team"], suffixes=("_p", "_b"))
    return {"p_backup_b2b": float(t.backup[t.rest == 1].mean()),
            "p_backup_other": float(t.backup[t.rest != 1].mean()),
            "avg_gap": float((gap.talent_p - gap.talent_b).mean()),
            "seasons": [min_season, max_season]}


# ---------------------------------------------------------------------------
# Goalie talent in the fitted units, for the 2026-27 freeze and live use
# ---------------------------------------------------------------------------
def _frozen_goalie_key() -> tuple:
    import json
    try:
        hp = json.loads((C.PARAMS / "ratings_hp.json").read_text())["hp"]
        return tuple(float(x) for x in hp["goalie"])
    except FileNotFoundError:
        return DEFAULT_GOALIE_KEY


def goalie_talent(V: int, goalie_key: tuple | None = None) -> pd.Series:
    """Preseason save talent for season V in the units the game model's
    goalie coefficient was fitted on (goals saved above average per shot on
    goal; MoneyPuck GSAx per shot, normalised to the league goals/xG ratio,
    Marcel weights 3/2/1 over V-1..V-3, regressed toward m0 with n0 weighted
    pseudo-shots). Reads seasons < V only. Goalies without NHL history in
    V-1..V-3 get the prior mean m0 (``goalie_talent_default``).
    Returns a Series player_id -> talent."""
    n0, m0, _, _ = goalie_key or _frozen_goalie_key()
    prior, _ = goalie_prior(V, n0, m0)
    return pd.Series({pid: a / b for pid, (a, b) in prior.items()}, name="talent")


def goalie_talent_default(goalie_key: tuple | None = None) -> float:
    """Talent assigned to a goalie with no NHL history (the prior mean m0)."""
    return float((goalie_key or _frozen_goalie_key())[1])


def goalie_talent_2027() -> pd.Series:
    """player_id -> preseason save talent for 2026-27 in the fitted units, as of
    the 2026-09-29 cutoff (MoneyPuck seasons through 2025-26 only, from the
    pre-cutoff snapshot). Use with gamemodel.goalie_offset (talent minus the
    team's usual starter) and gamemodel.b2b_goalie_offset (team_gap_2027)."""
    return goalie_talent(C.TARGET_SEASON)


def team_gap_2027(goalies: pd.DataFrame, talent: pd.Series | None = None) -> pd.Series:
    """Each team's primary-minus-backup save-talent gap in the fitted units.

    goalies: player_id, team, start_share [, p_present]. Goalies with
    p_present <= 0.5 are dropped; primary and backup are the two largest
    start shares. A team with one goalie gets 0. Same definition as the
    historical ``backup_usage`` avg_gap that gamemodel.b2b_goalie_offset is
    centred on."""
    t = goalie_talent_2027() if talent is None else talent
    g = goalies.copy()
    if "p_present" in g:
        g = g[g.p_present > 0.5]
    g = g.sort_values("start_share", ascending=False)
    g["t"] = g.player_id.map(t).fillna(goalie_talent_default())
    top2 = g.groupby("team").head(2)
    return top2.groupby("team").t.agg(lambda x: x.iloc[0] - x.iloc[1] if len(x) > 1 else 0.0)


def usual_starter_talent(goalies: pd.DataFrame, talent: pd.Series | None = None) -> pd.Series:
    """A team's usual-starter reference: start-share-weighted mean talent of
    its goalies (fitted units). The starter offset for a game is
    goalie_offset(P, talent[starter] - usual_starter_talent[team])."""
    t = goalie_talent_2027() if talent is None else talent
    g = goalies.copy()
    if "p_present" in g:
        g = g[g.p_present > 0.5]
    g["t"] = g.player_id.map(t).fillna(goalie_talent_default())
    w = g.start_share.clip(lower=0)
    return (g.assign(wt=w * g.t).groupby("team").wt.sum() / w.groupby(g.team).sum()).rename("ref")
