"""Team layer: preseason standings-points projections and the market blend.

Three independent views of a team's coming season are combined:

1. TOP-DOWN - what the team did: previous seasons' goal differential,
   score-and-venue-adjusted expected-goal differential and points, regressed
   by a walk-forward ridge regression fitted only on earlier seasons.
2. BOTTOM-UP - who the team has now: the summed projections of the players on
   the opening roster (hattrick.team_components), which sees trades, signings,
   injuries and aging that the top-down view cannot.
3. MARKET - the sportsbook season points line, de-vigged. Markets aggregate a
   great deal of information (camp news, depth charts, goalie health) and are
   hard to beat on their own; a model earns its keep by adding what the line
   has not yet absorbed.

The blend weights are fitted walk-forward on seasons with all three views, and
the result is a points target per team that the season simulator is then
calibrated to reproduce (hattrick.calibrate).
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import data as D


# ---------------------------------------------------------------------------
# Actual standings
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def standings_all() -> pd.DataFrame:
    """Regular-season standings for every season: pts, gp, w, rw, gf, ga."""
    g = D.regular_season()
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        x = pd.DataFrame({
            "season_end": g.season_end, "team": g[side],
            "gf": g[f"{side}_g"], "ga": g[f"{other}_g"],
        })
        win = x.gf > x.ga
        x["w"] = win.astype(int)
        x["rw"] = (win & (g.extra == "REG")).astype(int)
        x["otl"] = ((~win) & (g.extra != "REG")).astype(int)
        x["pts"] = 2 * x.w + x.otl
        x["gp"] = 1
        rows.append(x)
    s = pd.concat(rows).groupby(["season_end", "team"], as_index=False).sum()
    s["pts82"] = s.pts * 82.0 / s.gp
    s["gd_pg"] = (s.gf - s.ga) / s.gp
    s["pts_pct"] = s.pts / (2.0 * s.gp)
    return s


# ---------------------------------------------------------------------------
# Top-down features
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def team_history() -> pd.DataFrame:
    """Per team-season: standings + MoneyPuck process stats, per game."""
    s = standings_all()
    t = D.team_seasons()
    t = t.assign(
        xgd_pg=(t.xgf_all - t.xga_all) / t.gp,
        axgd_pg=(t.axgf_all - t.axga_all) / t.gp,
        ev_xg_share=t.xgf_ev / (t.xgf_ev + t.xga_ev),
        sd_pg=(t.sf_all - t.sa_all) / t.gp,
        cd_pg=(t.cf_all - t.ca_all) / t.gp,
        pp_xg60=t.xgf_pp / t.toi_pp * 60, pk_xga60=t.xga_sh / t.toi_sh * 60,
        pen_diff_pg=(t.pen_drawn_all - t.pen_taken_all) / t.gp
        if "pen_drawn_all" in t else 0.0,
    )
    keep = ["season_end", "team", "xgd_pg", "axgd_pg", "ev_xg_share", "sd_pg",
            "cd_pg", "pp_xg60", "pk_xga60", "pen_diff_pg"]
    return s.merge(t[keep], on=["season_end", "team"], how="left")


LAG_FEATURES = ["gd_pg", "axgd_pg", "pts_pct"]


def topdown_features(season_end: int, teams=None) -> pd.DataFrame:
    """Features for predicting `season_end` from the two previous seasons.

    A team missing a lagged season (expansion) gets league-average values and
    an indicator.
    """
    h = team_history()
    if teams is None:
        teams = sorted(h[h.season_end == season_end].team.unique())
    out = pd.DataFrame({"team": teams})
    for lag in (1, 2):
        prev = h[h.season_end == season_end - lag].set_index("team")
        for f in LAG_FEATURES:
            col = f"{f}_l{lag}"
            out[col] = out.team.map(prev[f])
            fill = 0.0 if f != "pts_pct" else 0.5
            out[f"miss_l{lag}"] = out[col].isna().astype(float)
            out[col] = out[col].fillna(fill)
    out["season_end"] = season_end
    return out


def _center(season_end, feats):
    """Remove each lagged season's league mean (pts_pct drifts with OT rules)."""
    f = feats.copy()
    for lag in (1, 2):
        c = f"pts_pct_l{lag}"
        f[c] = f[c] - f[c].mean()
    return f


# ---------------------------------------------------------------------------
# Walk-forward ridge
# ---------------------------------------------------------------------------
TRAIN_FROM = 2011
SCORED = [s for s in range(2011, 2027) if s not in C.BROKEN_SEASONS]


def _design(df, cols):
    return df[cols].to_numpy(float)


def fit_topdown(target_season: int, alpha: float = 2.0, extra_cols=(),
                extra_frame: pd.DataFrame | None = None):
    """Fit points-per-82 above league average on seasons before target_season.

    Returns (coef dict, intercept, residual sd).
    """
    rows = []
    for s in range(TRAIN_FROM, target_season):
        if s in C.BROKEN_SEASONS:
            continue
        f = _center(s, topdown_features(s))
        y = standings_all().query("season_end == @s").set_index("team").pts82
        f["y"] = f.team.map(y) - y.mean()
        if extra_frame is not None:
            f = f.merge(extra_frame[extra_frame.season_end == s], on=["season_end", "team"], how="left")
        rows.append(f)
    tr = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
    cols = [f"{f}_l{l}" for l in (1, 2) for f in LAG_FEATURES] + list(extra_cols)
    tr = tr.dropna(subset=cols)
    X = _design(tr, cols)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + alpha * len(Z) / 100.0 * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (tr.y.to_numpy() - tr.y.mean()))
    resid = tr.y.to_numpy() - tr.y.mean() - Z @ beta
    return {"cols": cols, "mu": mu, "sd": sd, "beta": beta,
            "b0": tr.y.mean(), "resid_sd": float(resid.std()), "n": len(tr)}


def predict_topdown(fit, season_end: int, teams=None, extra_frame=None) -> pd.DataFrame:
    f = _center(season_end, topdown_features(season_end, teams))
    if extra_frame is not None:
        f = f.merge(extra_frame[extra_frame.season_end == season_end],
                    on=["season_end", "team"], how="left")
    Z = (_design(f, fit["cols"]) - fit["mu"]) / fit["sd"]
    f["td_rel82"] = fit["b0"] + Z @ fit["beta"]       # points per 82 above league mean
    return f[["team", "season_end", "td_rel82"]]


def league_points_per_team(season_end: int, games: int) -> float:
    """Expected league-average points per team: 2 per game + extra-time points.

    Extra-time share is projected from the three previous full seasons."""
    g = D.regular_season()
    past = [s for s in range(season_end - 3, season_end) if s not in C.BROKEN_SEASONS] or [season_end - 1]
    ex = (g[g.season_end.isin(past)].extra != "REG").mean()
    return games * (1.0 + ex / 2.0)


# ---------------------------------------------------------------------------
# Offence / defence style (goals for and against per game)
# ---------------------------------------------------------------------------
def _gfga_frame(season_end: int, teams=None) -> pd.DataFrame:
    """Lagged per-game GF, GA, score-adjusted xGF, xGA for one target season,
    each expressed relative to that lag season's league mean."""
    h = team_history()
    t = D.team_seasons()
    t = t.assign(gf_pg=t.gf_all / t.gp, ga_pg=t.ga_all / t.gp,
                 axgf_pg=t.axgf_all / t.gp, axga_pg=t.axga_all / t.gp)
    if teams is None:
        teams = sorted(h[h.season_end == season_end].team.unique())
    out = pd.DataFrame({"team": teams, "season_end": season_end})
    for lag in (1, 2):
        prev = t[t.season_end == season_end - lag].set_index("team")
        for f in ("gf_pg", "ga_pg", "axgf_pg", "axga_pg"):
            v = prev[f] - prev[f].mean()
            out[f"{f}_l{lag}"] = out.team.map(v).fillna(0.0)
    return out


STYLE_COLS = [f"{f}_l{l}" for l in (1, 2) for f in ("gf_pg", "ga_pg", "axgf_pg", "axga_pg")]


def fit_style(target_season: int, alpha: float = 8.0) -> dict:
    """Walk-forward ridge for next-season GF/game and GA/game (relative to
    league), returned as log-rate offsets o_m, d_m by `predict_style`."""
    rows = []
    t = D.team_seasons()
    for s in range(TRAIN_FROM, target_season):
        if s in C.BROKEN_SEASONS or s not in set(t.season_end):
            continue
        f = _gfga_frame(s)
        cur = t[t.season_end == s].set_index("team")
        f["y_gf"] = f.team.map(cur.gf_all / cur.gp - (cur.gf_all / cur.gp).mean())
        f["y_ga"] = f.team.map(cur.ga_all / cur.gp - (cur.ga_all / cur.gp).mean())
        rows.append(f)
    tr = pd.concat(rows).dropna()
    X = tr[STYLE_COLS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + alpha * len(Z) / 100.0 * np.eye(Z.shape[1])
    fit = {"mu": mu, "sd": sd}
    for y in ("y_gf", "y_ga"):
        fit[y] = np.linalg.solve(A, Z.T @ tr[y].to_numpy())
    lg = t[t.season_end == target_season - 1]
    fit["league_gpg"] = float((lg.gf_all / lg.gp).mean())
    return fit


def predict_style(fit: dict, season_end: int, teams) -> pd.DataFrame:
    f = _gfga_frame(season_end, teams)
    Z = (f[STYLE_COLS].to_numpy(float) - fit["mu"]) / fit["sd"]
    gf = fit["league_gpg"] + Z @ fit["y_gf"]
    ga = fit["league_gpg"] + Z @ fit["y_ga"]
    return pd.DataFrame({"team": f.team, "gf_pg": gf, "ga_pg": ga,
                         "o_m": np.log(gf / fit["league_gpg"]),
                         "d_m": np.log(ga / fit["league_gpg"])})


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------
def market_history() -> pd.DataFrame:
    """Historical preseason points lines (82-game seasons), hand-collected from
    Hockey-Reference preseason-odds tables and cross-checked (see
    hattrick/data_sources.csv)."""
    m = pd.read_csv(C.PKG / "data_market_history.csv")
    m["team"] = D.norm_team(m.team)
    return m[["season_end", "team", "line", "confidence"]]


def market_rel82(season_end: int) -> pd.DataFrame:
    """Market view as points per 82 games above the league's mean line."""
    if season_end == C.TARGET_SEASON:
        m = D.market_totals_2027()
        mean = market_mean_from_line(m.line, m.p_over)
        games = C.GAMES_PER_TEAM[season_end]
        rel = (mean - mean.mean()) * 82.0 / games
        return pd.DataFrame({"team": m.team, "season_end": season_end, "mkt_rel82": rel})
    m = market_history()
    m = m[m.season_end == season_end]
    return pd.DataFrame({"team": m.team, "season_end": season_end,
                         "mkt_rel82": m.line - m.line.mean()})


def blend_frame(seasons, extra: pd.DataFrame | None = None, alpha: float = 8.0) -> pd.DataFrame:
    """Per team-season: actual, market and top-down points per 82 above league
    mean (plus any extra view columns, e.g. the bottom-up roster model)."""
    st = standings_all()
    rows = []
    for V in seasons:
        p = predict_topdown(fit_topdown(V, alpha=alpha), V)
        a = st[st.season_end == V].set_index("team")
        p["act_rel82"] = p.team.map(a.pts82 - a.pts82.mean())
        p = p.merge(market_rel82(V), on=["team", "season_end"], how="left")
        if extra is not None:
            p = p.merge(extra, on=["team", "season_end"], how="left")
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


def fit_blend(frame: pd.DataFrame, cols) -> np.ndarray:
    """Non-negative least-squares weights (no intercept: every view is already
    relative to the league mean)."""
    from scipy.optimize import nnls
    f = frame.dropna(subset=list(cols) + ["act_rel82"])
    w, _ = nnls(f[list(cols)].to_numpy(float), f.act_rel82.to_numpy(float))
    return w


def loso_blend(frame: pd.DataFrame, cols) -> pd.DataFrame:
    """Leave-one-season-out: weights fitted on the other seasons, scored on the
    held-out one. Returns per-season MAE/RMSE of the blend and of each view."""
    out = []
    for V in sorted(frame.season_end.unique()):
        tr, te = frame[frame.season_end != V], frame[frame.season_end == V].dropna(
            subset=list(cols) + ["act_rel82"])
        w = fit_blend(tr, cols)
        pred = te[list(cols)].to_numpy(float) @ w
        e = pred - te.act_rel82.to_numpy()
        row = {"season_end": V, "n": len(te), "blend_mae": np.abs(e).mean(),
               "blend_rmse": np.sqrt((e ** 2).mean()),
               **{f"w_{c}": wi for c, wi in zip(cols, w)}}
        for c in cols:
            row[f"{c}_mae"] = np.abs(te[c] - te.act_rel82).mean()
        out.append(row)
    return pd.DataFrame(out)


LUCK_SD_82 = 8.43   # SD of season points from game-outcome luck alone (82 games,
                    # P(W)~.5, P(OTL)~.11); talent SD = sqrt(RMSE^2 - luck^2)


def market_mean_from_line(line, p_over=0.5, sd=11.0):
    """Mean points implied by an O/U line and the no-vig over probability.

    Lines are half-points; with a roughly normal outcome of SD `sd`, the mean
    is line + sd * z(p_over).
    """
    from scipy.stats import norm
    return np.asarray(line, float) + sd * norm.ppf(np.clip(p_over, 0.02, 0.98))
