"""Player panels + player-level statistical modules for v2.

- Panels: skater/goalie player-seasons (situation=="all", traded players aggregated),
  per-team rows kept for majority-team logic; bios for birthDate.
- Goalie module: per-shot centered GSAx rates, DerSimonian-Laird consistency C,
  empirical-Bayes Marcel projection (posterior mean theta, predictive variance V).
- Skater module: pts/60 Marcel shrunk to position mean, age-curve adjusted, TOI projection.
- Age curves: TOI-weighted player-fixed-effects piecewise-linear regression (knots 23/27/31),
  with a delta-method cross-check.

Vantage convention: V = season_end of the last completed season. All functions take a vantage
and use ONLY seasons <= V (walk-forward safe by construction).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
PROC = PROJ / "data" / "processed"

MP_FRAN = {"ATL": "WPG", "ARI": "UTA", "L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}
MIN_SHOTS_SEASON = 300     # goalie season counts toward consistency if >= this many shots
GOALIE_WINDOW = 4          # seasons of history for goalie Marcel/consistency
SKATER_WINDOW = 3


# ------------------------------------------------------------------ panels
def build_panels():
    PROC.mkdir(parents=True, exist_ok=True)
    sk_cols = ["playerId", "season", "name", "team", "position", "situation",
               "games_played", "icetime", "I_F_points", "I_F_goals"]
    go_cols = ["playerId", "season", "name", "team", "position", "situation",
               "games_played", "icetime", "xGoals", "goals", "ongoal"]
    sk_frames, go_frames = [], []
    for season in range(2008, 2026):
        sk = pd.read_csv(RAW / f"mp_skaters_{season}.csv", usecols=sk_cols)
        go = pd.read_csv(RAW / f"mp_goalies_{season}.csv", usecols=go_cols)
        sk_frames.append(sk[sk.situation == "all"])
        go_frames.append(go[go.situation == "all"])
    sk = pd.concat(sk_frames, ignore_index=True)
    go = pd.concat(go_frames, ignore_index=True)
    for df in (sk, go):
        df["season_end"] = df.pop("season").astype(int) + 1
        df["team"] = df.team.replace(MP_FRAN)
        df["toi_min"] = df.pop("icetime") / 60.0

    # per-team rows (traded players appear once per team)
    sk_team = sk[["playerId", "season_end", "team", "toi_min", "games_played",
                  "I_F_points"]].copy()
    go_team = go[["playerId", "season_end", "team", "toi_min", "games_played",
                  "ongoal", "xGoals", "goals"]].copy()

    # player-season aggregates
    skaters = sk.groupby(["playerId", "season_end"], as_index=False).agg(
        name=("name", "first"), position=("position", "first"),
        gp=("games_played", "sum"), toi_min=("toi_min", "sum"),
        points=("I_F_points", "sum"), goals=("I_F_goals", "sum"))
    skaters["pts60"] = 60.0 * skaters.points / skaters.toi_min.clip(lower=1.0)
    skaters["pos_group"] = np.where(skaters.position == "D", "D", "F")

    goalies = go.groupby(["playerId", "season_end"], as_index=False).agg(
        name=("name", "first"), gp=("games_played", "sum"), toi_min=("toi_min", "sum"),
        shots=("ongoal", "sum"), xga=("xGoals", "sum"), ga=("goals", "sum"))

    bios = pd.read_csv(RAW / "mp_lookup.csv", usecols=["playerId", "birthDate", "position"])
    bios["birthDate"] = pd.to_datetime(bios.birthDate, errors="coerce")
    bios = bios.dropna(subset=["birthDate"]).drop_duplicates("playerId").set_index("playerId")

    # NHL roster backfill for missing birthdates
    ros = pd.read_csv(RAW / "nhl_rosters_20262027.csv")
    ros["birthDate"] = pd.to_datetime(ros.birthDate, errors="coerce")
    extra = ros.dropna(subset=["birthDate"]).drop_duplicates("playerId").set_index("playerId")
    missing_ids = set(skaters.playerId) | set(goalies.playerId)
    add = extra.loc[extra.index.isin(missing_ids - set(bios.index)), ["birthDate"]]
    if len(add):
        add["position"] = "?"
        bios = pd.concat([bios, add])

    # asserts
    assert skaters.duplicated(["playerId", "season_end"]).sum() == 0
    assert goalies.duplicated(["playerId", "season_end"]).sum() == 0
    big = skaters[skaters.toi_min >= 1000].playerId
    cov = big.isin(bios.index).mean()
    assert cov > 0.99, f"birthDate coverage for regulars only {cov:.3f}"
    team_toi = sk_team.groupby(["season_end", "team"]).toi_min.sum()
    ts = pd.read_csv(PROC / "team_seasons.csv").set_index(["season_end", "team"])
    gp = ts.gp.reindex(team_toi.index)
    ratio = team_toi / (gp * 300)  # ~5 skaters x 60 min; penalties/pulled-goalie move it +-10%
    frac_bad = ((ratio < 0.75) | (ratio > 1.15)).mean()
    assert frac_bad < 0.02, f"team skater TOI sanity fails for {frac_bad:.1%} of team-seasons"

    skaters.to_csv(PROC / "panel_skaters.csv", index=False)
    goalies.to_csv(PROC / "panel_goalies.csv", index=False)
    sk_team.to_csv(PROC / "panel_skater_team.csv", index=False)
    go_team.to_csv(PROC / "panel_goalie_team.csv", index=False)
    bios.reset_index().to_csv(PROC / "panel_bios.csv", index=False)
    print(f"panels: skaters {len(skaters)}, goalies {len(goalies)}, bios {len(bios)}, "
          f"seasons {skaters.season_end.min()}-{skaters.season_end.max()}")


@lru_cache(maxsize=1)
def load_panels():
    skaters = pd.read_csv(PROC / "panel_skaters.csv")
    goalies = pd.read_csv(PROC / "panel_goalies.csv")
    sk_team = pd.read_csv(PROC / "panel_skater_team.csv")
    go_team = pd.read_csv(PROC / "panel_goalie_team.csv")
    bios = pd.read_csv(PROC / "panel_bios.csv", parse_dates=["birthDate"]).set_index("playerId")
    return skaters, goalies, sk_team, go_team, bios


def age_of(bios: pd.DataFrame, player_ids, season_end: int) -> pd.Series:
    """Age at the fixed reference date Jan 1 of season_end. NaN if unknown."""
    ref = pd.Timestamp(season_end, 1, 1)
    bd = bios.birthDate.reindex(pd.Index(player_ids))
    return (ref - bd).dt.days / 365.25


def majority_team(team_rows: pd.DataFrame) -> pd.DataFrame:
    """(playerId, season_end) -> team with max TOI that season."""
    idx = team_rows.groupby(["playerId", "season_end"]).toi_min.idxmax()
    return team_rows.loc[idx, ["playerId", "season_end", "team"]].reset_index(drop=True)


# ------------------------------------------------------------------ goalie module
def goalie_rates(goalies: pd.DataFrame) -> pd.DataFrame:
    """Per-shot centered GSAx rate r and binomial sampling variance v, per goalie-season."""
    g = goalies.copy()
    lg = g.groupby("season_end").agg(N=("shots", "sum"), GA=("ga", "sum"), D=("xga", "sum"))
    lg["p_bar"] = lg.GA / lg.N
    lg["r_league"] = (lg.D - lg.GA) / lg.N
    raw_r = (g.xga - g.ga) / g.shots.clip(lower=1)
    g["r"] = raw_r - g.season_end.map(lg.r_league)
    p = g.season_end.map(lg.p_bar)
    g["v"] = (p * (1 - p)) / g.shots.clip(lower=1)
    return g


def _dl_tau2(r: np.ndarray, v: np.ndarray) -> float:
    """DerSimonian-Laird between-season variance for one goalie."""
    w = 1.0 / v
    rbar = (w * r).sum() / w.sum()
    Q = (w * (r - rbar) ** 2).sum()
    m = len(r)
    S1, S2 = w.sum(), (w ** 2).sum()
    denom = S1 - S2 / S1
    return max(0.0, (Q - (m - 1)) / denom) if denom > 0 else 0.0


def goalie_project(goalies: pd.DataFrame, vantage: int, delta: float = 0.7,
                   nu0: float = 4.0, window: int = GOALIE_WINDOW) -> pd.DataFrame:
    """EB Marcel per goalie at vantage V. Returns theta (per-shot talent), V_post
    (predictive variance for a future season), consistency C, trend slope, m, shots."""
    g = goalie_rates(goalies)
    hist = g[(g.season_end <= vantage)]
    win = hist[(hist.season_end > vantage - window) & (hist.shots >= MIN_SHOTS_SEASON)]

    # per-goalie DL tau2 on the window
    recs = []
    for pid, d in win.groupby("playerId"):
        m = len(d)
        tau2 = _dl_tau2(d.r.to_numpy(), d.v.to_numpy()) if m >= 2 else np.nan
        recs.append((pid, m, tau2, d.shots.sum()))
    tau = pd.DataFrame(recs, columns=["playerId", "m", "tau2", "shots_win"])

    est = tau[(tau.m >= 3) & tau.tau2.notna()]
    if len(est) < 5:
        est = tau[(tau.m >= 2) & tau.tau2.notna()]
    tau_bar2 = float(np.average(est.tau2, weights=est.shots_win)) if len(est) else 1e-6
    tau_bar2 = max(tau_bar2, 1e-8)

    def tau_tilde(row):
        if row.m >= 2 and pd.notna(row.tau2):
            return ((row.m - 1) * row.tau2 + nu0 * tau_bar2) / (row.m - 1 + nu0)
        return tau_bar2
    tau["tau_t2"] = tau.apply(tau_tilde, axis=1)
    tau["C"] = tau.tau_t2 / tau_bar2

    # population talent variance (method of moments on careers <= vantage, >=1000 shots)
    car = hist.groupby("playerId").agg(shots=("shots", "sum"))
    car = car[car.shots >= 1000]
    means, samp_v, wts = [], [], []
    for pid in car.index:
        d = hist[hist.playerId == pid]
        w = 1.0 / d.v
        means.append(float((w * d.r).sum() / w.sum()))
        samp_v.append(float(1.0 / w.sum()))
        wts.append(float(d.shots.sum()))
    means, samp_v, wts = np.array(means), np.array(samp_v), np.array(wts)
    mu = np.average(means, weights=wts)
    tau2_pop = max(np.average((means - mu) ** 2, weights=wts) - np.average(samp_v, weights=wts),
                   1e-8)

    # posterior per goalie (window seasons, recency x precision weights incl own volatility)
    out = []
    for pid, d in win.groupby("playerId"):
        row = tau[tau.playerId == pid].iloc[0]
        lag = vantage - d.season_end.to_numpy()
        om = delta ** lag / (d.v.to_numpy() + row.tau_t2)
        theta = float((om * d.r.to_numpy()).sum() / (om.sum() + 1.0 / tau2_pop))
        V_post = 1.0 / (om.sum() + 1.0 / tau2_pop) + row.tau_t2
        if len(d) >= 2:
            x = d.season_end.to_numpy().astype(float)
            wls = 1.0 / (d.v.to_numpy() + row.tau_t2)
            xm = np.average(x, weights=wls)
            ym = np.average(d.r.to_numpy(), weights=wls)
            trend = float((wls * (x - xm) * (d.r.to_numpy() - ym)).sum()
                          / max((wls * (x - xm) ** 2).sum(), 1e-12))
        else:
            trend = 0.0
        out.append({"playerId": pid, "theta": theta, "V_post": float(V_post),
                    "C": float(row.C), "trend": trend, "m": int(row.m),
                    "shots_win": float(row.shots_win)})
    res = pd.DataFrame(out)
    res.attrs["tau_bar2"] = tau_bar2
    res.attrs["tau2_pop"] = tau2_pop
    res.attrs["n0_implied"] = float(0.076 * (1 - 0.076) / tau2_pop)
    return res


# ------------------------------------------------------------------ age curves
KNOTS = (23.0, 27.0, 31.0)


def _basis(a: np.ndarray) -> np.ndarray:
    return np.column_stack([a] + [np.clip(a - k, 0, None) for k in KNOTS])


def age_curve_fe(skaters: pd.DataFrame, bios: pd.DataFrame, max_season: int,
                 pos_group: str) -> np.ndarray:
    """TOI-weighted player-FE piecewise-linear age curve. Returns coefs for _basis.
    Only differences f(a2)-f(a1) are meaningful."""
    d = skaters[(skaters.season_end <= max_season) & (skaters.pos_group == pos_group)
                & (skaters.toi_min >= 100)].copy()
    ref = pd.to_datetime(d.season_end.astype(str) + "-01-01")
    bd = bios.birthDate.reindex(d.playerId).to_numpy()
    d["age"] = (ref.to_numpy() - bd).astype("timedelta64[D]").astype(float) / 365.25
    d = d.dropna(subset=["age"])
    d = d[(d.age >= 18) & (d.age <= 42)]
    # multi-season players only (FE needs within variation)
    counts = d.playerId.value_counts()
    d = d[d.playerId.isin(counts[counts >= 2].index)]

    y = d.pts60.to_numpy()
    X = np.column_stack([_basis(d.age.to_numpy()),
                         pd.get_dummies(d.season_end, drop_first=True).to_numpy(float)])
    w = d.toi_min.to_numpy()
    # weighted within-player demeaning (Frisch-Waugh for player FE)
    dfX = pd.DataFrame(X)
    dfX["y"] = y
    dfX["pid"] = d.playerId.to_numpy()
    dfX["w"] = w
    def wdemean(g):
        ww = g["w"].to_numpy()[:, None]
        cols = [c for c in g.columns if c not in ("pid", "w")]
        vals = g[cols].to_numpy(float)
        mean = (vals * ww).sum(0) / ww.sum()
        return pd.DataFrame(vals - mean, columns=cols, index=g.index)
    dem = dfX.groupby("pid", group_keys=False).apply(wdemean, include_groups=False)
    dem = dem.loc[dfX.index]
    Xd = dem.drop(columns=["y"]).to_numpy(float) * np.sqrt(w)[:, None]
    yd = dem["y"].to_numpy(float) * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    return beta[:4]  # age-basis coefficients only


def curve_value(coefs: np.ndarray, ages) -> np.ndarray:
    return _basis(np.atleast_1d(np.asarray(ages, dtype=float))) @ coefs


def age_curve_delta(skaters: pd.DataFrame, bios: pd.DataFrame, max_season: int,
                    pos_group: str) -> pd.DataFrame:
    """Delta-method cross-check: mean TOI-weighted year-over-year change in pts60 by age."""
    d = skaters[(skaters.season_end <= max_season) & (skaters.pos_group == pos_group)
                & (skaters.toi_min >= 200)].copy()
    ref = pd.to_datetime(d.season_end.astype(str) + "-01-01")
    bd = bios.birthDate.reindex(d.playerId).to_numpy()
    d["age"] = (ref.to_numpy() - bd).astype("timedelta64[D]").astype(float) / 365.25
    d = d.dropna(subset=["age"])
    cur = d.copy()
    nxt = d.copy()
    nxt["season_end"] -= 1
    m = cur.merge(nxt, on=["playerId", "season_end"], suffixes=("_t", "_n"))
    m["w"] = 2 / (1 / m.toi_min_t + 1 / m.toi_min_n)
    m["age_i"] = m.age_t.round().astype(int)
    m["dr"] = m.pts60_n - m.pts60_t
    rows = [(a, np.average(g.dr, weights=g.w), len(g))
            for a, g in m.groupby("age_i") if len(g) >= 20]
    return pd.DataFrame(rows, columns=["age", "delta_pts60", "n"])


# ------------------------------------------------------------------ skater Marcel
def skater_marcel(skaters: pd.DataFrame, bios: pd.DataFrame, vantage: int,
                  delta: float = 0.8, window: int = SKATER_WINDOW,
                  fe_coefs: dict | None = None, horizon: int = 1) -> pd.DataFrame:
    """Projected pts60 and TOI for season vantage+horizon, for players active at vantage."""
    hist = skaters[(skaters.season_end <= vantage) & (skaters.season_end > vantage - window)
                   & (skaters.toi_min >= 60)].copy()
    lg = skaters[(skaters.season_end <= vantage) & (skaters.toi_min >= 200)]
    pos_stats = {}
    for pg, d in lg.groupby("pos_group"):
        mean_rate = float(np.average(d.pts60, weights=d.toi_min))
        means_by_player = d.groupby("playerId").apply(
            lambda x: np.average(x.pts60, weights=x.toi_min), include_groups=False)
        shots_v = 60.0 * mean_rate / d.groupby("playerId").toi_min.sum()
        tau2 = max(float(means_by_player.var(ddof=1) - shots_v.mean()), 0.01)
        toi82 = float(np.average(d.toi_min / d.gp.clip(lower=1) * 82, weights=d.toi_min))
        pos_stats[pg] = (mean_rate, tau2, toi82)

    out = []
    for (pid), d in hist.groupby("playerId"):
        pg = d.pos_group.iloc[0]
        mean_rate, tau2, toi82_pos = pos_stats[pg]
        lag = vantage - d.season_end.to_numpy()
        v = 60.0 * max(mean_rate, 0.5) / d.toi_min.to_numpy()
        om = delta ** lag / v
        rc = d.pts60.to_numpy() - mean_rate
        theta_c = (om * rc).sum() / (om.sum() + 1.0 / tau2)
        theta = mean_rate + theta_c
        last = d[d.season_end == d.season_end.max()].iloc[0]
        toi82_own = last.toi_min / max(last.gp, 1) * 82
        toi82 = 0.75 * toi82_own + 0.25 * toi82_pos
        out.append({"playerId": pid, "pos_group": pg, "theta_pts60": float(theta),
                    "toi82_proj": float(toi82), "last_season": int(d.season_end.max())})
    res = pd.DataFrame(out)
    # age adjustment
    if fe_coefs is not None:
        a_now = age_of(bios, res.playerId, vantage).to_numpy()
        adj = np.ones(len(res))
        for pg in ("F", "D"):
            mask = (res.pos_group == pg).to_numpy() & ~np.isnan(a_now)
            f_now = curve_value(fe_coefs[pg], a_now[mask])
            f_fut = curve_value(fe_coefs[pg], a_now[mask] + horizon)
            base = np.maximum(res.theta_pts60.to_numpy()[mask], 0.8)
            adj[mask] = np.clip(1.0 + (f_fut - f_now) / base, 0.6, 1.25)
        res["age_adj"] = adj
        res["proj_pts60"] = res.theta_pts60 * res.age_adj
    else:
        res["age_adj"] = 1.0
        res["proj_pts60"] = res.theta_pts60
    res["proj_points"] = res.proj_pts60 * res.toi82_proj / 60.0
    return res


def replacement_rates(skaters: pd.DataFrame, max_season: int) -> dict:
    lg = skaters[(skaters.season_end <= max_season) & (skaters.toi_min >= 200)]
    return {pg: float(d.pts60.quantile(0.20)) for pg, d in lg.groupby("pos_group")}


if __name__ == "__main__":
    build_panels()
