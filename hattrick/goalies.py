"""Goalie projections: save talent (GSAx) and the within-team start share.

Talent
------
GSAx is measured against LEAGUE-CALIBRATED expected goals: MoneyPuck's xG
model runs hot or cold by season (league goals/xG ranged 0.97-1.06 in
2009-26), so each season's xGA is rescaled by that season's league G/xG and
an average goalie is exactly 0. The rate is GSAx per unblocked shot faced
(Fenwick against), the exposure that the xG model prices.

  rate_hat = (sum_j w_j GSAx_j + K * prior) / (sum_j w_j FA_j + K)

with season weights decay**(j-1) over four seasons, a prior that depends on
workload (teams give the net to goalies they believe are better, so a
2,000-shot starter's expected rate is above a 300-shot call-up's), and K
(shots) from the method of moments: talent variance tau^2 is the
FA-weighted covariance of a goalie's residual rates in consecutive seasons,
noise is binomial (xGA/FA per shot). Goalie seasons are notoriously noisy, so
K comes out in the thousands of shots: even a full season is regressed
about halfway. An additive age curve (per shot) is fitted walk-forward on
residuals of the un-aged projection.

Start share
-----------
Among the goalies a team carries, goalie i gets a share proportional to
  w_i = exp(b1 * workload_i + b2 * talent_z_i + b3 * [age>=35] + b4 * [No. 1])
where workload = recent-season minutes as a share of a full season. The
coefficients are a conditional logit of the NIGHTLY starter choice among the
goalies available around that game (box scores 2011-2024, walk-forward), so
they describe coaches' choices among healthy goalies; injuries are layered
on separately. There is NO cap on starts: an elite starter's share
reaches 0.75-0.80 when his workload and talent say so. Injuries are a
two-state chain per goalie (league injury prior + own absence history, as
for skaters); on a given night the healthy goalies split the start in
proportion to w, and if none is healthy a call-up starts. Special statuses
(injured to start, holdouts) enter as games out and a probability of being
present at all, drawn per simulated season.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import data as D

LAGS = 4
FIRST = 2009
GOALIE_DECAY = 0.85                # season weight ratio (chosen in goalies_bt on V<=2017 from 0.5/0.7/0.85)
FALLBACK_SHARE_B = np.array([2.0, 0.0, 0.0, 0.5])
CHOICE_WINDOW = 4                  # +-team games defining "available" around a start
REPL_SHARE_W = 0.02                # call-up weight when every rostered goalie is out
# Connor Hellebuyck (WPG), suspended for not reporting after a public trade
# request (coordinator's brief, 2026-09-29): P(plays for WPG in 2026-27)=0.10,
# and if so from about team game 13. No other team is assigned here; the team
# layer handles a trade destination. Exposed so the team layer can override.
HOLDOUTS = {8476945: {"team": "WPG", "p_present": 0.10, "games_out": 12}}
PARAMS_PATH = C.PARAMS / "goalies.json"


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def goalie_panel() -> pd.DataFrame:
    g = D.goalie_seasons().copy()
    lg = g.groupby("season_end")[["ga_all", "xga_all", "sa_all", "fa_all", "toi_all"]].sum()
    rho = lg.ga_all / lg.xga_all
    g["rho"] = g.season_end.map(rho)
    g["gsax_adj"] = g.xga_all * g.rho - g.ga_all            # league-calibrated GSAx
    tg = D.team_seasons().groupby("season_end").gp.median()
    g["sched"] = g.season_end.map(tg).fillna(82)
    g["workload"] = g.toi_all / (g.sched * 60.0)             # share of a full season
    b = D.bios().drop_duplicates("player_id").set_index("player_id").birth
    g["birth"] = g.player_id.map(b)
    return g


@functools.lru_cache(maxsize=None)
def league_goalie(V: int) -> dict:
    """League per-shot ratios for V: the last completed season's."""
    g = D.goalie_seasons()
    y = max(s for s in g.season_end.unique() if s < V)
    d = g[g.season_end == y]
    s = d[["ga_all", "xga_all", "sa_all", "fa_all", "toi_all"]].sum()
    t = D.team_seasons()
    t = t[t.season_end == y]
    return {"season": int(y), "sv": 1 - s.ga_all / s.sa_all, "fa_per_sa": s.fa_all / s.sa_all,
            "fa_per60": s.fa_all / s.toi_all * 60, "sa_per_game": float(t.sa_all.sum() / t.gp.sum()),
            "xga_per_fa": s.xga_all / s.fa_all}


def _age(birth, V):
    return ((pd.Timestamp(f"{V}-02-01") - pd.to_datetime(birth)).dt.days / 365.25).fillna(28.0)


# ---------------------------------------------------------------------------
# Talent
# ---------------------------------------------------------------------------
def _weighted(V: int, decay: float) -> pd.DataFrame:
    g = goalie_panel()
    h = g[(g.season_end < V) & (g.season_end >= V - LAGS)].copy()
    h["w"] = decay ** (V - 1 - h.season_end)
    agg = h.assign(n=h.w * h.gsax_adj, e=h.w * h.fa_all, wl=h.w * h.workload,
                   ww=h.w).groupby("player_id")[["n", "e", "wl", "ww"]].sum()
    agg["workload"] = agg.wl / agg.ww
    last = h.sort_values("season_end").groupby("player_id").tail(1).set_index("player_id")
    agg["last_team"] = last.team
    agg["last_season"] = last.season_end
    return agg


@functools.lru_cache(maxsize=None)
def talent_prior(V: int, window: int = 8) -> dict:
    """Workload-dependent prior and EB constant from seasons V-window..V-1."""
    g = goalie_panel()
    h = g[(g.season_end < V) & (g.season_end >= V - window) & (g.fa_all > 0)].copy()
    h["r"] = h.gsax_adj / h.fa_all
    X = np.column_stack([np.ones(len(h)), np.log(h.workload.clip(0.01, 1.2))])
    w = h.fa_all.to_numpy()
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], h.r.to_numpy() * np.sqrt(w), rcond=None)[0]
    h["d"] = h.r - X @ beta
    nxt = h[["player_id", "season_end", "d", "fa_all"]].assign(season_end=lambda d: d.season_end - 1)
    pr = h.merge(nxt, on=["player_id", "season_end"], suffixes=("", "_n"))
    pr = pr[(pr.fa_all > 300) & (pr.fa_all_n > 300)]
    hw = 2 / (1 / pr.fa_all + 1 / pr.fa_all_n)
    tau2 = max(float((hw * pr.d * pr.d_n).sum() / hw.sum()), 1e-7)
    p = float((h.xga_all * h.rho).sum() / h.fa_all.sum())       # goals per FA
    sig2 = p * (1 - p)
    return {"beta": beta.tolist(), "tau2": tau2, "k_shots": sig2 / tau2, "p_goal": p}


@functools.lru_cache(maxsize=None)
def age_curve(V: int, decay: float = GOALIE_DECAY) -> np.ndarray:
    """Additive per-shot age effect: quadratic fit (ages 22-40) of the
    FA-weighted mean residual actual - un-aged projection, seasons < V."""
    g = goalie_panel()
    rows = []
    for s in range(FIRST + 2, V):
        pj = _talent_unaged(s, decay)
        a = g[(g.season_end == s) & (g.fa_all > 0)].set_index("player_id")
        m = a.join(pj[["rate"]], how="inner")
        m["age"] = _age(m.birth, s)
        m["res"] = m.gsax_adj / m.fa_all - m.rate
        rows.append(m[["age", "res", "fa_all"]])
    if not rows:                                   # no earlier season: no aging
        return np.zeros(3)
    d = pd.concat(rows)
    d["a"] = d.age.clip(22, 40) - 30.0
    X = np.column_stack([np.ones(len(d)), d.a, d.a ** 2])
    w = d.fa_all.to_numpy()
    return np.linalg.solve(X.T @ (X * w[:, None]) + np.diag([0, 1, 1]) * 1e3,
                           X.T @ (w * d.res.to_numpy()))


@functools.lru_cache(maxsize=None)
def _talent_unaged(V: int, decay: float = GOALIE_DECAY) -> pd.DataFrame:
    pr = talent_prior(V)
    a = _weighted(V, decay)
    prior = pr["beta"][0] + pr["beta"][1] * np.log(a.workload.clip(0.01, 1.2))
    K = pr["k_shots"]
    a["prior"] = prior
    a["rate"] = (a.n + K * prior) / (a.e + K)
    a["rate_sd"] = np.sqrt(pr["tau2"] * K / (K + a.e))
    return a


def project_goalies(V: int, ids=None, decay: float = GOALIE_DECAY,
                    extra: pd.DataFrame | None = None, use_age: bool = True) -> pd.DataFrame:
    """Per-goalie talent for V: gsax_per_fa (+sd), gsax60, sv_pct, workload.

    ids: restrict to these goalies; goalies with no NHL history (in `extra`
    or in ids) get the prior at a call-up workload."""
    pr = talent_prior(V)
    lg = league_goalie(V)
    t = _talent_unaged(V, decay).copy()
    ids = list(t.index) if ids is None else list(ids)
    t = t.reindex(ids)
    new = t.rate.isna()
    wl0 = 0.05
    t.loc[new, "workload"] = wl0
    t.loc[new, "rate"] = pr["beta"][0] + pr["beta"][1] * np.log(wl0)
    t.loc[new, "rate_sd"] = np.sqrt(pr["tau2"])
    t["e"] = t.e.fillna(0.0)
    b = D.bios().drop_duplicates("player_id").set_index("player_id").birth
    birth = pd.Series(t.index, index=t.index).map(b)
    if extra is not None and "birth" in extra:
        eb = extra.set_index("player_id").birth
        birth = birth.fillna(pd.Series(t.index, index=t.index).map(eb))
    t["age"] = _age(birth, V).to_numpy()
    if use_age:
        c = age_curve(V, decay)
        a = t.age.clip(22, 40) - 30.0
        t["age_adj"] = c[0] + c[1] * a + c[2] * a ** 2
    else:
        t["age_adj"] = 0.0
    t["gsax_per_fa"] = t.rate + t.age_adj
    t["gsax_per_fa_sd"] = t.rate_sd
    t["gsax60"] = t.gsax_per_fa * lg["fa_per60"]
    t["gsax60_sd"] = t.rate_sd * lg["fa_per60"]
    # sv% = league sv% + GSAx per shot on goal
    t["sv_pct"] = lg["sv"] + t.gsax_per_fa * lg["fa_per_sa"]
    t["has_hist"] = ~new
    t.index.name = "player_id"
    return t.reset_index()[["player_id", "age", "has_hist", "workload", "e", "gsax_per_fa",
                            "gsax_per_fa_sd", "gsax60", "gsax60_sd", "sv_pct", "last_team"]]


# ---------------------------------------------------------------------------
# Start shares
# ---------------------------------------------------------------------------
def _share_features(p: pd.DataFrame, pr: dict) -> np.ndarray:
    z = p.gsax_per_fa.fillna(0) / np.sqrt(pr["tau2"])
    # workload enters linearly: the nightly split between two goalies whose
    # workloads differ by 0.3-0.7 of a season is ~69/31 in 2015-19 box
    # scores, while near-equal workloads split 50/50 whatever their level
    # (a log scale mistakes 0.05 vs 0.10 for a large gap)
    wl = p.workload.fillna(0.0).clip(0, 1.0)
    lead = (wl >= wl.max() - 1e-12).astype(float) if len(p) else wl
    return np.column_stack([wl, z, (p.age >= 35).astype(float), lead])


@functools.lru_cache(maxsize=None)
def _start_choices(s: int) -> pd.DataFrame:
    """Per team-game of season s (box scores): the starter and the choice
    set = goalies who started for that team within +-CHOICE_WINDOW team
    games (a proxy for 'with the club and healthy'; wider windows let an
    injured starter into the set and flatten the split)."""
    gg = D.goalie_games()
    gg = gg[gg.season_end == s]
    sch = D.fr_schedule()
    sch = sch[sch.season_end == s][["game_id", "date", "home", "away"]]
    gg = gg.merge(sch, on="game_id")
    gg["team"] = np.where(gg.home_away.str.lower() == "home", gg.home, gg.away)
    gg = gg.sort_values(["team", "date", "game_id"])
    gg["k"] = gg.groupby("team").cumcount()
    rows = []
    for team, t in gg.groupby("team"):
        ks = t.k.to_numpy()
        gid = t.goalie_id.to_numpy()
        for i in range(len(t)):
            m = np.abs(ks - ks[i]) <= CHOICE_WINDOW
            cs = np.unique(gid[m])
            if len(cs) >= 2:
                rows.append((team, ks[i], gid[i], tuple(cs)))
    return pd.DataFrame(rows, columns=["team", "k", "starter", "choices"])


@functools.lru_cache(maxsize=None)
def share_model(V: int, window: int = 8) -> np.ndarray:
    """Conditional-logit weights b for the nightly choice of starter among
    the goalies available, fitted on box-score seasons V-window..V-1 (box
    scores end with 2024's first weeks). Features are each goalie's
    PRESEASON projection for that season (log workload, talent z, age>=35),
    so the fit learns how coaches split the net given what was known in
    September; injuries are handled separately by the availability chain."""
    from scipy.optimize import minimize
    Xs, ys, grp = [], [], []
    gi = 0
    for s in range(max(V - window, 2011), V):
        if s in (2013, 2020, 2021) or s > 2024:
            continue
        ch = _start_choices(s)
        if not len(ch):
            continue
        ids = sorted({g for c in ch.choices for g in c})
        pj = project_goalies(s, ids=ids).set_index("player_id")
        pr_s = talent_prior(s)
        for r in ch.itertuples():
            sub = pj.loc[list(r.choices)].reset_index()
            for g, x in zip(sub.player_id, _share_features(sub, pr_s)):
                Xs.append(x)
                ys.append(1.0 if g == r.starter else 0.0)
                grp.append(gi)
            gi += 1
    if gi < 200:
        # No box-score season before V (V <= 2012): a fixed prior split --
        # workload 2.0, No. 1 +0.5 per unit, i.e. ~0.66/0.34 for a
        # 0.65/0.30-workload duo. Used only for 2011-12 history rows.
        return FALLBACK_SHARE_B.copy()
    X = np.array(Xs)
    y = np.array(ys)
    grp = np.array(grp)

    def nll(b):
        u = X @ b
        mx = np.zeros(gi)
        np.maximum.at(mx, grp, u)
        e = np.exp(u - mx[grp])
        den = np.bincount(grp, weights=e, minlength=gi)
        return -float((y * (u - mx[grp] - np.log(den[grp]))).sum())
    b = minimize(nll, np.array([1.0, 0.3, 0.0, 0.5]), method="BFGS").x
    return b * share_sharpness(V, tuple(b))


@functools.lru_cache(maxsize=None)
def share_sharpness(V: int, b: tuple, window: int = 8) -> float:
    """One multiplier on the choice-model coefficients so the projected
    season start share of each team's No. 1 goalie (injuries included)
    matches what opening-roster No. 1s actually got, seasons V-8..V-1.

    The nightly choice sets can only contain a backup when he starts nearby,
    which flattens the fitted split; this restores the season-level level
    without touching the relative ordering."""
    from hattrick import deploy as DP
    gg = D.goalie_games()
    sch = D.fr_schedule()[["game_id", "home", "away"]]
    gg = gg.merge(sch, on="game_id")
    gg["team"] = np.where(gg.home_away.str.lower() == "home", gg.home, gg.away)
    emp, sets = [], []
    for s in range(max(V - window, 2011), V):
        if s in (2013, 2020, 2021) or s > 2024:
            continue
        try:
            r = D.opening_rosters(s)
        except FileNotFoundError:
            continue
        r = r[r.grp == "G"]
        pj = project_goalies(s, ids=r.player_id.unique())
        st = gg[gg.season_end == s].groupby(["team", "goalie_id"]).size()
        ng = gg[gg.season_end == s].groupby("team").size()
        for team, t in r.merge(pj, on="player_id").groupby("team"):
            if len(t) < 2 or team not in ng.index:
                continue
            top = t.sort_values("workload", ascending=False).player_id.iloc[0]
            emp.append(st.get((team, top), 0) / ng[team])
            q = DP.health(s, t.player_id.to_numpy(), t.age.to_numpy())
            sets.append((t.reset_index(drop=True), q, s))
    target = float(np.mean(emp))
    best = (1.0, 9.0)
    for lam in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        sh = []
        for t, q, s in sets:
            tt = t.assign(games_out=0.0, p_present=1.0)
            stt, _ = expected_starts(tt, 82, np.array(b) * lam, talent_prior(s), q)
            sh.append(stt[np.argmax(tt.workload.to_numpy())] / 82)
        gap = abs(np.mean(sh) - target)
        if gap < best[1]:
            best = (lam, gap)
    return best[0]


def expected_starts(team_goalies: pd.DataFrame, games: int, b: np.ndarray, pr: dict,
                    q: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact expected starts: per block of games with constant availability,
    enumerate which goalies are healthy (2^n states); the healthy split the
    start in proportion to w; nobody healthy -> a call-up."""
    X = _share_features(team_goalies, pr)
    w = np.exp(X @ b)
    n = len(w)
    go = team_goalies.games_out.to_numpy(float)
    pres = team_goalies.p_present.to_numpy(float)
    cuts = sorted({0, games, *[int(g) for g in go if 0 < g < games]})
    starts = np.zeros(n)
    rep = 0.0
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        h = np.where(go > a_, 0.0, (1 - q)) * pres
        for state in itertools.product((0, 1), repeat=n):
            s = np.array(state, bool)
            p = np.prod(np.where(s, h, 1 - h))
            if p == 0:
                continue
            if s.any():
                starts[s] += p * (b_ - a_) * w[s] / w[s].sum()
            else:
                rep += p * (b_ - a_)
    return starts, rep


def deploy_goalies(V: int, roster: pd.DataFrame, games: int,
                   games_out: pd.Series | None = None,
                   p_present: pd.Series | None = None,
                   q_scale: float = 1.0) -> pd.DataFrame:
    """Expected starts, shots, goals against and GSAx per rostered goalie.

    roster: player_id, team (goalies only)."""
    from hattrick import deploy as DP
    b = share_model(V)
    pr = talent_prior(V)
    lg = league_goalie(V)
    pj = project_goalies(V, ids=roster.player_id.unique(), extra=roster)
    d = roster[["player_id", "team"]].drop_duplicates("player_id").merge(pj, on="player_id")
    d["games_out"] = d.player_id.map(games_out if games_out is not None else {}).fillna(0.0)
    d["p_present"] = d.player_id.map(p_present if p_present is not None else {}).fillna(1.0)
    d["q"] = DP.health(V, d.player_id.to_numpy(), d.age.to_numpy(), q_scale)
    out = []
    for team, t in d.groupby("team"):
        t = t.reset_index(drop=True)
        st, rep = expected_starts(t, games, b, pr, t.q.to_numpy())
        t["starts"] = st
        t["callup_starts"] = rep
        out.append(t)
    d = pd.concat(out, ignore_index=True)
    d["start_share"] = d.starts / games
    d["gp"] = d.starts * 1.04                  # relief appearances (~4% extra GP)
    d["sa"] = d.starts * lg["sa_per_game"]
    d["fa"] = d.sa * lg["fa_per_sa"]
    d["gsax"] = d.gsax_per_fa * d.fa
    d["ga"] = d.sa * (1 - d.sv_pct)
    d["wins_placeholder"] = d.starts * 0.5     # team layer supplies real win rates
    return d


def simulate_goalies(dep: pd.DataFrame, V: int, games: int, S: int = 2000,
                     spell_len: float = 5.2, seed: int = C.SEED) -> pd.DataFrame:
    """p10/p50/p90 of starts, SA, GA, SV% and GSAx per goalie: injury
    spells, holdout presence and games-out drawn per season; talent drawn
    from its posterior; goals against binomial given the shots."""
    rng = np.random.default_rng(seed)
    b = share_model(V)
    pr = talent_prior(V)
    lg = league_goalie(V)
    res = []
    for team, t in dep.groupby("team"):
        t = t.reset_index(drop=True)
        n = len(t)
        w = np.exp(_share_features(t, pr) @ b)
        q = np.clip(t.q.to_numpy(), 1e-4, 0.9)
        p_hi, p_ih = q / ((1 - q) * spell_len), 1 / spell_len
        st = rng.random((S, n)) >= q
        pres = rng.random((S, n)) < t.p_present.to_numpy()
        starts = np.zeros((S, n))
        for g in range(games):
            u = rng.random((S, n))
            st = np.where(st, u >= p_hi, u < p_ih)
            av = st & pres & (g >= t.games_out.to_numpy())[None, :]
            ww = av * w[None, :]
            tot = ww.sum(1, keepdims=True)
            pick = rng.random((S, 1)) * np.where(tot > 0, tot, 1)
            idx = (np.cumsum(ww, 1) < pick).sum(1)
            ok = tot[:, 0] > 0
            starts[np.arange(S)[ok], idx[ok]] += 1
        rate = t.gsax_per_fa.to_numpy()[None, :] + t.gsax_per_fa_sd.to_numpy()[None, :] \
            * rng.standard_normal((S, n))
        sa = rng.poisson(starts * lg["sa_per_game"])
        sv = np.clip(lg["sv"] + rate * lg["fa_per_sa"], 0.8, 0.97)
        ga = rng.binomial(sa, 1 - sv)
        # goals saved vs a league-average goalie on the same shots (equals
        # GSAx against league-calibrated xG at league-average shot quality)
        gsax = sa * (1 - lg["sv"]) - ga
        o = pd.DataFrame({"player_id": t.player_id})
        for name, x in (("starts", starts), ("sa", sa), ("ga", ga), ("gsax", gsax)):
            for qq in (10, 50, 90):
                o[f"{name}_p{qq}"] = np.percentile(x, qq, axis=0)
        svp = np.where(sa > 0, 1 - ga / np.maximum(sa, 1), np.nan)
        for qq in (10, 50, 90):
            o[f"sv_pct_p{qq}"] = np.nanpercentile(np.where(sa >= 300, svp, np.nan), qq, axis=0) \
                if (sa >= 300).any() else np.nan
        res.append(o)
    return pd.concat(res, ignore_index=True)


# ---------------------------------------------------------------------------
# 2026-27
# ---------------------------------------------------------------------------
def goalie_status_2027(roster: pd.DataFrame, games: int) -> tuple[pd.Series, pd.Series, dict]:
    """Games out and presence for 2026-27 goalies: holdout mixture first,
    then the skater rules in hattrick.deploy (research file, overrides,
    return dates, DailyFaceoff statuses, LTIR)."""
    from hattrick import deploy as DP
    go, rng_, _ = DP.games_out_2027(roster, games, with_range=True)
    pres = pd.Series(1.0, index=go.index)
    for pid, h in HOLDOUTS.items():
        if pid in go.index:
            go[pid] = h["games_out"]
            pres[pid] = h["p_present"]
    return go, pres, rng_


def run_2027(out_dir, games: int = 84, sims: int = 4000) -> pd.DataFrame:
    from hattrick import players as PL
    r = PL.roster_2027()          # opening rosters + injured non-roster players
    g = r[r.grp == "G"][["player_id", "team", "name", "birth"]]
    go, pres, _ = goalie_status_2027(g, games)
    dep = deploy_goalies(C.TARGET_SEASON, g, games, go, pres)
    q = simulate_goalies(dep, C.TARGET_SEASON, games, S=sims)
    dep = dep.merge(q, on="player_id").merge(g[["player_id", "name"]], on="player_id")
    dep["holdout_p_present"] = dep.p_present
    cols = ["player_id", "name", "team", "age", "has_hist", "workload", "gsax_per_fa",
            "gsax_per_fa_sd", "gsax60", "gsax60_sd", "sv_pct", "q", "games_out",
            "p_present", "start_share", "starts", "starts_p10", "starts_p90", "gp",
            "callup_starts", "sa", "sa_p10", "sa_p90", "ga", "ga_p10", "ga_p90",
            "gsax", "gsax_p10", "gsax_p90", "sv_pct_p10", "sv_pct_p90", "wins_placeholder"]
    dep = dep[cols].sort_values(["team", "starts"], ascending=[True, False])
    out_dir.mkdir(parents=True, exist_ok=True)
    dep.round(5).to_csv(out_dir / "goalie_rates_2027.csv", index=False)
    D.write_json({"share_model_b": share_model(C.TARGET_SEASON).tolist(),
                  "talent_prior": talent_prior(C.TARGET_SEASON),
                  "age_curve": age_curve(C.TARGET_SEASON).tolist(),
                  "goalie_decay": GOALIE_DECAY, "holdouts": {str(k): v for k, v in HOLDOUTS.items()},
                  "league": league_goalie(C.TARGET_SEASON)}, PARAMS_PATH)
    return dep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=C.TARGET_SEASON)
    a = ap.parse_args()
    d = run_2027(C.OUT / f"freeze_{a.season}")
    print(d.head(40).to_string())
