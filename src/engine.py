"""Rating engine + season simulator for NHL projections.

Components:
  run_elo            MOV-adjusted Elo over all games (regular + playoffs), season carryover
  fit_logistic       tiny IRLS logistic regression (numpy only)
  fit_outcome        game-outcome submodels: P(reach OT), P(reg win|d), P(OT win|d), SO share
  fit_xg_beta        maps xG share onto the Elo scale (cross-sectional OLS)
  project_ratings    preseason blended + shrunk ratings (one- or two-season ahead)
  analytic_xpts      expected points for a schedule without simulation (fast tuning)
  simulate_season    vectorized Monte Carlo: standings, tiebreakers, playoffs to the Cup
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
INIT, MEAN = 1505.0, 1505.0
EXPANSION_INIT = 1470.0

DIVISIONS_CURRENT = {
    "Atlantic": ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"],
    "Metropolitan": ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
    "Central": ["CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"],
    "Pacific": ["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"],
}
# 2014-2021 alignment (pre-Seattle): ARI/UTA franchise in Pacific, no SEA
DIVISIONS_2014_2020 = {
    "Atlantic": ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"],
    "Metropolitan": ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
    "Central": ["CHI", "COL", "DAL", "MIN", "NSH", "STL", "WPG"],
    "Pacific": ["ANA", "CGY", "EDM", "LAK", "SJS", "UTA", "VAN", "VGK"],
}
CONFS = {"East": ["Atlantic", "Metropolitan"], "West": ["Central", "Pacific"]}


def divisions_for(season_end: int) -> dict:
    return DIVISIONS_CURRENT if season_end >= 2022 else DIVISIONS_2014_2020


def load():
    g = pd.read_csv(PROJ / "data/processed/games.csv", keep_default_na=False,
                    parse_dates=["date"])
    ts = pd.read_csv(PROJ / "data/processed/team_seasons.csv")
    return g, ts


# ---------------------------------------------------------------- Elo
def run_elo(g: pd.DataFrame, K=8.0, H=30.0, phi_s=0.7, expansion_init=EXPANSION_INIT):
    """Chronological Elo. Returns (preds, end_ratings, pre_ratings).

    end_ratings[s][team]: rating after season s (incl. playoffs), before carryover.
    pre_ratings[s][team]: rating entering season s (after carryover).
    preds: one row per game with pregame ratings (for fitting/evaluation).
    """
    ratings: dict[str, float] = {}
    cur_season = None
    rows = []
    end_ratings: dict[int, dict] = {}
    pre_ratings: dict[int, dict] = {}
    for game in g.itertuples(index=False):
        s = game.season_end
        if cur_season is None:
            cur_season = s
        if s != cur_season:
            end_ratings[cur_season] = dict(ratings)
            for t in ratings:
                ratings[t] = MEAN + phi_s * (ratings[t] - MEAN)
            pre_ratings[s] = dict(ratings)
            cur_season = s
        for t in (game.home, game.away):
            if t not in ratings:
                ratings[t] = INIT if s <= 2006 else expansion_init
        rh, ra = ratings[game.home], ratings[game.away]
        neutral = (s == 2020 and game.game_type == "P")
        h = 0.0 if neutral else H
        d = rh + h - ra
        e_home = 1.0 / (1.0 + 10 ** (-d / 400.0))
        home_win = game.home_g > game.away_g
        gd = abs(game.home_g - game.away_g)
        winner_d = d if home_win else -d
        mov = np.log(gd + 1.0) * (2.2 / (2.2 + 0.001 * winner_d))
        delta = K * mov * ((1.0 if home_win else 0.0) - e_home)
        ratings[game.home] = rh + delta
        ratings[game.away] = ra - delta
        rows.append((s, game.game_type, game.home, game.away, rh, ra, d - h,
                     e_home, home_win, game.went_ot, game.went_so, neutral))
    end_ratings[cur_season] = dict(ratings)
    preds = pd.DataFrame(rows, columns=[
        "season_end", "game_type", "home", "away", "rh", "ra", "d_ex_hfa",
        "e_home", "home_win", "went_ot", "went_so", "neutral"])
    return preds, end_ratings, pre_ratings


# ------------------------------------------------- logistic (IRLS, numpy only)
def fit_logistic(X: np.ndarray, y: np.ndarray, iters=25):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        w = np.clip(p * (1 - p), 1e-6, None)
        z = X @ b + (y - p) / w
        WX = X * w[:, None]
        b_new = np.linalg.solve(X.T @ WX, X.T @ (w * z))
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new
            break
        b = b_new
    return b  # [intercept, slopes...]


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------- outcome submodels
def fit_outcome(preds: pd.DataFrame, seasons: list[int]) -> dict:
    """Fit game-outcome models on regular-season games in `seasons` (causal set).

    Eras matter for OT (3v3 from 2016): OT params are fit on the 3v3-era subset when
    the causal set includes any (for projecting modern seasons); else on all.
    """
    sub = preds[(preds.game_type == "R") & preds.season_end.isin(seasons)].copy()
    past_reg = sub.went_ot | sub.went_so
    d = sub.d_ex_hfa.to_numpy() / 100.0  # per-100-Elo units for conditioning

    # P(reach OT | |d|)
    b_pot = fit_logistic(np.abs(d)[:, None], past_reg.to_numpy().astype(float))

    # regulation winner
    rg = sub[~past_reg]
    b_reg = fit_logistic((rg.d_ex_hfa.to_numpy() / 100.0)[:, None],
                         (rg.home_win).to_numpy().astype(float))

    # OT/SO winner + SO share (3v3-era subset if available)
    ot_pool_seasons = [s for s in seasons if s >= 2016] or seasons
    ot = preds[(preds.game_type == "R") & preds.season_end.isin(ot_pool_seasons)]
    ot = ot[ot.went_ot | ot.went_so]
    b_ot = fit_logistic((ot.d_ex_hfa.to_numpy() / 100.0)[:, None],
                        (ot.home_win).to_numpy().astype(float))
    so_share = float(ot.went_so.mean())

    # playoff game model (all playoff games in causal window, non-neutral)
    pl = preds[(preds.game_type == "P") & preds.season_end.isin(seasons) & ~preds.neutral]
    if len(pl) > 200:
        b_pl = fit_logistic((pl.d_ex_hfa.to_numpy() / 100.0)[:, None],
                            (pl.home_win).to_numpy().astype(float))
    else:
        b_pl = b_reg
    return {"b_pot": b_pot, "b_reg": b_reg, "b_ot": b_ot, "b_pl": b_pl,
            "so_share": so_share}


# ------------------------------------------------------------- xG on Elo scale
def fit_xg_beta(end_ratings: dict, ts: pd.DataFrame, seasons: list[int]) -> float:
    """OLS through the mean: (end Elo - MEAN) ~ beta * (xg_pct_all - 0.5)."""
    xs, ys = [], []
    for s in seasons:
        t = ts[(ts.season_end == s)].dropna(subset=["xg_pct_all"])
        for row in t.itertuples(index=False):
            if row.team in end_ratings.get(s, {}):
                xs.append(row.xg_pct_all - 0.5)
                ys.append(end_ratings[s][row.team] - MEAN)
    xs, ys = np.array(xs), np.array(ys)
    return float((xs @ ys) / (xs @ xs))


def project_ratings(end_ratings: dict, ts: pd.DataFrame, beta: float,
                    from_season: int, w: float, phi: float) -> dict:
    """Preseason ratings for a future season from end-of-`from_season` state."""
    t = ts[ts.season_end == from_season].set_index("team")
    out = {}
    for team, r_elo in end_ratings[from_season].items():
        xg = t.loc[team, "xg_pct_all"] if team in t.index else np.nan
        r_xg = MEAN + beta * (xg - 0.5) if pd.notna(xg) else r_elo
        blend = w * (r_elo - MEAN) + (1 - w) * (r_xg - MEAN)
        out[team] = MEAN + phi * blend
    return out


def fill_missing(ratings: dict, teams, val=EXPANSION_INIT) -> dict:
    """Complete a ratings dict for teams debuting in the projected season."""
    out = dict(ratings)
    for t in teams:
        out.setdefault(t, val)
    return out


# ------------------------------------------------------------------ schedules
def actual_schedule(g: pd.DataFrame, season_end: int) -> pd.DataFrame:
    return g[(g.season_end == season_end) & (g.game_type == "R")][["home", "away"]].copy()


def synthetic_schedule_84(divisions: dict, rng: np.random.Generator) -> pd.DataFrame:
    """New-CBA 84-game matrix: 4x each division rival (2H/2A), 3x conference
    non-division (2H/1A for half the opponents, 1H/2A for the other half),
    2x inter-conference (1H/1A). 42 home / 42 away each."""
    conf_of = {}
    for conf, divs in CONFS.items():
        for dv in divs:
            for t in divisions[dv]:
                conf_of[t] = conf
    games = []
    teams = [t for dv in divisions.values() for t in dv]
    # division: 2 home each way
    for dv, ts_ in divisions.items():
        for i, a in enumerate(ts_):
            for b in ts_[i + 1:]:
                games += [(a, b), (a, b), (b, a), (b, a)]
    # conference non-division: 3 games; balanced extra-home via 4-regular 0/1 matrix
    for conf, divs in CONFS.items():
        d1, d2 = divisions[divs[0]], divisions[divs[1]]
        base = np.array([[(jj - ii) % 8 < 4 for jj in range(8)] for ii in range(8)], dtype=int)
        mat = base[np.ix_(rng.permutation(8), rng.permutation(8))]  # row & col sums stay 4
        for i, a in enumerate(d1):
            for j, b in enumerate(d2):
                games += [(a, b), (b, a)]
                games.append((a, b) if mat[i, j] else (b, a))
    # inter-conference: home and home
    east = [t for dv in CONFS["East"] for t in divisions[dv]]
    west = [t for dv in CONFS["West"] for t in divisions[dv]]
    for a in east:
        for b in west:
            games += [(a, b), (b, a)]
    df = pd.DataFrame(games, columns=["home", "away"])
    counts = pd.concat([df.home, df.away]).value_counts()
    assert (counts == 84).all(), f"bad schedule totals:\n{counts[counts != 84]}"
    home_counts = df.home.value_counts()
    assert (home_counts == 42).all(), "home/away not exactly 42/42"
    return df


# ------------------------------------------------------------------ analytics
def game_probs(d_elo: np.ndarray, om: dict):
    """d_elo: home-away rating diff EXCLUDING home ice (it lives in intercepts).
    Returns (p_ot, p_home_reg, p_home_ot) arrays."""
    d = d_elo / 100.0
    p_ot = logistic(om["b_pot"][0] + om["b_pot"][1] * np.abs(d))
    p_reg = logistic(om["b_reg"][0] + om["b_reg"][1] * d)
    p_otw = logistic(om["b_ot"][0] + om["b_ot"][1] * d)
    return p_ot, p_reg, p_otw


def analytic_xpts(ratings: dict, sched: pd.DataFrame, om: dict) -> pd.Series:
    missing = set(sched.home) | set(sched.away) - set(ratings)
    assert set(sched.home) <= set(ratings) and set(sched.away) <= set(ratings), \
        f"schedule teams missing from ratings: {missing} (use fill_missing)"
    teams = sorted(ratings)
    idx = {t: i for i, t in enumerate(teams)}
    r = np.array([ratings[t] for t in teams])
    hi = sched.home.map(idx).to_numpy()
    ai = sched.away.map(idx).to_numpy()
    p_ot, p_reg, p_otw = game_probs(r[hi] - r[ai], om)
    xp_home = (1 - p_ot) * 2 * p_reg + p_ot * (2 * p_otw + (1 - p_otw))
    xp_away = (1 - p_ot) * 2 * (1 - p_reg) + p_ot * (2 * (1 - p_otw) + p_otw)
    out = np.zeros(len(teams))
    np.add.at(out, hi, xp_home)
    np.add.at(out, ai, xp_away)
    return pd.Series(out, index=teams)


# ------------------------------------------------------------------ simulator
def _series_win_prob(p_games: np.ndarray) -> float:
    """P(higher seed wins best-of-7), p_games: win prob for higher seed, games 1..7
    (2-2-1-1-1 venues already applied)."""
    # DP over (games_played, wins); stop at 4
    probs = {(0, 0): 1.0}
    win = 0.0
    for gm in range(7):
        nxt = {}
        for (g_, w_), pr in probs.items():
            if g_ != gm:
                nxt[(g_, w_)] = nxt.get((g_, w_), 0) + pr
                continue
            p = p_games[gm]
            for won in (1, 0):
                w2 = w_ + won
                l2 = gm + 1 - w2
                pr2 = pr * (p if won else 1 - p)
                if w2 == 4:
                    win += pr2
                elif l2 == 4:
                    pass
                else:
                    nxt[(gm + 1, w2)] = nxt.get((gm + 1, w2), 0) + pr2
        probs = nxt
    return win


def _playoff_game_p(d_elo: float, om: dict) -> float:
    return float(logistic(om["b_pl"][0] + om["b_pl"][1] * d_elo / 100.0))


HOME_PATTERN = np.array([1, 1, 0, 0, 1, 0, 1])  # higher seed home games


def _series(higher: int, lower: int, strengths: np.ndarray, om: dict,
            rng: np.random.Generator) -> int:
    d = strengths[higher] - strengths[lower]
    p_home = _playoff_game_p(d, om)          # higher seed at home
    p_away = 1.0 - _playoff_game_p(-d, om)   # higher seed on the road
    p_games = np.where(HOME_PATTERN == 1, p_home, p_away)
    p_series = _series_win_prob(p_games)
    return higher if rng.random() < p_series else lower


def simulate_season(ratings_mean: dict, sigma, sched: pd.DataFrame, om: dict,
                    divisions: dict, n_sims: int, rng: np.random.Generator,
                    playoffs: bool = True, chunk: int = 2000, extra_noise=None):
    """sigma: scalar Elo strength noise, or dict team->sigma (heteroscedastic).
    extra_noise: optional callable(m, rng) -> (m, n_teams) Elo noise aligned to sorted
    team order (e.g. availability draws); added to the per-sim strength draw."""
    teams = sorted(ratings_mean)
    if isinstance(sigma, dict):
        sigma = np.array([sigma[t] for t in teams], dtype=float)
        assert sigma.shape == (len(teams),) and (sigma >= 0).all()
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    div_of = {t: dv for dv, ts_ in divisions.items() for t in ts_}
    conf_of = {t: c for c, dvs in CONFS.items() for dv in dvs for t in divisions.get(dv, [])}
    mu = np.array([ratings_mean[t] for t in teams])
    hi = sched.home.map(idx).to_numpy()
    ai = sched.away.map(idx).to_numpy()

    pts_all = np.zeros((n_sims, n), dtype=np.float32)
    rw_all = np.zeros((n_sims, n), dtype=np.float32)
    row_all = np.zeros((n_sims, n), dtype=np.float32)
    made_po = np.zeros((n_sims, n), dtype=bool)
    won_div = np.zeros((n_sims, n), dtype=bool)
    won_conf = np.zeros((n_sims, n), dtype=bool)
    won_cup = np.zeros((n_sims, n), dtype=bool)
    strengths_store = np.zeros((n_sims, n), dtype=np.float32)

    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        S = mu[None, :] + sigma * rng.standard_normal((m, n))
        if extra_noise is not None:
            S = S + extra_noise(m, rng)
        strengths_store[done:done + m] = S
        d = S[:, hi] - S[:, ai]
        p_ot, p_reg, p_otw = game_probs(d, om)
        u1, u2, u3 = rng.random((3, m, len(hi)))
        is_ot = u1 < p_ot
        home_w = np.where(is_ot, u2 < p_otw, u2 < p_reg)
        is_so = is_ot & (u3 < om["so_share"])
        hp = np.where(home_w, 2, np.where(is_ot, 1, 0)).astype(np.float32)
        ap = np.where(~home_w, 2, np.where(is_ot, 1, 0)).astype(np.float32)
        base = (np.arange(m, dtype=np.int64) * n)[:, None]
        hflat = (base + hi[None, :]).ravel()
        aflat = (base + ai[None, :]).ravel()

        def scatter(vals_h, vals_a):
            arr = np.zeros(m * n, np.float32)
            np.add.at(arr, hflat, vals_h.ravel().astype(np.float32))
            np.add.at(arr, aflat, vals_a.ravel().astype(np.float32))
            return arr.reshape(m, n)

        pts_all[done:done + m] = scatter(hp, ap)
        rw_all[done:done + m] = scatter(home_w & ~is_ot, ~home_w & ~is_ot)
        row_all[done:done + m] = scatter(home_w & ~is_so, ~home_w & ~is_so)
        done += m

    # standings, playoffs (python loop over sims)
    conf_names = list(CONFS)
    if playoffs:
        missing = {t for ts_ in divisions.values() for t in ts_} - set(idx)
        assert not missing, f"division teams missing from ratings: {missing}"
    div_teams = {dv: [idx[t] for t in ts_ if t in idx] for dv, ts_ in divisions.items()}
    rng_tb = rng
    for k in range(n_sims if playoffs else 0):
        key = pts_all[k] * 1e8 + rw_all[k] * 1e4 + row_all[k] + rng_tb.random(n) * 0.5
        champs = {}
        for conf in conf_names:
            seeds = []  # (bracket) per division
            div_ranks = {}
            for dv in CONFS[conf]:
                order = sorted(div_teams[dv], key=lambda i: -key[i])
                div_ranks[dv] = order
            winners = sorted([div_ranks[dv][0] for dv in CONFS[conf]], key=lambda i: -key[i])
            won_div[k, winners[0]] = won_div[k, winners[1]] = True
            rest = [i for dv in CONFS[conf] for i in div_ranks[dv][1:]]
            rest_sorted = sorted(rest, key=lambda i: -key[i])
            d23 = {dv: div_ranks[dv][1:3] for dv in CONFS[conf]}
            wc = [i for i in rest_sorted if all(i not in d23[dv] for dv in CONFS[conf])][:2]
            qual = [winners[0], winners[1]] + [i for dv in CONFS[conf] for i in d23[dv]] + wc
            made_po[k, qual] = True
            # brackets: best winner vs wc2, its div 2v3; other winner vs wc1, its div 2v3
            b = []
            for wi, wcx in ((0, 1), (1, 0)):
                wteam = winners[wi]
                dv = div_of[teams[wteam]]
                b.append(((wteam, wc[wcx]), tuple(d23[dv])))
            semi = []
            for (m1, m2) in b:
                w1 = _series(*sorted(m1, key=lambda i: -key[i]), strengths=strengths_store[k], om=om, rng=rng_tb)
                w2 = _series(*sorted(m2, key=lambda i: -key[i]), strengths=strengths_store[k], om=om, rng=rng_tb)
                semi.append((w1, w2))
            finalists = [_series(*sorted(s, key=lambda i: -key[i]), strengths=strengths_store[k], om=om, rng=rng_tb)
                         for s in semi]
            champ = _series(*sorted(finalists, key=lambda i: -key[i]), strengths=strengths_store[k], om=om, rng=rng_tb)
            won_conf[k, champ] = True
            champs[conf] = champ
        if playoffs:
            cup = _series(*sorted(champs.values(), key=lambda i: -key[i]),
                          strengths=strengths_store[k], om=om, rng=rng_tb)
            won_cup[k, cup] = True

    return {
        "teams": teams, "pts": pts_all, "rw": rw_all, "row": row_all,
        "made_po": made_po, "won_div": won_div, "won_conf": won_conf, "won_cup": won_cup,
    }
