"""Availability and deployment: who dresses, how often, and for how long.

A team's season is a set of budgets that must be spent exactly:

* **Dressed games.** Every game 12 forwards and 6 defencemen dress. A player
  dresses when he is healthy AND ranks inside his position's top 12 / top 6
  among the healthy players on the roster (coaches do not scratch a healthy
  first-liner; the 13th forward plays when somebody above him is hurt).
  Expected games therefore come from a Poisson-binomial recursion over the
  depth chart (exact, no simulation noise), averaged over noisy orderings
  because coaches' depth charts are not our projected-TOI ranking. Games
  nobody on the roster can fill go to call-ups (the replacement pool).
* **Ice time.** Each game the dressed skaters' minutes by situation sum to the
  team's minutes in that situation times the number of skaters of that
  position on the ice (e.g. 5v5 forwards: ~49.5 min x 3). Projected minutes
  per game are therefore rescaled within team, position and situation so the
  season budget is met, after capping individual loads at the observed
  ceilings (F 23.5, D 28 min/game) and handing the excess to teammates.
* **Goals.** Team goals are then exactly the sum of its skaters' projected
  goals (the team layer reads them from here).

Health: each player has a per-game probability of being out injured,
estimated by empirical Bayes from his own history of injury-like absence
spells (hattrick.data.absences: runs of >=3 missed team games) shrunk toward
an age-dependent league prior; the prior weight is set from the observed
year-to-year persistence of injury rates. Current injuries (2026-27) remove
games up front: researched overrides win, then the listed return date, then a
DailyFaceoff status; LTIR and suspended-not-reporting players get none.

Season totals come with means (from the exact recursion) and p10/p90 bands
from a Monte Carlo that also draws injury spells, talent uncertainty and
Poisson scoring noise.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import data as D

N_DRESS = {"F": 12, "D": 6}
TOI_CAP = {"F": 23.5, "D": 28.0}        # min/game ceilings observed 2009-26
SITS = ("ev", "pp", "sh", "oth")
# DailyFaceoff statuses with no return date: expected games out at the start.
# ir:dtd -> 1 (day-to-day); ir:out -> 5 (out, no timeline: median short-term
# IR stint); ir:ir -> 10 (placed on IR, minimum 7 days / ~3 games, typically
# longer). Chosen as round medians, not fitted (no labelled history).
DF_STATUS_GAMES = {"ir:dtd": 1.0, "ir:out": 5.0, "ir:ir": 10.0}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _absence_panel() -> pd.DataFrame:
    a = D.absences()
    g = a.groupby(["player_id", "season_end"], as_index=False)[
        ["dressed", "window_games", "injury_spell_games"]].sum()
    return g


@functools.lru_cache(maxsize=None)
def health_prior(V: int) -> dict:
    """League injury-spell rate by age and its year-to-year persistence.

    Fitted on absence seasons < V (2012+): rate = injury-spell games / window
    games, linear in age (clipped 20..36). The EB prior weight k (in games)
    comes from the lag-1 correlation r of per-season rates among players
    with >=60-game windows: reliability n/(n+k) = r at n ~ 82.
    """
    a = _absence_panel()
    a = a[(a.season_end < V) & (a.season_end >= V - 8)]
    if len(a) < 500:                              # before absences exist
        return {"b0": 0.12, "b1": 0.002, "k": 350.0, "fallback": True}
    b = D.bios().drop_duplicates("player_id").set_index("player_id").birth
    age = D.age_on(a.player_id.map(b), 2000) + (a.season_end - 2000)
    age = age.fillna(27.0).clip(20, 36)
    w = a.window_games.to_numpy(float)
    y = (a.injury_spell_games / a.window_games).to_numpy(float)
    X = np.column_stack([np.ones(len(a)), age.to_numpy() - 27.0])
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    big = a[a.window_games >= 60].assign(r=lambda d: d.injury_spell_games / d.window_games)
    pair = big.merge(big.assign(season_end=big.season_end - 1),
                     on=["player_id", "season_end"], suffixes=("", "_n"))
    r = float(np.clip(np.corrcoef(pair.r, pair.r_n)[0, 1], 0.05, 0.9)) if len(pair) > 50 else 0.2
    k = 82.0 * (1 - r) / r
    return {"b0": float(beta[0]), "b1": float(beta[1]), "k": k, "r": r, "fallback": False}


def health(V: int, ids, age: np.ndarray, q_scale: float = 1.0) -> np.ndarray:
    """Per-game probability of being out injured in V, per player.

    q = (sum_w injury games + k q0(age)) / (sum_w window games + k), with
    seasons V-1..V-4 weighted 1, .8, .6, .4 (injury proneness is persistent
    but weakly so). q_scale calibrates the absence-spell definition (which is
    censored at the edges of a player's season) to real missed games.
    """
    pr = health_prior(V)
    q0 = np.clip(pr["b0"] + pr["b1"] * (np.clip(age, 20, 36) - 27.0), 0.02, 0.4)
    a = _absence_panel()
    a = a[(a.season_end < V) & (a.season_end >= V - 4)]
    wt = {V - 1: 1.0, V - 2: 0.8, V - 3: 0.6, V - 4: 0.4}
    a = a.assign(w=a.season_end.map(wt))
    agg = a.assign(ig=a.injury_spell_games * a.w, wg=a.window_games * a.w) \
        .groupby("player_id")[["ig", "wg"]].sum()
    ids = pd.Index(ids)
    ig = agg.ig.reindex(ids).fillna(0).to_numpy()
    wg = agg.wg.reindex(ids).fillna(0).to_numpy()
    q = (ig + pr["k"] * q0) / (wg + pr["k"])
    return np.clip(q * q_scale, 0.0, 0.6)


# ---------------------------------------------------------------------------
# Dressing competition (exact Poisson-binomial over the depth chart)
# ---------------------------------------------------------------------------
def dress_probs(score: np.ndarray, h: np.ndarray, n_slots: int,
                noise_sd: float, draws: int = 64, seed: int = 0):
    """P(player dressed in a game) and expected call-ups per game.

    score: depth-chart score (projected min/game); h: P(healthy/available)
    per player for this block of games. Each draw perturbs the scores with
    N(0, noise_sd) and walks down the resulting depth chart keeping the
    distribution of how many dressed slots are already filled.
    """
    n = len(score)
    if n == 0:
        return np.zeros(0), float(n_slots)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((draws, n)) * noise_sd
    order = np.argsort(-(score[None, :] + eps), axis=1)          # (R, n)
    R = draws
    state = np.zeros((R, n_slots + 1))
    state[:, 0] = 1.0
    pd_ = np.zeros((R, n))
    rows = np.arange(R)
    for t in range(n):
        who = order[:, t]
        ht = h[who]
        free = 1.0 - state[:, n_slots]
        pd_[rows, who] = ht * free
        new = state * (1 - ht)[:, None]
        new[:, 1:n_slots] += state[:, 0:n_slots - 1] * ht[:, None]
        new[:, n_slots] = state[:, n_slots] + state[:, n_slots - 1] * ht
        state = new
    filled = (state * np.arange(n_slots + 1)[None, :]).sum(1)
    return pd_.mean(0), float(n_slots - filled.mean())


# ---------------------------------------------------------------------------
# Team deployment (means)
# ---------------------------------------------------------------------------
def games_out_2027(roster: pd.DataFrame, games: int) -> pd.Series:
    """Games each 2026-27 player misses at the start of the season.

    Priority: manual status (LTIR / suspended-not-reporting: the whole
    season) > researched override_games_out > MoneyPuck injured flag with a
    return date (team games scheduled before that date) > DailyFaceoff status
    without a date (DF_STATUS_GAMES) > 0.
    """
    av = D.availability_raw_2027().drop_duplicates("player_id").set_index("player_id")
    sched = D.schedule_2027()
    out = pd.Series(0.0, index=roster.player_id.to_numpy())
    for pid, team in zip(roster.player_id, roster.team):
        if pid not in av.index:
            continue
        r = av.loc[pid]
        ms = str(r.get("manual_status") or "")
        if ms in ("LTIR", "SUSPENDED_NOT_REPORTING"):
            out[pid] = float(games)
            continue
        if pd.notna(r.get("override_games_out")):
            out[pid] = float(r["override_games_out"])
            continue
        if bool(r.get("injured") is True or r.get("injured") == "True") and \
                pd.notna(r.get("return_date")):
            tg = sched[(sched.home == team) | (sched.away == team)]
            out[pid] = float((tg.date < pd.Timestamp(r["return_date"])).sum())
            continue
        st = r.get("df_status")
        if isinstance(st, str) and st in DF_STATUS_GAMES:
            out[pid] = DF_STATUS_GAMES[st]
    return out.clip(0, games)


def _blocks(games_out: np.ndarray, games: int):
    """Split the season into blocks of games where availability is constant."""
    cuts = sorted({0, games, *[int(round(g)) for g in games_out if 0 < g < games]})
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def deploy_team(t: pd.DataFrame, games: int, budgets: dict, repl_tpg: dict,
                noise_sd: float, draws: int = 64) -> tuple[pd.DataFrame, dict]:
    """Expected games and ice time for one team's roster.

    t columns: player_id, pos, score, q (injury rate), games_out,
    tpg_ev/pp/sh/oth (un-normalised minutes per game).
    budgets[(pos, sit)]: skater-minutes per team-game; repl_tpg likewise
    for a call-up. Returns per-player gp and toi_<sit> season minutes, plus
    team-level replacement games and scale factors.
    """
    t = t.copy()
    t["gp"] = 0.0
    info = {}
    for pos in ("F", "D"):
        m = (t.pos == pos).to_numpy()
        sub = t[m]
        gp = np.zeros(len(sub))
        rep = 0.0
        score = sub.score.to_numpy(float)
        for a, b in _blocks(sub.games_out.to_numpy(float), games):
            h = np.where(sub.games_out.to_numpy() > a, 0.0, 1.0 - sub.q.to_numpy(float))
            p, r = dress_probs(score, h, N_DRESS[pos], noise_sd, draws,
                               seed=hash((pos, a, len(sub))) % 2**31)
            gp += p * (b - a)
            rep += r * (b - a)
        t.loc[m, "gp"] = gp
        info[f"rep_gp_{pos}"] = rep
        # --- ice-time conservation per situation, with individual caps
        tpg0 = np.column_stack([sub[f"tpg_{k}"].to_numpy(float) for k in SITS])
        tpg, scale = _conserve(tpg0, gp, rep, games, pos, budgets, repl_tpg)
        for i, k in enumerate(SITS):
            t.loc[m, f"tpg_{k}_n"] = tpg[:, i]
            t.loc[m, f"toi_{k}"] = gp * tpg[:, i]
            info[f"scale_{pos}_{k}"] = scale[i]
    return t, info


def _conserve(tpg0: np.ndarray, gp: np.ndarray, rep: float, games: int,
              pos: str, budgets: dict, repl_tpg: dict):
    """Rescale minutes per game so each situation's season budget is spent.

    Players whose all-situation load would pass the positional ceiling are
    pinned at it (in proportion across situations) and the remainder of the
    budget is shared by the others, iterating until nobody is over.
    """
    tpg = tpg0.copy()
    capped = np.zeros(len(tpg), bool)
    scale = np.ones(len(SITS))
    for _ in range(6):
        for i, k in enumerate(SITS):
            need = games * budgets[(pos, k)] - rep * repl_tpg[(pos, k)]
            fixed = float((gp[capped] * tpg[capped, i]).sum())
            have = float((gp[~capped] * tpg[~capped, i]).sum())
            f = (need - fixed) / have if have > 0 else 1.0
            f = max(f, 0.0)
            tpg[~capped, i] *= f
            scale[i] = f if not capped.any() else scale[i] * f
        tot = tpg.sum(1)
        over = (tot > TOI_CAP[pos] + 1e-9) & ~capped
        if not over.any():
            break
        tpg[over] *= (TOI_CAP[pos] / tot[over])[:, None]
        capped |= over
    return tpg, scale


def budgets_for(V: int) -> tuple[dict, dict]:
    """Skater-minute budgets per team-game (league level projected for V)
    and a call-up's minutes per game, by position and situation."""
    from hattrick import players as PL
    lg = PL.project_league(V)
    bud = {(p, k): float(lg[f"sk_min_{p}_{k}"]) for p in ("F", "D") for k in SITS}
    rep = PL.usage_prior_tpg(V)
    return bud, {(p, k): rep[(p, k)] for p in ("F", "D") for k in SITS}


def deploy(proj: pd.DataFrame, roster: pd.DataFrame, V: int, games: int,
           noise_sd: float, q_scale: float = 1.0,
           games_out: pd.Series | None = None, draws: int = 64) -> pd.DataFrame:
    """Expected GP and season TOI by situation for every rostered skater.

    proj: hattrick.players.project() output (indexed by player_id or with a
    player_id column). roster: player_id, team (skaters only).
    """
    pj = proj.set_index("player_id") if "player_id" in proj.columns else proj
    r = roster[["player_id", "team"]].drop_duplicates("player_id").copy()
    r = r[r.player_id.isin(pj.index)]
    cols = ["pos", "age", "score"] + [f"tpg_{k}" for k in SITS]
    r = r.join(pj[cols], on="player_id")
    r["q"] = health(V, r.player_id.to_numpy(), r.age.to_numpy(), q_scale)
    go = games_out if games_out is not None else pd.Series(dtype=float)
    r["games_out"] = r.player_id.map(go).fillna(0.0).clip(0, games)
    bud, rep = budgets_for(V)
    outs = []
    for team, t in r.groupby("team"):
        dt, info = deploy_team(t, games, bud, rep, noise_sd, draws)
        dt["rep_gp_F"] = info["rep_gp_F"]
        dt["rep_gp_D"] = info["rep_gp_D"]
        outs.append(dt)
    return pd.concat(outs, ignore_index=True)


def add_totals(dep: pd.DataFrame, proj: pd.DataFrame, stats=None) -> pd.DataFrame:
    """Season means: per-situation TOI x per-60 rates (+ all-situation
    peripherals). Adds g, a1, a2, a, p, sog, ixg, ppg, ppa and peripherals."""
    from hattrick import players as PL
    pj = proj.set_index("player_id") if "player_id" in proj.columns else proj
    d = dep.copy()
    rc = [c for c in pj.columns if c.startswith("r60_")]
    d = d.join(pj[rc], on="player_id")
    for s in PL.SIT_STATS:
        d[s] = sum(d[f"toi_{k}"] * d[f"r60_{s}_{k}"] / 60.0 for k in SITS)
    d["ppg"] = d.toi_pp * d.r60_g_pp / 60.0
    d["ppa"] = d.toi_pp * (d.r60_a1_pp + d.r60_a2_pp) / 60.0
    d["toi"] = sum(d[f"toi_{k}"] for k in SITS)
    for s in PL.PERIPH:
        d[s] = d.toi * d[f"r60_{s}"] / 60.0
    d["a"] = d.a1 + d.a2
    d["p"] = d.g + d.a
    return d
