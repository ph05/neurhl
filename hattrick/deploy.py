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
RESEARCH_FILE = C.PKG / "data_injuries_2027.csv"
RESEARCH_CONFIDENCE = ("high", "medium")
RESEARCH_CUTOFF = "2026-09-29"


def _norm_name(z) -> str:
    import unicodedata
    z = unicodedata.normalize("NFKD", str(z)).encode("ascii", "ignore").decode()
    return z.lower().replace(".", "").replace("'", "").replace("-", " ").strip()


def injury_research_2027(roster: pd.DataFrame) -> pd.DataFrame:
    """Researched 2026-27 injury estimates (hattrick/data_injuries_2027.csv,
    reports dated on or before the cutoff), matched to roster player ids by
    name (team only breaks ties: the roster file's team is authoritative)."""
    if not RESEARCH_FILE.exists():
        return pd.DataFrame(columns=["player_id", "est_games_out", "est_min", "est_max",
                                     "confidence"])
    r = pd.read_csv(RESEARCH_FILE)
    r = r[pd.to_datetime(r.report_date) <= pd.Timestamp(RESEARCH_CUTOFF)].copy()
    if {"name", "team"} <= set(roster.columns):
        ros = roster[["player_id", "team", "name"]].dropna(subset=["name"]).copy()
    else:
        ros = D.rosters_2027()[["player_id", "team", "name"]].copy()
    ros["nn"] = ros.name.map(_norm_name)
    r["nn"] = r.name.map(_norm_name)
    m = r.merge(ros[["player_id", "nn", "team"]], on="nn", suffixes=("", "_roster"))
    m["tie"] = (m.team == m.team_roster).astype(int)
    m = m.sort_values("tie", ascending=False).drop_duplicates("nn")
    return m[["player_id", "name", "est_games_out", "est_min", "est_max", "confidence"]]


def games_out_2027(roster: pd.DataFrame, games: int, with_range: bool = False):
    """Games each 2026-27 player misses at the start of the season.

    Priority: manual status (LTIR / suspended-not-reporting: the whole
    season; goalie holdouts are refined in hattrick.goalies) > injury
    research with high/medium confidence (est_games_out, range est_min..
    est_max) > NeurHL's researched override_games_out > MoneyPuck injured
    flag with a return date (team games scheduled before that date) >
    DailyFaceoff status without a date (DF_STATUS_GAMES) > 0.
    With with_range=True also returns {player_id: (min, max)} for the
    research rows and the source label of each decision.
    """
    av = D.availability_raw_2027().drop_duplicates("player_id").set_index("player_id")
    res = injury_research_2027(roster).set_index("player_id")
    sched = D.schedule_2027()
    out = pd.Series(0.0, index=roster.player_id.to_numpy())
    rng_, src = {}, {}
    for pid, team in zip(roster.player_id, roster.team):
        r = av.loc[pid] if pid in av.index else pd.Series(dtype=object)
        ms = str(r.get("manual_status") or "")
        if ms in ("LTIR", "SUSPENDED_NOT_REPORTING"):
            out[pid], src[pid] = float(games), f"manual:{ms}"
            continue
        if pid in res.index and str(res.loc[pid, "confidence"]) in RESEARCH_CONFIDENCE:
            e = res.loc[pid]
            out[pid] = float(e.est_games_out)
            rng_[pid] = (float(e.est_min), float(e.est_max))
            src[pid] = f"research:{e.confidence}"
            continue
        if pd.notna(r.get("override_games_out")):
            out[pid], src[pid] = float(r["override_games_out"]), "override"
            continue
        inj = r.get("injured")
        if (inj is True or str(inj) == "True") and pd.notna(r.get("return_date")):
            tg = sched[(sched.home == team) | (sched.away == team)]
            out[pid] = float((tg.date < pd.Timestamp(r["return_date"])).sum())
            src[pid] = "moneypuck_return_date"
            continue
        st = r.get("df_status")
        if isinstance(st, str) and st in DF_STATUS_GAMES:
            out[pid], src[pid] = DF_STATUS_GAMES[st], f"dailyfaceoff:{st}"
    out = out.clip(0, games)
    if with_range:
        return out, rng_, src
    return out


def _blocks(games_out: np.ndarray, games: int):
    """Split the season into blocks of games where availability is constant."""
    cuts = sorted({0, games, *[int(round(g)) for g in games_out if 0 < g < games]})
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def deploy_team(t: pd.DataFrame, games: int, budgets: dict, repl_tpg: dict,
                noise_sd: float, draws: int = 64,
                cover: dict | None = None,
                gp_map: dict | None = None,
                gp_override: dict | None = None) -> tuple[pd.DataFrame, dict]:
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
        pres = sub["presence"].to_numpy(float) if "presence" in sub else np.ones(len(sub))
        if gp_override is None and gp_map is None:
            for a, b in _blocks(sub.games_out.to_numpy(float), games):
                h = np.where(sub.games_out.to_numpy() > a, 0.0, 1.0 - sub.q.to_numpy(float)) * pres
                p, r = dress_probs(score, h, N_DRESS[pos], noise_sd, draws,
                                   seed=hash((pos, a, len(sub))) % 2**31)
                gp += p * (b - a)
                rep += r * (b - a)
        if gp_override is not None:
            gp = sub.player_id.map(gp_override).fillna(0.0).to_numpy(float)
            rep = 0.0
            if cover is not None and len(sub):
                target = min(cover[pos] * N_DRESS[pos] * games, 0.999 * games * len(sub))
                ceil = np.full(len(sub), games, float)
                for a, b in _blocks(sub.games_out.to_numpy(float), games):
                    ceil -= np.where(sub.games_out.to_numpy() > a, b - a, 0.0)
                cap = np.clip(ceil / games, 1e-3, 1.0)
                gp = games * _logit_shift(np.clip(gp / games, 1e-4, 0.999), target / games, cap)
                rep = max(N_DRESS[pos] * games - gp.sum(), 0.0)
        elif gp_map is not None and len(sub):
            # team-agnostic games (ex-post rosters): no depth competition, no
            # call-ups; ice time is still conserved within the team below
            hh = sub.has_hist.to_numpy(bool)
            share = np.where(hh, gp_map[True].predict(score), gp_map[False].predict(score))
            gp = np.zeros(len(sub))
            for a, b in _blocks(sub.games_out.to_numpy(float), games):
                h = np.where(sub.games_out.to_numpy() > a, 0.0, 1.0 - sub.q.to_numpy(float))
                gp += h * share * (b - a)
            rep = 0.0
        elif cover is not None and len(sub):
            # dressed-games conservation for this kind of roster: the roster
            # plays cover[pos] of the team's N x games dressed slots
            target = min(cover[pos] * N_DRESS[pos] * games, 0.999 * games * len(sub))
            ceil = np.full(len(sub), games, float)
            for a, b in _blocks(sub.games_out.to_numpy(float), games):
                ceil -= np.where(sub.games_out.to_numpy() > a, b - a, 0.0)
            cap = np.clip((ceil * (1 - sub.q.to_numpy(float) * 0.5)) / games, 1e-3, 1.0)
            gp = games * _logit_shift(gp / games, target / games, cap)
            rep = max(N_DRESS[pos] * games - gp.sum(), 0.0)
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


# ---------------------------------------------------------------------------
# Rosters and presence (fraction of the season with the club)
# ---------------------------------------------------------------------------
def historical_roster(V: int, kind: str) -> tuple[pd.DataFrame, str]:
    """Skater roster for a past season V.

    kind='expost': every skater who played in V, on his main (most-TOI)
    team -- NeurHL's backtest information. kind='opening': everyone who
    dressed in one of his team's first 10 games (2011-2024); seasons without
    boxes fall back to 'expost' and are flagged 'season_team'.
    """
    from hattrick import players as PL
    if kind == "opening":
        try:
            r = D.opening_rosters(V)
            return r[r.grp != "G"][["player_id", "team"]].reset_index(drop=True), "first10"
        except FileNotFoundError:
            pass
    P = PL.panel()
    j = P.j(V)
    m = P.x["gp"][:, j] > 0
    return pd.DataFrame({"player_id": P.ids[m], "team": P.team[m, j]}), "season_team"


RANK_BUCKETS = {"F": [0, 6, 9, 12, 14, 16, 99], "D": [0, 4, 6, 7, 8, 99]}


def depth_rank(r: pd.DataFrame) -> pd.Series:
    """Rank (1 = most projected ice time) within team and position."""
    return r.groupby(["team", "pos"]).score.rank(ascending=False, method="first")


def _bucket(pos, rank):
    return [int(np.searchsorted(RANK_BUCKETS[p][1:], k, side="left")) for p, k in zip(pos, rank)]


_PRESENCE: dict = {}


def presence_table(V: int, kind: str, prm) -> dict:
    """E[share of the season a skater is with the club | position, depth
    rank on this kind of roster], from seasons V-8..V-1 (absence windows:
    first to last game dressed for the team / team games). Walk-forward;
    empty before absence data exist (2012), meaning presence 1."""
    from hattrick import players as PL
    key = (V, kind, prm.key())
    if key in _PRESENCE:
        return _PRESENCE[key]
    ab = D.absences()
    rows = []
    for s in range(max(2012, V - 8), V):
        if s in C.BROKEN_SEASONS:
            continue
        ros, flag = historical_roster(s, kind)
        if kind == "opening" and flag != "first10":
            continue
        pj = PL.project(s, prm, ids=ros.player_id)
        r = ros.merge(pj[["player_id", "pos", "score"]], on="player_id")
        r["rank"] = depth_rank(r)
        a = ab[ab.season_end == s][["player_id", "team", "window_games"]]
        r = r.merge(a, on=["player_id", "team"], how="left")
        tg = D.team_seasons()
        tg = tg[tg.season_end == s].set_index("team").gp
        r["pi"] = (r.window_games.fillna(0) / r.team.map(tg)).clip(0, 1)
        r["b"] = _bucket(r.pos, r["rank"])
        rows.append(r)
    tab = {}
    if rows:
        d = pd.concat(rows)
        raw = d.groupby(["pos", "b"]).pi.mean()
        # relative to the top depth bucket: the top bucket's missing window
        # games are injuries at either end of the season, which the health
        # model already charges, so only the EXTRA absence of lower ranks
        # (late call-ups, demotions, deadline arrivals) is presence.
        tab = {(p, b): float(min(1.0, v / raw[(p, 0)])) for (p, b), v in raw.items()}
    _PRESENCE[key] = tab
    return tab


_COVER: dict = {}


def coverage(V: int, kind: str) -> dict:
    """Games the players on this kind of roster play in total (for any
    team: a traded player keeps playing), as a share of the team's dressed
    skater-games (12 F + 6 D per game), by position, averaged over seasons
    V-8..V-1 (walk-forward). Opening rosters miss later call-ups and
    acquisitions, and gain the games their traded players play elsewhere."""
    key = (V, kind)
    if key in _COVER:
        return _COVER[key]
    st = D.skater_team_seasons()
    from hattrick import players as PL
    P = PL.panel()
    pos = dict(zip(P.ids, P.pos))
    tg = D.team_seasons()
    vals = {"F": [], "D": []}
    for s in range(max(2011, V - 8), V):
        if s in C.BROKEN_SEASONS:
            continue
        ros, flag = historical_roster(s, kind)
        if kind == "opening" and flag != "first10":
            continue
        g = tg[tg.season_end == s].set_index("team").gp
        tot_gp = st[st.season_end == s].groupby("player_id").gp.sum()
        x = ros.assign(gp=ros.player_id.map(tot_gp).fillna(0.0))
        x["pos"] = x.player_id.map(pos)
        for p in ("F", "D"):
            tot = x[x.pos == p].groupby("team").gp.sum()
            vals[p] += list((tot / (N_DRESS[p] * g.reindex(tot.index))).to_numpy())
    out = {p: float(np.clip(np.mean(v), 0.5, 1.0)) if v else 1.0 for p, v in vals.items()}
    _COVER[key] = out
    return out


def _logit_shift(p: np.ndarray, target: float, cap: np.ndarray) -> np.ndarray:
    """Shift probabilities on the logit scale so they sum to `target`,
    keeping each within [0, cap] (cap: availability ceiling from health)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo, hi = -10.0, 10.0
    f = lambda d: np.minimum(1 / (1 + np.exp(-(np.log(p / (1 - p)) + d))), cap)
    if f(hi).sum() < target:
        return f(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid).sum() > target:
            hi = mid
        else:
            lo = mid
    return f(0.5 * (lo + hi))


_GPMAP: dict = {}


def league_gp_map(V: int, prm, window: int = 6):
    """Team-agnostic games model for EX-POST rosters (NeurHL's backtest
    information: a player's season-V team, not his path through it).

    Among skaters who played in season s (s = V-window..V-1), the share of
    the schedule they played divided by their health availability (1-q),
    as an isotonic function of the projected depth-chart score, separately
    for players with and without NHL history. A traded player keeps his
    full-season games; teams' dressed-game totals are then not enforced
    (they cannot be without knowing when he moved)."""
    from sklearn.isotonic import IsotonicRegression
    from hattrick import players as PL
    key = (V, prm.key(), prm.q_scale)
    if key in _GPMAP:
        return _GPMAP[key]
    P = PL.panel()
    xs, ys, hs = [], [], []
    for s in range(max(2010, V - window), V):
        if s in C.BROKEN_SEASONS:
            continue
        j = P.j(s)
        m = P.x["gp"][:, j] > 0
        pj = PL.project(s, prm, ids=P.ids[m])
        q = health(s, pj.player_id.to_numpy(), pj.age.to_numpy(), prm.q_scale)
        gp = pj.player_id.map(dict(zip(P.ids, P.x["gp"][:, j]))).to_numpy()
        sched = float(D.team_seasons().query("season_end == @s").gp.median())
        xs.append(pj.score.to_numpy())
        ys.append(np.clip(gp / (sched * (1 - q)), 0, 1))
        hs.append(pj.has_hist.to_numpy())
    x, y, h = map(np.concatenate, (xs, ys, hs))
    maps = {}
    for flag in (True, False):
        k = h == flag
        maps[flag] = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x[k], y[k])
    _GPMAP[key] = maps
    return maps


# ---------------------------------------------------------------------------
# Games stacking: structural games + the player's own recent games played
# ---------------------------------------------------------------------------
_STRUCT: dict = {}
_STACK: dict = {}


def structural_games(s: int, prm, kind: str) -> pd.DataFrame:
    """Structural expected games (depth chart / team-agnostic map, no
    stacking) for a past season's roster of this kind."""
    from hattrick import players as PL
    key = (s, kind, prm.key(), prm.dress_noise, prm.q_scale)
    if key in _STRUCT:
        return _STRUCT[key]
    ros, flag = historical_roster(s, kind)
    proj = PL.project(s, prm, ids=ros.player_id)
    ros = ros[ros.player_id.isin(proj.player_id)]
    gmap = league_gp_map(s, prm) if flag == "season_team" else None
    cover = coverage(s, "opening") if flag == "first10" else None
    dep = deploy(proj, ros, s, int(round(_sched(s))), prm.dress_noise, prm.q_scale,
                 gp_map=gmap, cover=cover)
    out = dep[["player_id", "team", "gp", "has_hist"]].assign(flag=flag)
    _STRUCT[key] = out
    return out


def _sched(s: int) -> float:
    t = D.team_seasons()
    return float(t[t.season_end == s].gp.median())


def gp_history(V: int, ids) -> tuple[np.ndarray, np.ndarray]:
    """Games played in V-1 and V-2, restated to an 82-game schedule."""
    from hattrick import players as PL
    P = PL.panel()
    out = []
    for lag in (1, 2):
        y = V - lag
        if y < P.seasons[0] or y > P.seasons[-1]:
            out.append(np.zeros(len(ids)))
            continue
        g = dict(zip(P.ids, P.x["gp"][:, P.j(y)] * 82.0 / _sched(y)))
        out.append(np.array([g.get(p, 0.0) for p in ids]))
    return out[0], out[1]


def gp_stack(V: int, prm, kind: str, window: int = 8):
    """Walk-forward OLS of realised games on [1, structural games (82-game
    basis), GP(V-1), GP(V-2)] for players with NHL history on this kind of
    roster, seasons V-window..V-1 (broken seasons and, for opening rosters,
    seasons without box scores skipped). None when no season qualifies."""
    from hattrick import players as PL
    key = (V, kind, prm.key(), prm.dress_noise, prm.q_scale)
    if key in _STACK:
        return _STACK[key]
    P = PL.panel()
    Xs, ys = [], []
    for s in range(max(2011, V - window), V):
        if s in C.BROKEN_SEASONS or s > P.seasons[-1]:
            continue
        if kind == "opening":
            try:
                D.opening_rosters(s)
            except FileNotFoundError:
                continue
        d = structural_games(s, prm, kind)
        d = d[d.has_hist]
        g1, g2 = gp_history(s, d.player_id.to_numpy())
        act = dict(zip(P.ids, P.x["gp"][:, P.j(s)] * 82.0 / _sched(s)))
        Xs.append(np.column_stack([np.ones(len(d)), d.gp * 82.0 / _sched(s), g1, g2]))
        ys.append(np.array([act.get(p, 0.0) for p in d.player_id]))
    b = None
    if Xs:
        X, y = np.vstack(Xs), np.concatenate(ys)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
    _STACK[key] = b
    return b


def apply_stack(dep: pd.DataFrame, V: int, prm, kind: str, games: int) -> dict:
    """Stacked expected games per player (history players only; others and
    players with no fitted stack keep their structural games). Games already
    ruled out (injured at the start) are removed in proportion."""
    b = gp_stack(V, prm, kind)
    gp = dep.set_index("player_id").gp.copy()
    if b is None:
        return gp.to_dict()
    h = dep.has_hist.to_numpy(bool)
    ids = dep.player_id.to_numpy()
    g1, g2 = gp_history(V, ids)
    avail = (games - dep.games_out.to_numpy(float)) / games
    base = dep.gp.to_numpy(float) / np.maximum(avail, 1e-6) * 82.0 / games   # healthy-start, 82 basis
    pred = b[0] + b[1] * base + b[2] * g1 + b[3] * g2
    new = np.clip(pred, 0.0, 82.0) * games / 82.0 * avail
    out = np.where(h, new, dep.gp.to_numpy(float))
    return dict(zip(ids, out))


NEURHL_ROSTER_SHARE = 0.968     # neurhl/models/player_proj.py ROSTER_SHARE_OF_GAMES
NEURHL_MAX_GP = 78.0            # ... MAX_EXPECTED_GP (82-game basis)


def neurhl_gp_budget(gp: pd.Series, n_teams: int, games: int) -> pd.Series:
    """NeurHL's league-wide games identity (to_totals): scale expected games
    of the players it projects so they sum to 18 x games x teams x 0.968,
    capped at 78/82 of the schedule, iterating as NeurHL does. Used only by
    the matched backtest protocol, to share NeurHL's conventions."""
    budget = 18 * games * n_teams * NEURHL_ROSTER_SHARE
    cap = NEURHL_MAX_GP * games / 82.0
    g = gp.clip(lower=0.02 * games).copy()
    for _ in range(4):
        tot = float(g.sum())
        g = (g * budget / tot).clip(upper=cap)
        if abs(float(g.sum()) - budget) < 5:
            break
    return g


def presence(r: pd.DataFrame, tab: dict) -> np.ndarray:
    if not tab:
        return np.ones(len(r))
    b = _bucket(r.pos, depth_rank(r))
    return np.array([tab.get((p, k), 1.0) for p, k in zip(r.pos, b)])


def deploy(proj: pd.DataFrame, roster: pd.DataFrame, V: int, games: int,
           noise_sd: float, q_scale: float = 1.0,
           games_out: pd.Series | None = None, draws: int = 64,
           games_out_range: dict | None = None,
           presence_tab: dict | None = None,
           cover: dict | None = None, gp_map: dict | None = None,
           gp_override: dict | None = None) -> pd.DataFrame:
    """Expected GP and season TOI by situation for every rostered skater.

    proj: hattrick.players.project() output (indexed by player_id or with a
    player_id column). roster: player_id, team (skaters only).
    """
    pj = proj.set_index("player_id") if "player_id" in proj.columns else proj
    r = roster[["player_id", "team"]].drop_duplicates("player_id").copy()
    r = r[r.player_id.isin(pj.index)]
    cols = ["pos", "age", "score", "has_hist"] + [f"tpg_{k}" for k in SITS]
    r = r.join(pj[cols], on="player_id")
    r["q"] = health(V, r.player_id.to_numpy(), r.age.to_numpy(), q_scale)
    # presence enters as a per-game availability factor (means only)
    r["presence"] = presence(r, presence_tab or {})
    go = games_out if games_out is not None else pd.Series(dtype=float)
    r["games_out"] = r.player_id.map(go).fillna(0.0).clip(0, games)
    rg = games_out_range or {}
    r["games_out_min"] = r.player_id.map({k: v[0] for k, v in rg.items()}).fillna(r.games_out)
    r["games_out_max"] = r.player_id.map({k: v[1] for k, v in rg.items()}).fillna(r.games_out)
    bud, rep = budgets_for(V)
    outs = []
    for team, t in r.groupby("team"):
        dt, info = deploy_team(t, games, bud, rep, noise_sd, draws, cover, gp_map,
                               gp_override)
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


# ---------------------------------------------------------------------------
# Monte Carlo season draws (for intervals)
# ---------------------------------------------------------------------------
MC_STATS = ("gp", "toi", "g", "a", "p", "sog", "ppg", "ppa", "hits", "blk",
            "pim", "fow", "fol", "tk", "gv")


def _markov_health(q, L, games, S, rng, out_first=None, present=None):
    """(S, n, games) bool availability. Injuries follow a two-state Markov
    chain with stationary out-fraction q and mean spell length L games;
    `out_first` (S, n) forces the first k games out; `present` (S, n) bool
    removes a player for the whole season (e.g. an unresolved holdout)."""
    n = len(q)
    q = np.clip(q, 1e-4, 0.9)
    p_hi = q / ((1 - q) * L)                     # healthy -> injured
    p_ih = 1.0 / L                               # injured -> healthy
    st = rng.random((S, n)) >= q                 # True = healthy
    avail = np.empty((S, n, games), bool)
    for g in range(games):
        u = rng.random((S, n))
        st = np.where(st, u >= p_hi, u < p_ih)
        avail[:, :, g] = st
    if out_first is not None:
        gidx = np.arange(games)[None, None, :]
        avail &= gidx >= out_first[:, :, None]
    if present is not None:
        avail &= present[:, :, None]
    return avail


def simulate_team(t: pd.DataFrame, games: int, budgets: dict, repl_tpg: dict,
                  mc: dict, S: int, rng) -> dict:
    """Season draws for one team's skaters.

    t: deploy_team output joined with projection columns (r60_*, sd60_*),
    plus q, score, games_out, optional games_out_min/max (triangular draw)
    and p_present. mc: noise_sd, spell_len, usage_sd, sd_scale, team_sd,
    rho_ga. Returns {stat: (S, n) array} in the row order of t.
    """
    n = len(t)
    res = {s: np.zeros((S, n)) for s in MC_STATS}
    team_mult = np.exp(mc["team_sd"] * rng.standard_normal(S) - mc["team_sd"] ** 2 / 2)
    for pos in ("F", "D"):
        idx = np.where((t.pos == pos).to_numpy())[0]
        if not len(idx):
            continue
        sub = t.iloc[idx]
        k_ = len(idx)
        go = sub.games_out.to_numpy(float)
        lo = sub.get("games_out_min", pd.Series(go, index=sub.index)).fillna(sub.games_out).to_numpy(float)
        hi = sub.get("games_out_max", pd.Series(go, index=sub.index)).fillna(sub.games_out).to_numpy(float)
        mode = np.clip(3 * go - lo - hi, lo, hi)
        draw = np.where(hi > lo, rng.triangular(np.minimum(lo, mode), mode,
                                                np.maximum(hi, mode + 1e-9), (S, k_)), go)
        pres = rng.random((S, k_)) < sub.get("p_present", pd.Series(1.0, index=sub.index)).fillna(1.0).to_numpy()
        avail = _markov_health(sub.q.to_numpy(float), mc["spell_len"], games, S, rng,
                               out_first=np.round(draw), present=pres)
        # presence < 1: the player is with the club for one contiguous stretch
        # of that share of the season, placed uniformly at random
        pi = sub.get("presence", pd.Series(1.0, index=sub.index)).fillna(1.0).to_numpy()
        if (pi < 1).any():
            L = np.maximum(np.round(pi * games), 1).astype(int)[None, :]
            start = (rng.random((S, k_)) * (games - L + 1)).astype(int)
            gi = np.arange(games)[None, None, :]
            avail &= (gi >= start[:, :, None]) & (gi < (start + L)[:, :, None])
        score = sub.score.to_numpy(float)[None, :] + mc["noise_sd"] * rng.standard_normal((S, k_))
        ms = np.where(avail, score[:, :, None], -np.inf)          # (S, k, G)
        N = N_DRESS[pos]
        if k_ > N:
            kth = -np.sort(-ms, axis=1)[:, N - 1:N, :]            # N-th best
            dressed = avail & (ms >= kth)
        else:
            dressed = avail.copy()
        n_rep = np.clip(N - dressed.sum(1), 0, None)              # (S, G)
        u = np.exp(mc["usage_sd"] * rng.standard_normal((S, k_)) - mc["usage_sd"] ** 2 / 2)
        toi_tot = np.zeros((S, k_))
        toi_k = {}
        for k in SITS:
            w = sub[f"tpg_{k}"].to_numpy(float)[None, :] * u         # (S, k)
            denom = (dressed * w[:, :, None]).sum(1) + n_rep * repl_tpg[(pos, k)]
            share = np.where(dressed, w[:, :, None] / np.maximum(denom[:, None, :], 1e-9), 0)
            tk = budgets[(pos, k)] * share.sum(2)                   # (S, k) season min
            toi_k[k] = tk
            toi_tot += tk
        gp = dressed.sum(2).astype(float)
        # --- talent draws (log-normal, correlated goals/assists)
        def rel_sd(stat_cols):
            num = sum(toi_k[k].mean(0) * sub[f"sd60_{c}_{k}"].to_numpy(float) for c in stat_cols for k in SITS)
            den = sum(toi_k[k].mean(0) * sub[f"r60_{c}_{k}"].to_numpy(float) for c in stat_cols for k in SITS)
            return mc["sd_scale"] * num / np.maximum(den, 1e-9)
        sg, sa, ss = rel_sd(["g"]), rel_sd(["a1", "a2"]), rel_sd(["sog"])
        z1 = rng.standard_normal((S, k_))
        z2 = mc["rho_ga"] * z1 + np.sqrt(1 - mc["rho_ga"] ** 2) * rng.standard_normal((S, k_))
        mg = np.exp(sg * z1 - sg ** 2 / 2) * team_mult[:, None]
        ma = np.exp(sa * z2 - sa ** 2 / 2) * team_mult[:, None]
        msog = np.exp(ss * z1 - ss ** 2 / 2)
        lam = {}
        for stat, cols, mult in (("g", ["g"], mg), ("a", ["a1", "a2"], ma), ("sog", ["sog"], msog)):
            lam[stat] = mult * sum(toi_k[k] * sub[f"r60_{c}_{k}"].to_numpy(float)[None, :] / 60
                                   for c in cols for k in SITS)
            lam[stat + "_pp"] = mult * sum(toi_k["pp"] * sub[f"r60_{c}_pp"].to_numpy(float)[None, :] / 60
                                           for c in cols)
        G = rng.poisson(lam["g"])
        A = rng.poisson(lam["a"])
        res["gp"][:, idx] = gp
        res["toi"][:, idx] = toi_tot
        res["g"][:, idx] = G
        res["a"][:, idx] = A
        res["p"][:, idx] = G + A
        res["sog"][:, idx] = rng.poisson(lam["sog"])
        res["ppg"][:, idx] = rng.binomial(G, np.clip(lam["g_pp"] / np.maximum(lam["g"], 1e-9), 0, 1))
        res["ppa"][:, idx] = rng.binomial(A, np.clip(lam["a_pp"] / np.maximum(lam["a"], 1e-9), 0, 1))
        for s in ("hits", "blk", "pim", "fow", "fol", "tk", "gv"):
            r = sub[f"r60_{s}"].to_numpy(float)[None, :]
            sd = sub[f"sd60_{s}"].to_numpy(float)[None, :] * mc["sd_scale"]
            sig = np.clip(sd / np.maximum(r, 1e-9), 0, 1.5)
            m = np.exp(sig * rng.standard_normal((S, k_)) - sig ** 2 / 2)
            res[s][:, idx] = rng.poisson(toi_tot * r / 60 * m)
    return res


def simulate(dep: pd.DataFrame, proj: pd.DataFrame, V: int, games: int, mc: dict,
             S: int = 400, seed: int = C.SEED, quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """p10/p50/p90 (and SD) of season totals per skater, rescaled so each
    player's simulated mean equals the exact deterministic mean in `dep`
    (add_totals output). The rescaling keeps the interval shape from the
    simulation but anchors the level to the conservation-law means."""
    pj = proj.set_index("player_id") if "player_id" in proj.columns else proj
    rc = [c for c in pj.columns if c.startswith(("r60_", "sd60_")) and c not in dep.columns]
    d = dep.join(pj[rc], on="player_id")
    bud, rep = budgets_for(V)
    rng = np.random.default_rng(seed)
    out = []
    for team, t in d.groupby("team", sort=True):
        t = t.reset_index(drop=True)
        r = simulate_team(t, games, bud, rep, mc, S, rng)
        o = pd.DataFrame({"player_id": t.player_id})
        for s in MC_STATS:
            x = r[s]
            det = t[s].to_numpy(float) if s in t.columns else x.mean(0)
            if s == "gp":
                # additive shift keeps the spread of a bounded count; clip to
                # the games the player can still play
                hi = games - t.games_out.to_numpy(float)
                x = np.clip(x + (det - x.mean(0))[None, :], 0, hi[None, :])
            else:
                f = np.where(x.mean(0) > 1e-9, det / np.maximum(x.mean(0), 1e-9), 0.0)
                x = x * f[None, :]
            for qq in quantiles:
                o[f"{s}_p{int(qq * 100):02d}"] = np.quantile(x, qq, axis=0)
            o[f"{s}_sd"] = x.std(0)
        out.append(o)
    return pd.concat(out, ignore_index=True)
