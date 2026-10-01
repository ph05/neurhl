"""L1 tasks 1-3: the starter-choice dataset, the choice model, expected offsets.

Dataset (``situations``): one row per (team-game, candidate goalie) for every
regular-season game 2011-2024 with box scores, built strictly from
information dated before the game's date (games on the same date are
"predicted" from the state at the end of the previous date, as in the
filter). Candidates of team T: goalies whose most recent NHL appearance
(any team) was for T, within the last ``W`` of T's games (seasons carry
over). One extra "other" alternative per team-game stands for a goalie who is
not a candidate (call-up, new signing, first appearance).

Goalie talent and the team's usual-starter reference are reproduced EXACTLY
as ``orr.structural.goalie_game_talent`` computes them (same loop, same
dates, same in-season evidence), but for every candidate, not only the
actual starter; ``check_talent`` verifies that the actual starter's
talent - reference equals ``goalie_game_talent``'s gdiff.

Choice model: a conditional logit over the candidates plus "other"
(``fit_clogit``), L2-penalised, features per candidate:
  lsh_s, lsh_l   log EWMA start share over the team's games (half-lives hs, hl)
  last           started the team's previous game
  last_b2b       last x the team is on the second night of a back-to-back
  lstreak        log1p(consecutive starts) if last
  app_yday       appeared in any game the previous day (fatigue)
  lgsa           log1p(team games since his last appearance for the team)
  pulled         started the previous game and was relieved
  relief         came on in relief in the previous game
  home_l         home x lsh_l
  tgap           his talent minus the best candidate's (x100; goals saved / 100 shots)
"other": o_const, o_early = 1 / (1 + team games played this season).

Expected starter talent difference (the quantity ``gamemodel.goalie_offset``
multiplies): E[gdiff] = sum_c p_c (t_c - ref) + p_other (m_other - ref),
m_other = mean talent of "other" starters in the training seasons.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from orr import gametable as T
from orr import structural as S

CAND_FEATS = ["lsh_s", "lsh_l", "last", "last_b2b", "lstreak", "app_yday", "lgsa",
              "pulled", "relief", "home_l", "tgap"]
OTHER_FEATS = ["o_const", "o_early"]


@functools.lru_cache(maxsize=None)
def situations(goalie_key: tuple, hs: float = 4.0, hl: float = 30.0, W: int = 60) -> pd.DataFrame:
    """Candidate rows (one per team-game x candidate, plus one 'other' row).

    Columns: sid (team-game id), gid, season_end, date, team, side (0 home,
    1 away), k_season (team games played this season before this one),
    goalie_id (-1 for other), is_other, chosen, t (candidate talent; NaN for
    other), ref (team usual-starter reference before the game), t_act / gdiff_act
    (actual starter's talent and talent - ref, as goalie_game_talent),
    both_known (both starters of the game recorded), features."""
    n0, m0, w_in, halflife = (float(x) for x in goalie_key)
    g = S.game_frame()
    log = T.goalie_game_log()
    ids = g[["gid", "game_id", "date", "season_end", "home", "away", "gk_h", "gk_a",
             "rest_h", "rest_a"]]
    log = log.merge(ids[["game_id", "gid", "date", "season_end", "home", "away"]], on="game_id")
    log["team"] = np.where(log.side == "Home", log.home, log.away)
    alpha = 1 - 0.5 ** (1.0 / halflife)          # reference EWMA (goalie_game_talent)
    a_s = 1 - 0.5 ** (1.0 / hs)
    a_l = 1 - 0.5 ** (1.0 / hl)
    ref: dict = {}
    # per-team state (carries across seasons)
    ew_s: dict = {}          # team -> {goalie: share}
    ew_l: dict = {}
    k_team: dict = {}        # team -> number of known team games so far
    last_app: dict = {}      # team -> {goalie: k index of last appearance}
    prev: dict = {}          # team -> (starter, streak, set(appeared), pulled starter?)
    k_season: dict = {}      # (team, season) -> games
    g_team: dict = {}        # goalie -> team of last appearance
    g_date: dict = {}        # goalie -> date of last appearance
    rows = []
    sid = 0
    known_any = ids[ids.gk_h.notna() | ids.gk_a.notna()]
    for V, gv in known_any.groupby("season_end"):
        prior, sv = S.goalie_prior(V, n0, m0)
        ins_num: dict = {}
        ins_den: dict = {}

        def talent(pid):
            a, b = prior.get(pid, (n0 * m0, n0))
            return (a + w_in * ins_num.get(pid, 0.0)) / (b + w_in * ins_den.get(pid, 0.0))

        season_log = log[log.season_end == V]
        by_date = {d: x for d, x in season_log.groupby("date")}
        for d in sorted(set(gv.date)):
            day = gv[gv.date == d]
            yday = d - pd.Timedelta(days=1)
            act = {}
            for r in day.itertuples(index=False):
                both = not (np.isnan(r.gk_h) or np.isnan(r.gk_a))
                for side, pid, team, rest in ((0, r.gk_h, r.home, r.rest_h),
                                              (1, r.gk_a, r.away, r.rest_a)):
                    if np.isnan(pid):
                        continue
                    pid = int(pid)
                    t_act = talent(pid)
                    rf = ref.get(team, t_act)
                    act[(r.gid, side)] = t_act
                    kt = k_team.get(team, 0)
                    la = last_app.get(team, {})
                    cands = [c for c, kk in la.items()
                             if g_team.get(c) == team and kt - kk <= W]
                    ks = k_season.get((team, V), 0)
                    pv = prev.get(team)
                    b2b = float(rest == 1)
                    tal = {c: talent(c) for c in cands}
                    tmax = max(tal.values()) if tal else np.nan
                    for c in cands:
                        s_s = ew_s[team].get(c, 0.0)
                        s_l = ew_l[team].get(c, 0.0)
                        last = float(pv is not None and pv[0] == c)
                        rows.append((sid, r.gid, V, d, team, side, ks, c, 0, int(c == pid),
                                     tal[c], rf, t_act, t_act - rf, both,
                                     np.log(s_s + 0.01), np.log(s_l + 0.01), last, last * b2b,
                                     np.log1p(pv[1]) * last if pv is not None else 0.0,
                                     float(g_date.get(c) == yday),
                                     np.log1p(kt - 1 - la[c]),
                                     float(last and pv[3]),
                                     float(pv is not None and c in pv[2] and pv[0] != c),
                                     (1.0 - side) * np.log(s_l + 0.01),
                                     100.0 * (tal[c] - tmax), 0.0, 0.0))
                    rows.append((sid, r.gid, V, d, team, side, ks, -1, 1, int(pid not in cands),
                                 np.nan, rf, t_act, t_act - rf, both,
                                 *([0.0] * len(CAND_FEATS)), 1.0, 1.0 / (1.0 + ks)))
                    sid += 1
            # ---- after the day's games: reference (exactly goalie_game_talent)
            for r in day.itertuples(index=False):
                for side, pid, team in ((0, r.gk_h, r.home), (1, r.gk_a, r.away)):
                    if not np.isnan(pid):
                        t_ = act[(r.gid, side)]
                        rf = ref.get(team, t_)
                        ref[team] = rf + alpha * (t_ - rf)
            # ---- team usage states (known starters only)
            x = by_date.get(d)
            for r in day.itertuples(index=False):
                for side, pid, team in ((0, r.gk_h, r.home), (1, r.gk_a, r.away)):
                    if np.isnan(pid):
                        continue
                    pid = int(pid)
                    app = set() if x is None else set(
                        x[(x.gid == r.gid) & (x.team == team)].goalie_id.astype(int))
                    app.add(pid)
                    es, el = ew_s.setdefault(team, {}), ew_l.setdefault(team, {})
                    for c in set(es) | {pid}:
                        es[c] = (1 - a_s) * es.get(c, 0.0) + a_s * (c == pid)
                        el[c] = (1 - a_l) * el.get(c, 0.0) + a_l * (c == pid)
                    kt = k_team.get(team, 0)
                    la = last_app.setdefault(team, {})
                    for c in app:
                        la[c] = kt
                    k_team[team] = kt + 1
                    k_season[(team, V)] = k_season.get((team, V), 0) + 1
                    pv = prev.get(team)
                    streak = (pv[1] + 1) if (pv is not None and pv[0] == pid) else 1
                    prev[team] = (pid, streak, app, len(app) > 1)
            # every appearance on the day: last team / date, in-season evidence
            if x is not None:
                for pid, tm in zip(x.goalie_id.astype(int), x.team):
                    g_team[pid] = tm
                    g_date[pid] = d
                for pid, sh, svs in zip(x.goalie_id, x.shots, x.saves):
                    ins_num[pid] = ins_num.get(pid, 0.0) + (svs - sv * sh)
                    ins_den[pid] = ins_den.get(pid, 0.0) + sh
    cols = ["sid", "gid", "season_end", "date", "team", "side", "k_season", "goalie_id",
            "is_other", "chosen", "t", "ref", "t_act", "gdiff_act", "both_known",
            *CAND_FEATS, *OTHER_FEATS]
    out = pd.DataFrame(rows, columns=cols)
    return out


def check_talent(sit: pd.DataFrame, goalie_key: tuple) -> float:
    """Max |gdiff_act - goalie_game_talent gdiff| over the actual starters."""
    gt = S.goalie_game_talent(*goalie_key)
    s = sit.drop_duplicates("sid")
    m = s.merge(gt, on="gid")
    ref = np.where(m.side == 0, m.gdiff_h, m.gdiff_a)
    return float(np.nanmax(np.abs(m.gdiff_act.to_numpy() - ref)))


# ---------------------------------------------------------------------------
# Conditional logit
# ---------------------------------------------------------------------------
def _design(sit: pd.DataFrame, feats: list[str]):
    X = sit[feats + OTHER_FEATS].to_numpy(float)
    grp = sit.sid.to_numpy()
    _, gi = np.unique(grp, return_inverse=True)
    return X, gi, sit.chosen.to_numpy(float)


def _probs(X, gi, beta):
    u = X @ beta
    mx = np.full(gi.max() + 1, -np.inf)
    np.maximum.at(mx, gi, u)
    e = np.exp(u - mx[gi])
    den = np.bincount(gi, e)
    return e / den[gi]


def fit_clogit(sit: pd.DataFrame, feats: list[str], lam: float = 1.0) -> dict:
    """Penalised ML (L2 ``lam`` on every coefficient). Only team-games whose
    actual starter is recorded and with at least one alternative chosen."""
    X, gi, y = _design(sit, feats)
    n = gi.max() + 1

    def f(b):
        p = _probs(X, gi, b)
        ll = np.sum(y * np.log(np.clip(p, 1e-300, None)))
        grad = X.T @ (y - p)          # sum_i (x_chosen - E_p x)
        return -ll + 0.5 * lam * b @ b, -grad + lam * b

    res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000})
    t_other = sit[(sit.is_other == 1) & (sit.chosen == 1)].t_act
    return {"feats": feats, "beta": res.x, "lam": lam, "n": int(n),
            "converged": bool(res.success),
            "m_other": float(t_other.mean()) if len(t_other) else np.nan}


def predict(sit: pd.DataFrame, model: dict) -> np.ndarray:
    X, gi, _ = _design(sit, model["feats"])
    return _probs(X, gi, model["beta"])


def choice_scores(sit: pd.DataFrame, p: np.ndarray) -> dict:
    """Per team-game: log loss of the actual choice, top-1 accuracy."""
    s = sit.assign(p=p)
    ch = s[s.chosen == 1]
    top = s.loc[s.groupby("sid").p.idxmax()]
    return {"n": int(len(ch)), "logloss": float(-np.mean(np.log(np.clip(ch.p, 1e-12, None)))),
            "acc": float(top.chosen.mean()), "p_other_actual": float(s[s.is_other == 1].chosen.mean())}


def mix_probs(sit: pd.DataFrame, col: str = "lsh_l", p_other: float = 0.02) -> np.ndarray:
    """The start-share mix as a choice model: candidates in proportion to
    their EWMA start share (the shrunk exp(lsh) - 0.01), fixed p_other."""
    sh = np.where(sit.is_other == 1, 0.0, np.exp(sit[col].to_numpy()) - 0.01).clip(0)
    tot = pd.Series(sh).groupby(sit.sid.to_numpy()).transform("sum").to_numpy()
    has = tot > 0
    p = np.where(sit.is_other == 1, np.where(has, p_other, 1.0),
                 np.where(has, (1 - p_other) * sh / np.where(has, tot, 1.0), 0.0))
    return p


def expected_gdiff(sit: pd.DataFrame, p: np.ndarray, m_other: float,
                   ref_kind: str = "ewma") -> pd.DataFrame:
    """Per (gid, side): E[starter talent] - ref, and the actual gdiff.

    ref_kind 'ewma': ref = the team's usual-starter reference of
      goalie_game_talent (EWMA of past starters' talents at the time).
    ref_kind 'mix': ref = the start-share mix's expected talent with CURRENT
      talents (the live definition, inseason.starter_diffs /
      structural.usual_starter_talent): E_model[t] - E_mix[t], mix =
      ``mix_probs`` on the situations' long EWMA share. This removes the
      in-season talent drift (current talent minus past reference) that the
      filter already sees through goals against."""
    t = np.where(sit.is_other == 1, m_other, sit.t.to_numpy())
    if ref_kind == "ewma":
        s = sit.assign(pt=p * (t - sit.ref.to_numpy()))
    elif ref_kind == "mix":
        s = sit.assign(pt=(p - mix_probs(sit)) * t)
    else:
        raise ValueError(ref_kind)
    e = s.groupby("sid").agg(gid=("gid", "first"), side=("side", "first"),
                             season_end=("season_end", "first"), egd=("pt", "sum"),
                             gdiff_act=("gdiff_act", "first"), both=("both_known", "first"))
    return e.reset_index(drop=True)
