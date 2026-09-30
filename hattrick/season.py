"""Season simulator: standings, playoff bracket and Cup odds.

Design choices (and why they differ from NeurHL)
- Team strength is UNCERTAIN, not fixed. Each simulated season draws every
  team's offence/defence from its preseason posterior, then lets it drift in a
  random walk through the season. A season simulator that holds strength fixed
  and only rolls game-level dice understates how far teams can finish from
  their projection, which makes playoff and Cup odds overconfident.
- Every game is simulated from the same scoring model the game forecasts use,
  so standings, goals and the game file agree by construction.
- Completed games are taken as played, so the same code serves the preseason
  freeze and every in-season update.
- Standings follow the NHL rules in force since 2019-20: 2 points for a win,
  1 for an overtime/shootout loss; ranking by points, then regulation wins,
  regulation+OT wins, wins, goal differential, goals for (head-to-head points
  are approximated by a random draw, which only matters for exact ties that
  survive five criteria). Playoffs: top three per division plus two wild cards
  per conference, division winners meet the wild cards, best-of-seven with
  2-2-1-1-1 home ice and sudden-death overtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hattrick import config as C


# ---------------------------------------------------------------------------
# Game-probability interface
# ---------------------------------------------------------------------------
@dataclass
class ScoringModel:
    """Minimal scoring model used by the simulator.

    lam = exp(mu + h*home + o_att + d_def) regulation goals.  Regulation goals
    are drawn from a Poisson pair with an extra probability mass on ties
    (`tie_inflation`: the share of would-be one-goal games that end level, a
    stand-in for late-game score effects).  Overtime is won with a logistic
    function of the log-rate difference; shootouts are coin flips tilted by
    `so_skill`.  hattrick.gamemodel provides the fitted version; this default
    is what the simulator falls back to.
    """
    mu: float = np.log(3.05)
    home: float = 0.045
    tie_inflation: float = 0.07
    p_ot_given_tie: float = 0.68        # share of tied games decided in OT (vs SO)
    ot_skill: float = 1.2               # logistic slope of OT win on (lam_h - lam_a)/(lam_h + lam_a) * 2
    so_skill: float = 0.3
    extra: dict = field(default_factory=dict)

    def rates(self, o_h, d_h, o_a, d_a, adj_h=0.0, adj_a=0.0):
        lam_h = np.exp(self.mu + self.home + o_h + d_a + adj_h)
        lam_a = np.exp(self.mu + o_a + d_h + adj_a)
        return lam_h, lam_a

    def probs(self, lam_h, lam_a, kmax: int = 15):
        """Analytic outcome probabilities matching `sample`.

        Returns dict of arrays: hreg, areg, tie, h_ot, h_so (unconditional
        probabilities that home wins in OT / in the SO), p_home, exp_pts_home,
        exp_pts_away.
        """
        from scipy.stats import poisson
        lam_h = np.atleast_1d(np.asarray(lam_h, float))
        lam_a = np.atleast_1d(np.asarray(lam_a, float))
        k = np.arange(kmax)
        ph = poisson.pmf(k[None, :], lam_h[:, None])
        pa = poisson.pmf(k[None, :], lam_a[:, None])
        joint = ph[:, :, None] * pa[:, None, :]
        diff = k[:, None] - k[None, :]
        p_tie0 = (joint * (diff == 0)).sum((1, 2))
        p_h1 = (joint * (diff == 1)).sum((1, 2))
        p_a1 = (joint * (diff == -1)).sum((1, 2))
        p_hw0 = (joint * (diff > 0)).sum((1, 2))
        p_aw0 = (joint * (diff < 0)).sum((1, 2))
        t = self.tie_inflation
        tie = p_tie0 + t * (p_h1 + p_a1)
        hreg = p_hw0 - t * p_h1
        areg = p_aw0 - t * p_a1
        edge = (lam_h - lam_a) / (lam_h + lam_a)
        p_ot_home = 1.0 / (1.0 + np.exp(-self.ot_skill * 2.0 * edge))
        p_so_home = 1.0 / (1.0 + np.exp(-self.so_skill * 2.0 * edge))
        h_ot = tie * self.p_ot_given_tie * p_ot_home
        a_ot = tie * self.p_ot_given_tie * (1 - p_ot_home)
        h_so = tie * (1 - self.p_ot_given_tie) * p_so_home
        p_home = hreg + h_ot + h_so
        exp_h = 2 * p_home + (tie - h_ot - h_so)
        exp_a = 2 * (1 - p_home) + (h_ot + h_so)
        return {"hreg": hreg, "areg": areg, "tie": tie, "h_ot": h_ot, "h_so": h_so,
                "a_ot": a_ot, "p_home": p_home, "exp_pts_home": exp_h,
                "exp_pts_away": exp_a}

    def sample(self, lam_h, lam_a, rng):
        """Return (home regulation goals, away regulation goals, extra, home_wins).

        extra: 0 regulation, 1 overtime, 2 shootout.
        """
        n = lam_h.shape[0]
        gh = rng.poisson(lam_h)
        ga = rng.poisson(lam_a)
        # tie inflation: a share of one-goal games are pulled level (the
        # trailing team scores late / the leader fails to extend).
        one = np.abs(gh - ga) == 1
        pull = one & (rng.random(n) < self.tie_inflation)
        gh = np.where(pull & (gh > ga), gh - 1, gh)
        ga = np.where(pull & (ga > gh), ga - 1, ga)
        tie = gh == ga
        edge = (lam_h - lam_a) / (lam_h + lam_a)
        ot = tie & (rng.random(n) < self.p_ot_given_tie)
        so = tie & ~ot
        p_ot_home = 1.0 / (1.0 + np.exp(-self.ot_skill * 2.0 * edge - 0.0))
        p_so_home = 1.0 / (1.0 + np.exp(-self.so_skill * 2.0 * edge))
        u = rng.random(n)
        home_wins = np.where(ot, u < p_ot_home,
                             np.where(so, u < p_so_home, gh > ga))
        extra = np.where(ot, 1, np.where(so, 2, 0))
        return gh, ga, extra, home_wins


class FittedModel:
    """Adapter exposing hattrick.gamemodel (fitted end-game layer, era OT/SO
    model, rest/travel offsets) through the simulator's interface."""

    def __init__(self, P: dict | None = None, season_end: int = C.TARGET_SEASON):
        from hattrick import gamemodel as GM
        self.GM = GM
        self.P = P if P is not None else GM.load_params()
        self.season_end = season_end
        self.tie_inflation = 0.0   # the fitted layer is inside GM; kept for API parity

    def rates(self, o_h, d_h, o_a, d_a, adj_h=0.0, adj_a=0.0):
        lh, la = self.GM.rates(self.P, np.asarray(o_h), np.asarray(d_h),
                               np.asarray(o_a), np.asarray(d_a))
        return lh * np.exp(adj_h), la * np.exp(adj_a)

    def sample(self, lam_h, lam_a, rng):
        s = self.GM.sample(lam_h, lam_a, self.P, rng)
        ext = np.asarray(s["extra"])
        if ext.dtype.kind in "OUS":
            ext = np.select([ext == "OT", ext == "SO"], [1, 2], 0)
        return (np.asarray(s["reg_h"]), np.asarray(s["reg_a"]), ext.astype(int),
                np.asarray(s["home_win"]).astype(bool))

    def probs(self, lam_h, lam_a, kmax: int = 15):
        p = self.GM.outcome_probs(np.atleast_1d(lam_h), np.atleast_1d(lam_a), self.P)
        return _standard_probs(p)

    def game_adjustments(self, schedule) -> pd.DataFrame:
        """Rest/travel log-rate offsets for every scheduled game."""
        sch = self.GM.schedule_features(schedule, self.season_end)
        ah, aa = self.GM.ctx_offsets(self.P, sch)
        return pd.DataFrame({"game_id": schedule.game_id.to_numpy(),
                             "adj_h": np.asarray(ah, float), "adj_a": np.asarray(aa, float)})


def _standard_probs(p: dict) -> dict:
    """Map gamemodel.outcome_probs output onto the simulator's keys."""
    keys = {k.lower(): k for k in p}

    def get(*names):
        for n in names:
            if n in p:
                return np.asarray(p[n], float)
        raise KeyError(f"none of {names} in outcome_probs keys {list(p)}")

    hreg = get("p_home_reg", "hreg")
    areg = get("p_away_reg", "areg")
    tie = 1.0 - hreg - areg
    h_ot = get("p_home_ot", "h_ot")
    h_so = get("p_home_so", "h_so")
    a_ot = get("p_away_ot", "a_ot")
    p_home = hreg + h_ot + h_so
    return {"hreg": hreg, "areg": areg, "tie": tie, "h_ot": h_ot, "h_so": h_so,
            "a_ot": a_ot, "p_home": p_home,
            "exp_pts_home": 2 * p_home + (tie - h_ot - h_so),
            "exp_pts_away": 2 * (1 - p_home) + (h_ot + h_so)}


# ---------------------------------------------------------------------------
# Strength paths
# ---------------------------------------------------------------------------
def strength_paths(teams, o, d, o_sd, d_sd, n_sims, n_steps, drift_sd, rng,
                   rho_od=0.0):
    """Draw season strength per simulation, plus a random walk within season.

    Returns arrays of shape (n_steps, n_sims, n_teams) for o and d.
    `drift_sd` is the SD of the total drift over a full season (split over
    n_steps increments), in log-rate units.
    """
    T = len(teams)
    z1 = rng.standard_normal((n_sims, T))
    z2 = rho_od * z1 + np.sqrt(1 - rho_od ** 2) * rng.standard_normal((n_sims, T))
    o0 = o[None, :] + o_sd[None, :] * z1
    d0 = d[None, :] + d_sd[None, :] * z2
    if n_steps <= 1 or drift_sd <= 0:
        return o0[None], d0[None]
    step = drift_sd / np.sqrt(n_steps)
    inc_o = rng.standard_normal((n_steps, n_sims, T)) * step
    inc_d = rng.standard_normal((n_steps, n_sims, T)) * step
    # centre the walk so the season-average strength equals the draw
    wo = np.cumsum(inc_o, axis=0)
    wd = np.cumsum(inc_d, axis=0)
    wo -= wo.mean(axis=0, keepdims=True)
    wd -= wd.mean(axis=0, keepdims=True)
    return o0[None] + wo, d0[None] + wd


# ---------------------------------------------------------------------------
# Regular season
# ---------------------------------------------------------------------------
@dataclass
class SeasonResult:
    teams: list
    points: np.ndarray      # (n_sims, T)
    wins: np.ndarray
    rw: np.ndarray
    row: np.ndarray
    otl: np.ndarray
    gf: np.ndarray          # includes OT goals and one goal per SO win
    ga: np.ndarray
    gp: np.ndarray
    sol: np.ndarray | None = None            # shootout losses (their SO "goal" is in ga)
    playoff_seed: np.ndarray | None = None   # (n_sims, T) seed 1..8 in conf, 0 = out
    rounds: np.ndarray | None = None         # (n_sims, T) rounds won 0..4
    game_home_win: np.ndarray | None = None  # (n_games,) P(home win) over sims


def simulate(schedule: pd.DataFrame, ratings: pd.DataFrame, model: ScoringModel,
             n_sims: int = 20000, seed: int = C.SEED, drift_sd: float = 0.06,
             n_steps: int = 8, completed: pd.DataFrame | None = None,
             game_adj: pd.DataFrame | None = None, playoffs: bool = True,
             rho_od: float = 0.0) -> SeasonResult:
    """Simulate the regular season (and playoffs).

    schedule: game_id, date, home, away (all games of the season)
    ratings: team, o, d, o_sd, d_sd (log-rate scale)
    completed: game_id, home_g, away_g, extra ('REG'/'OT'/'SO') for games
        already played; these are fixed, not simulated.
    game_adj: optional game_id, adj_h, adj_a log-rate adjustments per game
        (rest, travel, goalie starts) from hattrick.gamemodel.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(ratings.team)
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    r = ratings.set_index("team").loc[teams]
    sch = schedule.sort_values(["date", "game_id"]).reset_index(drop=True)
    G = len(sch)
    step_of = np.minimum((np.arange(G) * n_steps) // max(G, 1), n_steps - 1)
    O, D = strength_paths(teams, r.o.to_numpy(float), r.d.to_numpy(float),
                          r.o_sd.to_numpy(float), r.d_sd.to_numpy(float),
                          n_sims, n_steps, drift_sd, rng, rho_od)
    if O.shape[0] == 1:
        step_of = np.zeros(G, int)
    adj = {}
    if game_adj is not None:
        adj = game_adj.set_index("game_id")[["adj_h", "adj_a"]].to_dict("index")
    done = {}
    if completed is not None and len(completed):
        done = completed.set_index("game_id")[["home_g", "away_g", "extra"]].to_dict("index")

    pts = np.zeros((n_sims, T), np.int32)
    w = np.zeros_like(pts); rw = np.zeros_like(pts); row = np.zeros_like(pts)
    otl = np.zeros_like(pts); gf = np.zeros_like(pts); ga = np.zeros_like(pts)
    gp = np.zeros_like(pts)
    sol = np.zeros_like(pts)
    p_home = np.zeros(G)
    for k, g in enumerate(sch.itertuples(index=False)):
        h, a = idx[g.home], idx[g.away]
        if g.game_id in done:
            res = done[g.game_id]
            hg, ag, ex = int(res["home_g"]), int(res["away_g"]), res["extra"]
            hw = hg > ag
            extra = {"REG": 0, "OT": 1, "SO": 2}[ex]
            # standings convention: a shootout counts as one goal for the winner
            hg_s = np.full(n_sims, hg); ag_s = np.full(n_sims, ag)
            if extra == 2:
                # the recorded SO score already includes the decisive goal
                pass
            hw = np.full(n_sims, hw)
            ext = np.full(n_sims, extra)
            p_home[k] = float(hw[0])
        else:
            s = step_of[k]
            ah, aa = (adj[g.game_id]["adj_h"], adj[g.game_id]["adj_a"]) if g.game_id in adj else (0.0, 0.0)
            lam_h, lam_a = model.rates(O[s, :, h], D[s, :, h], O[s, :, a], D[s, :, a], ah, aa)
            gh, ga_, ext, hw = model.sample(lam_h, lam_a, rng)
            # add the deciding extra-time goal (OT goal or SO "goal") to the winner
            hg_s = gh + ((ext > 0) & hw)
            ag_s = ga_ + ((ext > 0) & ~hw)
            p_home[k] = hw.mean()
        _book(h, a, hg_s, ag_s, ext, hw, pts, w, rw, row, otl, gf, ga, gp)
        so = ext == 2
        sol[:, h] += so & ~hw
        sol[:, a] += so & hw
    res = SeasonResult(teams, pts, w, rw, row, otl, gf, ga, gp, sol=sol, game_home_win=p_home)
    if playoffs:
        seeds = seed_playoffs(res, rng)
        res.playoff_seed = seeds
        # playoff strength: each team's end-of-season strength in that sim
        res.rounds = simulate_playoffs(res, O[-1], D[-1], model, rng)
    return res


def _book(h, a, hg, ag, ext, hw, pts, w, rw, row, otl, gf, ga, gp):
    aw = ~hw
    reg = ext == 0
    pts[:, h] += 2 * hw + ((~hw) & ~reg)
    pts[:, a] += 2 * aw + ((~aw) & ~reg)
    w[:, h] += hw; w[:, a] += aw
    rw[:, h] += hw & reg; rw[:, a] += aw & reg
    row[:, h] += hw & (ext < 2); row[:, a] += aw & (ext < 2)
    otl[:, h] += (~hw) & ~reg; otl[:, a] += (~aw) & ~reg
    gf[:, h] += hg; ga[:, h] += ag
    gf[:, a] += ag; ga[:, a] += hg
    gp[:, h] += 1; gp[:, a] += 1


# ---------------------------------------------------------------------------
# Ranking and playoffs
# ---------------------------------------------------------------------------
def _rank_key(res: SeasonResult, rng) -> np.ndarray:
    """Composite sort key (bigger is better) encoding the NHL tiebreakers."""
    pct = res.points / np.maximum(2 * res.gp, 1)
    gd = res.gf - res.ga
    noise = rng.random(res.points.shape)
    # lexicographic: points pct, RW, ROW, W, GD, GF, random
    return (pct * 1e12 + res.rw * 1e9 + res.row * 1e6 + res.wins * 1e3
            + (gd + 500) * 1.0 + res.gf * 1e-3 + noise * 1e-6)


def seed_playoffs(res: SeasonResult, rng) -> np.ndarray:
    """Seed 1-8 per conference for each simulation (0 = missed).

    Seeds 1-3 / 4-6: division places (the better division winner is seed 1),
    7-8: wild cards. Stored as 'conference seed' where 1 = best division
    winner, 2 = other division winner, 3-6 division 2nd/3rd, 7-8 wild cards.
    """
    key = _rank_key(res, rng)
    res.key = key
    teams = res.teams
    n = key.shape[0]
    seeds = np.zeros(key.shape, np.int8)
    col = {t: i for i, t in enumerate(teams)}
    for conf, divs in C.CONFERENCES.items():
        div_top = {}
        qualified = np.zeros((n, len(teams)), bool)
        winners = []
        for dv in divs:
            ids = np.array([col[t] for t in C.DIVISIONS[dv] if t in col])
            order = np.argsort(-key[:, ids], axis=1)
            top3 = ids[order[:, :3]]
            div_top[dv] = top3
            winners.append(top3[:, 0])
            for j in range(3):
                qualified[np.arange(n), top3[:, j]] = True
        conf_ids = np.array([col[t] for dv in divs for t in C.DIVISIONS[dv] if t in col])
        rest_key = np.where(qualified[:, conf_ids], -np.inf, key[:, conf_ids])
        wc = conf_ids[np.argsort(-rest_key, axis=1)[:, :2]]
        w0, w1 = winners
        first_is_0 = key[np.arange(n), w0] > key[np.arange(n), w1]
        d0, d1 = div_top[divs[0]], div_top[divs[1]]
        ar = np.arange(n)
        # seeds: 1 best div winner, 2 other div winner, 3/4 = 2nd of div(1)/(2),
        # 5/6 = 3rd of div(1)/(2) (div(1) = division of seed 1), 7/8 wild cards
        best_d = np.where(first_is_0[:, None], d0, d1)
        other_d = np.where(first_is_0[:, None], d1, d0)
        seeds[ar, best_d[:, 0]] = 1
        seeds[ar, other_d[:, 0]] = 2
        seeds[ar, best_d[:, 1]] = 3
        seeds[ar, other_d[:, 1]] = 4
        seeds[ar, best_d[:, 2]] = 5
        seeds[ar, other_d[:, 2]] = 6
        seeds[ar, wc[:, 0]] = 7
        seeds[ar, wc[:, 1]] = 8
    return seeds


def _series(p_game_home_team, rng, n):
    """Best-of-7 with 2-2-1-1-1; p_game_home_team[i] = P(higher seed wins at home),
    tuple of (p at home, p on road). Returns bool array: higher seed wins."""
    ph, pr = p_game_home_team
    home_games = np.array([1, 1, 0, 0, 1, 0, 1], bool)
    wins = np.zeros(n, np.int8); losses = np.zeros(n, np.int8)
    for hg in home_games:
        live = (wins < 4) & (losses < 4)
        p = ph if hg else pr
        u = rng.random(n) < p
        wins += (live & u).astype(np.int8)
        losses += (live & ~u).astype(np.int8)
    return wins == 4


def _p_playoff_game(model, o_h, d_h, o_a, d_a):
    """P(home team wins a playoff game): regulation + sudden-death overtime
    (no shootouts), from the same scoring model as the regular season."""
    lam_h, lam_a = model.rates(o_h, d_h, o_a, d_a)
    p = model.probs(lam_h, lam_a)
    p_ot_home = p["h_ot"] / np.maximum(p["h_ot"] + p["a_ot"], 1e-12)
    return p["hreg"] + p["tie"] * p_ot_home


def simulate_playoffs(res: SeasonResult, O, D, model, rng) -> np.ndarray:
    """Rounds won (0-4) per team per simulation."""
    n, T = res.points.shape
    rounds = np.zeros((n, T), np.int8)
    seeds = res.playoff_seed
    ar = np.arange(n)
    key = res.key

    def team_with_seed(conf_ids, s):
        m = seeds[:, conf_ids] == s
        return conf_ids[np.argmax(m, axis=1)]

    def play(hi, lo):
        """hi has home ice (better regular-season key)."""
        swap = key[ar, lo] > key[ar, hi]
        h = np.where(swap, lo, hi); a = np.where(swap, hi, lo)
        ph = _p_playoff_game(model, O[ar, h], D[ar, h], O[ar, a], D[ar, a])
        pr = 1 - _p_playoff_game(model, O[ar, a], D[ar, a], O[ar, h], D[ar, h])
        hw = _series((ph, pr), rng, n)
        win = np.where(hw, h, a)
        rounds[ar, win] += 1
        return win

    col = {t: i for i, t in enumerate(res.teams)}
    finalists = []
    for conf, divs in C.CONFERENCES.items():
        conf_ids = np.array([col[t] for dv in divs for t in C.DIVISIONS[dv] if t in col])
        s = {k: team_with_seed(conf_ids, k) for k in range(1, 9)}
        # 1 v WC2 (8), 2 v WC1 (7), 3 v 5, 4 v 6 (division brackets)
        a1 = play(s[1], s[8]); b1 = play(s[3], s[5])
        a2 = play(s[2], s[7]); b2 = play(s[4], s[6])
        sf1 = play(a1, b1); sf2 = play(a2, b2)
        finalists.append(play(sf1, sf2))
    play(finalists[0], finalists[1])
    return rounds


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarise(res: SeasonResult) -> pd.DataFrame:
    """Per-team distribution summary."""
    rows = []
    for i, t in enumerate(res.teams):
        p = res.points[:, i]
        row = {"team": t, "conf": C.CONF_OF.get(t), "div": C.DIV_OF.get(t),
               "points": p.mean(), "points_sd": p.std(),
               "points_p10": np.percentile(p, 10), "points_p50": np.percentile(p, 50),
               "points_p90": np.percentile(p, 90),
               "w": res.wins[:, i].mean(), "l": (res.gp[:, i] - res.wins[:, i] - res.otl[:, i]).mean(),
               "otl": res.otl[:, i].mean(), "rw": res.rw[:, i].mean(), "row": res.row[:, i].mean(),
               "gf": res.gf[:, i].mean(), "ga": res.ga[:, i].mean(), "gp": res.gp[:, i].mean(),
               "gf_sd": res.gf[:, i].std(), "ga_sd": res.ga[:, i].std(),
               "so_losses": res.sol[:, i].mean() if res.sol is not None else np.nan}
        if res.playoff_seed is not None:
            s = res.playoff_seed[:, i]
            row["playoff_pct"] = 100 * (s > 0).mean()
            row["division_pct"] = 100 * np.isin(s, [1, 2]).mean()
        if res.rounds is not None:
            r = res.rounds[:, i]
            row["round2_pct"] = 100 * (r >= 1).mean()
            row["conf_final_pct"] = 100 * (r >= 2).mean()
            row["cup_final_pct"] = 100 * (r >= 3).mean()
            row["cup_pct"] = 100 * (r >= 4).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    if res.playoff_seed is not None:
        best = res.key.argmax(axis=1)
        out["presidents_pct"] = [100 * (best == i).mean() for i in range(len(res.teams))]
    return out.sort_values("points", ascending=False).reset_index(drop=True)
