"""NeurHL 1.0: one model for every level of the 2026-27 season (PLAN_NeurHL_1_0.md).

The game engine (the NeurHL-G bundle named in configs/live_models.json, stacked
with Elo and NeurHL-H exactly as the game-day forecasts are) is evaluated on
all 1,344 scheduled games K times, each time with lineups drawn by
sim/availability_2027.py (absences, injuries, goalie starts). Every other
number is a sum over those evaluations, so the levels agree by construction:

  player-games  the engine's expectations for each dressed skater and goalie,
                averaged over the draws (a player who does not dress in a
                draw contributes zero, so games played is a sum of dress
                probabilities)
  games         stacked home-win probability, outcome4, team goals, shots,
                xG and power plays, averaged over the draws
  season        20,000 simulated seasons from the per-game outcome4 with one
                team-strength shock per team and season (sd TEAM_SIGMA = 0.07
                on the log goal rate, calibrated in NeurHL-2), acting through
                the engine's own sensitivity of the win logit to the goal-rate
                ratio; NHL division format, tiebreakers, playoff bracket and
                Cup odds from ratings fitted to the game probabilities
  totals        team and player season totals are sums of the game and
                player-game expectations; the stat sheet's goal level uses the
                frozen A1 multiplier (live/goal_calibration.py); additional
                statistics are per-60 rates (data/build_player_rates.py) times
                the engine's ice time, with faceoffs balanced within each game

Preseason convention: no 2026-27 game has been played, so every game is
evaluated with the state as of opening night and days-into-season at its
opening value (a mid-season date with an empty season would be outside
anything the engine was trained on); NeurHL-H's projection for each team is
its opening-night projection from the expected lineup.

Writes neurhl/output/neurhl_1_0/: games_2027.csv, teams_2027.csv,
skaters_2027.csv, goalies_2027.csv, player_games_2027.csv.gz, consistency_2027.json
and run_2027.json.

Usage: uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow \
         --with numba --with torch --with scipy --with scikit-learn==1.9.1 --with requests \
         python neurhl/sim/unified_2027.py --rosters-date 2026-09-28 [--draws 32] [--sims 20000]
"""
import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT))
from common import CONFIGS, NOUT, TENSORS  # noqa: E402
import sim.g_live as GL  # noqa: E402
from sim.g_forecast_core import load_bundle, raw_outputs, sha  # noqa: E402
from sim.game_model import TEAM_SIGMA  # noqa: E402
from sim.project_2027 import load_schedule  # noqa: E402
from sim.schedule_context import build as sched_ctx  # noqa: E402

OUT = NOUT / "neurhl_1_0"
SEASON = 2027
P_SO_GIVEN_TIE = 0.38      # 2016-2026 share of ties decided in a shootout (sim/boxscore_mc)
FO_PER_GAME = 57.0         # fallback when the rates file carries no league value
SK_STATS = ["toi_ev", "toi_pp", "toi_sh", "sog", "att", "ixg", "g", "a", "oi_xgf", "oi_xga"]


def _live(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "live" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def divisions() -> tuple[dict, dict]:
    h = pd.read_csv(PROJ / "output" / "projections_2026_27_howe.csv")
    div = dict(zip(h.Abbr, h.Division))
    conf = {t: ("E" if d in ("Atlantic", "Metropolitan") else "W") for t, d in div.items()}
    return div, conf


# ------------------------------------------------------------------ lineups
class FallbackAvailability:
    """Used only when sim/availability_2027.py is absent: every game dresses the
    resolver's opening-night lineup and the starter plays every game."""

    def __init__(self, games, rosters_date):
        R = _live("lineup_resolver")
        nodf = tempfile.mkdtemp()
        self.lu = {}
        for r in games.sort_values("date").itertuples():
            for team, side in ((r.home, "home"), (r.away, "away")):
                if team in self.lu:
                    continue
                x = R.resolve(r.game_id, r.date, r.home, r.away, snapshot_dir=nodf,
                              rosters_date=rosters_date, use_api=False,
                              as_of=dt.datetime.fromisoformat(f"{r.date}T15:00:00+00:00"))
                self.lu[team] = {"skaters": list(x[side]["skaters"]), "goalie": x[side]["goalie"]}

    def draws(self, games, k, seed):
        one = {r.game_id: {"home": self.lu[r.home], "away": self.lu[r.away]} for r in games.itertuples()}
        return [one] * k

    def expected(self, games):
        return dict(self.lu)


class ModelAvailability:
    def __init__(self, AV, rosters_date, games):
        self.AV = AV
        self.roster = AV.load(rosters_date)
        self.games = games

    def draws(self, games, k, seed):
        return self.AV.draw_lineups(games, self.roster, k, seed)

    def expected(self, games):
        ex = self.AV.expected_lineup(games, self.roster)
        first = {}
        for r in games.sort_values("date").itertuples():
            for team, side in ((r.home, "home"), (r.away, "away")):
                first.setdefault(team, ex[r.game_id][side])
        return first


def availability(games, rosters_date):
    try:
        import sim.availability_2027 as AV
        return ModelAvailability(AV, rosters_date, games), "sim/availability_2027.py"
    except ImportError:
        return FallbackAvailability(games, rosters_date), "fallback (resolver lineup, starter always)"


# ------------------------------------------------------------------ NeurHL-H
def h_probabilities(games, expected, elo_logit):
    """Opening-night NeurHL-H projection for every team at home and away, then
    NeurHL-H's head for every scheduled game."""
    from sim.h_live import forecast as h_forecast, lineup_projection
    teams = sorted(expected)
    half = len(teams) // 2
    open_date = str(games.date.min())
    proj = {}
    for rot in (0, 1):
        pairs = [(teams[i], teams[i + half]) if rot == 0 else (teams[i + half], teams[i])
                 for i in range(half)]
        g = pd.DataFrame([{"game_id": 9_900_000 + 100 * rot + i, "date": open_date,
                           "home": h, "away": a} for i, (h, a) in enumerate(pairs)])
        lus = {r.game_id: {"home": expected[r.home], "away": expected[r.away]} for r in g.itertuples()}
        piv = lineup_projection(g, lus)
        for r in g.itertuples():
            q = piv.loc[r.game_id]
            proj[(r.home, "h")] = (q.cfpct_h, q.clsh_h)
            proj[(r.away, "a")] = (q.cfpct_a, q.clsh_a)
    piv = pd.DataFrame({"cfpct_h": [proj[(h, "h")][0] for h in games.home],
                        "clsh_h": [proj[(h, "h")][1] for h in games.home],
                        "cfpct_a": [proj[(a, "a")][0] for a in games.away],
                        "clsh_a": [proj[(a, "a")][1] for a in games.away]},
                       index=games.game_id.to_numpy())
    sc = sched_ctx(load_schedule(), SEASON)[["game_id", "home_rest", "away_rest"]]
    return h_forecast(games, None, elo_logit, sc, piv), piv


# ------------------------------------------------------------------ engine
class Canonical:
    """Opening-night state rows for every candidate skater, goalie and team.

    The state builders treat each appended row as a game played, so evaluating
    84 future games per team in one call would feed later games a state
    diluted by earlier, statless ones. Here each entity appears exactly once
    per build (a player in one chunk of at most NS skaters, each team once at
    home and once away), and every game is assembled from those rows plus its
    own schedule context (rest, back-to-back, travel, time zones)."""

    def __init__(self, games, cand):
        teams = sorted(cand)
        half = len(teams) // 2
        n_chunks = max(max(-(-len(c["skaters"]) // GL.NS) for c in cand.values()),
                       max(len(c["goalies"]) for c in cand.values()), 1)
        open_date = str(games.date.min())
        self.sk, self.gk, self.tm = {}, {}, {}
        for ch in range(n_chunks):
            for rot in (0, 1):
                pairs = [(teams[i], teams[i + half]) if rot == 0 else (teams[i + half], teams[i])
                         for i in range(half)]
                g = pd.DataFrame([{"game_id": 9_800_000 + 1000 * ch + 100 * rot + i, "date": open_date,
                                   "home": h, "away": a} for i, (h, a) in enumerate(pairs)])

                def side_lu(team):
                    sks = cand[team]["skaters"][ch * GL.NS:(ch + 1) * GL.NS] or cand[team]["skaters"][:GL.NS]
                    gks = cand[team]["goalies"]
                    return {"skaters": sks, "goalie": gks[ch] if ch < len(gks) else (gks[0] if gks else None)}

                lus = {r.game_id: {"home": side_lu(r.home), "away": side_lu(r.away)} for r in g.itertuples()}
                A, _, names = GL.build(g, lus)
                for n, r in enumerate(g.itertuples()):
                    for side, team in ((0, r.home), (1, r.away)):
                        home = 1 - side
                        for slot in np.nonzero(A["SKM"][n, side] > 0)[0]:
                            pid = int(A["SKID"][n, side, slot])
                            self.sk[(pid, home)] = (A["SK"][n, side, slot], A["SKB"][n, side, slot],
                                                    A["SKP"][n, side, slot])
                        if A["GKID"][n, side] > 0:
                            self.gk[(int(A["GKID"][n, side]), home)] = A["GK"][n, side]
                        self.tm[(team, home)] = A["TM"][n, side]
        self.names = names
        self.shapes = {k: A[k].shape[2:] for k in ("SK", "SKB")}
        self.n_gk, self.n_tm = A["GK"].shape[-1], A["TM"].shape[-1]

    def assemble(self, games, lineups, sc, elo, days0):
        names, NS = self.names, GL.NS
        N = len(games)
        SK = np.full((N, 2, NS) + self.shapes["SK"][1:], np.nan, np.float32)
        SKB = np.full((N, 2, NS) + self.shapes["SKB"][1:], np.nan, np.float32)
        SKM = np.zeros((N, 2, NS), np.float32)
        SKP = np.zeros((N, 2, NS), np.int8)
        SKID = np.zeros((N, 2, NS), np.int64)
        GK = np.full((N, 2, self.n_gk), np.nan, np.float32)
        GKID = np.zeros((N, 2), np.int64)
        TM = np.full((N, 2, self.n_tm), np.nan, np.float32)
        tf = names["tm_feat"]
        ix = {c: tf.index(c) for c in ("rest", "b2b", "km3d", "dtz")}
        missing = 0
        for n, r in enumerate(games.itertuples()):
            for side, team, key in ((0, r.home, "home"), (1, r.away, "away")):
                home = 1 - side
                lu = lineups[r.game_id][key]
                rows = [(int(pid), self.sk[(int(pid), home)]) for pid in lu["skaters"]
                        if (int(pid), home) in self.sk]
                missing += len(lu["skaters"]) - len(rows)
                rows.sort(key=lambda x: -np.nan_to_num(x[1][1][0], nan=-1.0))
                for slot, (pid, (f, b, pg)) in enumerate(rows[:NS]):
                    SK[n, side, slot], SKB[n, side, slot], SKP[n, side, slot] = f, b, pg
                    SKM[n, side, slot], SKID[n, side, slot] = 1.0, pid
                gid_ = lu.get("goalie")
                if gid_ and (int(gid_), home) in self.gk:
                    GK[n, side], GKID[n, side] = self.gk[(int(gid_), home)], int(gid_)
                t = self.tm[(team, home)].copy()
                pre = "home" if side == 0 else "away"
                rest = float(sc[f"{pre}_rest"].get(r.game_id, np.nan))
                t[ix["rest"]], t[ix["b2b"]] = rest, float(rest <= 1)
                t[ix["km3d"]] = float(sc[f"{pre}_km3d"].get(r.game_id, np.nan))
                t[ix["dtz"]] = float(sc[f"{pre}_dtz"].get(r.game_id, np.nan))
                TM[n, side] = t
        era = GL.era_2027()
        CTX = np.array([[elo[n] if c == "elo_logit" else (days0 if c == "days_in" else era[c])
                         for c in names["ctx"]] for n in range(N)], np.float32)
        A = {"SK": SK, "SKB": SKB, "SKM": SKM, "SKP": SKP, "SKID": SKID,
             "GK": GK, "GKID": GKID, "TM": TM, "CTX": CTX}
        return A, missing


def evaluate(canon, games, lineups, bundle, sc, elo, days0, lh):
    A, missing = canon.assemble(games, lineups, sc, elo, days0)
    o, meta, _ = raw_outputs(bundle, A)
    st = meta["stack"]
    cols = {"elo_logit": A["CTX"][:, 0], "lg": logit(o["p_home_win"]), "lh": lh}
    z = st["intercept"] + sum(c * cols[k] for k, c in zip(st["cols"], st["coef"]))
    if "lh" in st["cols"]:
        fb = st.get("fallback", st)
        zf = fb["intercept"] + sum(c * cols[k] for k, c in zip(fb["cols"], fb["coef"]))
        z = np.where(np.isfinite(lh), z, zf)
    p = sigmoid(z)
    o4 = o["o4"].astype(float).copy()
    hm, am = o4[:, 0] + o4[:, 2], o4[:, 1] + o4[:, 3]
    o4[:, [0, 2]] *= (p / np.maximum(hm, 1e-9))[:, None]
    o4[:, [1, 3]] *= ((1 - p) / np.maximum(am, 1e-9))[:, None]
    return A, o, p, o4, missing


def sensitivity(bundle, lam):
    """d logit P(home win) / d log(lam_h / lam_a) from the engine's own hazard
    integration, seed-averaged, at each game's goal rates."""
    import torch
    models, _, _ = load_bundle(bundle)
    eps = 0.05
    lh = torch.as_tensor(lam[:, 0], dtype=torch.float32)
    la = torch.as_tensor(lam[:, 1], dtype=torch.float32)
    up, dn = np.exp(eps / 2), np.exp(-eps / 2)
    ks = []
    with torch.no_grad():
        for m in models:
            p1 = m.outcome(lh * up, la * dn)["p_home_win"].numpy()
            p0 = m.outcome(lh * dn, la * up)["p_home_win"].numpy()
            ks.append((logit(p1) - logit(p0)) / eps)
    return np.mean(ks, 0)


# ------------------------------------------------------------------ season
def bt_ratings(games, teams, lp):
    """Least-squares ratings r (sum zero) and home edge h with logit p = h + r_home - r_away."""
    ti = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(games), len(teams) + 1))
    X[np.arange(len(games)), games.home.map(ti).to_numpy()] += 1
    X[np.arange(len(games)), games.away.map(ti).to_numpy()] -= 1
    X[:, -1] = 1
    X = np.vstack([X, np.r_[np.ones(len(teams)), 0.0]])
    y = np.r_[lp, 0.0]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[:-1], beta[-1]


def series_prob(p_hi_home, p_hi_away):
    """P(higher seed wins a best-of-seven) with home games 1, 2, 5, 7."""
    home = [1, 1, 0, 0, 1, 0, 1]
    dist = {(0, 0): 1.0}
    win = 0.0
    for g in range(7):
        nxt = {}
        for (a, b), pr in dist.items():
            pw = p_hi_home if home[g] else p_hi_away
            for (da, db, q) in ((1, 0, pw), (0, 1, 1 - pw)):
                s = (a + da, b + db)
                if s[0] == 4:
                    win += pr * q
                elif s[1] < 4:
                    nxt[s] = nxt.get(s, 0.0) + pr * q
        dist = nxt
    return win


def season_mc(games, p, o4, kg, teams, div, conf, sims, seed, sigma):
    rng = np.random.default_rng(seed)
    tie_rng = np.random.default_rng(seed + 7)
    po_rng = np.random.default_rng(seed + 11)
    T, N = len(teams), len(games)
    ti = {t: i for i, t in enumerate(teams)}
    hi = games.home.map(ti).to_numpy()
    ai = games.away.map(ti).to_numpy()
    H = np.zeros((N, T), np.float32)
    H[np.arange(N), hi] = 1
    Aw = np.zeros((N, T), np.float32)
    Aw[np.arange(N), ai] = 1
    lp = logit(p)
    rh = o4[:, 0] / np.maximum(o4[:, 0] + o4[:, 2], 1e-9)     # home wins decided in regulation
    ra = o4[:, 1] / np.maximum(o4[:, 1] + o4[:, 3], 1e-9)
    stats = {k: np.zeros((sims, T), np.int16) for k in ("pts", "w", "rw", "row", "otl")}
    shocks = np.zeros((sims, T), np.float32)
    C = 1000
    for c0 in range(0, sims, C):
        c = min(C, sims - c0)
        s = rng.normal(0.0, sigma, (c, T)).astype(np.float32)
        shocks[c0:c0 + c] = s
        pp = sigmoid(lp[None, :] + kg[None, :] * (s[:, hi] - s[:, ai]))
        hw = rng.random((c, N)) < pp
        reg = np.where(hw, rng.random((c, N)) < rh[None, :], rng.random((c, N)) < ra[None, :])
        so = rng.random((c, N)) < P_SO_GIVEN_TIE
        f = lambda m_h, m_a: (m_h.astype(np.float32) @ H + m_a.astype(np.float32) @ Aw)
        stats["w"][c0:c0 + c] = f(hw, ~hw)
        stats["rw"][c0:c0 + c] = f(hw & reg, ~hw & reg)
        stats["row"][c0:c0 + c] = f(hw & (reg | ~so), ~hw & (reg | ~so))
        stats["otl"][c0:c0 + c] = f(~hw & ~reg, hw & ~reg)
    stats["pts"] = 2 * stats["w"] + stats["otl"]
    key = (stats["pts"].astype(np.float64) * 1e7 + stats["rw"] * 1e5 + stats["row"] * 1e3
           + stats["w"] * 10 + tie_rng.random((sims, T)))
    divs = sorted(set(div[t] for t in teams))
    res = {k: np.zeros(T) for k in ("playoff", "division", "presidents", "r2", "cf", "final", "cup")}
    r, hedge = bt_ratings(games, teams, lp)
    kbar = float(np.median(kg))
    res["presidents"] += np.bincount(key.argmax(1), minlength=T)
    teams_of_div = {d: np.array([ti[t] for t in teams if div[t] == d]) for d in divs}
    for s_i in range(sims):
        ks = key[s_i]
        bracket = {}
        for cf in ("E", "W"):
            dvs = [d for d in divs if conf[teams[teams_of_div[d][0]]] == cf]
            tops, rest = {}, []
            for d in dvs:
                ids = teams_of_div[d][np.argsort(-ks[teams_of_div[d]])]
                tops[d] = ids[:3]
                res["division"][ids[0]] += 1
                rest.extend(ids[3:])
            rest = np.array(rest)
            wc = rest[np.argsort(-ks[rest])][:2]
            q = np.r_[np.concatenate([tops[d] for d in dvs]), wc]
            res["playoff"][q] += 1
            # division winner with more points meets the second wildcard
            d1, d2 = sorted(dvs, key=lambda d: -ks[tops[d][0]])
            bracket[cf] = [((tops[d1][0], wc[1]), (tops[d1][1], tops[d1][2])),
                           ((tops[d2][0], wc[0]), (tops[d2][1], tops[d2][2]))]
        sh = shocks[s_i]

        def play(a, b):
            hi_, lo_ = (a, b) if ks[a] > ks[b] else (b, a)
            d_ = r[hi_] - r[lo_] + kbar * (sh[hi_] - sh[lo_])
            pw = series_prob(sigmoid(hedge + d_), sigmoid(-hedge + d_))
            return hi_ if po_rng.random() < pw else lo_

        champs = []
        for cf in ("E", "W"):
            sides = []
            for (m1, m2) in bracket[cf]:
                a, b = play(*m1), play(*m2)
                res["r2"][[a, b]] += 1
                sides.append(play(a, b))
            res["cf"][sides] += 1
            champs.append(play(*sides))
        res["final"][champs] += 1
        res["cup"][play(*champs)] += 1
    for k in res:
        res[k] = res[k] / sims
    return stats, res, r, hedge, shocks


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rosters-date", required=True)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=711)
    ap.add_argument("--rookie-priors", help="rookies' goals and assists per game from this table "
                                            "(configs/rookie_priors_2027.csv; PLAN_NeurHL_1_1 A15)")
    ap.add_argument("--out-dir", help="write the season files here instead of output/neurhl_1_0 "
                                      "(a dated file set, e.g. output/neurhl_1_0/v2_20260929)")
    ap.add_argument("--rookie-weight", type=float, default=1.0,
                    help="weight of the translated record in rookies' goals and assists "
                         "(w x translated + (1 - w) x engine; PLAN_NeurHL_1_2 R5)")
    ap.add_argument("--rookie-weight-a", type=float, help="the assists weight, when it differs (R5 chooses "
                                                          "per statistic); default: --rookie-weight")
    ap.add_argument("--level-ratios", help="season-level ratios for the opening-night convention "
                                           "(output/neurhl_1_2/season_convention.json; PLAN_NeurHL_1_2 R1)")
    ap.add_argument("--extra-time", action="store_true",
                    help="add overtime and shootout-deciding goals to season totals (PLAN_NeurHL_1_2 R2)")
    ap.add_argument("--goal-state", help="goal-level state file with m0 (default: the live state; "
                                         "PLAN_NeurHL_1_2 R3)")
    ap.add_argument("--shot-level-last-season", action="store_true",
                    help="shots and attempts scaled so their league means over the schedule equal the "
                         "previous season's (PLAN_NeurHL_1_3 R1)")
    ap.add_argument("--goal-level-full", action="store_true",
                    help="m0 = L / M_full: L from the goal state, M_full the engine's mean regulation goals "
                         "over every scheduled game (PLAN_NeurHL_1_2 R3)")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((CONFIGS / "live_models.json").read_text())["neurhl_g"]
    games = load_schedule()[["game_id", "date", "home", "away"]].sort_values(
        ["date", "game_id"]).reset_index(drop=True)
    games["date"] = games.date.astype(str)
    N = len(games)
    teams = sorted(set(games.home))
    div, conf = divisions()
    sc = sched_ctx(load_schedule(), SEASON).set_index("game_id")
    days0 = float(sc.days_in.min())
    gc = _live("goal_calibration")
    m0 = float(json.loads((Path(a.goal_state) if a.goal_state else gc.STATE).read_text())["m0"])
    ratios = (json.loads(Path(a.level_ratios).read_text())["decision"]["ratios"] if a.level_ratios else {})
    r_ppo, r_ppm, r_xg, r_sog = (float(ratios.get(k, 1.0)) for k in ("pp_opps", "pp_m", "xgf", "sogf"))
    r_att = r_sog
    av, av_src = availability(games, a.rosters_date)
    expected = av.expected(games)
    elo = GL.elo_logits(games)
    p_h, hpiv = h_probabilities(games, expected, elo)
    lh = logit(p_h)
    print(f"[unified] {N} games, {len(teams)} teams; availability: {av_src}; "
          f"NeurHL-H ready ({time.time() - t0:.0f}s)", flush=True)

    draws = av.draws(games, a.draws, a.seed)
    cand = {t: {"skaters": [], "goalies": []} for t in teams}
    for lineups in list(draws) + [{r.game_id: {"home": expected[r.home], "away": expected[r.away]}
                                   for r in games.itertuples()}]:
        for r in games.itertuples():
            for team, key in ((r.home, "home"), (r.away, "away")):
                lu = lineups[r.game_id][key]
                for pid in lu["skaters"]:
                    if int(pid) not in cand[team]["skaters"]:
                        cand[team]["skaters"].append(int(pid))
                if lu.get("goalie") and int(lu["goalie"]) not in cand[team]["goalies"]:
                    cand[team]["goalies"].append(int(lu["goalie"]))
    canon = Canonical(games, cand)
    print(f"[unified] canonical opening-night states: {len(canon.sk)} skater rows, "
          f"{len(canon.gk)} goalie rows ({time.time() - t0:.0f}s)", flush=True)
    acc = {k: 0.0 for k in ("p", "o4", "goals", "sogf", "attf", "xgf", "xgf_ev", "xgf_pp", "xgf_sh",
                            "pp_opps")}
    pg_parts, gk_parts, checks = [], [], []
    gid = games.game_id.to_numpy()
    n_missing = 0
    for k, lineups in enumerate(draws):
        A, o, p, o4, miss = evaluate(canon, games, lineups, bundle, sc, elo, days0, lh)
        n_missing += miss
        acc["p"] = acc["p"] + p
        acc["o4"] = acc["o4"] + o4
        for key in ("goals", "sogf", "attf", "xgf", "xgf_ev", "xgf_pp", "xgf_sh", "pp_opps"):
            acc[key] = acc[key] + o[key].astype(float)
        m = A["SKM"] > 0
        n_i, s_i, _ = np.nonzero(m)
        part = pd.DataFrame({"game_id": gid[n_i], "side": s_i, "player_id": A["SKID"][m],
                             "toi_ev": o["toi_ev"][m], "toi_pp": o["toi_pp"][m],
                             "toi_sh": o["toi_sh"][m], "sog": o["isog"][m], "att": o["iatt"][m],
                             "ixg": o["ixg"][m], "g": o["g"][m], "a": o["a"][m],
                             "oi_xgf": o["oi_xgf"][m], "oi_xga": o["oi_xga"][m]})
        pg_parts.append(part)
        for side in (0, 1):
            opp = 1 - side
            gk_parts.append(pd.DataFrame({
                "game_id": gid, "side": side, "player_id": A["GKID"][:, side],
                "sa": o["sogf"][:, opp].astype(float), "ga": o["goals"][:, opp].astype(float),
                "win": p if side == 0 else 1 - p}))
        # conservation inside every evaluation: team totals equal the sum of the skaters
        for key, tkey in (("g", "goals"), ("isog", "sogf"), ("ixg", "xgf")):
            diff = np.abs(np.nansum(np.where(m, o[key], 0), -1) - o[tkey]).max()
            checks.append({"draw": k, "stat": key, "max_abs_team_minus_players": float(diff)})
        print(f"[unified] draw {k + 1}/{len(draws)} ({time.time() - t0:.0f}s)", flush=True)
    K = len(draws)
    for key in acc:
        acc[key] = acc[key] / K
    p, o4 = acc["p"], acc["o4"]
    M_full = float(acc["goals"].mean())
    shot_level = {}
    if a.shot_level_last_season:
        # 1.3 R1: the season's league shot and attempt levels are last season's (the opening-night
        # states lag a league-wide change, and the season set never updates them)
        prev = pd.read_parquet(TENSORS / f"tgx_{SEASON - 1}.parquet", columns=["sogf", "attf"]).mean()
        M_sog, M_att = float(acc["sogf"].mean()), float(acc["attf"].mean())
        r_sog, r_att = float(prev.sogf) / M_sog, float(prev.attf) / M_att
        shot_level = {"L_sog": float(prev.sogf), "M_sog": M_sog, "m_sog": r_sog,
                      "L_att": float(prev.attf), "M_att": M_att, "m_att": r_att}
        print(f"[unified] shot level: shots {M_sog:.3f} -> {prev.sogf:.3f} (x{r_sog:.4f}), "
              f"attempts {M_att:.3f} -> {prev.attf:.3f} (x{r_att:.4f})", flush=True)
    if a.goal_level_full:
        L_ = float(json.loads((Path(a.goal_state) if a.goal_state else gc.STATE).read_text())["L"])
        m0 = L_ / M_full
        print(f"[unified] goal level: L {L_:.4f} / M_full {M_full:.4f} = m0 {m0:.4f}", flush=True)
    kg = sensitivity(bundle, acc["goals"])

    # ---- player-games: expectations over the draws (zero when not dressed)
    pg = pd.concat(pg_parts, ignore_index=True)
    pg["dress"] = 1.0
    pg = pg.groupby(["game_id", "side", "player_id"], as_index=False)[SK_STATS + ["dress"]].sum()
    pg[SK_STATS + ["dress"]] = pg[SK_STATS + ["dress"]] / K
    pg["g"] *= m0
    pg["a"] *= m0
    gsz = games.set_index("game_id")
    pg["team"] = np.where(pg.side == 0, pg.game_id.map(gsz.home), pg.game_id.map(gsz.away))
    pg["date"] = pg.game_id.map(gsz.date)
    if ratios or shot_level:
        # R1: every game carries opening-night season-progress inputs, which overstate power
        # plays for a whole season (history: +33% opportunities, +24% PP minutes). PP and SH
        # minutes are scaled; the clock time freed goes to even strength in proportion to each
        # player's even-strength minutes. Shots and xG take their own ratios.
        keys = [pg.game_id, pg.side]
        d_pp = pg.toi_pp.groupby(keys).transform("sum") * (1 - r_ppm) / 5.0     # clock minutes
        d_sh = pg.toi_sh.groupby(keys).transform("sum") * (1 - r_ppm) / 4.0
        ev_tot = pg.toi_ev.groupby(keys).transform("sum")
        pg["toi_ev"] = pg.toi_ev + pg.toi_ev / ev_tot.clip(lower=1e-9) * 5.0 * (d_pp + d_sh)
        pg["toi_pp"] *= r_ppm
        pg["toi_sh"] *= r_ppm
        for c in ("ixg", "oi_xgf", "oi_xga"):
            pg[c] *= r_xg
        pg["sog"] *= r_sog
        pg["att"] *= r_att
    pg["toi"] = pg.toi_ev + pg.toi_pp + pg.toi_sh
    if a.rookie_priors:
        # A15: a rookie's goals and assists per game come from his translated pre-NHL record
        # (times his chance of dressing); his teammates in the same team-game are rescaled so
        # the team's totals stay the engine's
        rp = pd.read_csv(a.rookie_priors).drop_duplicates("player_id").set_index("player_id")
        isr = pg.player_id.isin(rp.index).to_numpy()
        keys = [pg.game_id, pg.side]
        for st, col in (("g", "pred_g"), ("a", "pred_a")):
            new = pg[st].to_numpy().copy()
            tr_ = pg.loc[isr, "player_id"].map(rp[col]).to_numpy() * pg.loc[isr, "dress"].to_numpy()
            w_ = a.rookie_weight_a if (st == "a" and a.rookie_weight_a is not None) else a.rookie_weight
            new[isr] = w_ * tr_ + (1 - w_) * new[isr]     # R5 blend
            tot = pg[st].groupby(keys).transform("sum").to_numpy()
            r_new = pd.Series(np.where(isr, new, 0.0)).groupby(keys).transform("sum").to_numpy()
            nr_old = pd.Series(np.where(isr, 0.0, pg[st].to_numpy())).groupby(keys).transform("sum").to_numpy()
            scale = np.clip((tot - r_new) / np.maximum(nr_old, 1e-9), 0.0, None)
            pg[st] = np.where(isr, new, pg[st].to_numpy() * scale)
        print(f"[unified] rookie priors applied to {pg.loc[isr, 'player_id'].nunique()} rookies", flush=True)
    rates_p = OUT / "player_rates_2027.csv"
    extra = []
    if rates_p.exists():
        rt = pd.read_csv(rates_p).drop_duplicates("player_id").set_index("player_id")
        league = {}
        vj = OUT / "player_rates_validation.json"
        if vj.exists():
            league = json.loads(vj.read_text()).get("league_per_team_game", {})
        for c_rate, c_out in (("hits60", "hits"), ("blocks60", "blocks"), ("giveaways60", "giveaways"),
                              ("takeaways60", "takeaways"), ("pim60", "pim"),
                              ("pen_taken60", "pen_taken"), ("pen_drawn60", "pen_drawn"),
                              ("fo_taken60", "fo_taken")):
            if c_rate in rt:
                pos_mean = rt.groupby("pos_group")[c_rate].mean() if "pos_group" in rt else None
                rate = pg.player_id.map(rt[c_rate])
                if pos_mean is not None:
                    rate = rate.fillna(pg.player_id.map(rt.pos_group).map(pos_mean)).fillna(rt[c_rate].mean())
                pg[c_out] = rate.fillna(rt[c_rate].mean()) * pg.toi / 60.0
                extra.append(c_out)
        if "fo_taken" in pg and "fo_win_share" in rt:
            # each faceoff is taken by one player per team: team faceoffs equal the game's
            # faceoffs, and the two teams' wins sum to them
            fo_game = float(league.get("faceoffs_per_game", FO_PER_GAME))
            tot = pg.groupby(["game_id", "side"]).fo_taken.transform("sum")
            dress_team = pg.groupby(["game_id", "side"]).dress.transform("sum") / 18.0
            pg["fo_taken"] = np.where(tot > 0, pg.fo_taken / tot * fo_game * dress_team.clip(upper=1), 0)
            share = pg.player_id.map(rt.fo_win_share).fillna(0.5)
            pg["fo_won_raw"] = pg.fo_taken * share
            won = pg.groupby(["game_id", "side"]).fo_won_raw.sum().unstack()
            tot_g = won.sum(1)
            scale = {}
            for (g_, s_), v in pg.groupby(["game_id", "side"]).fo_won_raw.sum().items():
                scale[(g_, s_)] = (fo_game * won.loc[g_, s_] / max(tot_g.loc[g_], 1e-9)) / max(v, 1e-9)
            pg["fo_won"] = pg.fo_won_raw * [scale[(g_, s_)] for g_, s_ in zip(pg.game_id, pg.side)]
            pg["fo_lost"] = pg.fo_taken - pg.fo_won
            pg = pg.drop(columns="fo_won_raw")
            extra += ["fo_won", "fo_lost"]

    # ---- goalie-games
    gk = pd.concat(gk_parts, ignore_index=True)
    gk = gk[gk.player_id > 0]
    gk["shutout"] = np.exp(-gk.ga * m0)          # per draw, at the final goal level
    gk["starts"] = 1.0
    gk = gk.groupby(["game_id", "side", "player_id"], as_index=False)[
        ["starts", "sa", "ga", "win", "shutout"]].sum()
    gk[["starts", "sa", "ga", "win", "shutout"]] = gk[["starts", "sa", "ga", "win", "shutout"]] / K
    gk["ga"] *= m0
    gk["sa"] *= r_sog
    gk["team"] = np.where(gk.side == 0, gk.game_id.map(gsz.home), gk.game_id.map(gsz.away))

    # ---- games
    gm = games.copy()
    gm["p_home_win"] = p
    for j, c in enumerate(("p_home_reg", "p_away_reg", "p_home_ot", "p_away_ot")):
        gm[c] = o4[:, j]
    gm["p_ot"] = o4[:, 2] + o4[:, 3]
    gm["p_home_win_elo"] = sigmoid(elo)
    gm["p_home_win_h"] = p_h
    for key, lab in (("goals", "goals"), ("sogf", "sog"), ("xgf", "xgf"), ("pp_opps", "pp_opps"),
                     ("attf", "attempts")):
        mult = {"goals": m0, "sogf": r_sog, "attf": r_att, "xgf": r_xg, "pp_opps": r_ppo}[key]
        gm[f"{lab}_home"] = acc[key][:, 0] * mult
        gm[f"{lab}_away"] = acc[key][:, 1] * mult
    # strength split of xG: with R1 ratios, PP and SH xG follow the scaled PP minutes and
    # even-strength xG the freed clock time (pp_clock: the scaled PP minutes per team-game)
    xs = {k: acc[k].astype(float).copy() for k in ("xgf_ev", "xgf_pp", "xgf_sh")}
    if ratios:
        clk = pg.assign(pp=pg.toi_pp / 5.0, sh=pg.toi_sh / 4.0).groupby(["game_id", "side"])[["pp", "sh"]].sum()
        for j in (0, 1):
            c = clk.xs(j, level="side").reindex(gm.game_id)
            new_st = (c.pp + c.sh).to_numpy()
            old_st = new_st / r_ppm
            xs["xgf_pp"][:, j] *= r_ppm
            xs["xgf_sh"][:, j] *= r_ppm
            xs["xgf_ev"][:, j] *= (60.0 - new_st) / np.maximum(60.0 - old_st, 1e-9)
    # power-play goals: each team's goals split by strength in proportion to its xG
    for j, side in ((0, "home"), (1, "away")):
        share = xs["xgf_pp"][:, j] / np.maximum(xs["xgf_ev"][:, j] + xs["xgf_pp"][:, j] + xs["xgf_sh"][:, j], 1e-9)
        gm[f"pp_goals_{side}"] = gm[f"goals_{side}"] * share
    if a.extra_time:
        # R2: the engine's goals are regulation goals. A game won past regulation adds one goal
        # to the winner's GF (overtime or shootout, as in the standings); only overtime goals
        # count for skaters and goalies. Skater goals and assists in the team-game grow by the
        # same factor (the team's overtime goals over its regulation goals).
        for side, j in (("home", 2), ("away", 3)):
            gm[f"goals_{side}_reg"] = gm[f"goals_{side}"]
            gm[f"ot_goals_{side}"] = o4[:, j] * (1 - P_SO_GIVEN_TIE)
            gm[f"so_goals_{side}"] = o4[:, j] * P_SO_GIVEN_TIE
            gm[f"goals_{side}"] = gm[f"goals_{side}_reg"] + gm[f"ot_goals_{side}"] + gm[f"so_goals_{side}"]
        g_ix = gm.set_index("game_id")
        ot_for = np.where(pg.side == 0, pg.game_id.map(g_ix.ot_goals_home), pg.game_id.map(g_ix.ot_goals_away))
        reg_for = np.where(pg.side == 0, pg.game_id.map(g_ix.goals_home_reg), pg.game_id.map(g_ix.goals_away_reg))
        f_ = 1 + ot_for / np.maximum(reg_for, 1e-9)
        pg["g"] *= f_
        pg["a"] *= f_
        ot_against = np.where(gk.side == 0, gk.game_id.map(g_ix.ot_goals_away), gk.game_id.map(g_ix.ot_goals_home))
        gk["ga"] = gk.ga + gk.starts * ot_against
    gm["rate_sensitivity"] = kg
    old = pd.read_csv(NOUT / "games_2027.csv").set_index("game_id")
    gm["p_home_win_0925"] = gm.game_id.map(old.p_home_win)

    # ---- season
    stats, res, rating, hedge, shocks = season_mc(games, p, o4, kg, teams, div, conf,
                                                 a.sims, a.seed, TEAM_SIGMA)
    tm = pd.DataFrame({"team": teams, "conf": [conf[t] for t in teams], "div": [div[t] for t in teams]})
    pts = stats["pts"].astype(float)
    tm["points"] = pts.mean(0)
    for q in (10, 50, 90):
        tm[f"points_p{q}"] = np.percentile(pts, q, axis=0)
    for k in ("w", "otl", "rw"):
        tm[k] = stats[k].mean(0)
    gp_team = games.home.value_counts().add(games.away.value_counts(), fill_value=0)
    tm["gp"] = tm.team.map(gp_team)
    tm["l"] = tm.gp - tm.w - tm.otl
    for k, lab in (("playoff", "playoff_pct"), ("division", "division_pct"),
                   ("presidents", "presidents_pct"), ("r2", "round2_pct"), ("cf", "conf_final_pct"),
                   ("final", "cup_final_pct"), ("cup", "cup_pct")):
        tm[lab] = res[k] * 100
    tm["rating"] = rating
    side_sum = lambda col: (gm.groupby("home")[f"{col}_home"].sum().add(
        gm.groupby("away")[f"{col}_away"].sum(), fill_value=0))
    side_against = lambda col: (gm.groupby("home")[f"{col}_away"].sum().add(
        gm.groupby("away")[f"{col}_home"].sum(), fill_value=0))
    for col in ("goals", "sog", "xgf", "pp_opps", "attempts", "pp_goals"):
        tm[f"{col}_for"] = tm.team.map(side_sum(col))
        tm[f"{col}_against"] = tm.team.map(side_against(col))
    if a.extra_time:
        for col in ("goals_reg", "ot_goals", "so_goals"):
            src = "goals" if col == "goals_reg" else col
            if col == "goals_reg":
                tm["goals_for_reg"] = tm.team.map(gm.groupby("home").goals_home_reg.sum().add(
                    gm.groupby("away").goals_away_reg.sum(), fill_value=0))
                tm["goals_against_reg"] = tm.team.map(gm.groupby("home").goals_away_reg.sum().add(
                    gm.groupby("away").goals_home_reg.sum(), fill_value=0))
            else:
                tm[f"{col}_for"] = tm.team.map(side_sum(src))
                tm[f"{col}_against"] = tm.team.map(side_against(src))
    tm["pp_pct"] = 100 * tm.pp_goals_for / tm.pp_opps_for
    tm["pk_pct"] = 100 * (1 - tm.pp_goals_against / tm.pp_opps_against)
    tm["shooting_pct"] = 100 * tm.goals_for / tm.sog_for
    tm["save_pct"] = 1 - tm.goals_against / tm.sog_against
    tm["exp_wins_from_games"] = tm.team.map(
        gm.groupby("home").p_home_win.sum().add(gm.groupby("away").p_home_win.apply(lambda x: (1 - x).sum()),
                                                  fill_value=0))
    for c in extra:
        tm[c] = tm.team.map(pg.groupby("team")[c].sum())
    tm = tm.sort_values("points", ascending=False)
    # the full points distribution per team (percentiles 1-99), for season-end CRPS
    pq = pd.DataFrame(np.percentile(pts, np.arange(1, 100), axis=0).T,
                      columns=[f"q{q:02d}" for q in range(1, 100)])
    pq.insert(0, "team", teams)

    # ---- skaters
    # names: the rosters snapshot first, then the availability and rates tables
    # (injured-reserve players and call-ups are not on the NHL roster), then every
    # earlier snapshot
    name = {}
    snaps = sorted((PROJ / "data" / "raw" / "rosters").glob("*/rosters.csv"))
    for f in snaps[::-1]:
        r_ = pd.read_csv(f)
        for pid, nm in zip(r_.player_id, r_["first"].astype(str) + " " + r_["last"].astype(str)):
            name.setdefault(int(pid), nm)
    for f in (OUT / "player_rates_2027.csv", OUT / "availability_2027.csv"):
        if f.exists():
            r_ = pd.read_csv(f)
            if "name" in r_:
                for pid, nm in zip(r_.player_id, r_.name):
                    if isinstance(nm, str) and nm.strip():
                        name[int(pid)] = nm
    rost = pd.read_csv(PROJ / "data" / "raw" / "rosters" / a.rosters_date / "rosters.csv")
    name.update({int(pid): nm for pid, nm in zip(rost.player_id, rost["first"] + " " + rost["last"])})
    agg_cols = SK_STATS + ["dress", "toi"] + extra
    sk = pg.groupby(["player_id", "team"], as_index=False)[agg_cols].sum()
    sk = sk.rename(columns={"dress": "gp"})
    sk["points"] = sk.g + sk.a
    sk["name"] = sk.player_id.map(name)
    bios = pd.read_parquet(ROOT / "data" / "tensors" / "career_bios.parquet").set_index("player_id")
    sk["pos"] = np.where(sk.player_id.map(bios.pos_group).fillna(0) == 1, "D", "F")
    sk["toi_per_gp"] = sk.toi / sk.gp.clip(lower=1e-9)
    # season distributions: the team shock scales scoring, counts are Poisson given it
    rng = np.random.default_rng(a.seed + 23)
    tix = {t: i for i, t in enumerate(teams)}
    sh = shocks[rng.integers(0, len(shocks), 4000)]
    mult = np.exp(sh[:, sk.team.map(tix).to_numpy()])
    for col in ("g", "a"):
        draws_ = rng.poisson(sk[col].to_numpy()[None, :] * mult)
        sk[f"{col}_p10"], sk[f"{col}_p90"] = np.percentile(draws_, 10, 0), np.percentile(draws_, 90, 0)
        if col == "g":
            gd = draws_
        else:
            pdraw = gd + draws_
    sk["points_p10"], sk["points_p90"] = np.percentile(pdraw, 10, 0), np.percentile(pdraw, 90, 0)
    sk = sk.rename(columns={"g": "goals", "a": "assists"}).sort_values("points", ascending=False)
    sk["points_per_gp"] = sk.points / sk.gp.clip(lower=1e-9)
    sk["shooting_pct"] = 100 * sk.goals / sk.sog.clip(lower=1e-9)
    sk["oi_xg_diff"] = sk.oi_xgf - sk.oi_xga

    # ---- goalies
    gl = gk.groupby(["player_id", "team"], as_index=False)[["starts", "sa", "ga", "win", "shutout"]].sum()
    gl["name"] = gl.player_id.map(name)
    gl["saves"] = gl.sa - gl.ga
    gl["sv_pct"] = 1 - gl.ga / gl.sa.clip(lower=1e-9)
    gl["gaa"] = gl.ga / gl.starts.clip(lower=1e-9)
    gl = gl.rename(columns={"win": "wins", "shutout": "shutouts"}).sort_values("starts", ascending=False)

    # ---- consistency
    cons = {"engine_conservation_max": max(c["max_abs_team_minus_players"] for c in checks),
            "lineup_players_without_state": int(n_missing)}
    team_goals = tm.set_index("team").goals_for
    if a.extra_time:        # skaters score overtime goals, not shootout-deciding ones
        team_goals = team_goals - tm.set_index("team").so_goals_for
    player_goals = sk.groupby("team").goals.sum()
    cons["team_goals_minus_player_goals_max"] = float((team_goals - player_goals).abs().max())
    team_sog = tm.set_index("team").sog_for
    cons["team_sog_minus_player_sog_max"] = float((team_sog - sk.groupby("team").sog.sum()).abs().max())
    cons["skater_games_per_team"] = {t: round(float(v), 3) for t, v in sk.groupby("team").gp.sum().items()}
    cons["goalie_starts_per_team"] = {t: round(float(v), 3) for t, v in gl.groupby("team").starts.sum().items()}
    per_season = stats["pts"].sum(1)
    n_ot = (2 * stats["w"] + stats["otl"]).sum(1) - 2 * N
    cons["one_win_per_game_every_season"] = bool(np.all(stats["w"].sum(1) == N))
    cons["points_identity_holds_every_season"] = bool(np.all(per_season == 2 * N + stats["otl"].sum(1)))
    cons["league_points_mean"] = float(per_season.mean())
    cons["league_ot_games_mean"] = float(n_ot.mean())
    cons["mc_wins_minus_game_sum_max"] = float((tm.w - tm.exp_wins_from_games).abs().max())
    cons["league_goals_per_team_game"] = float(gm[["goals_home", "goals_away"]].to_numpy().mean())
    cons["ot_share_predicted"] = float(gm.p_ot.mean())
    cons["home_win_mean"] = float(gm.p_home_win.mean())
    cons["corr_points_vs_0925"] = float(np.corrcoef(
        tm.points, tm.team.map(pd.read_csv(NOUT / "projection_2027.csv").set_index("team").proj_points))[0, 1])
    cons["corr_game_prob_vs_0925"] = float(gm[["p_home_win", "p_home_win_0925"]].corr().iloc[0, 1])

    # ---- write
    W_ = Path(a.out_dir).resolve() if a.out_dir else OUT
    if W_ == OUT.resolve():
        from common import refuse_if_frozen_1_0
        refuse_if_frozen_1_0("the NeurHL 1.0 season files")
    W_.mkdir(parents=True, exist_ok=True)
    gm.to_csv(W_ / "games_2027.csv", index=False, float_format="%.5f")
    tm.to_csv(W_ / "teams_2027.csv", index=False, float_format="%.4f")
    sk.to_csv(W_ / "skaters_2027.csv", index=False, float_format="%.4f")
    gl.to_csv(W_ / "goalies_2027.csv", index=False, float_format="%.4f")
    pq.to_csv(W_ / "team_points_quantiles_2027.csv", index=False, float_format="%.1f")
    pg.to_csv(W_ / "player_games_2027.csv.gz", index=False, float_format="%.5f")
    (W_ / "consistency_2027.json").write_text(json.dumps(cons, indent=1))
    code = subprocess.run(["git", "-C", str(PROJ), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    run = {"rosters_date": a.rosters_date, "draws": K, "sims": a.sims, "seed": a.seed,
           "team_sigma": TEAM_SIGMA, "goal_mult_m0": m0, "days_in_opening": days0,
           "goal_state": a.goal_state, "M_full": M_full, "goal_level_full": bool(a.goal_level_full), "level_ratios": ratios, "shot_level": shot_level or None,
           "extra_time": bool(a.extra_time),
           "rookie_priors": a.rookie_priors, "rookie_weight": ({"g": a.rookie_weight, "a": a.rookie_weight if a.rookie_weight_a is None else a.rookie_weight_a}
                             if a.rookie_priors else None),
           "bundle": bundle, "bundle_sha": sha(ROOT / "checkpoints" / "g" / bundle / "bundle.json")[:16],
           "availability": av_src, "home_edge_rating": float(hedge), "code": code,
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "seconds": round(time.time() - t0, 1)}
    (W_ / "run_2027.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(cons, indent=1)[:2500])
    print(tm[["team", "points", "points_p10", "points_p90", "playoff_pct", "cup_pct", "goals_for",
              "goals_against"]].head(10).to_string(index=False))
    print(f"[unified] done in {time.time() - t0:.0f}s -> {W_}")


if __name__ == "__main__":
    main()
