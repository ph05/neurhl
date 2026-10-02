"""Calibration of rest-of-season skater intervals (ORR 1.7).

1.6's interval for rest-of-season points is the conjugate negative-binomial
predictive of the Gamma posterior on the player's points rate, for a FIXED
number of games. 1.7 adds two terms, simulated per player:
  * games-played variance: games ~ Binomial(team games left, q), with q the
    player's preseason games share;
  * a rate-variance multiplier v: the Gamma posterior's variance times v (same
    mean).
At 25/50/75% of a season (by game id) every projected skater's remaining
points are predicted (prior: the strict preseason projection, updated with
his box-score games so far as in 1.2) and scored with the 80% interval score
(width + 2/0.2 * miss) and coverage. Tuned on 2012 and 2014-2017 over GRID;
the test (2022, 2023) runs once. Pre-declared rule: the tuned configuration
is adopted (params/ros_interval.json) only if its test interval score beats
1.6's (v = 1, no games variance).

Run: python3 -m orr.backtest.ros_interval_bt
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from orr import config as C
from orr import players as PL
from orr import player_update as PU
from orr.backtest import players_bt as PB

TUNE, TEST = [2012, 2014, 2015, 2016, 2017], [2022, 2023]
CHECKPOINTS = (0.25, 0.5, 0.75)
GRID = list(itertools.product([1.0, 1.5, 2.0, 3.0], [False, True]))
N_DRAWS = 400
OUT = C.OUT / "backtest" / "ros_interval_bt.json"
PARAMS = C.PARAMS / "ros_interval.json"


def season_rows(V, prm, k):
    tot = PB.run_season(V, prm, "strict")
    tot = tot[(tot.gp > 0) & tot.team.notna()]
    b = PU.season_boxes(V)
    # team of each box row: the team the player's game belongs to, via the season's dressed table
    from orr import lineups as LU
    d = LU.dressed()
    d = d[d.season_end == V][["gid", "team", "player_id"]]
    g = __import__("orr.structural", fromlist=["x"]).game_frame()
    gid2 = g[g.season_end == V][["gid", "game_id"]].dropna().astype({"game_id": "int64"})
    team_games = d.merge(gid2, on="gid").drop_duplicates(["game_id", "team"])[["game_id", "team"]]
    gids = np.sort(b.game_id.unique())
    n_team = 82
    rows = []
    for c in CHECKPOINTS:
        cut = gids[int(len(gids) * c)]
        bf, af = PU.totals(b[b.game_id < cut]).set_index("player_id"), PU.totals(b[b.game_id >= cut]).set_index("player_id")
        left = team_games[team_games.game_id >= cut].groupby("team").size()
        x = tot[["player_id", "team", "gp", "g", "a"]].copy()
        x["n_b"] = x.player_id.map(bf.n).fillna(0)
        x["p_b"] = x.player_id.map(bf.g + bf.a).fillna(0)
        x["p_f"] = x.player_id.map(af.g + af.a).fillna(0)
        x["G"] = x.team.map(left).fillna(0)
        x["q"] = np.clip(x.gp / n_team, 0.01, 1)
        r0 = (x.g + x.a) / x.gp.clip(lower=1)
        x["a_post"] = k * r0 + x.p_b
        x["rate"] = k + x.n_b
        rows.append(x.assign(season=V, checkpoint=c))
    return pd.concat(rows, ignore_index=True)


def interval(x: pd.DataFrame, v: float, games_var: bool, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(x)
    G = x.G.to_numpy().astype(int)
    q = x.q.to_numpy()
    games = rng.binomial(G[:, None], q[:, None], size=(n, N_DRAWS)) if games_var else (G * q)[:, None]
    lam = rng.gamma(np.maximum(x.a_post.to_numpy() / v, 1e-9)[:, None], (v / x.rate.to_numpy())[:, None], size=(n, N_DRAWS))
    pts = rng.poisson(lam * games)
    return np.percentile(pts, 10, axis=1), np.percentile(pts, 90, axis=1)


def score(x, v, gv):
    lo, hi = interval(x, v, gv)
    y = x.p_f.to_numpy()
    s = (hi - lo) + (2 / 0.2) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return s, (y >= lo) & (y <= hi)


def main():
    prm, n0 = PL.load_params(), PU.load_n0()
    k = n0["g"]
    data = {V: season_rows(V, prm, k) for V in TUNE + TEST}
    tune = pd.concat([data[V] for V in TUNE])
    test = pd.concat([data[V] for V in TEST])
    ledger = []
    for v, gv in GRID:
        s, cov = score(tune, v, gv)
        ledger.append({"v": v, "games_var": gv, "tune_score": float(s.mean()), "tune_cover80": float(cov.mean())})
    best = min(ledger, key=lambda r: r["tune_score"])
    s1, c1 = score(test, best["v"], best["games_var"])
    s0, c0 = score(test, 1.0, False)
    d = s1 - s0
    rng = np.random.default_rng(7)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    adopted = bool(d.mean() < 0)
    res = {"protocol": __doc__, "ledger": ledger, "best": best,
           "test": {"n": int(len(test)), "score_best": float(s1.mean()), "cover_best": float(c1.mean()),
                    "score_1_6": float(s0.mean()), "cover_1_6": float(c0.mean()), "diff": float(d.mean()),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
           "adopted": adopted}
    OUT.write_text(json.dumps(res, indent=1))
    PARAMS.write_text(json.dumps({"v": best["v"] if adopted else 1.0, "games_var": best["games_var"] if adopted else False,
                                  "adopted": adopted, "source": "orr/backtest/ros_interval_bt.py"}, indent=1))
    for r in ledger:
        print(r)
    print("best", best, "test", res["test"], "adopted", adopted)


if __name__ == "__main__":
    main()
