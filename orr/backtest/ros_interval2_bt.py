"""Rest-of-season skater intervals, second pass (ORR 1.8).

1.7's best rate-variance multiplier (v = 3) sat at the edge of its grid with
73% test coverage. This pass extends the grid, v in {3, 4, 6, 9}, and adds an
injury-spell term. Each player has a historical probability of a long
absence: p_spell is the smoothed share of his last three seasons with 15+
games missed, (k + 0.5) / (n + 2), or the league mean without history. A
spell occurs in the remaining games with probability p_spell * (games left /
82). When it occurs, it removes a uniform 0-60% of the player's remaining
games in one block. Setup and scoring as in ros_interval_bt.py (80% interval
score, coverage). Tuned on 2012 and 2014-2017; the test (2022, 2023) runs
once. Pre-declared rule: adopt the tuned configuration
(params/ros_interval.json) only if its test interval score beats 1.7's
(v = 3, games variance, no spells).

Run: python3 -m orr.backtest.ros_interval2_bt
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from orr import config as C
from orr import data as D
from orr import players as PL
from orr import player_update as PU
from orr.backtest import ros_interval_bt as R1

GRID = list(itertools.product([3.0, 4.0, 6.0, 9.0], [False, True]))
OUT = C.OUT / "backtest" / "ros_interval2_bt.json"


def spell_prob(V: int) -> pd.Series:
    s = D.skater_seasons().groupby(["player_id", "season_end"]).gp.sum().reset_index()
    s = s[(s.season_end >= V - 3) & (s.season_end < V) & ~s.season_end.isin(C.BROKEN_SEASONS)]
    s["spell"] = (82 - s.gp) >= 15
    g = s.groupby("player_id").agg(k=("spell", "sum"), n=("spell", "size"))
    return (g.k + 0.5) / (g.n + 2)


def interval(x: pd.DataFrame, v: float, spell: bool, seed: int = 7):
    rng = np.random.default_rng(seed)
    n, N = len(x), R1.N_DRAWS
    G = x.G.to_numpy().astype(int)
    games = rng.binomial(G[:, None], x.q.to_numpy()[:, None], size=(n, N)).astype(float)
    if spell:
        p = np.clip(x.p_spell.to_numpy() * G / 82.0, 0, 1)
        hit = rng.random((n, N)) < p[:, None]
        games = games * np.where(hit, 1 - rng.uniform(0, 0.6, (n, N)), 1.0)
    lam = rng.gamma(np.maximum(x.a_post.to_numpy() / v, 1e-9)[:, None], (v / x.rate.to_numpy())[:, None], size=(n, N))
    pts = rng.poisson(lam * games)
    return np.percentile(pts, 10, axis=1), np.percentile(pts, 90, axis=1)


def score(x, v, sp):
    lo, hi = interval(x, v, sp)
    y = x.p_f.to_numpy()
    return (hi - lo) + 10 * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0)), (y >= lo) & (y <= hi)


def main():
    prm, n0 = PL.load_params(), PU.load_n0()
    data = {}
    for V in R1.TUNE + R1.TEST:
        x = R1.season_rows(V, prm, n0["g"])
        ps = spell_prob(V)
        x["p_spell"] = x.player_id.map(ps).fillna(float(ps.mean()))
        data[V] = x
    tune = pd.concat([data[V] for V in R1.TUNE])
    test = pd.concat([data[V] for V in R1.TEST])
    ledger = []
    for v, sp in GRID:
        s, c = score(tune, v, sp)
        ledger.append({"v": v, "spell": sp, "tune_score": float(s.mean()), "tune_cover80": float(c.mean())})
    best = min(ledger, key=lambda r: r["tune_score"])
    s1, c1 = score(test, best["v"], best["spell"])
    s0, c0 = score(test, 3.0, False)           # 1.7
    d = s1 - s0
    rng = np.random.default_rng(7)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    adopted = bool(d.mean() < 0)
    res = {"protocol": __doc__, "ledger": ledger, "best": best,
           "test": {"n": int(len(test)), "score_best": float(s1.mean()), "cover_best": float(c1.mean()),
                    "score_1_7": float(s0.mean()), "cover_1_7": float(c0.mean()), "diff": float(d.mean()),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}, "adopted": adopted}
    OUT.write_text(json.dumps(res, indent=1))
    if adopted:
        R1.PARAMS.write_text(json.dumps({"v": best["v"], "games_var": True, "spell": best["spell"], "adopted": True,
                                         "source": "orr/backtest/ros_interval2_bt.py"}, indent=1))
    for r in ledger:
        print(r)
    print("best", best, "test", res["test"], "adopted", adopted)


if __name__ == "__main__":
    main()
