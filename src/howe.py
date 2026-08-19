"""HOWE — Hockey Outcomes via Weighted Ensemble (named for Gordie Howe, Mr. Hockey).

The flagship model, formerly "ENS" (renamed 2026-08-19; math, params and seeds are
byte-identical). An Elo-space equal blend of the two production engines, locked at
prereg (PLAN_V4 I3 @ 4bb92d9) BEFORE computation:

    r_howe = 1505 + 0.5*(r_v1 - 1505) + 0.5*(r_v4 - 1505)

v1 = Elo + xG projection; v4 = 19-feature walk-forward ridge + amended overlay.
report4.py produces the ratings (output/v4_prior_ratings.csv, column rating_howe);
this module is the single place downstream tooling rebuilds the deterministic HOWE
2026-27 season sim (same closures as the report4 h1 run, seed 411 + 1 = 412).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import goalie_game as GG
import players as P
from features import FeatureBuilder
from report4 import prod_noise, real_schedule_flags

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
NAME = "HOWE"
BACKRONYM = "Hockey Outcomes via Weighted Ensemble"
SEED = 412   # report4 HOWE h1 seed (411 + 1)


def ratings() -> dict:
    """The locked 50/50 blend, as produced by report4.py."""
    return dict(pd.read_csv(OUT / "v4_prior_ratings.csv", index_col=0)["rating_howe"])


def rebuild_sim(n_sims: int = 10_000, playoffs: bool = False):
    """Deterministically rebuild the HOWE 2026-27 sim (report4 h1 seeds/closures)."""
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p4 = json.loads((OUT / "params_v4.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    ship = p4["shipped"]
    g, ts = E.load()
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"],
                        skater_delta=p2["skater_delta"])
    om = E.fit_outcome(preds, list(range(2006, 2027)))
    sk, go, skt, got, bios = P.load_panels()
    noise_fn, *_, tand, frag = prod_noise(fb, sk, got, bios, ts, p2["k"], 1,
                                          ship["n0_g"])
    sched = real_schedule_flags()
    gn = GG.make_game_noise(sched, tand, p2["k"]) if ship["goalie_layer"] else None
    return E.simulate_season(ratings(), ship["sigma_c4"],
                             sched[["home", "away", "d_adj"]], om,
                             E.DIVISIONS_CURRENT, n_sims,
                             np.random.default_rng(SEED), playoffs=playoffs,
                             extra_noise=noise_fn, game_noise=gn)
