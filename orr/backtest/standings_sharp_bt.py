"""In-season standings sharpness (ORR 1.5).

ORR 1.4's in-season 80% intervals covered 0.86 of final points on test, so
they are too wide. This scales the filter's rating uncertainty used in the
rest-of-season simulation (o_sd, d_sd by m; od_cov by m^2), on the setup of
standings_inseason_bt.py (checkpoints 25/50/75%, drift k from 1.4).

m is chosen on 2012 and 2014-2017 by CRPS from GRID; the test (2022, 2023)
runs once. Pre-declared rule: adopt m (params/standings_inseason.json
"sd_mult") only if test CRPS improves on m = 1; coverage is reported.

Run: python3 -m orr.backtest.standings_sharp_bt
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from orr import config as C
from orr.backtest import standings_inseason_bt as B

GRID = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15]
OUT = C.OUT / "backtest" / "standings_sharp_bt.json"


def sim_m(s: B.Season, c: float, k: float, m: float) -> pd.DataFrame:
    st = s.state(c)
    orig = st["cur"]
    sc = orig.copy()
    sc["o_sd"], sc["d_sd"], sc["od_cov"] = orig.o_sd * m, orig.d_sd * m, orig.od_cov * m * m
    st["cur"] = sc
    try:
        return s.sim(c, k)
    finally:
        st["cur"] = orig


def summ(df):
    return {"n": int(len(df)), "crps": float(df.crps.mean()), "mae": float(df.abs_err.mean()),
            "cover80": float(df.cover80.mean())}


def main():
    params = json.loads(B.PARAMS.read_text())
    k = float(params["drift_k"])
    seasons = {V: B.Season(V) for V in B.TUNE + B.TEST}
    tune = {m: summ(pd.concat([sim_m(seasons[V], c, k, m) for V in B.TUNE for c in B.CHECKPOINTS])) for m in GRID}
    m_best = min(GRID, key=lambda m: tune[m]["crps"])
    t1 = pd.concat([sim_m(seasons[V], c, k, m_best) for V in B.TEST for c in B.CHECKPOINTS])
    t0 = pd.concat([sim_m(seasons[V], c, k, 1.0) for V in B.TEST for c in B.CHECKPOINTS])
    d = t1.crps.to_numpy() - t0.crps.to_numpy()
    rng = np.random.default_rng(7)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    adopted = bool(d.mean() < 0)
    res = {"protocol": __doc__, "drift_k": k, "tune": {str(m): v for m, v in tune.items()}, "m_best": m_best,
           "test": {"m_best": summ(t1), "m_1": summ(t0), "diff": float(d.mean()),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
           "adopted": adopted}
    OUT.write_text(json.dumps(res, indent=1))
    params["sd_mult"] = m_best if adopted else 1.0
    params["sd_mult_tuned"] = m_best
    B.PARAMS.write_text(json.dumps(params, indent=1))
    print({m: (round(v["crps"], 3), round(v["cover80"], 3)) for m, v in tune.items()})
    print("m_best", m_best, "test", res["test"], "adopted", adopted)


if __name__ == "__main__":
    main()
