"""Tuning-window diagnostics for L3 (V = 2012, 2014-2017 only): GP error by
player group, GP RMSE and correlation, for the current pipeline and a logged
configuration. No new configuration is scored here.

  python3 -m orr.experiments.L3.diag '<config json>'
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from orr import players as PL
from orr.experiments.L3 import gpmodel as G
from orr.experiments.L3.tune import OBJ_V


def frame(cfg, prm):
    out = []
    for V in OBJ_V:
        t = G.gp_rows(G.run_season_l3(V, prm, cfg))
        t["V"] = V
        out.append(t)
    return pd.concat(out, ignore_index=True)


def groups(t):
    h = t.has_hist.fillna(False).astype(bool)
    big = h & (t.player_id.map(lambda _: True))
    return {"all_rostered": np.ones(len(t), bool), "has_hist": h.to_numpy(),
            "rookies": (~h).to_numpy()}


def summary(t):
    e = t.gp - t.act_gp0
    r = {}
    for name, m in groups(t).items():
        r[name] = {"n": int(m.sum()), "mae": round(float(e[m].abs().mean()), 3),
                   "rmse": round(float(np.sqrt((e[m] ** 2).mean())), 3),
                   "bias": round(float(e[m].mean()), 3),
                   "corr": round(float(np.corrcoef(t.gp[m], t.act_gp0[m])[0, 1]), 4)}
    # by projected-games band of the baseline
    return r


def main():
    prm = PL.load_params()
    cfg = G.GPConfig(**json.loads(sys.argv[1]))
    b = frame(G.GPConfig(model="none"), prm)
    n = frame(cfg, prm).set_index(["V", "player_id"]).loc[
        b.set_index(["V", "player_id"]).index].reset_index()
    out = {"baseline": summary(b), "config": summary(n)}
    # error by actual-games band (where the errors are)
    band = pd.cut(b.act_gp0, [-1, 20, 40, 60, 75, 90], labels=["0-20", "21-40", "41-60", "61-75", "76+"])
    out["by_actual_band"] = {
        str(k): {"n": int((band == k).sum()),
                 "mae_base": round(float((b.gp - b.act_gp0)[band == k].abs().mean()), 2),
                 "mae_cfg": round(float((n.gp - n.act_gp0)[band == k].abs().mean()), 2)}
        for k in band.cat.categories}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
