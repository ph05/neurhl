"""NeurHL 1.1 C2b: backtest of the in-season standings projection
(sim/live_standings_1_1.py; decision rule in PLAN_NeurHL_1_1 A2).

For each season and checkpoint (4, 8, 12, 16 and 20 weeks after opening
night), the final points of every team are projected from:
  - the points already earned;
  - the frozen preseason probabilities of the remaining games
    (data/tensors/_season_bt/pre_<V>.parquet);
  - team strength either left at its preseason prior ("no update", which is
    what adding actual points to the frozen preseason simulation does) or
    updated from the games played (posterior plus drift).
Scored by mean points CRPS against the final table.

Writes neurhl/output/neurhl_1_1/live_standings_c2b.json.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import season_layer_c2 as S  # noqa: E402
from sim.live_standings_1_1 import posterior, project  # noqa: E402

OUT = ROOT / "output" / "neurhl_1_1" / "live_standings_c2b.json"
CHECK_WEEKS = [4, 8, 12, 16, 20]
GRID = list(itertools.product([0.05, 0.07, 0.09, 0.11, 0.13], [0, 0.005, 0.01, 0.02]))
SIMS = 4000


def points_of(o, hi, ai, T):
    pts = np.zeros(T)
    np.add.at(pts, hi, np.where(np.isin(o, [0, 2]), 2, np.where(o == 3, 1, 0)))
    np.add.at(pts, ai, np.where(np.isin(o, [1, 3]), 2, np.where(o == 2, 1, 0)))
    return pts


def run(seasons, variant, update):
    sigma0, sw = variant
    rows = []
    for V in seasons:
        d = S.load(V)
        teams, hi, ai, y = S.realised(d)
        T = len(teams)
        z = np.log(d.p / (1 - d.p)).to_numpy()
        k = d.k.to_numpy()
        o = d.outcome4.to_numpy()
        o4 = d[["o4_hr", "o4_ar", "o4_ho", "o4_ao"]].to_numpy()
        w = d.week.to_numpy()
        for cw in CHECK_WEEKS:
            pl, rem = w < cw, w >= cw
            if rem.sum() == 0:
                continue
            pts_now = points_of(o[pl], hi[pl], ai[pl], T)
            if update:
                yp = np.isin(o[pl], [0, 2]).astype(float)
                m, C = posterior(z[pl], k[pl], hi[pl], ai[pl], yp, T, sigma0)
            else:
                m, C = np.zeros(T), np.eye(T) * sigma0 ** 2
            # without an update the drift runs from opening night, as in the season layer
            wk_now = cw if update else 0
            sim = project(z[rem], k[rem], hi[rem], ai[rem], o4[rem], w[rem], wk_now, pts_now,
                          m, C, sw, SIMS, 711 + V + cw)
            c, ae, cov = S.scores(sim, y)
            rows.append(pd.DataFrame({"season": V, "week": cw, "team": teams, "crps": c, "ae": ae,
                                      "cov": cov}))
    return pd.concat(rows, ignore_index=True)


def summ(df):
    return {"crps": float(df.crps.mean()), "mae": float(df.ae.mean()), "coverage": float(df["cov"].mean()),
            "by_week": {int(k): float(v) for k, v in df.groupby("week").crps.mean().items()}}


def main():
    out = {"checkpoints_weeks": CHECK_WEEKS, "grid": [list(g) for g in GRID], "sims": SIMS}
    fit_u = {g: run(S.FIT, g, True) for g in GRID}
    fit_n = {g: run(S.FIT, g, False) for g in GRID}
    bu = min(GRID, key=lambda g: fit_u[g].crps.mean())
    bn = min(GRID, key=lambda g: fit_n[g].crps.mean())
    ju, jn = run(S.JUDGE, bu, True), run(S.JUDGE, bn, False)
    d = ju.crps.to_numpy() - jn.crps.to_numpy()
    j = ju.assign(diff_=d)
    out.update({
        "selected_update": {"sigma0": bu[0], "sw": bu[1]},
        "selected_no_update": {"sigma0": bn[0], "sw": bn[1]},
        "fit": {"update": summ(fit_u[bu]), "no_update": summ(fit_n[bn])},
        "judge": {"update": summ(ju), "no_update": summ(jn), "crps_diff": float(d.mean()),
                  "crps_diff_ci95_season_boot": S.season_boot(j)},
    })
    e = out["judge"]
    out["decision"] = {"adopt": bool(e["crps_diff"] < 0 and abs(e["update"]["coverage"] - 0.80) <= 0.05),
                       "rule": "PLAN_NeurHL_1_1 A2"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("selected_update", "selected_no_update", "judge", "decision")}, indent=1))


if __name__ == "__main__":
    main()
