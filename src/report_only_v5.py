"""REPORT-ONLY v5 restatement (PLAN_V5; computed AFTER gates locked the config).

Deviation-space walk-forward MAE/Spearman on 2018-2026, h1+h2, for v1, v4, v5,
HOWE (0.5 v1+v4) and HOWE5 (0.5 v1+v5) — the honest out-of-window context for the
v5 train gains. Follows the v4 I3 precedent exactly: the decision (F3 gates) was
made on train only and is already locked; nothing here can change what ships.
Writes output/report_only_v5.json + output/report_only_v5.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import ridge as R
from backtest import fit_yoy_slope, project_season   # noqa: F401 (v1 projection)
from features_v5 import FEATURES_V5_ALL, FeatureBuilderV5

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
VALID = list(range(2018, 2027))


def main():
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p4 = json.loads((OUT / "params_v4.json").read_text())
    p5 = json.loads((OUT / "params_v5.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    g, ts = E.load()
    ACT = ts.set_index(["season_end", "team"])
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilderV5(end_r, ts, goalie_hp=p2["goalie_hp"],
                          skater_delta=p2["skater_delta"])

    rows = []
    for h in (1, 2):
        all_pred = list(range(2012 + (h - 1), 2027))
        X, y, meta = fb.feature_matrix(all_pred, h, feats=FEATURES_V5_ALL)
        seasons = meta["T"].to_numpy()
        cfg = {"v4": (p4["shipped"][f"features_h{h}"], p4["shipped"][f"lam_h{h}"]),
               "v5": (p5["shipped"][f"features_h{h}"], p5["shipped"][f"lam_h{h}"])}
        phi = v1["phi1"] if h == 1 else v1["phi2"]
        for T in VALID:
            devs = {}
            for name, (feats, lam) in cfg.items():
                idx = [FEATURES_V5_ALL.index(f) for f in feats]
                tr = (seasons <= T - h) & ~np.isnan(y)
                b = R.fit_ridge(X[tr][:, idx], y[tr], lam)
                te = seasons == T
                d = pd.Series(X[te][:, idx] @ b,
                              index=meta.loc[te, "team"].to_numpy())
                devs[name] = d - d.mean()
            _, xp, _, _ = project_season(end_r, preds, T, h, v1["w"], phi)
            act = ACT.loc[T]
            gp2 = 2 * act.gp.reindex(xp.index)
            v1_dev = ((xp / gp2 - (xp / gp2).mean()) * 164) \
                .reindex(devs["v4"].index).fillna(0)
            devs["v1"] = v1_dev
            devs["howe"] = 0.5 * v1_dev + 0.5 * devs["v4"]
            devs["howe5"] = 0.5 * v1_dev + 0.5 * devs["v5"]
            ydev = ((act.pts_pct - act.pts_pct.mean()) * 164) \
                .reindex(devs["v4"].index)
            for name, pr in devs.items():
                rows.append({"horizon": h, "season": T, "model": name,
                             "mae": float((pr - ydev).abs().mean()),
                             "spearman": float(pr.rank().corr(ydev.rank()))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "report_only_v5.csv", index=False)
    print("=== REPORT-ONLY: v5 restatement 2018-2026 (post-lock, context only) ===")
    for h in (1, 2):
        sub = df[df.horizon == h]
        piv = sub.pivot_table(index="model", values=["mae", "spearman"], aggfunc="mean")
        core = sub[~sub.season.isin([2020, 2021])]
        piv2 = core.pivot_table(index="model", values="mae", aggfunc="mean")
        print(f"h{h}: MAE {piv.mae.round(3).to_dict()} | excl 20/21 "
              f"{piv2.mae.round(3).to_dict()} | rho {piv.spearman.round(3).to_dict()}")
    blob = {"restatement": df.round(4).to_dict("records"),
            "locked_before_run": "PLAN_V5.md + params_v5.json (F3 gates, train-only) "
                                 "locked before this ran"}
    (OUT / "report_only_v5.json").write_text(json.dumps(blob, indent=2))
    print("wrote report_only_v5.json + report_only_v5.csv")


if __name__ == "__main__":
    main()
