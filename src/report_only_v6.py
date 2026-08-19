"""REPORT-ONLY v6 restatement (PLAN_V6; computed AFTER F4/T1 locked the config).

2018-2026 deviation-space walk-forward MAE/Spearman for v1, v4, v5, v6, HOWE,
HOWE5, HOWE6 — the out-of-window context for the v6 train gains (v4-I3/v5
precedent; nothing here can change what ships).
Writes output/report_only_v6.json + output/report_only_v6.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import ridge as R
from backtest import project_season
from features_v6 import FEATURES_V6_ALL, FeatureBuilderV6

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
VALID = list(range(2018, 2027))


def main():
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p4 = json.loads((OUT / "params_v4.json").read_text())
    p5 = json.loads((OUT / "params_v5.json").read_text())
    p6 = json.loads((OUT / "params_v6.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    g, ts = E.load()
    ACT = ts.set_index(["season_end", "team"])
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilderV6(end_r, ts, goalie_hp=p2["goalie_hp"],
                          skater_delta=p2["skater_delta"])

    rows = []
    for h in (1, 2):
        all_pred = list(range(2012 + (h - 1), 2027))
        X, y, meta = fb.feature_matrix(all_pred, h, feats=FEATURES_V6_ALL)
        seasons = meta["T"].to_numpy()
        cfg = {name: (p["shipped"][f"features_h{h}"], p["shipped"][f"lam_h{h}"])
               for name, p in (("v4", p4), ("v5", p5), ("v6", p6))}
        phi = v1["phi1"] if h == 1 else v1["phi2"]
        for T in VALID:
            devs = {}
            for name, (feats, lam) in cfg.items():
                idx = [FEATURES_V6_ALL.index(f) for f in feats]
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
            for vv in ("v4", "v5", "v6"):
                devs[f"howe{'' if vv == 'v4' else vv[1]}"] = \
                    0.5 * v1_dev + 0.5 * devs[vv]
            ydev = ((act.pts_pct - act.pts_pct.mean()) * 164) \
                .reindex(devs["v4"].index)
            for name, pr in devs.items():
                rows.append({"horizon": h, "season": T, "model": name,
                             "mae": float((pr - ydev).abs().mean()),
                             "spearman": float(pr.rank().corr(ydev.rank()))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "report_only_v6.csv", index=False)
    print("=== REPORT-ONLY: v6 restatement 2018-2026 (post-lock, context only) ===")
    for h in (1, 2):
        sub = df[df.horizon == h]
        piv = sub.pivot_table(index="model", values=["mae", "spearman"], aggfunc="mean")
        print(f"h{h}: MAE {piv.mae.round(3).to_dict()}")
        print(f"    rho {piv.spearman.round(3).to_dict()}")
    blob = {"restatement": df.round(4).to_dict("records"),
            "locked_before_run": "PLAN_V6 @ b719257 + params_v6 gates preceded this"}
    (OUT / "report_only_v6.json").write_text(json.dumps(blob, indent=2))
    print("wrote report_only_v6.json + report_only_v6.csv")


if __name__ == "__main__":
    main()
