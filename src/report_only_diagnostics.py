"""ONE-TIME REPORT-ONLY diagnostics (PLAN_V4 I2/I3 — decisions locked at prereg commit
4bb92d9 BEFORE this ran; nothing here may change what ships, and nothing here tunes).

1. Historical ensemble restatement, 2018-2026, h1+h2: deviation-space MAE/Spearman of
   v1, the v4 feature pipeline (walk-forward, shipped feats/lams), and ENS = 0.5/0.5.
   The ENS ships regardless (locked); these numbers are context for the live test.
2. CRPS restatement, 2018-2026: v1 frozen-pipeline simulated distributions vs actual
   points; baseline = regressed-prior point forecast (its CRPS equals its abs error).
   Quantifies the value of the distribution itself, which MAE cannot see.

Writes output/report_only_v4.json + output/report_only_ens.csv. Single scripted run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import players as P
import ridge as R
import scoring as SC
from backtest import fit_yoy_slope, project_season
from features import FEATURES_V4, FeatureBuilder

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
VALID = list(range(2018, 2027))


def main():
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p4 = json.loads((OUT / "params_v4.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    ship = p4["shipped"]
    g, ts = E.load()
    ACT = ts.set_index(["season_end", "team"])
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"],
                        skater_delta=p2["skater_delta"])

    # ---- 1. historical ENS (deviation space; overlay never in backtests)
    rows = []
    for h in (1, 2):
        feats = ship[f"features_h{h}"]
        lam = ship[f"lam_h{h}"]
        all_pred = list(range(2012 + (h - 1), 2027))
        X, y, meta = fb.feature_matrix(all_pred, h, feats=FEATURES_V4)
        idx = [FEATURES_V4.index(f) for f in feats]
        Xs = X[:, idx]
        seasons = meta["T"].to_numpy()
        phi = v1["phi1"] if h == 1 else v1["phi2"]
        for T in VALID:
            tr = (seasons <= T - h) & ~np.isnan(y)
            b = R.fit_ridge(Xs[tr], y[tr], lam)
            te = seasons == T
            v4_dev = pd.Series(Xs[te] @ b, index=meta.loc[te, "team"].to_numpy())
            v4_dev -= v4_dev.mean()
            _, xp, _, _ = project_season(end_r, preds, T, h, v1["w"], phi)
            act = ACT.loc[T]
            gp2 = 2 * act.gp.reindex(xp.index)
            v1_dev = (xp / gp2 - (xp / gp2).mean()) * 164
            ens_dev = 0.5 * v1_dev.reindex(v4_dev.index).fillna(0) + 0.5 * v4_dev
            ydev = ((act.pts_pct - act.pts_pct.mean()) * 164).reindex(v4_dev.index)
            for name, pr in (("v1", v1_dev.reindex(v4_dev.index).fillna(0)),
                             ("v4", v4_dev), ("ens", ens_dev)):
                rows.append({"horizon": h, "season": T, "model": name,
                             "mae": float((pr - ydev).abs().mean()),
                             "spearman": float(pr.rank().corr(ydev.rank()))})
    ens = pd.DataFrame(rows)
    ens.to_csv(OUT / "report_only_ens.csv", index=False)
    print("=== REPORT-ONLY: historical ENS restatement (2018-2026) ===")
    for h in (1, 2):
        sub = ens[ens.horizon == h]
        piv = sub.pivot_table(index="model", values=["mae", "spearman"], aggfunc="mean")
        core = sub[~sub.season.isin([2020, 2021])]
        piv2 = core.pivot_table(index="model", values="mae", aggfunc="mean")
        print(f"h{h}: MAE {piv.mae.round(3).to_dict()} | excl 20/21 "
              f"{piv2.mae.round(3).to_dict()} | rho {piv.spearman.round(3).to_dict()}")

    # ---- 2. CRPS restatement (v1 frozen sims vs regressed-prior point forecast)
    slope1, slope2 = fit_yoy_slope(2017, 1), fit_yoy_slope(2017, 2)
    rng = np.random.default_rng(4242)
    crps_rows = []
    for h, phi, sigma, slope in ((1, v1["phi1"], v1["sigma1"], slope1),
                                 (2, v1["phi2"], v1["sigma2"], slope2)):
        for T in VALID:
            ratings, xp, om, sched = project_season(end_r, preds, T, h, v1["w"], phi)
            sim = E.simulate_season(ratings, sigma, sched, om, E.divisions_for(T),
                                    3000, rng, playoffs=False)
            act = ACT.loc[T]
            crps_m = SC.crps_matrix(sim["pts"], sim["teams"], act.pts)
            m_hist = ts[ts.season_end < T].pts_pct.mean()
            prior = ts[ts.season_end == T - h].set_index("team").pts_pct
            errs = []
            for t in crps_m.index:
                pred = (m_hist + slope * (prior.get(t, m_hist) - m_hist)) \
                    * 2 * act.loc[t].gp
                errs.append(abs(pred - act.loc[t].pts))
            crps_rows.append({"horizon": h, "season": T,
                              "crps_model": float(crps_m.mean()),
                              "mae_regressed_point": float(np.mean(errs))})
    cr = pd.DataFrame(crps_rows)
    print("\n=== REPORT-ONLY: CRPS restatement (2018-2026, v1 frozen sims) ===")
    for h in (1, 2):
        sub = cr[cr.horizon == h]
        print(f"h{h}: model CRPS {sub.crps_model.mean():.3f} vs regressed point-"
              f"forecast CRPS(=MAE) {sub.mae_regressed_point.mean():.3f} "
              f"({100 * (1 - sub.crps_model.mean() / sub.mae_regressed_point.mean()):.0f}% "
              f"sharper)")
    blob = {"ens_restatement": ens.round(4).to_dict("records"),
            "crps_restatement": cr.round(4).to_dict("records"),
            "locked_before_run": "PLAN_V4 @ 4bb92d9; ENS ships regardless"}
    (OUT / "report_only_v4.json").write_text(json.dumps(blob, indent=2))
    print("\nwrote report_only_v4.json + report_only_ens.csv")


if __name__ == "__main__":
    main()
