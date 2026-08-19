"""v5 train-only gates (PLAN_V5 G, prereg'd 2026-08-19 BEFORE this ran).

GATE F3 per candidate, per horizon: add-one-in to the v4 shipped set, train LOSO
(predict 2012-2017 h1 / 2013-2017 h2): dMAE < 0.00 (strict — no v4 +0.02 tolerance)
AND sign stability >= 0.67.
Collinearity rule: among {flurry_xg_dev, corsi_dev} vs the shipped xg_dev, at most
the single best gater ships; its replacement variant (swap for xg_dev) is also
evaluated and the better form kept.
GATE F3-joint: greedy forward addition of passers in add-one-in dMAE order, each
accepted only if joint LOSO improves; final set must beat the incumbent (dMAE < 0)
or v5 is a documented null.

Writes output/params_v5.json. 2018-2026 untouched.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import ridge as R
from features_v5 import FEATURES_CAND, FEATURES_V5_ALL, FeatureBuilderV5

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
TRAIN = {1: list(range(2012, 2018)), 2: list(range(2013, 2018))}
XG_FAMILY = ("flurry_xg_dev", "corsi_dev")   # collinearity rule vs shipped xg_dev

PREREG = {
    "gate_F3": "per candidate/horizon: add-one-in train LOSO dMAE < 0.00 AND sign "
               "stability >= 0.67 (stricter than v4 F2: empirically mined pool)",
    "collinearity": "among flurry_xg_dev/corsi_dev vs xg_dev: at most one ships; "
                    "replacement variant evaluated, better form kept",
    "joint": "greedy forward addition in dMAE order, accept only on joint LOSO "
             "improvement; final set must beat incumbent or v5 is null",
    "windows": "train 2012-2017 (h1) / 2013-2017 (h2); 2018-2026 untouched; "
               "live 2026-27 holdout unaffected (PLAN_V5 H)",
    "committed_before_results": "PLAN_V5.md written 2026-08-19 before this ran",
}


def main():
    t0 = time.time()
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p4 = json.loads((OUT / "params_v4.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    ship = p4["shipped"]
    g, ts = E.load()
    _, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilderV5(end_r, ts, goalie_hp=p2["goalie_hp"],
                          skater_delta=p2["skater_delta"])

    p5 = {"prereg": PREREG, "gate_F3": {}, "shipped": {}}
    final_sets, final_lams = {}, {}
    for h in (1, 2):
        X, y, meta = fb.feature_matrix(TRAIN[h], h, feats=FEATURES_V5_ALL)
        seasons = meta["T"].to_numpy()
        inc = list(ship[f"features_h{h}"])
        cols_i = [FEATURES_V5_ALL.index(f) for f in inc]
        lam_i, tab_i = R.loso_cv(X[:, cols_i], y, seasons)
        mae_i = tab_i.mae.min()
        print(f"\n=== h{h}: incumbent ({len(inc)} feats) LOSO MAE {mae_i:.4f} "
              f"(lam {lam_i}) ===")
        rec = {"mae_incumbent": round(float(mae_i), 4), "lam_incumbent": float(lam_i),
               "candidates": {}}

        results = []
        for cand in FEATURES_CAND:
            cols_c = cols_i + [FEATURES_V5_ALL.index(cand)]
            lam_c, tab_c = R.loso_cv(X[:, cols_c], y, seasons)
            mae_c = tab_c.mae.min()
            stab = float(R.loso_coef_signs(X[:, cols_c], y, seasons, lam_c)[-1])
            dmae = float(mae_c - mae_i)
            passed = (dmae < 0.0) and (stab >= 0.67)
            results.append((cand, dmae, stab, passed, mae_c))
            print(f"  {cand:16s} dMAE {dmae:+.4f}  stab {stab:.2f}  "
                  f"-> {'PASS' if passed else 'fail'}")
            rec["candidates"][cand] = {"dmae": round(dmae, 4),
                                       "stability": round(stab, 2),
                                       "pass": bool(passed)}

        # collinearity rule: keep at most best of XG_FAMILY; try replacement form
        passers = [r for r in results if r[3]]
        fam = [r for r in passers if r[0] in XG_FAMILY]
        if len(fam) > 1:
            best = min(fam, key=lambda r: r[1])
            passers = [r for r in passers if r[0] not in XG_FAMILY or r is best]
        repl_note = None
        if fam:
            best = min(fam, key=lambda r: r[1])
            swap = [f for f in inc if f != "xg_dev"] + [best[0]]
            cols_s = [FEATURES_V5_ALL.index(f) for f in swap]
            lam_s, tab_s = R.loso_cv(X[:, cols_s], y, seasons)
            mae_s = tab_s.mae.min()
            repl_note = {"candidate": best[0], "mae_replace_xg_dev": round(float(mae_s), 4),
                         "mae_add": round(float(best[4]), 4)}
            print(f"  replacement test: {best[0]} for xg_dev -> MAE {mae_s:.4f} "
                  f"(add form {best[4]:.4f})")
            if mae_s < best[4]:
                inc = swap[:-1]      # xg_dev removed; candidate added in greedy below
                cols_i = [FEATURES_V5_ALL.index(f) for f in inc]
                repl_note["chosen"] = "replace"
            else:
                repl_note["chosen"] = "add"
        rec["xg_family"] = repl_note

        # greedy forward joint assembly
        cur, cur_cols = list(inc), list(cols_i)
        lam_j, tab_j = R.loso_cv(X[:, cur_cols], y, seasons)
        mae_j = tab_j.mae.min()
        for cand, dmae, stab, _, _ in sorted(passers, key=lambda r: r[1]):
            trial = cur_cols + [FEATURES_V5_ALL.index(cand)]
            lam_t, tab_t = R.loso_cv(X[:, trial], y, seasons)
            if tab_t.mae.min() < mae_j:
                cur.append(cand)
                cur_cols = trial
                lam_j, mae_j = lam_t, tab_t.mae.min()
                print(f"  joint: +{cand} -> {mae_j:.4f} (kept)")
            else:
                print(f"  joint: +{cand} -> {tab_t.mae.min():.4f} (rejected)")
        shipped_h = (mae_j < mae_i) and (set(cur) != set(ship[f"features_h{h}"]))
        rec["joint"] = {"final_set_new": sorted(set(cur) - set(ship[f"features_h{h}"])),
                        "removed": sorted(set(ship[f"features_h{h}"]) - set(cur)),
                        "mae_final": round(float(mae_j), 4),
                        "dmae_vs_incumbent": round(float(mae_j - mae_i), 4),
                        "lam": float(lam_j), "ships": bool(shipped_h)}
        print(f"  h{h} FINAL: MAE {mae_j:.4f} vs incumbent {mae_i:.4f} -> "
              f"{'SHIPS' if shipped_h else 'NULL (v4 set stands)'}")
        final_sets[h] = cur if shipped_h else list(ship[f"features_h{h}"])
        final_lams[h] = float(lam_j if shipped_h else ship[f"lam_h{h}"])
        p5["gate_F3"][f"h{h}"] = rec

    p5["shipped"] = {
        "features_h1": final_sets[1], "features_h2": final_sets[2],
        "lam_h1": final_lams[1], "lam_h2": final_lams[2],
        "v5_is_null": not (p5["gate_F3"]["h1"]["joint"]["ships"]
                           or p5["gate_F3"]["h2"]["joint"]["ships"]),
        "inherits": "everything else (overlay, availability2, goalie layer, sigma, "
                    "n0_g) from params_v4 shipped",
    }
    p5["seconds"] = round(time.time() - t0, 1)
    (OUT / "params_v5.json").write_text(json.dumps(p5, indent=2))
    print(f"\nparams_v5.json written ({p5['seconds']}s)")


if __name__ == "__main__":
    main()
