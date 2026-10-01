"""X3r: confirmation of ORR 1.1 on seasons no experiment has used (orr/PLAN_1_1.md).

ORR 1.1 (the fixed X1 configuration: dressed-skater lineup offsets plus known
starting goalies, in the filter's prediction and update) against ORR 1.0
(neither), in the same walk-forward in-season filter, on 2010-11 and 2020-21.
Both seasons have box-score lineups and were used by no ORR 1.1 experiment;
neither has market lines, so both arms start from the team-history prior.
Each arm takes its OT/SO parameters from its own run, as in X3.

Primary: pooled log-loss difference, goals-and-shots variant, paired bootstrap
95% CI (games_bt.paired). Secondary: goals-only, per season, and an
inverse-variance combination with X3's gate result (reported, not decisive).

Run ONCE: python3 -m orr.backtest.inseason_bt_x3r   (refuses to overwrite)
Writes orr/output/backtest/inseason_bt_x3r.json and inseason_bt_x3r_preds.csv.gz.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time

import numpy as np
import pandas as pd

from orr import config as C
from orr import lineups as LU
from orr import ratings as R
from orr.backtest.games_bt import paired

SEASONS = [2011, 2021]
OUT = C.OUT / "backtest"
OUTF = OUT / "inseason_bt_x3r.json"
PREDF = OUT / "inseason_bt_x3r_preds.csv.gz"


def arm(hp: R.HP, use_goalie: bool, lineup) -> pd.DataFrame:
    pred = R.run_filter(hp, SEASONS, use_goalie=use_goalie, lineup=lineup)
    p = R.predict_probs(pred[pred.season_end >= 2010], hp, ot_params={})
    return p[p.season_end.isin(SEASONS)][["gid", "p_home_win"]]


def main():
    if OUTF.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUTF} exists: X3r has already been run")
    t0 = time.time()
    hp = R.load_hp()
    variants = {"goals_shots": hp, "goals_only": dataclasses.replace(hp, use_shots=False)}
    lo = LU.backtest_offsets(LU.X1)
    arms = {"orr_1_0": (False, None), "orr_1_1": (LU.X1.use_goalie, lo)}
    g = R.S.game_frame()
    P = g[g.season_end.isin(SEASONS)][["gid", "game_id", "season_end", "home_win"]].copy()
    for vt, h in variants.items():
        for name, (gk, lu) in arms.items():
            P = P.merge(arm(h, gk, lu).rename(columns={"p_home_win": f"p_{name}_{vt}"}), on="gid", how="left")
            print(f"{vt} {name} done ({time.time() - t0:.0f}s)", flush=True)
    P["lineup_known"] = P.gid.isin(set(lo.gid))
    assert P.filter(like="p_").notna().all().all(), "missing predictions"
    P.to_csv(PREDF, index=False, float_format="%.6f")            # written before any scoring

    y = P.home_win.to_numpy()
    res = {"created_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "protocol": __doc__,
           "seasons": SEASONS, "n": int(len(P)), "lineup_known_share": float(P.lineup_known.mean()),
           "lineup_settings": LU.settings(LU.X1), "variants": {}}
    for vt in variants:          # fixed call order: pooled goals_shots first (primary)
        a, b = P[f"p_orr_1_1_{vt}"], P[f"p_orr_1_0_{vt}"]
        block = {"pooled": {"orr_1_1": R.logloss(a, y), "orr_1_0": R.logloss(b, y), "d": paired(a, b, y)}}
        for V, x in P.groupby("season_end"):
            yy = x.home_win.to_numpy()
            block[str(V)] = {"n": int(len(x)), "orr_1_1": R.logloss(x[f"p_orr_1_1_{vt}"], yy),
                             "orr_1_0": R.logloss(x[f"p_orr_1_0_{vt}"], yy),
                             "d": paired(x[f"p_orr_1_1_{vt}"], x[f"p_orr_1_0_{vt}"], yy)}
        res["variants"][vt] = block
    d = res["variants"]["goals_shots"]["pooled"]["d"]
    res["accepted"] = bool(d["ci95"][1] < 0)
    gate = json.loads((OUT / "inseason_bt_1_1.json").read_text())
    try:
        gd = gate["all"]["goals_shots"]["d_orr_1_1_vs_shipped"]
    except (KeyError, TypeError):
        gd = None
    if gd and gd.get("se") and d.get("se"):
        w1, w2 = 1 / gd["se"] ** 2, 1 / d["se"] ** 2
        est = (w1 * gd["diff"] + w2 * d["diff"]) / (w1 + w2)
        se = (w1 + w2) ** -0.5
        res["combined_with_gate"] = {"diff": est, "se": se, "ci95": [est - 1.96 * se, est + 1.96 * se],
                                     "note": "inverse-variance combination; reported, not decisive"}
    OUTF.write_text(json.dumps(res, indent=1, default=float))
    for vt, blk in res["variants"].items():
        for k, v in blk.items():
            print(f"{vt:12s} {k:7s} 1.1 {v['orr_1_1']:.5f}  1.0 {v['orr_1_0']:.5f}  d {v['d']['diff']:+.5f} "
                  f"[{v['d']['ci95'][0]:+.5f}, {v['d']['ci95'][1]:+.5f}]")
    print("combined with gate:", res.get("combined_with_gate"), " accepted:", res["accepted"])


if __name__ == "__main__":
    main()
