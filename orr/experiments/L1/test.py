"""L1 task 4: the single test run (configuration fixed in ledger.json["chosen"]).

Protocol = orr/backtest/inseason_bt.py, per arm: a team-history-start run
gives the walk-forward OT/SO parameters; the shipped run starts each clean
market season (2019, 2020, 2022-24) from the preseason pipeline's ratings
(gamefile_bt.game_file_season, views mkt+td) and uses those OT/SO parameters.
Scored on NeurHL-G's gate games 2019-24 (n = 6289).

Arms (offset sources of lfilter.run_filter_gk):
  mix     the start-share mix                    (inseason_bt no_starters; lower bound)
  actual  actual starters, prediction + update   (inseason_bt known_starters; upper bound)
  pred    the choice model's expected starter, prediction + update (L1)
  hybrid  expected starter for the prediction, actual starter in the update
for goals + shots (frozen HP; PRIMARY) and goals only (use_shots=False).

Primary: pred vs mix, goals + shots, paired bootstrap 95% CI.
Run once: python3 -m orr.experiments.L1.test   (refuses to overwrite its output)
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from orr import ratings as R
from orr import structural as S
from orr.backtest import gamefile_bt as GB
from orr.backtest.games_bt import PREDS, paired
from orr.experiments.L1 import lfilter as LF
from orr.experiments.L1 import tune as TU

HERE = Path(__file__).resolve().parent
OUTF = HERE / "test_output.json"
PREDF = HERE / "test_preds.csv.gz"
VIEWS = ["mkt_rel82", "td_rel82"]
DRY = "--dry" in sys.argv     # expected offsets zeroed (pred == mix): validates the pipeline
                              # against the published baselines without scoring L1
PUBLISHED = {("goals_shots", "mix"): 0.66131, ("goals_shots", "actual"): 0.66062,
             ("goals_only", "mix"): 0.66337}


def main():
    global OUTF, PREDF
    if DRY:
        OUTF, PREDF = HERE / "dry_output.json", HERE / "dry_preds.csv.gz"
    elif OUTF.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUTF} exists: the test has already been run")
    t0 = time.time()
    L = json.loads(TU.LEDGER.read_text())
    ch = L["chosen"]
    cfg = {k: ch["choice_model"][k] for k in ("feats", "hs", "hl", "lam")}
    off = {"ctx_kind": ch["offsets"]["ctx_kind"], "lam": ch["offsets"]["lam"]}
    ref_kind = ch["offsets"].get("ref_kind", "ewma")
    hp = R.load_hp()
    hp_go = dataclasses.replace(hp, use_shots=False)

    # expected starters, walk-forward (choice model refitted on seasons < V)
    egd, info = TU.walkforward_egd(hp.goalie, cfg, range(2012, 2025), ref_kind=ref_kind)
    if DRY:
        egd["egd_h"], egd["egd_a"] = 0.0, 0.0
    print(f"choice model walk-forward done ({time.time() - t0:.0f}s)", flush=True)

    # the shipped loop's starting ratings (as inseason_bt.py)
    hist = GB.hist_frame()
    over = {}
    for V in GB.SEASONS:
        P_V = R.fit_gamemodel_params(V, hp, write=False)
        _, lg, r = GB.game_file_season(V, hist, VIEWS, P_V, with_ratings=True)
        over[V] = (r, lg["mu"])
    print(f"preseason ratings rebuilt ({time.time() - t0:.0f}s)", flush=True)

    gate = pd.read_csv(PREDS / "g_gate_games.csv")
    g = S.game_frame()[["gid", "game_id", "season_end", "home_win"]]
    gt = S.goalie_game_talent(*hp.goalie)
    arms = {"mix": dict(pred_src="mix"), "actual": dict(pred_src="actual"),
            "pred": dict(pred_src="pred", **off),
            "hybrid": dict(pred_src="pred", upd_src="actual", **off)}
    cols = {}
    for mode, hpm in (("goals_shots", hp), ("goals_only", hp_go)):
        for arm, kw in arms.items():
            base = LF.run_filter_gk(hpm, GB.SEASONS, egd=egd, **kw)
            ot = {}
            pb = R.predict_probs(base, hpm, ot_params=ot)
            ship = LF.run_filter_gk(hpm, GB.SEASONS, egd=egd, pre_override=over, **kw)
            ps = R.predict_probs(ship, hpm, ot_params=dict(ot))
            cols[f"{mode}_{arm}"] = ps.set_index("gid").p_home_win
            cols[f"{mode}_{arm}_teamhist"] = pb.set_index("gid").p_home_win
            print(f"arm {mode}/{arm} done ({time.time() - t0:.0f}s)", flush=True)
    m = g.merge(gate[["game_id", "p_stack", "p_elo"]], on="game_id")
    m = m[m.season_end.isin(GB.SEASONS)].reset_index(drop=True)
    for c, s in cols.items():
        m[c] = m.gid.map(s)
    m = m.merge(gt[["gid", "gdiff_h", "gdiff_a"]], on="gid", how="left")
    e = m[["gid"]].merge(egd, on="gid", how="left")
    m["covered"] = (m.gdiff_h.notna() & m.gdiff_a.notna() & e.egd_h.notna().to_numpy()
                    & e.egd_a.notna().to_numpy())

    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocol": __doc__, "config": {"choice_model": cfg, "offsets": {**off, "ref_kind": ref_kind}},
           "n": int(len(m)), "coverage": {"pooled": float(m.covered.mean()),
                                          **{int(V): float(x.covered.mean()) for V, x in m.groupby("season_end")}},
           "choice_model_by_season": {int(k): v for k, v in info.items()},
           "reproduction": {}, "modes": {}}
    for mode in ("goals_shots", "goals_only"):
        rows = []
        for V, x in list(m.groupby("season_end")) + [("all", m), ("covered", m[m.covered])]:
            y = x.home_win.to_numpy()
            r = {"season": V if isinstance(V, str) else int(V), "n": int(len(x))}
            for arm in arms:
                r[arm] = R.logloss(x[f"{mode}_{arm}"], y)
                r[f"{arm}_teamhist"] = R.logloss(x[f"{mode}_{arm}_teamhist"], y)
            r["neurhl_g"] = R.logloss(x.p_stack, y)
            r["neurhl_elo"] = R.logloss(x.p_elo, y)
            r["d_pred_vs_mix"] = paired(x[f"{mode}_pred"], x[f"{mode}_mix"], y)
            if V in ("all", "covered"):
                r["d_pred_vs_actual"] = paired(x[f"{mode}_pred"], x[f"{mode}_actual"], y)
                r["d_actual_vs_mix"] = paired(x[f"{mode}_actual"], x[f"{mode}_mix"], y)
                r["d_hybrid_vs_mix"] = paired(x[f"{mode}_hybrid"], x[f"{mode}_mix"], y)
                r["d_pred_vs_neurhl_g"] = paired(x[f"{mode}_pred"], x.p_stack, y)
                r["d_pred_vs_neurhl_elo"] = paired(x[f"{mode}_pred"], x.p_elo, y)
                r["d_pred_vs_mix_teamhist"] = paired(x[f"{mode}_pred_teamhist"],
                                                     x[f"{mode}_mix_teamhist"], y)
            rows.append(r)
        res["modes"][mode] = rows
        allr = [r for r in rows if r["season"] == "all"][0]
        for arm in ("mix", "actual"):
            if (mode, arm) in PUBLISHED:
                res["reproduction"][f"{mode}_{arm}"] = {"this_run": allr[arm],
                                                        "published": PUBLISHED[(mode, arm)]}
        print(f"\n[{mode}]")
        for r in rows:
            d = r["d_pred_vs_mix"]
            print(f"{str(r['season']):>8} n={r['n']:>5} mix {r['mix']:.5f} pred {r['pred']:.5f} "
                  f"actual {r['actual']:.5f} hybrid {r['hybrid']:.5f} G {r['neurhl_g']:.5f} "
                  f"Elo {r['neurhl_elo']:.5f}  pred-mix {d['diff']:+.5f} [{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}]")
    prim = [r for r in res["modes"]["goals_shots"] if r["season"] == "all"][0]["d_pred_vs_mix"]
    res["primary"] = {"pred_vs_mix_goals_shots": prim,
                      "accepted": bool(prim["ci95"][1] < 0),
                      "rule": L["pre_registration"]["test"]["accept_rule"]}
    res["secs"] = round(time.time() - t0)
    OUTF.write_text(json.dumps(res, indent=1, default=float))
    keep = ["gid", "game_id", "season_end", "home_win", "covered", "p_stack", "p_elo"] + list(cols)
    m[keep].to_csv(PREDF, index=False, float_format="%.6f")
    print("\nPRIMARY", json.dumps(res["primary"]), flush=True)


if __name__ == "__main__":
    main()
