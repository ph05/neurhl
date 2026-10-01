"""M2 step 4: the single test run on the NeurHL-G gate games 2019-24.

Configuration: config.json (fixed on 2012-2017). For each test season V in
2019, 2020, 2022, 2023, 2024 and each variant, the stack is fitted on the
team-history features of 2012 .. V-1 (2013 excluded; 2018 and 2021 included)
and applied to the features of the loop as run live (filter started from the
shipped market-anchored preseason ratings, inseason_bt). Primary comparison
(the acceptance rule): stack vs the same variant's in-season probability
alone, paired bootstrap 95% CI. Secondary (reported, not for acceptance): the
same on the team-history start; a stack trained on market-start features
where they exist; NeurHL-G and NeurHL's Elo on the same games.

Run once: python3 -m orr.experiments.M2.test
Writes test.json and test_preds.csv.gz.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from orr import ratings as R
from orr.backtest.games_bt import PREDS, paired
from orr.experiments.M2 import features as FT
from orr.experiments.M2 import stack as SK

HERE = Path(__file__).resolve().parent
GATE_SEASONS = [2019, 2020, 2022, 2023, 2024]


def table(m: pd.DataFrame, a: str, b: str) -> list:
    rows = []
    for V, x in list(m.groupby("season_end")) + [("all", m)]:
        y = x.home_win.to_numpy()
        rows.append({"season": V if V == "all" else int(V), "n": int(len(x)),
                     a: R.logloss(x[a], y), b: R.logloss(x[b], y),
                     "d": paired(x[a], x[b], y)})
    return rows


def main():
    out_f = HERE / "test.json"
    if out_f.exists():
        raise SystemExit("test.json exists: the test runs once")
    cfg = json.loads((HERE / "config.json").read_text())["chosen"]
    feat = FT.load()
    gate = pd.read_csv(PREDS / "g_gate_games.csv")[["game_id", "season_end", "y", "p_stack", "p_elo"]]
    gate = gate.rename(columns={"p_stack": "p_neurhl_g", "p_elo": "p_neurhl_elo",
                                "season_end": "season_gate"})
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocol": __doc__, "config": {v: {"family": cfg[v]["family"], "lambda": cfg[v]["lambda"]}
                                           for v in cfg},
           "variants": {}}
    keep = []
    for var in ("go", "gs"):
        c = cfg[var]
        runs = {
            "primary": SK.walk_forward(feat, var, GATE_SEASONS, c["lambda"], c["family"],
                                       train_src="ht", apply_src="ship"),
            "sec_ht_start": SK.walk_forward(feat, var, GATE_SEASONS, c["lambda"], c["family"],
                                            train_src="ht", apply_src="ht"),
            "sec_train_ship_where_available": SK.walk_forward(
                feat, var, GATE_SEASONS, c["lambda"], c["family"], apply_src="ship",
                train_ship_where_available=True)}
        vr = {}
        for tag, (pr, ws) in runs.items():
            m = gate.merge(pr, on="game_id", how="inner")
            assert len(m) == len(gate), f"{var}/{tag}: {len(m)} of {len(gate)} gate games matched"
            assert (m.y == m.home_win).all() and (m.season_gate == m.season_end).all()
            y = m.home_win.to_numpy()
            r = {"n": int(len(m)), "weights": ws,
                 "ll_stack": R.logloss(m.p_stack, y), "ll_in_season": R.logloss(m.p_in, y),
                 "ll_preseason": R.logloss(m.p_pre, y), "ll_orr_elo": R.logloss(m.p_elo, y),
                 "ll_neurhl_g": R.logloss(m.p_neurhl_g, y), "ll_neurhl_elo": R.logloss(m.p_neurhl_elo, y),
                 "stack_vs_in_season": table(m, "p_stack", "p_in"),
                 "stack_vs_neurhl_g": paired(m.p_stack, m.p_neurhl_g, y),
                 "in_season_vs_neurhl_g": paired(m.p_in, m.p_neurhl_g, y),
                 "stack_vs_neurhl_elo": paired(m.p_stack, m.p_neurhl_elo, y)}
            vr[tag] = r
            d = r["stack_vs_in_season"][-1]["d"]
            print(f"[{var} {tag}] n={r['n']} stack {r['ll_stack']:.5f} in-season {r['ll_in_season']:.5f} "
                  f"NeurHL-G {r['ll_neurhl_g']:.5f} NeurHL Elo {r['ll_neurhl_elo']:.5f}  "
                  f"stack-in {d['diff']:+.5f} [{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}]", flush=True)
            if tag == "primary":
                for row in r["stack_vs_in_season"][:-1]:
                    print(f"     {row['season']} n={row['n']} stack {row['p_stack']:.5f} "
                          f"in {row['p_in']:.5f} d {row['d']['diff']:+.5f}")
                keep.append(m.assign(variant=var)[["game_id", "season_end", "variant", "home_win",
                                                   "p_in", "p_pre", "p_elo", "p_stack",
                                                   "p_neurhl_g", "p_neurhl_elo"]])
        vr["accept_condition"] = {
            "upper_ci_below_0": vr["primary"]["stack_vs_in_season"][-1]["d"]["ci95"][1] < 0}
        res["variants"][var] = vr
    go = res["variants"]["go"]["accept_condition"]["upper_ci_below_0"]
    res["accepted"] = bool(go)
    res["accepted_rule"] = ("stack improves on the same variant without it with a 95% CI that "
                            "excludes 0, for at least the goals-only variant (primary comparison)")
    out_f.write_text(json.dumps(res, indent=1, default=float))
    pd.concat(keep).to_csv(HERE / "test_preds.csv.gz", index=False, float_format="%.6f")
    print("accepted:", res["accepted"])


if __name__ == "__main__":
    main()
