"""Collects M2's outputs into result.json.

Run: python3 -m orr.experiments.M2.make_result
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    led = json.loads((HERE / "ledger.json").read_text())
    con = json.loads((HERE / "confirm.json").read_text())
    tst = json.loads((HERE / "test.json").read_text())
    diag = json.loads((HERE / "diag_insample.json").read_text())
    chk = json.loads((HERE / "features_check.json").read_text())

    def prim(v, tag="primary"):
        r = tst["variants"][v][tag]
        d = r["stack_vs_in_season"][-1]["d"]
        return {"n": r["n"], "stack": r["ll_stack"], "in_season_alone": r["ll_in_season"],
                "diff": d["diff"], "ci95": d["ci95"], "se": d["se"],
                "by_season": [{"season": x["season"], "n": x["n"], "stack": x["p_stack"],
                               "in_season": x["p_in"], "diff": x["d"]["diff"], "ci95": x["d"]["ci95"]}
                              for x in r["stack_vs_in_season"][:-1]],
                "neurhl_g": r["ll_neurhl_g"], "neurhl_elo": r["ll_neurhl_elo"],
                "orr_elo": r["ll_orr_elo"], "orr_preseason": r["ll_preseason"],
                "stack_vs_neurhl_g": r["stack_vs_neurhl_g"],
                "stack_vs_neurhl_elo": r["stack_vs_neurhl_elo"],
                "weights_by_test_season": r["weights"]}

    res = {
        "id": "M2",
        "title": "Stack the in-season forecast with Elo",
        "accepted": bool(tst["accepted"]),
        "rule": "PLAN_1_1 M2: accept if the stack improves on the same variant without it, with a 95% CI "
                "that excludes 0, for at least the goals-only variant.",
        "rule_met": {"goals_only": tst["variants"]["go"]["accept_condition"]["upper_ci_below_0"],
                     "goals_and_shots": tst["variants"]["gs"]["accept_condition"]["upper_ci_below_0"]},
        "test_runs": 1,
        "test_window": "NeurHL-G gate games 2019, 2020, 2022-24 (n = 6289)",
        "config": {v: {k: cfg["chosen"][v][k] for k in ("stage", "family", "lambda")} for v in cfg["chosen"]},
        "tuning": {"seasons": led["tuning_seasons"], "n_configs": led["n_configs"],
                   "per_variant": {v: {"chosen_ll": cfg["chosen"][v]["tune_ll"],
                                       "in_season_alone_ll": cfg["chosen"][v]["baseline_in_season_ll"],
                                       "gain": cfg["chosen"][v]["gain"],
                                       "best_less_regularised": min(
                                           (r["gain"] for r in led["configs"]
                                            if r["variant"] == v and r["lambda"] < 1.0)),
                                       "elo_ll": cfg["chosen"][v]["elo_ll"],
                                       "preseason_ll": cfg["chosen"][v]["preseason_ll"]}
                                   for v in cfg["chosen"]}},
        "confirm_2018": {v: {"stack": con[v]["stack"], "in_season_alone": con[v]["in_season"],
                             "diff": con[v]["d_stack_vs_in_season"]["diff"],
                             "ci95": con[v]["d_stack_vs_in_season"]["ci95"]} for v in con},
        "test": {"goals_only": {"primary": prim("go"),
                                "secondary_ht_start": prim("go", "sec_ht_start"),
                                "secondary_train_ship_where_available": prim("go", "sec_train_ship_where_available")},
                 "goals_and_shots": {"primary": prim("gs"),
                                     "secondary_ht_start": prim("gs", "sec_ht_start"),
                                     "secondary_train_ship_where_available": prim("gs", "sec_train_ship_where_available")}},
        "diagnostic_insample_2012_2017": diag,
        "feature_check_vs_games_bt": chk,
        "integrate": "",
    }
    (HERE / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("accepted", "rule_met", "config")}, indent=1))
    for v in ("goals_only", "goals_and_shots"):
        p = res["test"][v]["primary"]
        print(v, round(p["stack"], 5), round(p["in_season_alone"], 5), p["diff"], p["ci95"])


if __name__ == "__main__":
    main()
