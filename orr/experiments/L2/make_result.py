"""Assemble result.json for L2 from ledger.json, config.json, confirm.json, test.json."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    led = json.loads((HERE / "ledger.json").read_text())
    cfg = json.loads((HERE / "config.json").read_text())
    con = json.loads((HERE / "confirm.json").read_text())
    tst = json.loads((HERE / "test.json").read_text())
    pB, pA, pall = tst["pooled"]["B"], tst["pooled"]["A"], tst["pooled"]["all"]
    cond = {
        "pooled_B_gain_ge_0.08": {"value": -pB["diff"], "met": -pB["diff"] >= 0.08},
        "B_improves_in_ge_3_of_5_seasons": {"value": pB["seasons_improved"],
                                            "met": pB["seasons_improved"] >= 3},
        "A_no_worse_than_0.02": {"value": pA["diff"], "met": pA["diff"] <= 0.02},
    }
    accepted = all(c["met"] for c in cond.values())
    moved = {k: v for k, v in cfg["mult"].items() if abs(v - 1.0) > 1e-9}
    out = {
        "id": "L2",
        "hypothesis": "Component-specific reliability for skater rates: per-component "
                      "(g, a1, a2, shots) x situation (EV, PP) x position (F, D) shrinkage "
                      "strengths improve per-82 points MAE (NeurHL sample B).",
        "accepted": accepted,
        "rule": "Accept if pooled sample-B MAE improves by at least 0.08 and in at least 3 of the "
                "5 test seasons, with sample-A MAE no worse by more than 0.02.",
        "rule_conditions": cond,
        "tuning": {"window": led["tune_seasons"], "objective": led["design"]["objective"],
                   "design": led["design"], "n_configurations": len(led["configs"]),
                   "baseline_B": cfg["baseline_tune"]["B"], "chosen_B": cfg["tune"]["B"],
                   "baseline_A": cfg["baseline_tune"]["A"], "chosen_A": cfg["tune"]["A"],
                   "baseline_all": cfg["baseline_tune"]["all"], "chosen_all": cfg["tune"]["all"],
                   "chosen_ledger_n": cfg["ledger_n"], "chosen_multipliers": cfg["mult"],
                   "multipliers_not_1": moved,
                   "K_minutes_at_mean_usage_2017": cfg["K_minutes_at_mean_usage_2017"]},
        "confirm_2018_19": {"pooled": con["pooled"], "seasons": con["seasons"]},
        "test_2022_26": {"metric": "strict protocol, sample B (per-82 points MAE), pooled "
                                   "n-weighted; paired bootstrap 95% CI over players within season",
                         "baseline_B": pB["shipped"], "candidate_B": pB["l2"], "diff_B": pB["diff"],
                         "ci95_B": pB["ci95"], "seasons_B_improved": pB["seasons_improved"],
                         "baseline_A": pA["shipped"], "candidate_A": pA["l2"], "diff_A": pA["diff"],
                         "ci95_A": pA["ci95"],
                         "baseline_all": pall["shipped"], "candidate_all": pall["l2"],
                         "diff_all": pall["diff"], "ci95_all": pall["ci95"],
                         "n": {s: tst["pooled"][s]["n"] for s in ("A", "B", "all")},
                         "seasons": tst["seasons"], "run_at": tst["run_at"]},
        "test_runs": 1,
        "files": [str(HERE / f) for f in ("rates.py", "run.py", "make_result.py", "ledger.json",
                                          "config.json", "confirm.json", "test.json", "tune.log",
                                          "result.json")],
    }
    (HERE / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("accepted", "rule_conditions")}, indent=1))


if __name__ == "__main__":
    main()
