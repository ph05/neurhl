"""Assemble orr/experiments/M1/result.json from ledger.json, config.json and
test_output.json. Reads only; computes nothing new on any window.
Run: python3 -m orr.experiments.M1.make_result
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    L = json.loads((HERE / "ledger.json").read_text())
    cfg = json.loads((HERE / "config.json").read_text())
    T = json.loads((HERE / "test_output.json").read_text())
    rows = {str(r["season"]): r for r in T["rows"]}
    P = rows["all"]
    d = P["d_candidate_vs_current"]
    hp = cfg["hp"]
    res = {
        "id": "M1",
        "hypothesis": L["pre_registration"]["hypothesis"],
        "accepted": T["accepted"],
        "rule": L["pre_registration"]["accept_rule"],
        "test_metric": "home-win log loss, NeurHL-G gate games 2019-24 (n=6289), shipped in-season "
                       "loop (inseason_bt goals_only_live protocol), use_shots=False, no starters",
        "baseline_current_goals_only": P["current"],
        "candidate": P["candidate"],
        "diff_candidate_vs_current": d["diff"],
        "ci95": d["ci95"],
        "gain": T["gain"],
        "gap_to_neurhl_g": T["gap_to_neurhl_g"],
        "gap_closed_fraction": T["gap_closed_fraction"],
        "neurhl_g": P["neurhl_g"],
        "neurhl_elo": P["neurhl_elo"],
        "d_candidate_vs_neurhl_g": P["d_candidate_vs_neurhl_g"],
        "d_current_vs_neurhl_g": P["d_current_vs_neurhl_g"],
        "d_candidate_vs_neurhl_elo": P["d_candidate_vs_neurhl_elo"],
        "teamhist_start": {"candidate": P["teamhist_candidate"], "current": P["teamhist_current"],
                           "d": P["d_teamhist_candidate_vs_current"]},
        "by_season": {k: {"n": r["n"], "candidate": r["candidate"], "current": r["current"],
                          "neurhl_g": r["neurhl_g"], "d_candidate_vs_current": r["d_candidate_vs_current"]}
                      for k, r in rows.items() if k != "all"},
        "tuning": {"window": L["pre_registration"]["tuning_window"], "n_configs": len(L["configs"]),
                   "selected": {"tag": cfg["tag"], "coords": cfg["coords"],
                                "q_s": hp["q_s"], "q_f": hp["q_f"], "q_mu": hp["q_mu"],
                                "phi_g": hp["phi_g"], "prior_scale": hp["prior_scale"]},
                   "tuning_ll": cfg["tuning_ll"], "tuning_ll_start": cfg["tuning_ll_start"],
                   "tuning_gain": cfg["tuning_gain"]},
        "confirm_2018": L.get("confirm_2018"),
        "test_runs": 1,
        "files": ["orr/experiments/M1/tune.py", "orr/experiments/M1/test.py",
                  "orr/experiments/M1/make_result.py", "orr/experiments/M1/ledger.json",
                  "orr/experiments/M1/config.json", "orr/experiments/M1/test_output.json",
                  "orr/experiments/M1/test_preds.csv.gz", "orr/experiments/M1/result.json",
                  "orr/experiments/M1/tune.log", "orr/experiments/M1/test.log"],
        "report_note": "REPORT.md was not written: the harness blocks report files from subagents; "
                       "the report was returned in the structured result instead.",
    }
    (HERE / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_season",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
