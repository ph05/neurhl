"""Assemble orr/experiments/L1/result.json from ledger.json and test_output.json."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    L = json.loads((HERE / "ledger.json").read_text())
    T = json.loads((HERE / "test_output.json").read_text())
    rows = {m: {str(r["season"]): r for r in rs} for m, rs in T["modes"].items()}
    gs, go = rows["goals_shots"], rows["goals_only"]
    prim = T["primary"]["pred_vs_mix_goals_shots"]
    out = {
        "id": "L1",
        "hypothesis": L["pre_registration"]["hypothesis"],
        "accepted": T["primary"]["accepted"],
        "rule": T["primary"]["rule"],
        "test_metric": "log loss, NeurHL-G gate games 2019-24 (n=6289), shipped in-season loop, goals+shots",
        "baseline_mix": gs["all"]["mix"], "candidate_pred": gs["all"]["pred"],
        "upper_bound_actual": gs["all"]["actual"], "hybrid": gs["all"]["hybrid"],
        "diff_pred_vs_mix": prim["diff"], "ci95": prim["ci95"],
        "neurhl_g": gs["all"]["neurhl_g"], "neurhl_elo": gs["all"]["neurhl_elo"],
        "coverage": T["coverage"],
        "by_season_goals_shots": {k: {a: v[a] for a in ("n", "mix", "pred", "actual", "hybrid")}
                                  | {"d_pred_vs_mix": v["d_pred_vs_mix"]}
                                  for k, v in gs.items()},
        "covered_games_goals_shots": {a: gs["covered"][a] for a in ("n", "mix", "pred", "actual")}
                                     | {"d_pred_vs_mix": gs["covered"]["d_pred_vs_mix"]},
        "secondary": {
            "goals_only": {a: go["all"][a] for a in ("mix", "pred", "actual", "hybrid")}
                          | {"d_pred_vs_mix": go["all"]["d_pred_vs_mix"],
                             "d_actual_vs_mix": go["all"]["d_actual_vs_mix"]},
            "goals_shots_actual_vs_mix": gs["all"]["d_actual_vs_mix"],
            "goals_shots_pred_vs_actual": gs["all"]["d_pred_vs_actual"],
            "goals_shots_hybrid_vs_mix": gs["all"]["d_hybrid_vs_mix"],
            "goals_shots_pred_vs_neurhl_g": gs["all"]["d_pred_vs_neurhl_g"],
            "goals_shots_teamhist_start_pred_vs_mix": gs["all"]["d_pred_vs_mix_teamhist"],
        },
        "reproduction_of_published_baselines": T["reproduction"],
        "choice_model_test_seasons": {k: v["choice"] for k, v in T["choice_model_by_season"].items()
                                      if int(k) >= 2019},
        "tuning": {
            "choice_model": L["chosen"]["choice_model"],
            "offsets": L["chosen"]["offsets"],
            "stage_a": [{k: c[k] for k in ("id", "logloss", "acc")} | {"gdiff_r2": c["gdiff"]["r2_mean"]}
                        for c in L["stage_a"]],
            "stage_b": [{k: c[k] for k in ("id", "pooled")} for c in L["stage_b"]],
            "confirm_2018_of_B4": L["confirm_2018"],
        },
    }
    (HERE / "result.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("accepted", "baseline_mix", "candidate_pred",
                                         "upper_bound_actual", "diff_pred_vs_mix", "ci95")}))


if __name__ == "__main__":
    main()
