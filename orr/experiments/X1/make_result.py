"""Write orr/experiments/X1/result.json from ledger.json and test_output.json."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    L = json.loads((HERE / "ledger.json").read_text())
    T = json.loads((HERE / "test_output.json").read_text())
    a = T["gate"]["all"]
    d = a["d_sh_x1_vs_base"]
    accepted = d["ci95"][1] < 0
    ch = L["chosen"]
    res = {
        "id": "X1",
        "hypothesis": "Adjusting both teams' ratings for who actually dresses (skater on-ice value and "
                      "ice time of present minus expected players, plus the starting goalie through the "
                      "goalie layer) improves ORR's in-season log loss.",
        "accepted": bool(accepted),
        "rule": "Accept if the improvement over ORR without lineups has a 95% CI that excludes 0.",
        "config": ch["config"],
        "tuning": {"seasons": L["protocol"]["tuning_seasons"], "n_configs": len(L["configs"]),
                   "tuning_ll": ch["tuning_ll"], "no_lineup_ll": ch["no_lineup_ll"],
                   "starters_only_ll": ch["starters_only_ll"], "skaters_only_ll": ch["skaters_only_ll"],
                   "paired": ch.get("tuning_paired")},
        "confirm_2018": L.get("confirm_2018"),
        "test": {
            "window": "NeurHL-G gate games 2019-24", "n": a["n"],
            "lineup_known_share": a["lineup_known_share"],
            "baseline_no_lineups": a["sh_base"], "starters_only": a["sh_starters"],
            "x1": a["sh_x1"], "x1_skaters_only": a["sh_x1_sk"], "neurhl_g": a["neurhl_g"],
            "neurhl_elo": a["neurhl_elo"],
            "diff_x1_vs_base": d["diff"], "ci95": d["ci95"],
            "diff_x1_vs_starters": a["d_sh_x1_vs_starters"],
            "diff_x1sk_vs_base": a["d_sh_x1sk_vs_base"],
            "diff_starters_vs_base": a["d_sh_starters_vs_base"],
            "goals_only": {"base": a["sh_base_go"], "x1": a["sh_x1_go"],
                           "diff": a["d_sh_x1go_vs_basego"]},
            "team_history_start": {"base": a["th_base"], "x1": a["th_x1"],
                                   "diff": a["d_th_x1_vs_base"]},
            "vs_neurhl_g": a["d_sh_x1_vs_neurhl_g"],
            "by_season": {V: {"n": b["n"], "lineup_known_share": b["lineup_known_share"],
                              "base": b["sh_base"], "x1": b["sh_x1"],
                              "diff": b["d_sh_x1_vs_base"]} for V, b in T["gate"]["by_season"].items()},
            "lineup_known_games": {k: T["gate"]["lineup_known_games"][k] for k in
                                   ("n", "sh_base", "sh_starters", "sh_x1", "d_sh_x1_vs_base",
                                    "d_sh_x1_vs_starters")},
        },
        "restatement_2018_24": T["restatement_2018_24"]["all"],
    }
    (HERE / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["test"], indent=1, default=float)[:3000])
    print("accepted:", accepted)


if __name__ == "__main__":
    main()
