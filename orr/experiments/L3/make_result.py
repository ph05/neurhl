"""Assemble result.json for L3 from config.json, ledger.json, confirm.json
and test.json (no model runs here)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    L = json.loads((HERE / "ledger.json").read_text())
    conf = json.loads((HERE / "confirm.json").read_text())
    test = json.loads((HERE / "test.json").read_text())
    t = test["first10_2022_24"]
    b, n = t["pooled"]["baseline"], t["pooled"]["l3"]
    gp_gain = b["gp_mae"] - n["gp_mae"]
    p_gain = b["p_mae_all"] - n["p_mae_all"]
    c1 = gp_gain >= 0.5
    c2 = n["p_mae_all"] < b["p_mae_all"]
    S = {**test["first10_2022_24"]["seasons"], **test["season_team_2025_26"]["seasons"]}

    def pooled(key, nk, m):
        tot = sum(S[v][m][nk] for v in S)
        return sum(S[v][m][key] * S[v][m][nk] for v in S) / tot
    st = test["season_team_2025_26"]
    res = {
        "id": "L3",
        "hypothesis": "Player-specific games-played projections (availability history, age, "
                      "position, usage tier) improve GP error and unconditional points error.",
        "accepted": bool(c1 and c2),
        "rule": "Accept if GP MAE (all rostered skaters) improves by at least 0.5 games AND "
                "sample-'all' points MAE improves, both pooled on 2022-24 (first-10 rosters).",
        "rule_check": {"gp_mae_gain": gp_gain, "gp_condition_met": c1,
                       "p_mae_all_gain": p_gain, "points_condition_met": c2},
        "test_2022_24": {"baseline": b, "l3": n, "paired": t["paired"],
                         "per_season": {v: {k: S[v][k] for k in
                                            ("d_gp_mae", "d_p_mae_all", "d_p_mae_A", "d_p_mae_B")}
                                        for v in test["first10_2022_24"]["seasons"]}},
        "report_2025_26_season_team": {"baseline": st["pooled"]["baseline"],
                                       "l3": st["pooled"]["l3"], "paired": st["paired"],
                                       "per_season": {v: {k: S[v][k] for k in
                                                          ("d_gp_mae", "d_p_mae_all",
                                                           "d_p_mae_A", "d_p_mae_B")}
                                                      for v in st["seasons"]}},
        "pooled_2022_26_strict": {
            m: {"p_mae_A": pooled("p_mae_A", "n_A", m), "p_mae_B": pooled("p_mae_B", "n_B", m),
                "p_mae_all": pooled("p_mae_all", "n_all", m),
                "gp_mae": pooled("gp_mae", "n_rostered", m)} for m in ("baseline", "l3")},
        "neurhl_2022_26": {"A": 9.543, "B": 9.082},
        "confirm_2018_19": {"baseline": conf["pooled"]["baseline"], "l3": conf["pooled"]["l3"],
                            "paired": conf["paired"]},
        "tuning": {"objective_seasons": [2012, 2014, 2015, 2016, 2017],
                   "diagnostic_season": 2011, "n_configs": len(L),
                   "n_by_stage": {s: sum(1 for r in L if r["stage"] == s)
                                  for s in ("baseline", "stage1", "stage2")},
                   "fixed": cfg},
        "test_runs": 1,
    }
    (HERE / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["rule_check"], indent=1), res["accepted"])


if __name__ == "__main__":
    main()
