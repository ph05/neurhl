"""NeurHL-H — the one-shot confirmation on 2018-2026 (PLAN_NeurHL A5.2).

Tests whether NeurHL-H has lower per-game log loss than v1 Elo on regular-season
games of season_end 2018-2026, with the model frozen exactly as it was scored on
the tune window. This is the only sanctioned use of the window. It runs once:

  * refuses if output/hier_restatement.json already exists;
  * refuses unless the tune-window record (output/hier_result.json) exists and
    amendment A5 is committed in PLAN_NeurHL.md.

Primary: scored set {2018-2020, 2022-2026} (NO_SCORE 2021), n = 10,184, paired
per-game difference two-sided p < 0.05, season-clustered direction agreeing.
Reported alongside, never as substitutes: the same battery including 2021, and
the sensitivity analyses in eval/hier_core.battery. Per-game predictions are
committed to output/preds/hier_restatement_games.csv.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, PROJ, TENSORS  # noqa: E402
from eval.backtest_hier import NO_SCORE  # noqa: E402
from eval.hier_core import battery, game_frame, predict  # noqa: E402

RESTATE = list(range(2018, 2027))


def a5_committed() -> bool:
    r = subprocess.run(["git", "-C", str(PROJ), "log", "--format=%H", "-1",
                        "-S", "## A5. AMENDMENT 5", "--", "PLAN_NeurHL.md"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def main():
    out_p = NOUT / "hier_restatement.json"
    assert not out_p.exists(), "the confirmation has already run (single use)"
    assert (NOUT / "hier_result.json").exists(), "tune-window record missing"
    assert a5_committed(), "PLAN_NeurHL amendment A5 must be committed first"

    d = game_frame(TENSORS, max(RESTATE))
    missing = [s for s in RESTATE if not (d.season_end == s).any()]
    assert not missing, f"missing Layer-1 projections for {missing}"
    P = predict(d, RESTATE)
    primary = P[~P.season.isin(NO_SCORE)]

    res = {
        "test": "NeurHL-H vs v1 Elo, paired per-game log loss (PLAN_NeurHL A5.2)",
        "window": "season_end 2018-2026, regular season; spent once",
        "primary": {"scored_seasons": sorted(int(s) for s in primary.season.unique()),
                    **battery(primary)},
        "including_2021": {"scored_seasons": sorted(int(s) for s in P.season.unique()),
                           **battery(P)},
        "head_C_by_season": {str(int(s)): float(c) for s, c in
                             P.groupby("season").head_C.first().items()},
    }
    res["verdict"] = "PASS" if res["primary"]["pass"] else "NULL"
    (NOUT / "preds").mkdir(exist_ok=True)
    P.round(6).to_csv(NOUT / "preds" / "hier_restatement_games.csv", index=False)
    out_p.write_text(json.dumps(res, indent=1))

    for k in ("primary", "including_2021"):
        v = res[k]
        print(f"{k:15s} n={v['n_games']:5d}  NeurHL-H {v['neurhl_h']:.5f}  "
              f"Elo {v['elo']:.5f}  diff {v['diff']:+.5f} "
              f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]  p={v['p_two_sided']:.4f}  "
              f"clustered p={v['clustered_p']:.4f}  {v['seasons_won']}/{v['seasons']}")
    print(f"verdict: {res['verdict']}")


if __name__ == "__main__":
    main()
