"""Write orr/experiments/X3/result.json and ledger.json from the single
hindcast (orr/output/backtest/inseason_bt_1_1.json) and the integration
checks (checks.json). Run: python3 -m orr.experiments.X3.make_result"""
from __future__ import annotations

import json
from pathlib import Path

from orr import config as C

HERE = Path(__file__).resolve().parent
BT = C.OUT / "backtest" / "inseason_bt_1_1.json"


def r5(x):
    return round(float(x), 5)


def ci(d):
    return [r5(d["ci95"][0]), r5(d["ci95"][1])]


def main():
    bt = json.loads(BT.read_text())
    ck = json.loads((HERE / "checks.json").read_text())
    a = bt["all"]
    gs, go = a["goals_shots"], a["goals_only"]
    dec = bt["decision"]
    ledger = {
        "item": "X3",
        "configurations_tried": [{"name": "ORR 1.1 = shipped loop + X1 (fixed X1 configuration)",
                                  "lineup_settings": bt["lineup_settings"], "tuned": False}],
        "note": "X3 tunes nothing: the only accepted item is X1, whose configuration was fixed before "
                "X1's test. No other configuration was run.",
        "runs": [{"what": "integration checks, tuning season 2016 only", "file": "checks.json",
                  "utc": ck["created_utc"]},
                 {"what": "hindcast on the gate games 2019-24, run once", "file": str(BT.relative_to(C.ROOT)),
                  "utc": bt["created_utc"], "secs": bt["secs"]}],
        "prereg": "prereg.json (declared 2026-10-01T03:59:19Z, amendment on bootstrap procedure before the run)"}
    (HERE / "ledger.json").write_text(json.dumps(ledger, indent=1))

    by = {V: {vt: {"diff": r5(r[vt]["d_orr_1_1_vs_shipped"]["diff"]), "ci95": ci(r[vt]["d_orr_1_1_vs_shipped"])}
              for vt in ("goals_shots", "goals_only")} for V, r in bt["by_season"].items()}
    known = bt["lineup_and_starters_known_games"]
    res = {
        "id": "X3",
        "accepted": bool(dec["accepted"]),
        "ship_x1_default_on": bool(dec["ship_x1_default_on"]),
        "summary": (
            "Only X1 of the eight experiments was accepted, so ORR 1.1 = the shipped in-season loop + X1 "
            "(dressed-skater lineup offsets and known starters in the filter's prediction and update), "
            "integrated in the core behind settings (ratings.run_filter(lineup=...), InSeasonFilter "
            "lo_h/lo_a and gdiff_h/gdiff_a, orr/lineups.py, inseason.py --model 1.1 default). "
            f"The single hindcast on the gate games 2019-24 (n={a['n']}) gives, goals+shots, "
            f"{r5(gs['orr_1_1'])} vs the shipped loop's {r5(gs['shipped'])} "
            f"({r5(gs['d_orr_1_1_vs_shipped']['diff'])}, CI {ci(gs['d_orr_1_1_vs_shipped'])}); goals-only "
            f"{r5(go['orr_1_1'])} vs {r5(go['shipped'])} ({r5(go['d_orr_1_1_vs_shipped']['diff'])}, "
            f"CI {ci(go['d_orr_1_1_vs_shipped'])}). The goals-only CI includes 0, so X3 is NOT accepted "
            "under the pre-registered reading (both variants); X1 ships default ON under the plan's "
            "fallback clause because it does not hurt in either variant."),
        "tuning": "None. X3 tunes nothing; ORR 1.1 uses X1's configuration fixed before X1's test "
                  "(beta_x 0.45, beta_st 0, beta_t 0.6, TOI scale 300, expected-lineup half-life 10, "
                  "shot_mult 0, use_goalie). One configuration, logged in ledger.json.",
        "test_metric": "Home-win log loss on the NeurHL-G gate games 2019-24 (n=6289), the shipped "
                       "in-season loop protocol of inseason_bt.py (preseason pipeline start, OT/SO from each "
                       "arm's team-history run), paired bootstrap 95% CI (games_bt.paired). Headline numbers "
                       "(baseline/candidate/diff/ci95) are the GOALS-ONLY variant, the one that decides the "
                       "verdict and that the live loop runs while results carry no shots; goals+shots in extra.",
        "baseline": r5(go["shipped"]), "candidate": r5(go["orr_1_1"]),
        "diff": r5(go["d_orr_1_1_vs_shipped"]["diff"]), "ci95": ci(go["d_orr_1_1_vs_shipped"]),
        "rule_check": (
            "Plan: \"Accept if the combined loop improves on the current shipped loop (same variant) with a "
            "95% CI that excludes 0. Otherwise ship only the individually accepted items that do not hurt "
            "in combination.\" Reading declared before the run (prereg.json): both variants must pass. "
            f"Goals+shots: {r5(gs['d_orr_1_1_vs_shipped']['diff'])}, CI {ci(gs['d_orr_1_1_vs_shipped'])} -> MET. "
            f"Goals-only: {r5(go['d_orr_1_1_vs_shipped']['diff'])}, CI {ci(go['d_orr_1_1_vs_shipped'])} -> "
            "NOT MET. X3 NOT ACCEPTED. Fallback: X1 does not hurt (point estimates <= 0 in both variants) -> "
            "X1 ships in ORR 1.1, default ON. Hypothesis clauses: 'better than any accepted item alone' is an "
            "identity (the combination is X1); 'better than NeurHL-G': goals+shots "
            f"{r5(gs['d_orr_1_1_vs_neurhl_g']['diff'])} (CI {ci(gs['d_orr_1_1_vs_neurhl_g'])}), a tie; "
            f"goals-only {r5(go['d_orr_1_1_vs_neurhl_g']['diff'])} (CI {ci(go['d_orr_1_1_vs_neurhl_g'])}), "
            "NeurHL-G better, not significantly."),
        "extra": {
            "gate_all": {vt: {"shipped": r5(a[vt]["shipped"]), "orr_1_1": r5(a[vt]["orr_1_1"]),
                              "d_vs_shipped": {"diff": r5(a[vt]["d_orr_1_1_vs_shipped"]["diff"]),
                                               "ci95": ci(a[vt]["d_orr_1_1_vs_shipped"])},
                              "d_vs_neurhl_g": {"diff": r5(a[vt]["d_orr_1_1_vs_neurhl_g"]["diff"]),
                                                "ci95": ci(a[vt]["d_orr_1_1_vs_neurhl_g"])},
                              "d_vs_neurhl_elo": {"diff": r5(a[vt]["d_orr_1_1_vs_neurhl_elo"]["diff"]),
                                                  "ci95": ci(a[vt]["d_orr_1_1_vs_neurhl_elo"])},
                              "teamhist_start": {"shipped": r5(a[vt]["shipped_teamhist"]),
                                                 "orr_1_1": r5(a[vt]["orr_1_1_teamhist"])}}
                         for vt in ("goals_shots", "goals_only")},
            "neurhl_g": r5(a["neurhl_g"]), "neurhl_elo": r5(a["neurhl_elo"]),
            "lineup_known_share": round(a["lineup_known_share"], 3),
            "by_season_orr_1_1_minus_shipped": by,
            "games_with_lineups_and_starters": {
                "n": known["n"],
                **{vt: {"diff": r5(known[vt]["d_orr_1_1_vs_shipped"]["diff"]),
                        "ci95": ci(known[vt]["d_orr_1_1_vs_shipped"])} for vt in ("goals_shots", "goals_only")}},
            "sensitivity_nb20000": {vt: ci(d) for vt, d in bt["sensitivity_nb20000"].items()},
            "reproduces_x1_test": "yes: 0.66001 / 0.66217 and identical CIs to X1's test_output.json",
            "integration_checks_2016": {k: v for k, v in ck.items() if k.endswith("maxdiff")},
            "today_forecast_2026_10_01": "regenerated with ORR 1.1 at 2026-10-01T04:09:30Z (before the "
                                          "15:00Z deadline); max |change| in p_home_win 0.0002, all from the "
                                          "09-29 starters entering the update; no lineup offset was non-zero.",
            "tests": "test_season 6/6, test_players 10/10, test_gamemodel 12/12, test_inseason 4/4 (2 new), "
                     "test_freeze 7/7: all pass."},
        "integrate": (
            "DONE in the core (defaults reproduce ORR 1.0 exactly when the options are off): "
            "orr/lineups.py (new: LineupHP/X1 settings, backtest_offsets, live_lineups, live_values, "
            "live_offsets); orr/ratings.py run_filter(lineup=None) and InSeasonFilter (lo_h/lo_a, "
            "gdiff_h/gdiff_a columns, _goal_offsets, predict_eta); orr/inseason.py MODELS {'1.1','1.0'}, "
            "--model (default 1.1), --lineup-dir, merge_starters; orr/backtest/inseason_bt_1_1.py (new). "
            "Parameters: beta_x 0.45, beta_st 0, beta_t 0.6, toi_scale 300, halflife 10, use_goalie True."),
        "caveats": [
            "Goals-only, the variant the live loop runs today (results carry no shots), does not clear the "
            "rule: the live gain is not established at 95%.",
            "Live lineups/starters come from NeurHL's pregame files (DailyFaceoff projections and "
            "confirmations), not box scores as in the backtest; past games' starters are the pregame "
            "starters, which may differ from who actually played. Expect a smaller live gain.",
            "Live starter talent uses inseason.starter_diffs (the 2027 talent table minus the start-share "
            "mix), the backtest uses structural.goalie_game_talent (walk-forward with in-season evidence, "
            "EWMA reference): same units and coefficient, different reference.",
            "A side without a lineup counts as its expected lineup (D=0) live; in the backtest both sides "
            "always come together from the box score.",
            "The bootstrap CIs depend on games_bt's module RNG call order; the order was fixed before the run "
            "(prereg amendment) and a 20,000-resample check agrees.",
            "run_2026-10-01.json records code '6cb0904' (HEAD) while the ORR 1.1 code is uncommitted; the "
            "commit that adds it should be pushed together with the regenerated forecast.",
            "RESULTS.md previously attributed '+0.0006 (CI -0.0019 to +0.0029)' vs NeurHL-G to the no-starter "
            "goals+shots loop; that figure is the known-starters mode (flagged by M2). Corrected to +0.0012 "
            "(CI -0.0011 to +0.0036).",
            "Pre-existing, not changed: build_dashboard keeps the EARLIEST live forecast per game while "
            "score.py keeps the latest before the deadline."],
        "report_note": "REPORT.md could not be written: the harness refuses report .md files from this "
                       "subagent ('subagents should return findings as text'). Its content (hypothesis, what "
                       "was done, tuning, the single test with CI, decision, integration) is in this file and "
                       "in the subagent's final response.",
        "files": [str(HERE / f) for f in ("__init__.py", "prereg.json", "checks.py", "checks.json",
                                         "make_result.py", "ledger.json", "result.json")]
        + [str(C.ROOT / f) for f in ("orr/lineups.py", "orr/ratings.py", "orr/inseason.py",
                                    "orr/backtest/inseason_bt_1_1.py", "orr/tests/test_inseason.py",
                                    "orr/output/backtest/inseason_bt_1_1.json",
                                    "orr/output/backtest/inseason_bt_1_1_preds.csv.gz",
                                    "orr/RESULTS.md", "orr/README.md", "orr/output/live/2026-10-01")]}
    (HERE / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("accepted", "baseline", "candidate", "diff", "ci95")}, indent=1))


if __name__ == "__main__":
    main()
