"""NeurHL-H tune-window result, reported every way A4.2 requires (A5.1).

The tune-window S1 result is exploratory (PLAN_NeurHL A5.1). A4.2 as committed
requires it to be reported with and without the anomalous seasons; this script
does that for the scored-set choices that changed on the way to the recorded
pass: with/without 2012 and with/without 2013.

  --artifacts original   Layer-1 projections as scored in hier_result.json
                         (kept under data/tensors/superseded_2026-09-25/);
                         must reproduce the recorded figures.
  --artifacts rebuilt    projections rebuilt from the current tensors
                         (train/run_confirm_chain.sh); a consistency check.

Writes configs/hier_both_ways_<artifacts>.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIGS, NOUT, TENSORS  # noqa: E402
from eval.hier_core import battery, game_frame, predict  # noqa: E402

TUNE = [2012, 2013, 2014, 2015, 2016, 2017]
SETS = {
    "2012-2017 (all six)": TUNE,
    "2012, 2014-2017 (recorded pass: NO_SCORE)": [2012, 2014, 2015, 2016, 2017],
    "2013-2017 (committed A4 run, with lockout)": [2013, 2014, 2015, 2016, 2017],
    "2014-2017 (committed A4 run, without lockout)": [2014, 2015, 2016, 2017],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", choices=("original", "rebuilt"),
                    required=True)
    a = ap.parse_args()
    src = (TENSORS / "superseded_2026-09-25" if a.artifacts == "original"
           else TENSORS)
    P = predict(game_frame(src, max(TUNE)), TUNE)
    out = {"artifacts": a.artifacts, "proj_dir": str(src.relative_to(TENSORS.parents[2])),
           "classification": "EXPLORATORY (PLAN_NeurHL A5.1)", "sets": {}}
    for name, seasons in SETS.items():
        b = battery(P[P.season.isin(seasons)])
        b.pop("per_season")
        out["sets"][name] = b
        print(f"{name:48s} n={b['n_games']:5d} diff {b['diff']:+.5f} "
              f"p={b['p_two_sided']:.4f} clustered p={b['clustered_p']:.4f} "
              f"{b['seasons_won']}/{b['seasons']}")
    if a.artifacts == "original":
        rec = json.loads((NOUT / "hier_result.json").read_text())
        mine = out["sets"]["2012, 2014-2017 (recorded pass: NO_SCORE)"]
        out["reproduces_record"] = {
            "recorded_diff": rec["diff"], "reproduced_diff": mine["diff"],
            "abs_err": abs(rec["diff"] - mine["diff"]),
            "ok": abs(rec["diff"] - mine["diff"]) < 1e-9}
        print("reproduces hier_result.json:", out["reproduces_record"])
    (CONFIGS / f"hier_both_ways_{a.artifacts}.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
