"""Season layer vs v1 on the SAME team-seasons (correction to an earlier claim).

NeurHL-2's season layer was reported as beating "both house benchmarks"
(standings MAE 9.61 vs HOWE 10.36). The HOWE figure is HOWE5's h1 restatement on
2018-2026, a different set of seasons, so that comparison is invalid. This
script compares on the seasons both records share: 2012 and 2014-2017 (2011 has
no v1 record; 2013 is NO_SCORE).

Inputs, all committed tune-window records:
  output/baselines_tune.json            v1 per-season MAE (per 82) and CRPS
  configs/season_backtest.json          NeurHL-2 season layer (shipped RAPM prior)
  configs/roster_prior_comparison.json  the same layer with other roster priors
Writes configs/season_matched_comparison.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIGS, NOUT  # noqa: E402

SEASONS = [2012, 2014, 2015, 2016, 2017]


def paired(a: dict, b: dict) -> dict:
    """a - b over SEASONS: mean, t (df = S - 1), two-sided p, wins for a."""
    from scipy import stats
    d = np.array([a[s] - b[s] for s in SEASONS])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean_diff": float(d.mean()), "t": float(t),
            "p_two_sided": float(2 * stats.t.sf(abs(t), df=len(d) - 1)),
            "a_better_in": int((d < 0).sum()), "of": len(d)}


def main():
    base = json.loads((NOUT / "baselines_tune.json").read_text())
    v1_mae = {int(s): r["v1_mae82"]
              for s, r in base["season_h1_per_season"].items()}
    v1_crps = {int(s): v for s, v in base["season_crps_v1"].items()}
    rp = json.loads((CONFIGS / "roster_prior_comparison.json").read_text())
    models = {"v1 (Elo + xG)": (v1_mae, v1_crps)}
    names = {"rapm": "NeurHL-2 season layer (shipped, RAPM roster prior)",
             "none": "NeurHL-2 season layer, no roster term",
             "raw_plain": "NeurHL-2 season layer, raw on-ice prior"}
    for k, label in names.items():
        rows = {r["season"]: r for r in rp["variants"][k]["rows"]}
        models[label] = ({s: rows[s]["mae"] for s in SEASONS},
                         {s: rows[s]["crps"] for s in SEASONS})

    out = {"seasons": SEASONS,
           "note": ("per-season standings-points MAE and CRPS; all five seasons "
                    "are 82-game seasons, so v1's per-82 MAE is on the same scale"),
           "models": {}, "vs_v1": {}}
    for label, (mae, crps) in models.items():
        out["models"][label] = {
            "mae": float(np.mean([mae[s] for s in SEASONS])),
            "crps": float(np.mean([crps[s] for s in SEASONS])),
            "per_season_mae": {str(s): round(mae[s], 3) for s in SEASONS}}
        if label != "v1 (Elo + xG)":
            out["vs_v1"][label] = {"mae": paired(mae, v1_mae),
                                   "crps": paired(crps, v1_crps)}
    (CONFIGS / "season_matched_comparison.json").write_text(
        json.dumps(out, indent=1))
    for label, m in out["models"].items():
        print(f"{label:55s} MAE {m['mae']:6.3f}  CRPS {m['crps']:6.3f}")
    for label, v in out["vs_v1"].items():
        print(f"  {label}: MAE diff {v['mae']['mean_diff']:+.3f} "
              f"(p={v['mae']['p_two_sided']:.3f}, better in "
              f"{v['mae']['a_better_in']}/5); CRPS diff "
              f"{v['crps']['mean_diff']:+.3f} (p={v['crps']['p_two_sided']:.3f})")


if __name__ == "__main__":
    main()
