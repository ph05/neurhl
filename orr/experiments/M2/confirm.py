"""M2: one confirmation of the fixed configuration on 2018 (2017-18).

The stack for 2018 is fitted on 2012, 2014-2017 (team-history features) and
scored on the 2018 games against the in-season probability alone, per
variant. Reported once; it does not change config.json. 2019 is not used
here: it belongs to the gate test window.

Run: python3 -m orr.experiments.M2.confirm
"""
from __future__ import annotations

import json
from pathlib import Path

from orr import ratings as R
from orr.backtest.games_bt import paired
from orr.experiments.M2 import features as FT
from orr.experiments.M2 import stack as SK

HERE = Path(__file__).resolve().parent


def main():
    out_f = HERE / "confirm.json"
    if out_f.exists():
        raise SystemExit("confirm.json exists: the 2018 confirmation runs once")
    cfg = json.loads((HERE / "config.json").read_text())["chosen"]
    feat = FT.load()
    feat = feat[feat.season_end <= 2018]
    res = {}
    for var in ("gs", "go"):
        c = cfg[var]
        pr, ws = SK.walk_forward(feat, var, [2018], c["lambda"], c["family"])
        y = pr.home_win.to_numpy()
        res[var] = {"n": int(len(pr)), "config": {"family": c["family"], "lambda": c["lambda"]},
                    "stack": R.logloss(pr.p_stack, y), "in_season": R.logloss(pr.p_in, y),
                    "elo": R.logloss(pr.p_elo, y), "preseason": R.logloss(pr.p_pre, y),
                    "d_stack_vs_in_season": paired(pr.p_stack, pr.p_in, y),
                    "weights": ws[2018]}
        r = res[var]
        d = r["d_stack_vs_in_season"]
        print(f"[{var}] 2018 n={r['n']}: stack {r['stack']:.5f} in-season {r['in_season']:.5f} "
              f"Elo {r['elo']:.5f}  diff {d['diff']:+.5f} [{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}]")
    out_f.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
