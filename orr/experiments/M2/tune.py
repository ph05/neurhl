"""M2 step 3: choose the regularisation on 2012-2017 only.

For each variant (gs, go) every configuration of prereg.json's grid is scored
by the pooled log loss of the walk-forward stack over the games of 2014,
2015, 2016 and 2017 (team-history features; the stack for V fitted on
2012 .. V-1, 2013 excluded). Every configuration is logged to ledger.json.
The best one per variant (ties within 1e-6 to the more regularised) is
written to config.json. No season after 2017 is read.

Run: python3 -m orr.experiments.M2.tune
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from orr import ratings as R
from orr.experiments.M2 import features as FT
from orr.experiments.M2 import stack as SK

HERE = Path(__file__).resolve().parent
TUNE = [2014, 2015, 2016, 2017]


def grid() -> list[tuple[int, str, float]]:
    """(stage, family, lambda). Stage 1 as pre-registered; stage 2 =
    prereg.json amendment_1 (added after stage 1, tuning window only)."""
    lams = [float(l) for l in json.loads((HERE / "prereg.json").read_text())
            ["regularisation_grid"]["lambda"]]
    out = [(1, "to_base", 0.0)]
    for fam in ("to_base", "to_zero"):
        out += [(1, fam, l) for l in lams if l > 0]
    out += [(2, "to_base_noint", l) for l in lams]
    out += [(2, "to_base_int", l) for l in lams if l > 0]
    return out


def main():
    feat = FT.load()
    feat = feat[feat.season_end <= 2017]          # nothing after the tuning window
    assert feat.season_end.max() <= 2017
    ledger, chosen = [], {}
    for var in ("gs", "go"):
        tune_games = feat[(feat.variant == var) & (feat.src == "ht") & feat.season_end.isin(TUNE)]
        y = tune_games.home_win.to_numpy()
        base = R.logloss(tune_games.p_in, y)
        elo = R.logloss(tune_games.p_elo, y)
        pre = R.logloss(tune_games.p_pre, y)
        print(f"[{var}] 2014-17 n={len(y)}: in-season alone {base:.5f}  preseason {pre:.5f}  Elo {elo:.5f}")
        rows = []
        for stage, fam, lam in grid():
            pr, ws = SK.walk_forward(feat, var, TUNE, lam, fam)
            ll = R.logloss(pr.p_stack, pr.home_win)
            by = {int(V): R.logloss(x.p_stack, x.home_win) for V, x in pr.groupby("season_end")}
            byb = {int(V): R.logloss(x.p_in, x.home_win) for V, x in pr.groupby("season_end")}
            row = {"variant": var, "stage": stage, "family": fam, "lambda": lam, "tune_ll": ll,
                   "baseline_in_season_ll": base, "gain": ll - base,
                   "by_season": by, "baseline_by_season": byb, "weights": ws}
            rows.append(row)
            ledger.append(row)
            w = ws[2017]
            print(f"  s{stage} {fam:>13} lam={lam:<7g} ll {ll:.5f} ({ll - base:+.5f})  "
                  f"w2017 a={w['a']:+.3f} in={w['b_in']:.3f} pre={w['b_pre']:+.3f} elo={w['b_elo']:+.3f}")
        best = min(rows, key=lambda r: (round(r["tune_ll"], 6), -r["lambda"], r["stage"]))
        chosen[var] = {"stage": best["stage"], "family": best["family"], "lambda": best["lambda"],
                       "tune_ll": best["tune_ll"], "baseline_in_season_ll": base,
                       "gain": best["gain"], "elo_ll": elo, "preseason_ll": pre,
                       "weights_by_season": best["weights"]}
        print(f"  -> chosen {best['family']} lam={best['lambda']} ll {best['tune_ll']:.5f}")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (HERE / "ledger.json").write_text(json.dumps(
        {"created_utc": stamp, "tuning_seasons": TUNE, "n_configs": len(ledger),
         "note": "stage 1 = the pre-registered grid (its first run is kept verbatim in "
                 "ledger_stage1.json); stage 2 = prereg.json amendment_1, added after stage 1",
         "configs": ledger}, indent=1, default=float))
    (HERE / "config.json").write_text(json.dumps(
        {"fixed_utc": stamp, "tuning_seasons": TUNE, "chosen": chosen}, indent=1, default=float))


if __name__ == "__main__":
    main()
