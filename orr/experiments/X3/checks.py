"""X3 integration checks, run BEFORE the hindcast. Tuning season 2016 only
(no test-window outcome is scored here; check 3 compares offsets, not
outcomes, for every gid).

  1. core ratings.run_filter(lineup=None) == X1's copy of the original
     run_filter (lfilter.run_filter_lineup, lineup=None)
  2. core run_filter(use_goalie=True, lineup=core offsets) == X1's
     run_filter_lineup(use_goalie=True, lineup=X1 offsets), goals+shots and
     goals-only
  3. orr.lineups.backtest_offsets() == X1 lineups.offsets(wide(10), 0.45, 0, 0.6)
  4. the live InSeasonFilter, fed 2016 day by day with gdiff_h/gdiff_a and
     lo_h/lo_a, reproduces run_filter's pregame linear predictors

Run: python3 -m orr.experiments.X3.checks  -> orr/experiments/X3/checks.json
"""
from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from orr import lineups as LU
from orr import ratings as R
from orr import structural as S
from orr.experiments.X1 import lfilter as XF
from orr.experiments.X1 import lineups as XL

HERE = Path(__file__).resolve().parent
V = 2016
COLS = ["eta_h", "eta_a", "v_h", "v_a", "c_ha", "feta_h", "feta_a", "fv_h", "fv_a", "fc_ha"]


def maxdiff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    m = a.merge(b, on="gid", suffixes=("", "_b"))
    assert len(m) == len(a) == len(b)
    return float(max(np.abs(m[c] - m[f"{c}_b"]).max() for c in COLS))


def replay_live(hp: R.HP, lo: pd.DataFrame) -> pd.DataFrame:
    """InSeasonFilter over season V with known starters and lineup offsets."""
    st = S.structural(V, hp.window, hp.h_halflife, goalie_key=hp.goalie)
    pre = R.preseason_table(V, hp.ridge, use_roster=hp.use_roster, sp=hp.pre_sp)
    g = S.game_frame()
    g = g[g.season_end == V].reset_index(drop=True)
    gt = S.goalie_game_talent(*hp.goalie)
    g = g.merge(gt[["gid", "gdiff_h", "gdiff_a"]], on="gid", how="left") \
         .merge(lo[["gid", "lo_h", "lo_a"]], on="gid", how="left")
    filt = R.InSeasonFilter(pre, {"hp": hp, "struct": st, "P": {"mu": st["mu0"], "h": st["h0"]}},
                            season_end=V, start_date=g.date.min())
    keep = ["gid", "date", "home", "away", "home_g", "away_g", "extra", "reg_h", "reg_a", "rest_h",
            "rest_a", "km_h", "km_a", "dtz_h", "dtz_a", "sh_h", "sh_a", "gdiff_h", "gdiff_a", "lo_h", "lo_a"]
    out = []
    for _, day in g[keep].groupby("date", sort=True):
        eh, ea = filt.predict_eta(day)
        out.append(pd.DataFrame({"gid": day.gid.to_numpy(), "eta_h_live": eh, "eta_a_live": ea}))
        filt.update_day(day)
    return pd.concat(out, ignore_index=True)


def main():
    t0 = time.time()
    hp = R.load_hp()
    hp_go = dataclasses.replace(hp, use_shots=False)
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "season": V}

    # 3. offsets
    core = LU.backtest_offsets()
    x1 = XL.offsets(XL.wide(10.0), 0.45, 0.0, 0.6)
    m = core.merge(x1, on="gid", suffixes=("", "_x1"), how="outer")
    res["offsets_n"] = [int(len(core)), int(len(x1)), int(len(m))]
    res["offsets_maxdiff"] = float(max(np.abs(m.lo_h - m.lo_h_x1).max(), np.abs(m.lo_a - m.lo_a_x1).max()))
    print("3 offsets", res["offsets_n"], res["offsets_maxdiff"], f"{time.time() - t0:.0f}s", flush=True)

    # 1. no lineup
    a = R.run_filter(hp, [V], first=V)
    b = XF.run_filter_lineup(hp, [V], first=V)
    res["no_lineup_maxdiff"] = maxdiff(a, b)
    print("1 no lineup", res["no_lineup_maxdiff"], flush=True)

    # 2. lineup + starters, both variants
    for tag, h in (("gs", hp), ("go", hp_go)):
        a = R.run_filter(h, [V], first=V, use_goalie=True, lineup=core)
        b = XF.run_filter_lineup(h, [V], first=V, use_goalie=True, lineup=x1)
        res[f"x1_{tag}_maxdiff"] = maxdiff(a, b)
        print(f"2 x1 {tag}", res[f"x1_{tag}_maxdiff"], flush=True)

    # 4. live filter replay
    for tag, h in (("gs", hp), ("go", hp_go)):
        a = R.run_filter(h, [V], first=V, use_goalie=True, lineup=core)
        lv = replay_live(h, core)
        m = a.merge(lv, on="gid")
        assert len(m) == len(a) == len(lv)
        res[f"live_{tag}_eta_maxdiff"] = float(max(np.abs(m.eta_h - m.eta_h_live).max(),
                                                   np.abs(m.eta_a - m.eta_a_live).max()))
        res[f"live_{tag}_eta_meanabs"] = float((np.abs(m.eta_h - m.eta_h_live).mean()
                                                + np.abs(m.eta_a - m.eta_a_live).mean()) / 2)
        print(f"4 live {tag}", res[f"live_{tag}_eta_maxdiff"], res[f"live_{tag}_eta_meanabs"], flush=True)
    res["secs"] = round(time.time() - t0, 1)
    (HERE / "checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
