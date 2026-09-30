"""Stat-sheet goal-level calibration for live NeurHL-G (PLAN_NeurHL4 A1).

  m0 = L / M   (fixed once, before the first game: `--freeze`)
       L = last season's league regulation goals per team-game from the era
           inputs (prior_gpg / 2 - (prior_ot_share + prior_so_share) / 2)
       M = the bundle's mean projected regulation goals over every game
           scheduled in the first 14 days (fallback lineups, inputs only)
  m  = (A + k L) / (P + k M),  k = 300 team-games of prior weight, in season:
       A and P are the actual and unscaled-predicted regulation goals over
       completed 2026-27 games that have a primary forecast (pregame, else
       morning). With no games m = m0; it moves toward A / P as games accrue.

m scales stat-sheet goal means only; the win probability never changes.
State: neurhl/configs/live_goal_calibration.json.

NeurHL 1.2 (PLAN_NeurHL_1_2 R3): the prior becomes m0 = L / M_full, M_full the engine's mean over
every scheduled game (configs/live_goal_calibration_1_2.json, from the 1.2 season set), and
forecasts made by earlier engines are converted to the live engine's units before pooling.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CONFIGS, NOUT  # noqa: E402

STATE = CONFIGS / "live_goal_calibration.json"
STATE_1_2 = CONFIGS / "live_goal_calibration_1_2.json"     # NeurHL 1.2 prior (PLAN_NeurHL_1_2 R3)
K = 300.0


def bundle_m() -> dict:
    """First-14-days mean projected regulation goals of each engine that made live forecasts
    (the frozen state for the live engine, configs/live_goal_calibration_<bundle>.json for earlier
    ones). Same window and lineups for every engine, so their ratios convert raw goals between
    engines (PLAN_NeurHL_1_2 R3)."""
    out = {}
    for f in sorted(CONFIGS.glob("live_goal_calibration_g*.json")) + [STATE]:
        st = json.loads(f.read_text())
        if "bundle" in st and "M" in st:
            out[st["bundle"]] = float(st["M"])
    return out


def freeze(first_days: int = 14):
    from common import refuse_if_frozen_1_0
    refuse_if_frozen_1_0(STATE.name + " (m0 and M are frozen inputs of NeurHL 1.0 and 1.1)")
    import importlib.util
    import sim.g_live as GL
    from sim.g_forecast_core import raw_outputs, sha
    from sim.project_2027 import load_schedule
    spec = importlib.util.spec_from_file_location("lineup_resolver", ROOT / "live" / "lineup_resolver.py")
    R = importlib.util.module_from_spec(spec)
    sys.modules["lineup_resolver"] = R
    spec.loader.exec_module(R)
    import tempfile
    bundle = json.loads((CONFIGS / "live_models.json").read_text())["neurhl_g"]
    sch = load_schedule()
    d0 = pd.Timestamp(sch.date.min())
    dates = sorted(d for d in sch.date.unique() if pd.Timestamp(d) < d0 + pd.Timedelta(days=first_days))
    no_df = tempfile.mkdtemp()      # an empty snapshot directory: fallback lineups only (A1)
    goals, sources = [], []
    for d in dates:
        day = sch[sch.date == d]
        lus, keep = {}, []
        for r in day.itertuples():
            lu = R.resolve(r.game_id, d, r.home, r.away, snapshot_dir=no_df, use_api=False,
                           as_of=dt.datetime.fromisoformat(f"{d}T15:00:00+00:00"))
            sources += [lu[s]["lineup_source"] for s in ("home", "away")]
            if lu["home"]["skaters"] and lu["away"]["skaters"]:
                lus[r.game_id] = {s: {"skaters": lu[s]["skaters"], "goalie": lu[s]["goalie"]}
                                  for s in ("home", "away")}
                keep.append(r.game_id)
        g = day[day.game_id.isin(keep)][["game_id", "date", "home", "away"]].reset_index(drop=True)
        if not len(g):
            continue
        A, _, _ = GL.build(g, lus)
        o, _, _ = raw_outputs(bundle, A)
        goals.append(o["goals"].reshape(-1))
        print(f"  {d}: {len(g)} games, mean projected goals {o['goals'].mean():.3f}", flush=True)
    M = float(np.mean(np.concatenate(goals)))
    era = GL.era_2027()
    L = era["prior_gpg"] / 2 - (era["prior_ot_share"] + era["prior_so_share"]) / 2
    st = {"m0": L / M, "L": L, "M": M, "k": K, "first_days": first_days,
          "n_team_games": int(sum(len(x) for x in goals)), "dates": [str(x) for x in dates],
          "lineup_sources": {k: sources.count(k) for k in sorted(set(sources))},
          "bundle": bundle,
          "bundle_sha": sha(ROOT / "checkpoints" / "g" / bundle / "bundle.json")[:16],
          "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    STATE.write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))
    return st


def current() -> float:
    """m for a forecast made now: m0 shrunk toward completed 2026-27 games. From NeurHL 1.2 the
    prior is L / M_full (STATE_1_2) and every forecast's raw goals are expressed in the live
    engine's units (M_live / M_engine) before pooling; forecasts by an engine with no recorded
    M are left out."""
    live = json.loads(STATE.read_text())
    st = json.loads(STATE_1_2.read_text()) if STATE_1_2.exists() else live
    res_p = NOUT / "live" / "results_2027.csv"
    base = NOUT / "live" / "2027"
    if not res_p.exists() or not base.exists():
        return float(st["m0"])
    res = pd.read_csv(res_p)
    if not len(res):
        return float(st["m0"])
    extra = res.last_period.isin(["OT", "SO"])
    hw = res.home_g > res.away_g
    res["reg_h"] = res.home_g - (extra & hw).astype(int)
    res["reg_a"] = res.away_g - (extra & ~hw).astype(int)
    preds = []
    for f in sorted(base.glob("*/pregame_*.csv")) + sorted(base.glob("*/morning.csv")):
        if f.name.endswith("_players.csv"):
            continue
        d = pd.read_csv(f)
        if "goals_home_raw" in d:
            if "bundle" not in d:
                d["bundle"] = live.get("bundle")
            preds.append(d[["game_id", "forecast", "goals_home_raw", "goals_away_raw", "bundle"]])
    if not preds:
        return float(st["m0"])
    p = pd.concat(preds, ignore_index=True)
    p = p.sort_values("forecast", key=lambda x: x.map({"pregame": 0, "morning": 1})) \
         .drop_duplicates("game_id")
    j = p.merge(res[["game_id", "reg_h", "reg_a"]], on="game_id")
    if STATE_1_2.exists():
        bm = bundle_m()
        conv = j.bundle.map(lambda b: bm[live["bundle"]] / bm[b] if b in bm else np.nan)
        j = j.assign(conv=conv)[conv.notna()]
    else:
        j = j.assign(conv=1.0)
    A = float(j.reg_h.sum() + j.reg_a.sum())
    P = float(((j.goals_home_raw + j.goals_away_raw) * j.conv).sum())
    return float((A + st["k"] * st["L"]) / (P + st["k"] * st["M"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    else:
        print(current())
