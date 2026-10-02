"""X1 tuning on 2012, 2014-2017 (2013 excluded), every configuration logged.

Metric: pooled home-win log loss of the in-season filter (goals + shots,
frozen ratings_hp.json, team-history start: the only start that exists
before 2019) on the regular-season games of 2012, 2014, 2015, 2016, 2017.
Coordinate search (pre-declared in ledger.json["protocol"] before any run):

  S0  baselines: no lineups; known starters only (goalie layer)
  S1  on-ice xG value (v_xgf, v_xga), expected-lineup half-life 40 games,
      offset on goals only, known starters on: beta_x in a grid
  S2  half-life of the expected lineup (dress-probability memory)
  S3  shot_mult (offset also on the shots rows)
  S4  special-teams on-ice value (beta_st = beta_x)
  S5  ice-time index (beta_t)
  S6  beta_x re-grid at the chosen settings
A component found in S3-S5 is kept only if it lowers the pooled tuning log
loss by at least 0.0001. 2018 is then scored once to confirm.

Run: python3 -m orr.experiments.X1.tune [--confirm]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from orr import ratings as R
from orr.experiments.X1 import lfilter as LF
from orr.experiments.X1 import lineups as LU

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"
TUNE = [2012, 2014, 2015, 2016, 2017]
MIN_GAIN = 1e-4

PROTOCOL = {
    "item": "X1 lineup-aware game forecasts (orr/PLAN_1_1.md)",
    "declared_utc": None,
    "tuning_seasons": TUNE, "excluded": [2013], "confirm": [2018],
    "metric": "pooled home-win log loss, in-season filter, goals+shots, team-history start",
    "search": __doc__,
    "test_primary": (
        "Gate games 2019-24 (NeurHL g_gate_games.csv, n=6289). Model: the shipped in-season loop "
        "(preseason pipeline start as orr/backtest/inseason_bt.py, goals+shots) with the lineup "
        "adjustment = skater lineup offsets (fixed config) + known starting goalies through the "
        "existing goalie layer (use_goalie). Baseline 'ORR without lineups': the same loop with no "
        "starters and no skater offsets (0.6613 reference). Paired bootstrap 95% CI "
        "(games_bt.paired). Accept iff the CI of (X1 - baseline) lies entirely below 0. Games with "
        "no box score (2020 partial, 2024 after 2023-11-07) fall back to no lineup."),
    "test_secondary_reported_not_for_acceptance": [
        "skater increment: X1 vs known-starters-only loop (same games)",
        "skaters only (no goalie layer) vs no lineups",
        "goals-only variant (use_shots=False) of both arms",
        "team-history start for both arms",
        "games where both lineups are known",
        "NeurHL restatement games 2018-24 vs NeurHL-H (team-history start: no market prior for 2018/2021)"],
}


def _load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    p = dict(PROTOCOL)
    p["declared_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {"protocol": p, "configs": []}


def _save(L: dict):
    LEDGER.write_text(json.dumps(L, indent=1, default=float))


def score(cfg: dict, seasons=TUNE, hp=None) -> dict:
    hp = hp or R.load_hp()
    if cfg.get("goals_only"):
        import dataclasses
        hp = dataclasses.replace(hp, use_shots=False)
    lineup = None
    if cfg.get("beta_x", 0) or cfg.get("beta_st", 0) or cfg.get("beta_t", 0):
        w = LU.wide(cfg.get("halflife", 40.0))
        lineup = LU.offsets(w, cfg.get("beta_x", 0.0), cfg.get("beta_st", 0.0), cfg.get("beta_t", 0.0))
    m = LF.probs(hp, range(min(seasons), max(seasons) + 1), use_goalie=cfg.get("goalie", True),
                 lineup=lineup, shot_mult=cfg.get("shot_mult", 0.0))
    m = m[m.season_end.isin(seasons)]
    out = {"ll": R.logloss(m.p_home_win, m.home_win), "n": int(len(m)),
           "by_season": {int(V): R.logloss(x.p_home_win, x.home_win) for V, x in m.groupby("season_end")}}
    out["_p"] = m
    return out


def run(L: dict, stage: str, cfg: dict, seasons=TUNE) -> float:
    for c in L["configs"]:
        if c["config"] == cfg and c["seasons"] == list(seasons):
            return c["ll"]
    t0 = time.time()
    r = score(cfg, seasons)
    rec = {"id": len(L["configs"]) + 1, "stage": stage, "config": cfg, "seasons": list(seasons),
           "ll": r["ll"], "n": r["n"], "by_season": r["by_season"],
           "secs": round(time.time() - t0, 1)}
    L["configs"].append(rec)
    _save(L)
    print(f"[{rec['id']:>3}] {stage:<4} {json.dumps(cfg):<110} ll={r['ll']:.5f}", flush=True)
    return r["ll"]


def tune():
    L = _load()
    _save(L)
    base = run(L, "S0", {"goalie": False})
    gk = run(L, "S0", {"goalie": True})
    cur = {"goalie": True, "beta_x": 0.0, "beta_st": 0.0, "beta_t": 0.0,
           "halflife": 40.0, "shot_mult": 0.0}
    # S1
    best = (gk, 0.0)
    for b in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        ll = run(L, "S1", {**cur, "beta_x": b})
        best = min(best, (ll, b))
    cur["beta_x"] = best[1]
    cur_ll = best[0]
    # S2
    for hl in (10.0, 20.0, 80.0, float(1e6)):
        ll = run(L, "S2", {**cur, "halflife": hl})
        if ll < cur_ll:
            cur_ll, cur["halflife"] = ll, hl
    # S3-S5: kept only with a gain of at least MIN_GAIN
    for stage, key, grid in (("S3", "shot_mult", (0.5, 1.0)),
                             ("S4", "beta_st", ("tie",)),
                             ("S5", "beta_t", (0.1, 0.2, 0.4))):
        cand = None
        for v in grid:
            val = cur["beta_x"] if v == "tie" else v
            ll = run(L, stage, {**cur, key: val})
            if ll < cur_ll - MIN_GAIN and (cand is None or ll < cand[0]):
                cand = (ll, val)
        if cand:
            cur_ll, cur[key] = cand
    # S6
    b0 = cur["beta_x"]
    for b in sorted({round(b0 * f, 4) for f in (0.6, 0.8, 1.2, 1.4)} - {b0}):
        ll = run(L, "S6", {**cur, "beta_x": b})
        if ll < cur_ll:
            cur_ll, cur["beta_x"] = ll, b
    L["chosen"] = {"config": cur, "tuning_ll": cur_ll, "no_lineup_ll": base,
                   "starters_only_ll": gk}
    # diagnostics on tuning seasons (not used for selection)
    L["chosen"]["skaters_only_ll"] = run(L, "D", {**cur, "goalie": False})
    _save(L)
    print("chosen", cur, cur_ll, "base", base, "starters", gk)


def extend():
    """S7/S8 (amendment, tuning window only): the S2 and S5 optima sat on a
    grid edge (half-life 10 = smallest tried, beta_t 0.4 = largest tried).
    S7 steps each edge parameter outward (beta_t x1.5 per step, half-life
    halved per step) until the pooled tuning loss stops improving; S8
    re-grids beta_x at x0.75 / x1.25 of the current value."""
    L = _load()
    L["protocol"].setdefault("amendments", []).append({
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text": extend.__doc__})
    _save(L)
    cur = dict(L["chosen"]["config"])
    cur_ll = L["chosen"]["tuning_ll"]
    for _ in range(4):                       # beta_t outward
        cand = round(cur["beta_t"] * 1.5, 4)
        ll = run(L, "S7", {**cur, "beta_t": cand})
        if ll < cur_ll:
            cur_ll, cur["beta_t"] = ll, cand
        else:
            break
    for _ in range(3):                       # half-life outward
        cand = cur["halflife"] / 2
        ll = run(L, "S7", {**cur, "halflife": cand})
        if ll < cur_ll:
            cur_ll, cur["halflife"] = ll, cand
        else:
            break
    b0 = cur["beta_x"]
    for b in (round(b0 * 0.75, 4), round(b0 * 1.25, 4)):
        ll = run(L, "S8", {**cur, "beta_x": b})
        if ll < cur_ll:
            cur_ll, cur["beta_x"] = ll, b
    L["chosen"].update({"config": cur, "tuning_ll": cur_ll})
    L["chosen"]["skaters_only_ll"] = run(L, "D", {**cur, "goalie": False})
    _save(L)
    print("chosen", cur, cur_ll)


def confirm():
    """Score the chosen configuration once on 2018 (confirmation only)."""
    L = _load()
    cur = L["chosen"]["config"]
    res = {}
    for tag, cfg in (("no_lineups", {"goalie": False}), ("starters_only", {"goalie": True}),
                     ("x1", cur)):
        r = score(cfg, [2018])
        res[tag] = {"ll": r["ll"], "n": r["n"]}
        res[f"_p_{tag}"] = r["_p"]
    from orr.backtest.games_bt import paired
    y = res["_p_x1"].home_win.to_numpy()
    m = res["_p_x1"][["gid", "p_home_win"]].merge(
        res["_p_no_lineups"][["gid", "p_home_win"]], on="gid", suffixes=("", "_b")).merge(
        res["_p_starters_only"][["gid", "p_home_win"]].rename(columns={"p_home_win": "p_s"}), on="gid")
    m = m.merge(res["_p_x1"][["gid", "home_win"]], on="gid")
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    out["d_x1_vs_no_lineups"] = paired(m.p_home_win, m.p_home_win_b, m.home_win)
    out["d_x1_vs_starters_only"] = paired(m.p_home_win, m.p_s, m.home_win)
    L["confirm_2018"] = out
    _save(L)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--extend", action="store_true")
    a = ap.parse_args()
    confirm() if a.confirm else extend() if a.extend else tune()
