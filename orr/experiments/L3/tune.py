"""L3 tuning, confirmation and the single test run.

Protocol (fixed before any configuration was scored)
---------------------------------------------------
* Tuning seasons: V = 2012, 2014, 2015, 2016, 2017 (strict protocol,
  first-10-game rosters). V = 2011 lies inside the allowed window but has no
  earlier opening-roster season to train a walk-forward model on, so the model
  is unused there and it equals the current pipeline; it is scored as a
  diagnostic only. 2013 is never scored.
* Every model is walk-forward: for season V it is fitted on the same kind of
  roster in seasons V-window..V-1 (broken seasons 2013/2020/2021 skipped as
  targets), with hyperparameters fixed by the tuning below.
* Stage 1 (games model): pooled GP MAE over every rostered skater (on an
  opening roster, with or without NHL history; actual GP 0 if he did not
  play) on the tuning seasons. Coordinate search over the GBM settings
  (iterations, learning rate, leaves, min leaf, l2), the training window and
  the feature set, then a GLM (fractional logit) ridge sweep; at most 40
  configurations.
* Stage 2 (projection step): on the stage-1 winner, rookies {model, old} x
  forecast-averaging paths {orr, all}; chosen by pooled sample-'all' points
  MAE on the tuning seasons (the existing skater tuner's objective).
* Confirmation: 2018-19 once with the fixed configuration (reported; the
  configuration is not changed after it).
* Test: 2022-24 (first-10 rosters) once, baseline and L3 in the same run;
  2025-26 (season-team fallback, 'expost' games) reported separately.
* Acceptance (PLAN_1_1.md, L3): GP MAE (all rostered skaters) improves by at
  least 0.5 games AND sample-'all' points MAE improves, both pooled on
  2022-24.

Usage
  python3 -m orr.experiments.L3.tune --verify
  python3 -m orr.experiments.L3.tune --stage1
  python3 -m orr.experiments.L3.tune --stage2
  python3 -m orr.experiments.L3.tune --confirm
  python3 -m orr.experiments.L3.tune --test
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from orr import players as PL
from orr.backtest import players_bt as BT
from orr.experiments.L3 import gpmodel as G

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"
CONFIG = HERE / "config.json"
OBJ_V = [2012, 2014, 2015, 2016, 2017]
DIAG_V = [2011]
CONFIRM_V = [2018, 2019]
TEST_FIRST10 = [2022, 2023, 2024]
TEST_SEASON_TEAM = [2025, 2026]
MAX_STAGE1 = 40


def load_ledger() -> list:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def save_ledger(L: list) -> None:
    LEDGER.write_text(json.dumps(L, indent=1))


def evaluate(cfg: G.GPConfig, prm: PL.Params, seasons) -> dict:
    per = {}
    used = {}
    for V in seasons:
        dg = {}
        tot = G.run_season_l3(V, prm, cfg, diag=dg)
        per[V] = G.season_metrics(tot)
        used[V] = dg.get("model_used", False)
    return per, used


def log_config(cfg, prm, stage, L, seen):
    k = json.dumps(asdict(cfg), sort_keys=True)
    if k in seen:
        return seen[k]
    t = time.time()
    per, used = evaluate(cfg, prm, DIAG_V + OBJ_V)
    pooled = G.pool(per, OBJ_V)
    row = {"n": len(L) + 1, "stage": stage, "config": asdict(cfg),
           "obj_seasons": OBJ_V, "pooled": {k_: round(v, 4) for k_, v in pooled.items()},
           "per_season": {str(V): {k_: round(v, 4) for k_, v in m.items()} for V, m in per.items()},
           "model_used": {str(V): u for V, u in used.items()},
           "seconds": round(time.time() - t, 1)}
    L.append(row)
    save_ledger(L)
    seen[k] = pooled
    print(f"[{row['n']:>2}] {stage:<9} gp_mae {pooled['gp_mae']:.3f} bias {pooled['gp_bias']:+.2f} "
          f"p_all {pooled['p_mae_all']:.4f} A {pooled['p_mae_A']:.4f} B {pooled['p_mae_B']:.4f} "
          f"| {cfg.model} {cfg.feats} w{cfg.window} lr{cfg.lr} it{cfg.iters} lv{cfg.leaves} "
          f"ml{cfg.min_leaf} l2{cfg.l2} r{cfg.ridge} rk:{cfg.rookies} paths:{cfg.paths} "
          f"({row['seconds']}s)", flush=True)
    return pooled


def stage1(prm):
    L = load_ledger()
    seen = {json.dumps(r["config"], sort_keys=True): r["pooled"] for r in L}
    base = log_config(G.GPConfig(model="none"), prm, "baseline", L, seen)
    print("baseline", base, flush=True)
    best = G.GPConfig(model="gbm")
    cur = log_config(best, prm, "stage1", L, seen)["gp_mae"]
    grid = {"iters": [100, 400], "lr": [0.025, 0.1], "leaves": [7, 31],
            "min_leaf": [20, 80, 160], "l2": [0.0, 10.0], "window": [6, 14],
            "feats": ["nostruct"]}

    def n1():
        return sum(1 for r in L if r["stage"] == "stage1")
    for sweep in range(2):
        improved = False
        for name, vals in grid.items():
            for v in vals:
                if n1() >= MAX_STAGE1 - 4:
                    break
                cand = replace(best, **{name: v})
                sc = log_config(cand, prm, "stage1", L, seen)["gp_mae"]
                if sc < cur - 1e-4:
                    cur, best, improved = sc, cand, True
        if not improved:
            break
    # GLM family (fractional logit), same window as the best GBM
    gbest, gcur = best, cur
    for r in (0.1, 1.0, 10.0, 100.0):
        cand = G.GPConfig(model="glm", ridge=r, window=best.window, feats=best.feats)
        sc = log_config(cand, prm, "stage1", L, seen)["gp_mae"]
        if sc < gcur - 1e-4:
            gcur, gbest = sc, cand
    print("stage1 best", asdict(gbest), gcur, flush=True)
    return gbest


def best_stage(L, stage, key):
    rows = [r for r in L if r["stage"] == stage]
    r = min(rows, key=lambda r: r["pooled"][key])
    return G.GPConfig(**r["config"]), r


def stage2(prm):
    L = load_ledger()
    seen = {json.dumps(r["config"], sort_keys=True): r["pooled"] for r in L}
    s1, row = best_stage(L, "stage1", "gp_mae")
    print("stage1 winner", asdict(s1), row["pooled"], flush=True)
    for rk in ("model", "old"):
        for paths in ("orr", "all"):
            cfg = replace(s1, rookies=rk, paths=paths)
            # the stage-1 winner itself is re-used from the ledger
            log_config(cfg, prm, "stage2" if cfg != s1 else "stage1", L, seen)
    cands = [r for r in L if r["stage"] == "stage2" or
             json.dumps(r["config"], sort_keys=True) == json.dumps(asdict(s1), sort_keys=True)]
    win = min(cands, key=lambda r: r["pooled"]["p_mae_all"])
    cfg = G.GPConfig(**win["config"])
    base = next(r for r in L if r["stage"] == "baseline")
    CONFIG.write_text(json.dumps({"config": asdict(cfg), "ledger_row": win["n"],
                                  "tune_pooled": win["pooled"],
                                  "baseline_tune_pooled": base["pooled"],
                                  "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print("FIXED", asdict(cfg), win["pooled"], "baseline", base["pooled"], flush=True)
    return cfg


def paired_mae(err_new, err_base, nb=2000, seed=0):
    """Mean |e_new| - |e_base| with a paired bootstrap 95% CI (players)."""
    d = np.abs(err_new) - np.abs(err_base)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (nb, len(d)))
    bs = d[idx].mean(1)
    return {"diff": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)),
                                              float(np.quantile(bs, 0.975))],
            "n": int(len(d))}


def run_window(cfg, prm, seasons, tag):
    out = {"seasons": {}, "config": asdict(cfg)}
    gp_n, gp_b, p_n, p_b = [], [], [], []
    for V in seasons:
        t = time.time()
        dg = {}
        tb = G.run_season_l3(V, prm, G.GPConfig(model="none"))
        tn = G.run_season_l3(V, prm, cfg, diag=dg)
        mb, mn = G.season_metrics(tb), G.season_metrics(tn)
        # paired rows
        rb, rn = G.gp_rows(tb).set_index("player_id"), G.gp_rows(tn).set_index("player_id")
        assert set(rb.index) == set(rn.index)
        rn = rn.loc[rb.index]
        gp_b.append((rb.gp - rb.act_gp0).to_numpy())
        gp_n.append((rn.gp - rn.act_gp0).to_numpy())
        sb, _ = BT.samples(tb)
        sn, _ = BT.samples(tn)
        ab = tb[sb["all"]].set_index("player_id")
        an = tn[sn["all"]].set_index("player_id").loc[ab.index]
        p_b.append((ab.p - ab.act_p).to_numpy())
        p_n.append((an.p - an.act_p).to_numpy())
        out["seasons"][V] = {"baseline": mb, "l3": mn, "model_used": dg.get("model_used"),
                             "roster_proxy": tn.roster_proxy.iloc[0],
                             "d_gp_mae": mn["gp_mae"] - mb["gp_mae"],
                             "d_p_mae_all": mn["p_mae_all"] - mb["p_mae_all"],
                             "d_p_mae_A": mn["p_mae_A"] - mb["p_mae_A"],
                             "d_p_mae_B": mn["p_mae_B"] - mb["p_mae_B"]}
        print(f"{tag} {V}: GP MAE {mb['gp_mae']:.3f} -> {mn['gp_mae']:.3f}  "
              f"P(all) {mb['p_mae_all']:.4f} -> {mn['p_mae_all']:.4f}  "
              f"A {mb['p_mae_A']:.4f} -> {mn['p_mae_A']:.4f}  "
              f"B {mb['p_mae_B']:.4f} -> {mn['p_mae_B']:.4f}  ({time.time() - t:.0f}s)",
              flush=True)
    per_b = {V: out["seasons"][V]["baseline"] for V in seasons}
    per_n = {V: out["seasons"][V]["l3"] for V in seasons}
    out["pooled"] = {"baseline": G.pool(per_b, seasons), "l3": G.pool(per_n, seasons)}
    out["paired"] = {"gp_mae": paired_mae(np.concatenate(gp_n), np.concatenate(gp_b)),
                     "p_mae_all": paired_mae(np.concatenate(p_n), np.concatenate(p_b))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    prm = PL.load_params()
    if a.verify:
        for V in DIAG_V + OBJ_V + CONFIRM_V:
            x = BT.run_season(V, prm, "strict").sort_values("player_id").reset_index(drop=True)
            y = G.run_season_l3(V, prm, G.GPConfig(model="none")).sort_values(
                "player_id").reset_index(drop=True)
            print(V, len(x) == len(y), float(np.abs(x.p - y.p).max()),
                  float(np.abs(x.gp - y.gp).max()), flush=True)
    if a.stage1:
        stage1(prm)
    if a.stage2:
        stage2(prm)
    if a.confirm:
        cfg = G.GPConfig(**json.loads(CONFIG.read_text())["config"])
        out = run_window(cfg, prm, CONFIRM_V, "confirm")
        (HERE / "confirm.json").write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps(out["pooled"], indent=1), json.dumps(out["paired"], indent=1))
    if a.test:
        dst = HERE / "test.json"
        if dst.exists():
            raise SystemExit("test.json exists: the test window is run once")
        cfg = G.GPConfig(**json.loads(CONFIG.read_text())["config"])
        out = {"first10_2022_24": run_window(cfg, prm, TEST_FIRST10, "test"),
               "season_team_2025_26": run_window(cfg, prm, TEST_SEASON_TEAM, "test-st"),
               "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        dst.write_text(json.dumps(out, indent=1, default=str))
        for k in ("first10_2022_24", "season_team_2025_26"):
            print(k, json.dumps(out[k]["pooled"], indent=1), json.dumps(out[k]["paired"], indent=1))


if __name__ == "__main__":
    main()
