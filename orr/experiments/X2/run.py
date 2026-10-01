"""X2 harness: strict skater protocol with the comparables blend.

`season_base` + `season_eval` together are a copy of
orr.backtest.players_bt.run_season (strict protocol, sims=0) and
orr.players.pipeline, split at the point where per-60 rates enter: the
deployment (games, ice time) does not read rates, so it is computed once per
season and every configuration of the comparables blend only re-does
rates -> totals -> forecast averaging. With w = 0 the output equals
players_bt.run_season exactly (checked by --verify).

Usage
  python3 -m orr.experiments.X2.run --verify      # w=0 reproduces players_bt (tune + confirm seasons)
  python3 -m orr.experiments.X2.run --tune s1_own_baseline   # coordinate search -> ledger.json
  python3 -m orr.experiments.X2.run --tune s2_orr_baseline   # second search (ORR-baseline changes)
  python3 -m orr.experiments.X2.run --select      # fix the best (w > 0) over both -> config.json
  python3 -m orr.experiments.X2.run --confirm     # 2018-19, once, with config.json
  python3 -m orr.experiments.X2.run --test        # 2022-26, ONCE, with config.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from orr import data as D
from orr import deploy as DP
from orr import players as PL
from orr.backtest import players_bt as B
from orr.experiments.X2 import comps as CX

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"
CONFIG = HERE / "config.json"
CONFIRM = HERE / "confirm.json"
TEST = HERE / "test.json"

GAMES = B.GAMES
# Objective seasons (pre-declared before any run): V = 2012, 2014-2017.
# V = 2011 is inside the allowed window but its comparables pool is a single
# transition (2009 -> 2010) with no lagged seasons, unlike the test pools of
# 12-16 transitions; it is scored and logged as a diagnostic only.
OBJ_V = [2012, 2014, 2015, 2016, 2017]
DIAG_V = [2011]
CONFIRM_V = B.CONFIRM_V
TEST_V = B.TEST_V
W_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


# ---------------------------------------------------------------------------
# Copy of players_bt.run_season / PL.pipeline, split at the rates
# ---------------------------------------------------------------------------
_BASE: dict = {}


def season_base(V: int, prm: PL.Params) -> dict:
    """Everything of the strict run that does not depend on rates."""
    if V in _BASE:
        return _BASE[V]
    act = B.actuals(V)
    roster, flag = DP.historical_roster(V, "opening")
    P = PL.panel()
    hist_ids = set(P.ids[PL._hist_gp(P, V) > 0])
    ids = set(roster.player_id) | (set(act.player_id) & hist_ids)
    kind = "opening" if flag == "first10" else "expost"
    games = GAMES
    # --- PL.pipeline (sims=0, stack=True, neurhl_budget=False, no games_out)
    proj = PL.calibrate_thin_rates(PL.project(V, prm, ids=ids, extra=None), prm)
    ros = roster[roster.player_id.isin(proj.player_id)]
    gmap = DP.league_gp_map(V, prm) if kind == "expost" else None
    cover = DP.coverage(V, "opening") if kind == "opening" else None
    dep = DP.deploy(proj, ros, V, games, prm.dress_noise, prm.q_scale, gp_map=gmap, cover=cover)
    gs = DP.apply_stack(dep, V, prm, kind, games) if B.STACK_GP else \
        dep.set_index("player_id").gp.to_dict()
    grp = dict(zip(proj.player_id, PL.thin_group(proj)))
    gm = PL.group_mults(prm, "gp")
    cap = dict(zip(dep.player_id, games - dep.games_out))
    gs = {k: min(v * gm[grp.get(k, "est")], cap.get(k, games)) for k, v in gs.items()}
    dep = DP.deploy(proj, ros, V, games, prm.dress_noise, prm.q_scale, gp_override=gs, cover=cover)
    out = {"V": V, "act": act, "flag": flag, "proj": proj, "dep": dep,
           "reserve_gp": B.reserve_gp(V)}
    _BASE[V] = out
    return out


def season_eval(base: dict, prm: PL.Params, comps: pd.DataFrame | None, w: float) -> pd.DataFrame:
    """Rates (ORR blended with comparables) -> totals -> forecast averaging,
    reserves, actuals: the rest of PL.pipeline and players_bt.run_season."""
    V, games = base["V"], GAMES
    proj = CX.blend_proj(base["proj"], V, comps, w)
    tot = DP.add_totals(base["dep"], proj)
    tot = PL.blend_paths(tot, V, prm, games)
    thin = tot.player_id.map(dict(zip(proj.player_id, PL.thin_group(proj)))) == "thin"
    if prm.thin_rate_mult != 1.0 and thin.any():
        tot = tot.copy()
        cols = [c for c in tot.columns if c in ("g", "a", "p", "a1", "a2", "ppg", "ppa")
                or c.startswith(("g_p", "a_p", "p_p")) or c in ("g_sd", "a_sd", "p_sd")]
        tot.loc[thin, cols] = tot.loc[thin, cols] * prm.thin_rate_mult
    tot = tot.copy()
    # --- players_bt.run_season (strict) from here
    tot["on_roster"] = True
    res = proj[~proj.player_id.isin(tot.player_id)].copy()
    if len(res):
        gp = base["reserve_gp"]
        res["gp"] = gp
        for k in PL.SITS:
            res[f"toi_{k}"] = gp * res[f"tpg_{k}"]
        rt = DP.add_totals(res[["player_id", "gp", *[f"toi_{k}" for k in PL.SITS]]], proj)
        rt["on_roster"] = False
        rt["team"] = None
        tot = pd.concat([tot, rt], ignore_index=True)
    tot = tot.drop(columns=["has_hist", "name", "age"], errors="ignore")
    tot = tot.merge(proj[["player_id", "has_hist", "name", "age", "pos"]]
                    .rename(columns={"pos": "pos_p"}), on="player_id", how="left")
    tot = tot.merge(base["act"], on="player_id", how="left").copy()
    tot["roster_proxy"] = base["flag"]
    tot["V"] = V
    return tot


def run_season_x2(V: int, prm: PL.Params, cp: CX.CompParams | None, w: float) -> pd.DataFrame:
    base = season_base(V, prm)
    comps = CX.project_comps(V, cp) if (cp is not None and w > 0) else None
    return season_eval(base, prm, comps, w)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def season_metrics(tot: pd.DataFrame) -> dict:
    m = B.metrics(tot)
    return {s: {"n": m[s]["n"], "mae": m[s]["mae"], "bias": m[s]["bias"],
                "mae_g": m[s]["mae_g"], "mae_a": m[s]["mae_a"]} for s in ("A", "B", "all")}


def pooled(per: dict, seasons, sample: str, key: str = "mae") -> float:
    n = sum(per[V][sample]["n"] for V in seasons)
    return sum(per[V][sample][key] * per[V][sample]["n"] for V in seasons) / n


def score(prm, cp, w, seasons, comps_cache=None) -> dict:
    per = {}
    for V in seasons:
        base = season_base(V, prm)
        comps = None
        if cp is not None and w > 0:
            comps = comps_cache[V] if comps_cache is not None else CX.project_comps(V, cp)
        per[V] = season_metrics(season_eval(base, prm, comps, w))
    return per


def summarise(per: dict, seasons) -> dict:
    a, b = pooled(per, seasons, "A"), pooled(per, seasons, "B")
    return {"A": round(a, 5), "B": round(b, 5), "obj": round((a + b) / 2, 5),
            "all": round(pooled(per, seasons, "all"), 5),
            "per_season": {int(V): {"A": round(per[V]["A"]["mae"], 4), "B": round(per[V]["B"]["mae"], 4),
                                    "all": round(per[V]["all"]["mae"], 4)} for V in seasons}}


# ---------------------------------------------------------------------------
# Verify: w = 0 reproduces players_bt.json
# ---------------------------------------------------------------------------
def verify(prm):
    ref = json.loads((B.OUT / "players_bt.json").read_text())["seasons"]
    for V in B.TUNE_V + CONFIRM_V:
        tot = run_season_x2(V, prm, None, 0.0)
        m = season_metrics(tot)
        r = ref[str(V)]["strict"]
        print(V, {s: (round(m[s]["mae"], 4), round(r[s]["mae"], 4), m[s]["n"], r[s]["n"])
                  for s in ("A", "B", "all")}, flush=True)
        for s in ("A", "B", "all"):
            assert abs(m[s]["mae"] - r[s]["mae"]) < 1e-6 and m[s]["n"] == r[s]["n"], (V, s)
    # the fast path equals a full re-run of the copied pipeline with a hook
    print("verify: w=0 reproduces players_bt.json strict A/B/all on tune+confirm seasons")


# ---------------------------------------------------------------------------
# Tuning (V <= 2017 only)
# ---------------------------------------------------------------------------
GRID = {
    "k": [25, 50, 100, 200, 400],
    "w_age": [0.5, 1.0, 2.0, 4.0],
    "w_rate": [0.5, 1.0, 2.0],
    "w_shot": [0.0, 0.5, 1.0],
    "w_use": [0.0, 0.5, 1.0, 2.0],
    "w_exp": [0.0, 0.5, 1.0],
}
START = CX.CompParams(k=100, w_age=2.0, w_rate=1.0, w_shot=0.5, w_use=1.0, w_exp=0.5)
MAX_SWEEPS = 2


SEARCHES = {
    "s1_own_baseline": CX.CompParams(k=100, w_age=2.0, w_rate=1.0, w_shot=0.5, w_use=1.0,
                                     w_exp=0.5, baseline="own"),
    # second search (added after s1, still tuning window only): comparables'
    # changes measured from ORR's own walk-forward projection of them
    "s2_orr_baseline": CX.CompParams(k=200, w_age=2.0, w_rate=1.0, w_shot=0.5, w_use=1.0,
                                     w_exp=0.5, baseline="orr"),
}


def tune(prm, search: str):
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    t0 = time.time()
    seasons = OBJ_V + DIAG_V
    if not ledger:
        base_per = score(prm, None, 0.0, seasons)
        ledger.append({"n": 0, "search": "baseline_w0", "comp_params": None, "w": 0.0,
                       "objective": "mean of pooled strict A and B MAE, V=2012,2014-17",
                       "tune": summarise(base_per, OBJ_V), "diag_2011": summarise(base_per, DIAG_V)})
    base_sum = ledger[0]["tune"]
    print(f"baseline (w=0): A {base_sum['A']:.4f} B {base_sum['B']:.4f} obj {base_sum['obj']:.4f}", flush=True)
    seen = {}

    def ev(cp):
        """Best blend weight for comparables config cp; every (cp, w) logged."""
        if cp.key() in seen:
            return seen[cp.key()]
        cc = {V: CX.project_comps(V, cp) for V in seasons}
        best = None
        for w in W_GRID:
            per = score(prm, cp, w, seasons, comps_cache=cc)
            sm = summarise(per, OBJ_V)
            ledger.append({"n": len(ledger), "search": search, "comp_params": asdict(cp), "w": w,
                           "tune": sm, "diag_2011": summarise(per, DIAG_V)})
            if best is None or sm["obj"] < best[0]:
                best = (sm["obj"], w, sm)
        seen[cp.key()] = best
        print(f"  [{len(seen):>2}] {asdict(cp)} -> best w {best[1]}: obj {best[0]:.4f} "
              f"(A {best[2]['A']:.4f} B {best[2]['B']:.4f}) [{time.time() - t0:.0f}s]", flush=True)
        LEDGER.write_text(json.dumps(ledger, indent=1))
        return best

    cur_cp = SEARCHES[search]
    cur = ev(cur_cp)
    for sweep in range(MAX_SWEEPS):
        improved = False
        for name, vals in GRID.items():
            for v in vals:
                cand = replace(cur_cp, **{name: v})
                b = ev(cand)
                if b[0] < cur[0] - 1e-4:
                    cur, cur_cp, improved = b, cand, True
        if not improved:
            break
    LEDGER.write_text(json.dumps(ledger, indent=1))
    print("best of", search, asdict(cur_cp), "w", cur[1], cur[2]["obj"], "baseline", base_sum["obj"])


def select():
    """Fix the configuration: the lowest tuning objective over every logged
    (comparables config, w > 0) of every search. Written to config.json."""
    ledger = json.loads(LEDGER.read_text())
    base = ledger[0]["tune"]
    cand = [r for r in ledger if r["comp_params"] is not None and r["w"] > 0]
    best = min(cand, key=lambda r: r["tune"]["obj"])
    by_search = {}
    for r in cand:
        b = by_search.get(r["search"])
        if b is None or r["tune"]["obj"] < b["tune"]["obj"]:
            by_search[r["search"]] = r
    cfg = {"comp_params": best["comp_params"], "w": best["w"], "search": best["search"],
           "ledger_n": best["n"], "tune": best["tune"], "baseline_tune": base,
           "best_by_search": {k: {"comp_params": v["comp_params"], "w": v["w"], "obj": v["tune"]["obj"],
                                  "A": v["tune"]["A"], "B": v["tune"]["B"]} for k, v in by_search.items()},
           "n_configurations": len(cand),
           "n_comp_configs": len({json.dumps(r["comp_params"], sort_keys=True) for r in cand}),
           "fixed": {"LAG_W": CX.LAG_W, "K_FEAT_MIN": CX.K_FEAT_MIN, "K_BASE_MIN": CX.K_BASE_MIN,
                     "K_OUT_MIN": CX.K_OUT_MIN, "MIN_TOI_HIST": CX.MIN_TOI_HIST,
                     "MIN_TOI_OUT": CX.MIN_TOI_OUT, "MASK_TOI": CX.MASK_TOI, "Y_CLIP": CX.Y_CLIP,
                     "F_CLIP": CX.F_CLIP, "ORR_MIN_OUT_SEASON": CX.ORR_MIN_OUT_SEASON,
                     "W_GRID": W_GRID},
           "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    CONFIG.write_text(json.dumps(cfg, indent=1))
    print(json.dumps({k: cfg[k] for k in ("comp_params", "w", "search", "tune", "baseline_tune",
                                          "best_by_search", "n_configurations")}, indent=1))
    return cfg


def load_cfg():
    c = json.loads(CONFIG.read_text())
    return CX.CompParams(**c["comp_params"]), float(c["w"])


# ---------------------------------------------------------------------------
# Confirm (2018-19) and test (2022-26)
# ---------------------------------------------------------------------------
def boot_ci(e0: np.ndarray, e1: np.ndarray, n_boot: int = 2000, seed: int = 7) -> list:
    """Paired bootstrap 95% CI of mean(|err1|) - mean(|err0|) over players."""
    rng = np.random.default_rng(seed)
    d = np.abs(e1) - np.abs(e0)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def evaluate(prm, cp, w, seasons, label):
    rows = {}
    per0, per1 = {}, {}
    errs = {s: ([], []) for s in ("A", "B", "all")}
    for V in seasons:
        t0 = run_season_x2(V, prm, None, 0.0)
        t1 = run_season_x2(V, prm, cp, w)
        per0[V], per1[V] = season_metrics(t0), season_metrics(t1)
        m0, sched = B.samples(t0)
        m1, _ = B.samples(t1)
        for s in ("A", "B", "all"):
            assert (m0[s] == m1[s]).all()
            f = 82.0 / sched if s == "B" else 1.0
            d0, d1 = t0[m0[s]], t1[m1[s]]
            assert (d0.player_id.to_numpy() == d1.player_id.to_numpy()).all()
            errs[s][0].append((d0.p - d0.act_p * f).to_numpy())
            errs[s][1].append((d1.p - d1.act_p * f).to_numpy())
        rows[int(V)] = {
            "orr": {s: round(per0[V][s]["mae"], 4) for s in ("A", "B", "all")},
            "x2": {s: round(per1[V][s]["mae"], 4) for s in ("A", "B", "all")},
            "n": {s: per0[V][s]["n"] for s in ("A", "B", "all")},
            "neurhl_A": B.NEURHL_A.get(V), "neurhl_B_blend": (B.NEURHL_V2.get(V) or [None] * 3)[2]}
        print(f"{label} {V}: A {per0[V]['A']['mae']:.3f} -> {per1[V]['A']['mae']:.3f}  "
              f"B {per0[V]['B']['mae']:.3f} -> {per1[V]['B']['mae']:.3f}  "
              f"all {per0[V]['all']['mae']:.3f} -> {per1[V]['all']['mae']:.3f}", flush=True)
    out = {"seasons": rows, "pooled": {}}
    for s in ("A", "B", "all"):
        e0, e1 = np.concatenate(errs[s][0]), np.concatenate(errs[s][1])
        out["pooled"][s] = {"n": int(len(e0)), "orr": float(np.abs(e0).mean()),
                            "x2": float(np.abs(e1).mean()),
                            "diff": float(np.abs(e1).mean() - np.abs(e0).mean()),
                            "ci95": boot_ci(e0, e1),
                            "seasons_improved": int(sum(rows[int(V)]["x2"][s] < rows[int(V)]["orr"][s]
                                                        for V in seasons))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tune", choices=list(SEARCHES))
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    prm = PL.load_params()
    if a.verify:
        verify(prm)
    if a.tune:
        tune(prm, a.tune)
    if a.select:
        select()
    if a.confirm:
        cp, w = load_cfg()
        res = evaluate(prm, cp, w, CONFIRM_V, "confirm")
        res.update({"comp_params": asdict(cp), "w": w, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        CONFIRM.write_text(json.dumps(res, indent=1))
        print(json.dumps(res["pooled"], indent=1))
    if a.test:
        if TEST.exists():
            raise SystemExit(f"{TEST} exists: the test window is run once")
        cp, w = load_cfg()
        res = evaluate(prm, cp, w, TEST_V, "test")
        res.update({"comp_params": asdict(cp), "w": w, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        TEST.write_text(json.dumps(res, indent=1))
        print(json.dumps(res["pooled"], indent=1))


if __name__ == "__main__":
    main()
