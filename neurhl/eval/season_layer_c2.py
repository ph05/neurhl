"""NeurHL 1.1 C2: season-layer selection and judging (PLAN_NeurHL_1_1 A1).

Reads data/tensors/_season_bt/pre_<V>.parquet (eval/backtest_unified_season.py)
and simulates each season's points under team-strength variants:

  s_team(week) = s0 + random walk (weekly innovation sd sw),  s0 ~ N(0, sigma0)
  game logit   = c + (logit p - c) * rho ** week + k * (s_home - s_away)

c is the season's mean logit and k the engine's rate sensitivity. The
regulation/overtime split of each win comes from the game's outcome4, as in
sim/unified_2027.py. The frozen NeurHL 1.0 layer is (sigma0, sw, rho) =
(0.07, 0, 1).

The variant with the lowest mean points CRPS on the fit seasons is selected,
then judged against (0.07, 0, 1) on the judge seasons (PLAN_NeurHL_1_1 A1).
The same grid on a preseason Elo-only model is reported, not used.

Writes neurhl/output/neurhl_1_1/season_layer_c2.json.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402

SRC = TENSORS / "_season_bt"
OUT = ROOT / "output" / "neurhl_1_1" / "season_layer_c2.json"
FIT = [2012, 2014, 2015, 2016, 2017, 2018]
JUDGE = [2019, 2020, 2022, 2023, 2024]
GRID = list(itertools.product([0, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13],
                              [0, 0.005, 0.01, 0.015, 0.02], [1, 0.995, 0.99, 0.98]))
FROZEN = (0.07, 0, 1)
SIMS, SEED = 4000, 711
P_SO_GIVEN_TIE = 0.38


def load(V):
    d = pd.read_parquet(SRC / f"pre_{V}.parquet")
    d["week"] = ((pd.to_datetime(d.date) - pd.to_datetime(d.date).min()).dt.days // 7).astype(int)
    return d


def realised(d):
    o = d.outcome4.to_numpy()
    teams = np.unique(np.r_[d.home_idx, d.away_idx])
    ti = {t: i for i, t in enumerate(teams)}
    hi, ai = d.home_idx.map(ti).to_numpy(), d.away_idx.map(ti).to_numpy()
    pts = np.zeros(len(teams))
    np.add.at(pts, hi, np.where(np.isin(o, [0, 2]), 2, np.where(o == 3, 1, 0)))
    np.add.at(pts, ai, np.where(np.isin(o, [1, 3]), 2, np.where(o == 2, 1, 0)))
    return teams, hi, ai, pts


def simulate(d, lp, k, hi, ai, T, sigma0, sw, rho, seed):
    rng = np.random.default_rng(seed)
    w = d.week.to_numpy()
    W = w.max() + 1
    c = lp.mean()
    base = c + (lp - c) * rho ** w
    s0 = rng.normal(0, sigma0, (SIMS, T)) if sigma0 > 0 else np.zeros((SIMS, T))
    if sw > 0:
        walk = np.cumsum(rng.normal(0, sw, (SIMS, W, T)), axis=1)
        walk = np.concatenate([np.zeros((SIMS, 1, T)), walk[:, :-1]], axis=1)
        S = s0[:, None, :] + walk                                  # sims x weeks x teams
        sh, sa = S[:, w, hi], S[:, w, ai]
    else:
        sh, sa = s0[:, hi], s0[:, ai]
    z = base[None, :] + k[None, :] * (sh - sa)
    hw = rng.random(z.shape) < 1 / (1 + np.exp(-z))
    o4 = d[["o4_hr", "o4_ar", "o4_ho", "o4_ao"]].to_numpy()
    rh = o4[:, 0] / np.maximum(o4[:, 0] + o4[:, 2], 1e-9)
    ra = o4[:, 1] / np.maximum(o4[:, 1] + o4[:, 3], 1e-9)
    reg = np.where(hw, rng.random(z.shape) < rh[None, :], rng.random(z.shape) < ra[None, :])
    hp = np.where(hw, 2, np.where(reg, 0, 1))
    ap = np.where(~hw, 2, np.where(reg, 0, 1))
    Hm = np.zeros((len(d), T))
    Hm[np.arange(len(d)), hi] = 1
    Am = np.zeros((len(d), T))
    Am[np.arange(len(d)), ai] = 1
    return hp @ Hm + ap @ Am                                        # sims x teams


def scores(sim, y):
    """Per team: CRPS (sample form), |mean - y|, inside the 10-90 interval."""
    n = sim.shape[0]
    crps = np.empty(sim.shape[1])
    for j in range(sim.shape[1]):
        x = np.sort(sim[:, j])
        e1 = np.abs(x - y[j]).mean()
        e2 = (2 * np.arange(1, n + 1) - n - 1) @ x / (n * n)       # E|X - X'| / 2
        crps[j] = e1 - e2
    lo, hi = np.quantile(sim, 0.1, axis=0), np.quantile(sim, 0.9, axis=0)
    return crps, np.abs(sim.mean(0) - y), ((y >= lo) & (y <= hi)).astype(float)


def _memo(model, V):
    import pickle
    f = SRC / f"c2memo_{model}_{V}.pkl"
    return f, (pickle.loads(f.read_bytes()) if f.exists() else {})


def run_grid(seasons, model, grid=None):
    """Per-season results are memoised on disk by (model, season, variant); the
    simulation is seeded, so a cached result equals a fresh one."""
    import pickle
    res = {}
    cache = {V: load(V) for V in seasons}
    memo = {V: _memo(model, V) for V in seasons}
    for g in (grid or GRID):
        rows = []
        for V in seasons:
            f, mm = memo[V]
            if g in mm:
                rows.append(mm[g])
                continue
            d = cache[V]
            teams, hi, ai, y = realised(d)
            if model == "engine":
                lp, k = np.log(d.p / (1 - d.p)).to_numpy(), d.k.to_numpy()
                var = g
            elif model == "blend":                      # C2c: g = (a, sigma0, sw, rho)
                a = g[0]
                lp = a * np.log(d.p / (1 - d.p)).to_numpy() + (1 - a) * d.elo_pre.to_numpy()
                k, var = d.k.to_numpy(), g[1:]
            else:
                lp, k = d.elo_pre.to_numpy(), np.full(len(d), float(d.k.mean()))
                var = g
            sim = simulate(d, lp, k, hi, ai, len(teams), *var, SEED + V)
            c, m, cov = scores(sim, y)
            mm[g] = pd.DataFrame({"season": V, "team": teams, "crps": c, "ae": m, "cov": cov})
            rows.append(mm[g])
        res[g] = pd.concat(rows, ignore_index=True)
    for V, (f, mm) in memo.items():
        f.write_bytes(pickle.dumps(mm))
    return res


def season_boot(d, draws=9999, seed=SEED):
    seas = d.season.unique()
    rng = np.random.default_rng(seed)
    by = {s: d[d.season == s].diff_.to_numpy() for s in seas}
    out = []
    for _ in range(draws):
        pick = rng.choice(seas, len(seas))
        v = np.concatenate([by[s] for s in pick])
        out.append(v.mean())
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def summary(df):
    return {"crps": float(df.crps.mean()), "mae": float(df.ae.mean()), "coverage": float(df["cov"].mean()),
            "n": int(len(df))}


def fit_only():
    """Fill the memo for the fit seasons (no judge season is simulated)."""
    for model in ("engine", "elo"):
        run_grid(FIT, model)
    run_grid(FIT, "blend", [(a_,) + g for a_ in (0, 0.25, 0.5, 0.75, 1) for g in GRID])


def main():
    if "--fit-only" in sys.argv:
        return fit_only()
    avail = [V for V in FIT + JUDGE if (SRC / f"pre_{V}.parquet").exists()]
    missing = sorted(set(FIT + JUDGE) - set(avail))
    assert not missing, f"missing preseason outputs: {missing}"
    out = {"grid_size": len(GRID), "sims": SIMS, "fit": FIT, "judge": JUDGE}
    for model in ("engine", "elo"):
        fit = run_grid(FIT, model)
        best = min(GRID, key=lambda g: fit[g].crps.mean())
        top = sorted(GRID, key=lambda g: fit[g].crps.mean())[:5]
        judge = run_grid(JUDGE, model, list(dict.fromkeys([best, FROZEN])))   # only the two compared
        a, b = judge[best].copy(), judge[FROZEN]
        a["diff_"] = a.crps.to_numpy() - b.crps.to_numpy()
        out[model] = {
            "selected": {"sigma0": best[0], "sw": best[1], "rho": best[2]},
            "fit": {"selected": summary(fit[best]), "frozen": summary(fit[FROZEN]),
                    "top5": [{"variant": list(g), **summary(fit[g])} for g in top]},
            "judge": {"selected": summary(judge[best]), "frozen": summary(judge[FROZEN]),
                      "crps_diff": float(a.diff_.mean()), "crps_diff_ci95_season_boot": season_boot(a),
                      "by_season": {int(s): {"selected": float(judge[best][judge[best].season == s].crps.mean()),
                                             "frozen": float(b[b.season == s].crps.mean())}
                                    for s in JUDGE}}}
        print(model, json.dumps(out[model]["selected"]), json.dumps(out[model]["judge"], indent=None)[:600], flush=True)
    # C2c (A4): blend weight crossed with the A1 grid
    BGRID = [(a_,) + g for a_ in (0, 0.25, 0.5, 0.75, 1) for g in GRID]
    fitb = run_grid(FIT, "blend", BGRID)
    bb = min(BGRID, key=lambda g: fitb[g].crps.mean())
    a1 = (1,) + tuple(out["engine"]["selected"].values())
    jb = run_grid(JUDGE, "blend", [bb, (1,) + FROZEN, a1])
    out["blend_c2c"] = {"selected": {"a": bb[0], "sigma0": bb[1], "sw": bb[2], "rho": bb[3]},
                        "fit": summary(fitb[bb]),
                        "judge": {"selected": summary(jb[bb]), "frozen": summary(jb[(1,) + FROZEN]),
                                  "a1_selected": summary(jb[a1])}}
    cb = out["blend_c2c"]["judge"]
    out["blend_c2c"]["adopt"] = bool(cb["selected"]["crps"] < min(cb["frozen"]["crps"], cb["a1_selected"]["crps"])
                                     and abs(cb["selected"]["coverage"] - 0.80) <= 0.05)
    print("c2c", json.dumps(out["blend_c2c"])[:700], flush=True)
    e = out["engine"]["judge"]
    out["decision"] = {"adopt": bool(e["crps_diff"] < 0 and abs(e["selected"]["coverage"] - 0.80) <= 0.05),
                       "rule": "PLAN_NeurHL_1_1 A1: judged CRPS difference < 0 and coverage within 0.80 +- 0.05"}
    out["engine_vs_elo_judge_frozen_layer"] = {"engine": out["engine"]["judge"]["frozen"],
                                               "elo": out["elo"]["judge"]["frozen"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["decision"]), f"-> {OUT}")


if __name__ == "__main__":
    main()
