"""L2 harness: strict skater protocol with component-specific shrinkage.

`season_base` + `season_eval` together are a copy of
orr.backtest.players_bt.run_season (strict protocol, sims=0) and
orr.players.pipeline, split where the per-60 rates enter: deployment (games,
ice time) reads only usage, so it is computed once per season with the
shipped parameters, and every shrinkage configuration only redoes
rates -> totals -> forecast averaging -> reserves. With every multiplier = 1
the output equals players_bt.run_season exactly (--verify, tuning seasons).

Usage
  python3 -m orr.experiments.L2.run --verify    # m = 1 reproduces players_bt.json (tuning seasons)
  python3 -m orr.experiments.L2.run --tune      # staged coordinate search -> ledger.json
  python3 -m orr.experiments.L2.run --fix-config  # end point of the logged search -> config.json
  python3 -m orr.experiments.L2.run --confirm   # 2018-19, once, with config.json
  python3 -m orr.experiments.L2.run --test      # 2022-26, ONCE, with config.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from orr import deploy as DP
from orr import players as PL
from orr.backtest import players_bt as B
from orr.experiments.L2 import rates as R

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"
CONFIG = HERE / "config.json"
CONFIRM = HERE / "confirm.json"
TEST = HERE / "test.json"

GAMES = B.GAMES
TUNE_V = B.TUNE_V            # 2011, 2012, 2014-2017
CONFIRM_V = B.CONFIRM_V      # 2018, 2019
TEST_V = B.TEST_V            # 2022-2026
MAX_CONFIGS = 80
ADOPT_GAIN = 5e-4            # a move is adopted if pooled B MAE falls by more than this

SEARCH_DESIGN = {
    "objective": "pooled (n-weighted) strict-protocol sample-B MAE (per-82 points, NeurHL "
                 "v2 restatement), V = 2011, 2012, 2014-2017; sample A and 'all' logged",
    "parameterisation": "K[c,pos] = kappa(0.35, shipped) * m[comp,sit,pos] * MoM K; comp in "
                        "g, a1, a2, shot (sog+ixg share one multiplier); sit in ev, pp; pos F, D: "
                        "16 multipliers, all 1 = shipped ORR. SH/other situations, peripherals, "
                        "finishing prior weight, usage, deployment, blend weights unchanged.",
    "stage_A": "component level (one multiplier per comp shared over pos and sit), comps in "
               "order a1, a2, g, shot: try x0.5 and x2; if the better one improves, adopt it "
               "and try one more step the same way (x0.25 or x4); <= 12 configs",
    "stage_B": "16 knobs in order F(ev: a1,a2,g,shot; pp: a1,a2,g,shot), then D likewise: "
               "try current x0.5 and x2, adopt the better if it improves; 32 configs",
    "stage_C": "the same 16 knobs with factors 1/sqrt2 and sqrt2; <= 32 configs",
    "adopt_rule": f"a move is adopted only if pooled B MAE improves by more than {ADOPT_GAIN}",
    "cap": f"at most {MAX_CONFIGS} configurations including the baseline (search stops at the cap)",
    "final": "the configuration with the lowest pooled tuning B MAE (the end point of the search)",
}


# ---------------------------------------------------------------------------
# Copy of players_bt.run_season / players.pipeline, split at the rates
# ---------------------------------------------------------------------------
_BASE: dict = {}


def season_base(V: int, prm: PL.Params) -> dict:
    """Everything of the strict run that does not read the scoring rates."""
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
    out = {"V": V, "act": act, "flag": flag, "ids": ids, "proj": proj, "dep": dep,
           "reserve_gp": B.reserve_gp(V)}
    _BASE[V] = out
    return out


def season_proj(base: dict, prm: PL.Params, m: dict) -> pd.DataFrame:
    """Projection with the L2 rates (rookie/thin calibration as shipped)."""
    proj = PL.calibrate_thin_rates(R.project_l2(base["V"], prm, m, base["ids"]), prm)
    b = base["proj"]
    assert (proj.player_id.to_numpy() == b.player_id.to_numpy()).all()
    for c in ("score", "tpg_ev", "tpg_pp", "tpg_sh", "tpg_oth", "has_hist"):
        assert np.array_equal(proj[c].to_numpy(), b[c].to_numpy()), c   # deployment inputs unchanged
    return proj


def season_eval(base: dict, prm: PL.Params, proj: pd.DataFrame) -> pd.DataFrame:
    """Rates -> totals -> forecast averaging, reserves, actuals: the rest of
    players.pipeline and players_bt.run_season."""
    V, games = base["V"], GAMES
    tot = DP.add_totals(base["dep"], proj)
    tot = PL.blend_paths(tot, V, prm, games)
    thin = tot.player_id.map(dict(zip(proj.player_id, PL.thin_group(proj)))) == "thin"
    if prm.thin_rate_mult != 1.0 and thin.any():
        tot = tot.copy()
        cols = [c for c in tot.columns if c in ("g", "a", "p", "a1", "a2", "ppg", "ppa")
                or c.startswith(("g_p", "a_p", "p_p")) or c in ("g_sd", "a_sd", "p_sd")]
        tot.loc[thin, cols] = tot.loc[thin, cols] * prm.thin_rate_mult
    tot = tot.copy()
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


def run_season_l2(V: int, prm: PL.Params, m: dict) -> pd.DataFrame:
    base = season_base(V, prm)
    return season_eval(base, prm, season_proj(base, prm, m))


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


def summarise(per: dict, seasons) -> dict:
    return {"B": round(pooled(per, seasons, "B"), 5), "A": round(pooled(per, seasons, "A"), 5),
            "all": round(pooled(per, seasons, "all"), 5),
            "B_mae_g": round(pooled(per, seasons, "B", "mae_g"), 5),
            "B_mae_a": round(pooled(per, seasons, "B", "mae_a"), 5),
            "B_bias": round(pooled(per, seasons, "B", "bias"), 4),
            "per_season": {int(V): {"A": round(per[V]["A"]["mae"], 4), "B": round(per[V]["B"]["mae"], 4),
                                    "all": round(per[V]["all"]["mae"], 4)} for V in seasons}}


def score(prm, m, seasons) -> dict:
    return {V: season_metrics(run_season_l2(V, prm, m)) for V in seasons}


# ---------------------------------------------------------------------------
# Verify: m = 1 reproduces players_bt.json (tuning seasons only)
# ---------------------------------------------------------------------------
def verify(prm):
    ref = json.loads((B.OUT / "players_bt.json").read_text())["seasons"]
    for V in TUNE_V:
        base = season_base(V, prm)
        p1 = R.project_l2(V, prm, R.ones(), base["ids"])
        p0 = PL.project(V, prm, ids=base["ids"])
        rc = [c for c in p0.columns if c.startswith(("r60_", "sd60_"))] + ["fin"]
        dmax = float(np.nanmax(np.abs(p1[rc].to_numpy(float) - p0[rc].to_numpy(float))))
        tot = run_season_l2(V, prm, R.ones())
        mm = season_metrics(tot)
        r = ref[str(V)]["strict"]
        print(V, "max |proj diff|", dmax,
              {s: (round(mm[s]["mae"], 4), round(r[s]["mae"], 4), mm[s]["n"], r[s]["n"])
               for s in ("A", "B", "all")}, flush=True)
        assert dmax < 1e-9
        for s in ("A", "B", "all"):
            assert abs(mm[s]["mae"] - r[s]["mae"]) < 1e-6 and mm[s]["n"] == r[s]["n"], (V, s)
    print("verify: m=1 reproduces players.project exactly and players_bt.json strict A/B/all "
          "on the tuning seasons")


# ---------------------------------------------------------------------------
# Tuning (V <= 2017 only)
# ---------------------------------------------------------------------------
def tune(prm):
    assert not LEDGER.exists(), "ledger exists: the search runs once"
    ledger = {"design": SEARCH_DESIGN, "written_before_first_run": time.strftime("%Y-%m-%d %H:%M:%S"),
              "tune_seasons": TUNE_V, "configs": []}
    LEDGER.write_text(json.dumps(ledger, indent=1))
    t0 = time.time()
    seen = {}

    def ev(m, stage, note):
        k = R.mkey(m)
        if k in seen:
            return seen[k]
        if len(seen) >= MAX_CONFIGS:
            return None
        per = score(prm, m, TUNE_V)
        sm = summarise(per, TUNE_V)
        seen[k] = sm["B"]
        ledger["configs"].append({"n": len(ledger["configs"]), "stage": stage, "move": note,
                                  "mult": {kk: round(float(m[kk]), 6) for kk in R.KNOBS},
                                  "tune": sm, "seconds_since_start": round(time.time() - t0, 1)})
        LEDGER.write_text(json.dumps(ledger, indent=1))
        print(f"  [{len(seen):>2}] {stage} {note:<26} B {sm['B']:.4f}  A {sm['A']:.4f}  "
              f"all {sm['all']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        return sm["B"]

    cur_m = R.ones()
    cur = ev(cur_m, "baseline", "all m=1")
    base_B = cur

    def setcomp(m, comp, v):
        out = dict(m)
        for p in R.POSS:
            for s in R.KSITS:
                out[f"{comp}_{s}_{p}"] = v
        return out

    # --- stage A: component level
    for comp in ("a1", "a2", "g", "shot"):
        v0 = cur_m[f"{comp}_ev_F"]
        res = {}
        for f in (0.5, 2.0):
            sc = ev(setcomp(cur_m, comp, v0 * f), "A", f"{comp} x{f}")
            if sc is not None:
                res[f] = sc
        if not res:
            break
        f_best = min(res, key=res.get)
        if res[f_best] < cur - ADOPT_GAIN:
            cur, cur_m = res[f_best], setcomp(cur_m, comp, v0 * f_best)
            f2 = f_best * f_best            # one more step the same way
            sc = ev(setcomp(cur_m, comp, v0 * f2), "A", f"{comp} x{f2}")
            if sc is not None and sc < cur - ADOPT_GAIN:
                cur, cur_m = sc, setcomp(cur_m, comp, v0 * f2)
    # --- stages B and C: per component x situation x position
    for stage, facs in (("B", (0.5, 2.0)), ("C", (1 / math.sqrt(2), math.sqrt(2)))):
        for knob in R.KNOBS:
            res = {}
            for f in facs:
                m2 = dict(cur_m)
                m2[knob] = cur_m[knob] * f
                sc = ev(m2, stage, f"{knob} x{f:.3f}")
                if sc is not None:
                    res[f] = (sc, m2)
            if not res:
                break
            f_best = min(res, key=lambda z: res[z][0])
            if res[f_best][0] < cur - ADOPT_GAIN:
                cur, cur_m = res[f_best]
    best = min(ledger["configs"], key=lambda r: r["tune"]["B"])
    # As run, this assertion FAILED after all 74 configurations had been
    # scored and logged: the lowest-B ledger entry (#69) is a sub-threshold
    # move (gain 0.00027 < ADOPT_GAIN) away from the search's end point (#66).
    # The configuration was then fixed from the ledger by fix_config(), on
    # tuning data only, before any confirm or test run.
    assert R.mkey(best["mult"]) == R.mkey(cur_m)
    cfg = {"mult": best["mult"], "ledger_n": best["n"], "tune": best["tune"],
           "baseline_tune": ledger["configs"][0]["tune"],
           "n_configurations": len(ledger["configs"]),
           "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "K_minutes_at_mean_usage_2017": k_table(prm, best["mult"], 2017)}
    CONFIG.write_text(json.dumps(cfg, indent=1))
    print(f"best: B {cur:.4f} (baseline {base_B:.4f}); {len(ledger['configs'])} configurations")
    print(json.dumps(best["mult"], indent=1))
    return cfg


def replay_endpoint(configs: list) -> dict:
    """Re-apply the search's adoption rule to the logged configurations (in
    logged order) and return the end point of the search."""
    C = configs
    cur, cur_m, i = C[0]["tune"]["B"], C[0]["mult"], 1
    adopted = []
    for comp in ("a1", "a2", "g", "shot"):           # stage A
        pair = C[i:i + 2]
        i += 2
        b = min(pair, key=lambda r: r["tune"]["B"])
        if b["tune"]["B"] < cur - ADOPT_GAIN:
            cur, cur_m = b["tune"]["B"], b["mult"]
            adopted.append(b["n"])
            nxt = C[i]
            i += 1
            assert nxt["stage"] == "A" and nxt["move"].startswith(comp)
            if nxt["tune"]["B"] < cur - ADOPT_GAIN:
                cur, cur_m = nxt["tune"]["B"], nxt["mult"]
                adopted.append(nxt["n"])
    for stage in ("B", "C"):                         # stages B and C
        while i < len(C) and C[i]["stage"] == stage:
            pair = C[i:i + 2]
            i += 2
            assert pair[0]["move"].split()[0] == pair[1]["move"].split()[0]
            b = min(pair, key=lambda r: r["tune"]["B"])
            if b["tune"]["B"] < cur - ADOPT_GAIN:
                cur, cur_m = b["tune"]["B"], b["mult"]
                adopted.append(b["n"])
    assert i == len(C)
    end = [r for r in C if R.mkey(r["mult"]) == R.mkey(cur_m)][0]
    return {"entry": end, "adopted": adopted}


def fix_config(prm):
    """Fix the configuration from the complete ledger (tuning data only):
    the end point of the search under its adoption rule. The lowest-B ledger
    entry is recorded alongside; it differs by one sub-threshold move."""
    assert not CONFIG.exists()
    ledger = json.loads(LEDGER.read_text())
    ep = replay_endpoint(ledger["configs"])
    end = ep["entry"]
    low = min(ledger["configs"], key=lambda r: r["tune"]["B"])
    cfg = {"mult": end["mult"], "ledger_n": end["n"], "tune": end["tune"],
           "baseline_tune": ledger["configs"][0]["tune"],
           "n_configurations": len(ledger["configs"]),
           "adopted_moves": [{"n": n, "move": ledger["configs"][n]["move"],
                              "B": ledger["configs"][n]["tune"]["B"]} for n in ep["adopted"]],
           "selection_note": "End point of the search under the adoption rule (gain > "
                             f"{ADOPT_GAIN}). The design's 'final' line assumed this equals the "
                             "lowest-B ledger entry; it does not: entry "
                             f"#{low['n']} ({low['move']}, B {low['tune']['B']}) is a "
                             f"sub-threshold step ({end['tune']['B'] - low['tune']['B']:.5f}) "
                             "from the end point. The end point is used, consistent with the "
                             "adoption rule; decided on tuning data only, before confirm/test.",
           "lowest_B_entry": {"n": low["n"], "move": low["move"], "mult": low["mult"],
                              "tune": low["tune"]},
           "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "K_minutes_at_mean_usage_2017": k_table(prm, end["mult"], 2017)}
    CONFIG.write_text(json.dumps(cfg, indent=1))
    print(json.dumps({k: cfg[k] for k in ("mult", "ledger_n", "adopted_moves")}, indent=1))
    print(f"tune B {cfg['baseline_tune']['B']} -> {end['tune']['B']}  "
          f"A {cfg['baseline_tune']['A']} -> {end['tune']['A']}")
    return cfg


def k_table(prm, m: dict, V: int) -> dict:
    """K (minutes of prior) at mean usage for each knob's columns, shipped vs L2."""
    pri = PL.rate_priors(V, prm.prior_window, prm.eb_window)
    tgt = PL.project_league(V)
    out = {}
    for knob in R.KNOBS:
        comp, sit, pos = knob.split("_")
        st = "ixg" if comp == "shot" else comp
        c = f"{st}_{sit}"
        e = pri[(c, pos)]
        X = np.array([1.0, 15.5 if pos == "F" else 19.0, 1.5, 1.0])
        prr = float(max(X @ e["beta"], 1e-3))
        k0 = prm.kappa * e["phi"] * prr / (float(tgt[c]) * e["tau2"])
        out[f"{c}_{pos}"] = {"shipped": round(k0, 1), "l2": round(k0 * m[knob], 1)}
    return out


def load_cfg() -> dict:
    return json.loads(CONFIG.read_text())["mult"]


# ---------------------------------------------------------------------------
# Confirm (2018-19) and test (2022-26)
# ---------------------------------------------------------------------------
def boot_ci(e0: list, e1: list, n_boot: int = 4000, seed: int = 11) -> list:
    """Paired bootstrap 95% CI of the pooled MAE difference (L2 - shipped),
    resampling players within each season (season sizes kept)."""
    rng = np.random.default_rng(seed)
    d = [np.abs(b) - np.abs(a) for a, b in zip(e0, e1)]
    n = sum(len(x) for x in d)
    tot = np.zeros(n_boot)
    for x in d:
        idx = rng.integers(0, len(x), (n_boot, len(x)))
        tot += x[idx].sum(1)
    m = tot / n
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def evaluate(prm, m: dict, seasons, label: str, check_ref: bool) -> dict:
    ref = json.loads((B.OUT / "players_bt.json").read_text())["seasons"]
    rows = {}
    errs = {s: ([], []) for s in ("A", "B", "all")}
    per0, per1 = {}, {}
    for V in seasons:
        t0 = run_season_l2(V, prm, R.ones())
        t1 = run_season_l2(V, prm, m)
        per0[V], per1[V] = season_metrics(t0), season_metrics(t1)
        if check_ref:       # the shipped arm equals the published players_bt.json
            for s in ("A", "B", "all"):
                assert abs(per0[V][s]["mae"] - ref[str(V)]["strict"][s]["mae"]) < 1e-6, (V, s)
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
            "shipped": {s: round(per0[V][s]["mae"], 4) for s in ("A", "B", "all")},
            "l2": {s: round(per1[V][s]["mae"], 4) for s in ("A", "B", "all")},
            "diff": {s: round(per1[V][s]["mae"] - per0[V][s]["mae"], 4) for s in ("A", "B", "all")},
            "n": {s: per0[V][s]["n"] for s in ("A", "B", "all")},
            "roster_proxy": str(t0.roster_proxy.iloc[0]),
            "neurhl_A": B.NEURHL_A.get(V), "neurhl_v2_blend": (B.NEURHL_V2.get(V) or [None] * 3)[2]}
        print(f"{label} {V}: A {per0[V]['A']['mae']:.4f} -> {per1[V]['A']['mae']:.4f}  "
              f"B {per0[V]['B']['mae']:.4f} -> {per1[V]['B']['mae']:.4f}  "
              f"all {per0[V]['all']['mae']:.4f} -> {per1[V]['all']['mae']:.4f}", flush=True)
    out = {"seasons": rows, "pooled": {}}
    for s in ("A", "B", "all"):
        e0, e1 = errs[s]
        a0 = float(np.abs(np.concatenate(e0)).mean())
        a1 = float(np.abs(np.concatenate(e1)).mean())
        out["pooled"][s] = {"n": int(sum(len(x) for x in e0)), "shipped": a0, "l2": a1,
                            "diff": a1 - a0, "ci95": boot_ci(e0, e1),
                            "seasons_improved": int(sum(per1[V][s]["mae"] < per0[V][s]["mae"]
                                                        for V in seasons))}
        print(f"{label} pooled {s}: {a0:.4f} -> {a1:.4f}  diff {a1 - a0:+.4f}  "
              f"CI {out['pooled'][s]['ci95']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--fix-config", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    prm = PL.load_params()
    if a.verify:
        verify(prm)
    if a.tune:
        tune(prm)
    if a.fix_config:
        fix_config(prm)
    if a.confirm:
        assert not CONFIRM.exists(), "confirm runs once"
        m = load_cfg()
        out = evaluate(prm, m, CONFIRM_V, "confirm", check_ref=True)
        out["mult"] = m
        out["run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        CONFIRM.write_text(json.dumps(out, indent=1))
    if a.test:
        assert not TEST.exists(), "the test runs once"
        m = load_cfg()
        out = evaluate(prm, m, TEST_V, "test", check_ref=True)
        out["mult"] = m
        out["run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        TEST.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
