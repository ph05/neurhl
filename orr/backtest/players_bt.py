"""Walk-forward backtests of the skater layer, on NeurHL's protocol and on a
strict preseason protocol.

Protocols
---------
matched  Reproduces the information AND budget conventions of NeurHL's
         published player backtests (neurhl/sim/project_players.py::backtest,
         player_season_v2.py::backtest, models/player_proj.py::to_totals):
         the population is every skater with NHL history who played in V,
         placed on the team he played for most IN season V; each team's
         ice-time budget is shared over those players (rookies excluded, as
         in NeurHL), and expected games are scaled league-wide to
         18 x 82 x teams x 0.968 with a 78-game cap. ORR's own games
         model here is team-agnostic (an isotonic map from projected usage
         to games share, fitted walk-forward on earlier seasons' players who
         played), because a traded player's games cannot be placed on one
         team's depth chart without knowing when he moved.
         Remaining difference: actual stats come from MoneyPuck season files,
         not NeurHL's gitignored player-game tensors (sample sizes agree
         within 0-2).
strict   Nothing from season V beyond the evaluation population: the roster is an opening-roster proxy (every
         skater who dressed in any of his team's first 10 games, fastRhockey
         boxes, 2011-2024). For 2025-2026 the boxes are missing and the roster
         falls back to each player's season-V main team ('season_team', which
         leaks like matched). Skaters in the evaluation sample who were on no
         opening roster are projected as reserves: their per-game projection
         times the mean games such players played in earlier seasons.

Samples (identical to NeurHL's)
-------------------------------
A  player_backtest.json: skaters with NHL history before V and >= 40 GP in V;
   actual raw points; projections for 82 games.
B  player_season_v2.json: >= 40*sched/82 GP with sched = the season's max GP
   of any skater, actual points restated x 82/sched (NeurHL's own quirk: in
   2023 a traded skater's 85 GP deflates every actual by 82/85).
all  every rostered skater with NHL history (unconditional; used for bias and
   interval coverage, since conditioning on >=40 GP selects on the outcome).

Tuning uses ONLY V in 2011, 2012, 2014-2017 (strict protocol, sample A).
2018-2019 are reported as a confirmation window, 2022-2026 as the held-out
test; 2013, 2020 and 2021 are shortened seasons and never scored.

Run:  python3 -m orr.backtest.players_bt            (final report)
      python3 -m orr.backtest.players_bt --tune     (search + ledger)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from orr import config as C
from orr import data as D
from orr import deploy as DP
from orr import players as PL

TUNE_V = [2011, 2012, 2014, 2015, 2016, 2017]
CONFIRM_V = [2018, 2019]
TEST_V = [2022, 2023, 2024, 2025, 2026]
OUT = C.OUT / "backtest"
LEDGER = OUT / "players_search_ledger.json"
GAMES = 82
RESERVE_GP_DEFAULT = 30.0
# Tuning objective: pooled points MAE over EVERY opening-roster skater with
# NHL history (strict protocol). The >=40-GP sample selects on the outcome
# and rewards over-projecting games; an earlier search on it drifted to
# +3 GP of unconditional over-projection. That search stays in the ledger.
OBJECTIVE_SAMPLE = "all"
STACK_GP = True       # games stacking layer (deploy.gp_stack); see ledger

NEURHL_A = {2022: 10.466, 2023: 9.597, 2024: 9.166, 2025: 9.092, 2026: 9.394}
NEURHL_A_N = {2022: 587, 2023: 603, 2024: 594, 2025: 586, 2026: 586}   # its sample sizes
NEURHL_V2 = {2011: (14.022, 9.775, 10.254), 2012: (10.399, 9.774, 9.561),
             2014: (9.385, 9.22, 8.966), 2015: (9.137, 9.143, 8.55),
             2016: (8.453, 8.64, 8.145), 2017: (9.313, 9.394, 8.904),
             2022: (10.619, 10.114, 9.981), 2023: (9.172, 9.582, 8.95),
             2024: (9.056, 9.468, 8.917), 2025: (8.846, 8.985, 8.515),
             2026: (9.238, 9.32, 9.037)}          # (A, B, blend)


# ---------------------------------------------------------------------------
# Actuals and rosters
# ---------------------------------------------------------------------------
def actuals(V: int) -> pd.DataFrame:
    P = PL.panel()
    j = P.j(V)
    x = P.x
    g = sum(x[f"g_{k}"][:, j] for k in PL.SITS)
    a = sum(x[f"a1_{k}"][:, j] + x[f"a2_{k}"][:, j] for k in PL.SITS)
    d = pd.DataFrame({"player_id": P.ids, "act_gp": x["gp"][:, j], "act_g": g,
                      "act_a": a, "act_p": g + a, "act_toi": x["toi_all"][:, j],
                      "act_team": P.team[:, j]})
    return d[d.act_gp > 0].reset_index(drop=True)


def roster_matched(V: int) -> pd.DataFrame:
    a = actuals(V)
    return a[["player_id", "act_team"]].rename(columns={"act_team": "team"})


def roster_strict(V: int) -> tuple[pd.DataFrame, str]:
    """Opening-roster proxy (first 10 team games) or the season-team fallback."""
    try:
        r = D.opening_rosters(V)
        r = r[r.grp != "G"][["player_id", "team"]]
        return r.reset_index(drop=True), "first10"
    except FileNotFoundError:
        return roster_matched(V), "season_team"


def reserve_gp(V: int) -> float:
    """Preseason expected GP (per 82) of a skater on no opening roster: over
    proxy seasons s < V (walk-forward), the mean season-s GP of every skater
    who played in s-1 but was on no opening roster in s, INCLUDING those who
    did not play in s (zeros). Conditioning on having played in s would use
    season-V participation, which a preseason projection cannot know."""
    vals = []
    for s in range(2011, V):
        if s in C.BROKEN_SEASONS or (s - 1) in C.BROKEN_SEASONS:
            continue
        try:
            r = D.opening_rosters(s)
        except FileNotFoundError:
            continue
        a, prev = actuals(s), actuals(s - 1)
        cand = sorted(set(prev.player_id) - set(r.player_id))
        gp = a.set_index("player_id").act_gp.reindex(cand).fillna(0.0)
        vals.append(gp.to_numpy() * 82.0 / a.act_gp.max())
    return float(np.concatenate(vals).mean()) if vals else RESERVE_GP_DEFAULT


# ---------------------------------------------------------------------------
# One season
# ---------------------------------------------------------------------------
def run_season(V: int, prm: PL.Params, protocol: str, sims: int = 0) -> pd.DataFrame:
    """Projections for season V under `protocol`, joined to actuals."""
    act = actuals(V)
    kind = "expost" if protocol == "matched" else "opening"
    roster, flag = DP.historical_roster(V, kind)
    P = PL.panel()
    hist_ids = set(P.ids[PL._hist_gp(P, V) > 0])
    if protocol == "matched":
        # NeurHL's population: players with NHL history who played in V
        roster = roster[roster.player_id.isin(hist_ids)]
    # ex-post rosters (matched, and the 2025-26 strict fallback) use the
    # team-agnostic games model; opening rosters the depth-chart recursion
    ids = set(roster.player_id) | (set(act.player_id) & hist_ids)
    tot, proj = PL.pipeline(V, prm, roster, GAMES,
                            "opening" if flag == "first10" else "expost",
                            eval_ids=ids, sims=sims, stack=STACK_GP,
                            neurhl_budget=(protocol == "matched"))
    tot = tot.copy()
    tot["on_roster"] = True
    # reserves: evaluation-sample players on no (proxy) roster
    res = proj[~proj.player_id.isin(tot.player_id)].copy()
    if len(res):
        gp = reserve_gp(V) if protocol == "strict" else 0.0
        res["gp"] = gp
        for k in PL.SITS:
            res[f"toi_{k}"] = gp * res[f"tpg_{k}"]
        rt = DP.add_totals(res[["player_id", "gp", *[f"toi_{k}" for k in PL.SITS]]], proj)
        rt["on_roster"] = False
        rt["team"] = None
        if sims:          # reserves: Poisson + talent-free band (rarely sampled)
            for s in ("g", "a", "p"):
                lam = rt[s].to_numpy()
                rt[f"{s}_p10"] = np.maximum(lam - 1.2816 * np.sqrt(lam + (0.5 * lam) ** 2), 0)
                rt[f"{s}_p90"] = lam + 1.2816 * np.sqrt(lam + (0.5 * lam) ** 2)
        tot = pd.concat([tot, rt], ignore_index=True)
    tot = tot.drop(columns=["has_hist", "name", "age"], errors="ignore")
    tot = tot.merge(proj[["player_id", "has_hist", "name", "age", "pos"]]
                    .rename(columns={"pos": "pos_p"}), on="player_id", how="left")
    tot = tot.merge(act, on="player_id", how="left").copy()
    tot["roster_proxy"] = flag
    tot["V"] = V
    return tot


def samples(tot: pd.DataFrame) -> dict:
    """Index masks for NeurHL's samples A and B, and 'all' (unconditional)."""
    played = tot.act_gp.fillna(0) > 0
    sched = float(tot.act_gp.max())
    h = tot.has_hist.fillna(False).astype(bool)
    return {"A": h & (tot.act_gp >= 40),
            "B": h & (tot.act_gp >= 40 * sched / 82.0),
            "all": h & played & tot.on_roster}, sched


def metrics(tot: pd.DataFrame, col_p="p", col_g="g", col_a="a") -> dict:
    m, sched = samples(tot)
    out = {}
    for name, mask in m.items():
        d = tot[mask]
        f = 82.0 / sched if name == "B" else 1.0
        ap, ag, aa = d.act_p * f, d.act_g * f, d.act_a * f
        e = d[col_p] - ap
        r = {"n": int(len(d)), "mae": float(e.abs().mean()),
             "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
             "corr": float(np.corrcoef(d[col_p], ap)[0, 1]),
             "mae_g": float((d[col_g] - ag).abs().mean()),
             "mae_a": float((d[col_a] - aa).abs().mean())}
        if name != "B":
            b = np.polyfit(d[col_p], ap, 1)
            r["calib_slope"] = float(b[0])           # actual on projected
        if "p_p10" in d.columns and name != "B":
            for s, a_ in (("p", ap), ("g", ag), ("a", aa)):
                ok = d[f"{s}_p10"].notna()
                r[f"cov80_{s}"] = float(((a_[ok] >= d[f"{s}_p10"][ok] - 1e-9)
                                         & (a_[ok] <= d[f"{s}_p90"][ok] + 1e-9)).mean())
        out[name] = r
    return out


def regression_slope(tot: pd.DataFrame, V: int) -> float:
    """Slope of projected P/GP on last season's P/GP for regulars (>=60 GP in
    V-1, restated per 82). A calibrated preseason projection should sit
    clearly below 1, because players do not repeat last season."""
    P = PL.panel()
    j = P.j(V - 1)
    lv = PL.league_levels()
    sched = float(lv.loc[V - 1, "games"])
    last = pd.DataFrame({"player_id": P.ids, "gp1": P.x["gp"][:, j] * 82 / sched,
                         "ppg1": sum(P.x[f"{s}_{k}"][:, j] for s in ("g", "a1", "a2") for k in PL.SITS)
                         / np.maximum(P.x["gp"][:, j], 1)})
    d = tot[tot.gp > 1].merge(last[last.gp1 >= 60], on="player_id")
    return float(np.polyfit(d.ppg1, d.p / d.gp, 1)[0])


# ---------------------------------------------------------------------------
# Marcel baseline
# ---------------------------------------------------------------------------
def marcel(V: int) -> pd.DataFrame:
    """Marcel-style baseline (orr.players.marcel_baseline)."""
    return PL.marcel_baseline(V)[["player_id", "gp", "g", "a", "p"]]


# ---------------------------------------------------------------------------
# Tuning (V <= 2017 only)
# ---------------------------------------------------------------------------
def score_config(prm: PL.Params, seasons=TUNE_V, protocol="strict") -> dict:
    rows = []
    for V in seasons:
        tot = run_season(V, prm, protocol)
        mm = metrics(tot)
        m = mm[OBJECTIVE_SAMPLE]
        rows.append((m["n"], m["mae"], m["bias"], mm["A"]["mae"]))
    n = sum(r[0] for r in rows)
    return {"mae": sum(r[0] * r[1] for r in rows) / n,
            "bias": sum(r[0] * r[2] for r in rows) / n,
            "mae_A": float(np.mean([r[3] for r in rows])),
            "per_season": {V: round(r[1], 4) for V, r in zip(seasons, rows)}}


def tune(max_configs: int = 20) -> PL.Params:
    """Coordinate search over a small grid, strict protocol, sample A MAE
    pooled over TUNE_V. Every configuration is appended to the ledger."""
    # Second search (objective: strict/all). It starts from the best rate
    # settings of the first search (strict/A), whose rate choices do not
    # depend on the games model, and re-sweeps every knob once.
    grid = {
        "kappa": [0.35, 0.5, 0.75],
        "rate_decay": [0.6, 0.75, 0.9],
        "fin_weight": [0.3, 0.6, 0.9],
        "toi_decay": [0.1, 0.2, 0.35],
        "k_toi_games": [3.0, 6.0, 12.0],
        "prosp_alpha": [0.4, 0.7, 1.0],
        "dress_noise": [1.0, 2.0, 4.0],
        "q_scale": [0.8, 1.0, 1.25],
    }
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    for r in ledger:
        r.setdefault("objective", "strict/A (superseded: selects on outcome)")
    best = PL.Params(kappa=0.5, rate_decay=0.75, fin_weight=0.6, toi_decay=0.2,
                     k_toi_games=6.0, prosp_alpha=0.4, dress_noise=2.0, q_scale=1.0)
    seen = {}

    def ev(p):
        k = json.dumps(asdict(p), sort_keys=True)
        if k not in seen:
            t = time.time()
            s = score_config(p)
            seen[k] = s
            ledger.append({"config": asdict(p), "objective": f"strict/{OBJECTIVE_SAMPLE}",
                           "tune_mae": round(s["mae"], 4), "tune_mae_A_strict": round(s["mae_A"], 4),
                           "tune_bias": round(s["bias"], 3), "per_season": s["per_season"],
                           "seconds": round(time.time() - t, 1)})
            print(f"  [{len(ledger):>2}] mae {s['mae']:.4f}  bias {s['bias']:+.2f}  "
                  f"{ {kk: getattr(p, kk) for kk in grid} }", flush=True)
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            LEDGER.write_text(json.dumps(ledger, indent=1))
        return seen[k]["mae"]

    cur = ev(best)
    for sweep in range(1):
        improved = False
        for name, vals in grid.items():
            for v in vals:
                if len(seen) >= max_configs:
                    break
                cand = replace(best, **{name: v})
                sc = ev(cand)
                if sc < cur - 1e-4:
                    cur, best, improved = sc, cand, True
        if not improved or len(seen) >= max_configs:
            break
    return best


def calibrate_mc(prm: PL.Params):
    """Fit the Monte Carlo spread parameters on the tuning window only, in
    sequence, each against the unconditional 'all' sample (strict):

    1. spell_len (mean injury-spell length) so 80% GP intervals cover 80%;
    2. usage_sd: log-SD of realised vs projected min/game for skaters with
       >= 40 GP (measured directly);
    3. sd_scale (multiplier on posterior talent SDs) so 80% intervals of
       season points cover 80%.
    """
    def cov(p, key):
        c, n = 0.0, 0
        for V in TUNE_V:
            tot = run_season(V, p, "strict", sims=300)
            m, _ = samples(tot)
            d = tot[m["all"]]
            act = {"gp": d.act_gp, "p": d.act_p}[key]
            ok = ((act >= d[f"{key}_p10"] - 1e-9) & (act <= d[f"{key}_p90"] + 1e-9)).mean()
            c += ok * len(d)
            n += len(d)
        return c / n
    log = []
    best = None
    for L in (14.0, 25.0, 40.0):
        c = cov(replace(prm, spell_len=L), "gp")
        log.append({"spell_len": L, "cov80_gp_all": round(c, 4)})
        print(f"  spell_len {L}: GP coverage {c:.3f}", flush=True)
        if best is None or abs(c - 0.8) < best[1]:
            best = (L, abs(c - 0.8))
    prm = replace(prm, spell_len=best[0])
    logs = []
    for V in TUNE_V:
        tot = run_season(V, prm, "strict")
        d = tot[(tot.act_gp >= 40) & tot.on_roster & (tot.gp > 1)]
        logs.append(np.log((d.act_toi / d.act_gp) / (d.toi / d.gp)))
    usage_sd = float(np.sqrt(max(np.concatenate(logs).var(), 1e-4)))
    prm = replace(prm, usage_sd=round(usage_sd, 4))
    best = None
    for sc in (2.0, 2.5, 3.0):
        c = cov(replace(prm, sd_scale=sc), "p")
        log.append({"sd_scale": sc, "cov80_p_all": round(c, 4)})
        print(f"  sd_scale {sc}: points coverage {c:.3f}", flush=True)
        if best is None or abs(c - 0.8) < best[1]:
            best = (sc, abs(c - 0.8))
    return replace(prm, sd_scale=best[0]), log, usage_sd


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
def pooled(rows, key):
    n = sum(r[key]["n"] for r in rows)
    out = {"n": n}
    for m in rows[0][key]:
        if m == "n":
            continue
        out[m] = sum(r[key][m] * r[key]["n"] for r in rows) / n
    return out


def report(prm: PL.Params, seasons, sims=400) -> dict:
    res = {"params": asdict(prm), "seasons": {}}
    for V in seasons:
        t = time.time()
        row = {}
        for proto in ("matched", "strict"):
            tot = run_season(V, prm, proto, sims=sims)
            row[proto] = metrics(tot)
            row[proto]["roster_proxy"] = tot.roster_proxy.iloc[0]
            row[proto]["slope_ppg_on_last"] = regression_slope(tot[tot.on_roster], V)
            row[proto]["n_offroster_in_A"] = int((samples(tot)[0]["A"] & ~tot.on_roster).sum())
            if proto == "strict":
                # rookies: no NHL history, >=40 GP, on an opening roster
                rk = tot[~tot.has_hist.fillna(False).astype(bool) & (tot.act_gp >= 40) & tot.on_roster]
                row["rookies_strict"] = {"n": int(len(rk)),
                                         "mae": float((rk.p - rk.act_p).abs().mean()) if len(rk) else None,
                                         "bias": float((rk.p - rk.act_p).mean()) if len(rk) else None}
        mc = marcel(V).merge(actuals(V), on="player_id")
        mt = run_season(V, prm, "matched")[["player_id", "has_hist", "on_roster"]]
        mc = mc.merge(mt, on="player_id", how="left")
        mc["on_roster"] = True
        mc["has_hist"] = mc.has_hist.fillna(False)
        row["marcel"] = metrics(mc)
        row["neurhl"] = {"A_player_backtest": NEURHL_A.get(V),
                         "v2_A_B_blend": NEURHL_V2.get(V)}
        res["seasons"][V] = row
        print(f"{V}: matched A {row['matched']['A']['mae']:.2f}  strict A "
              f"{row['strict']['A']['mae']:.2f}  marcel A {row['marcel']['A']['mae']:.2f}  "
              f"NeurHL A {NEURHL_A.get(V)}  ({time.time() - t:.0f}s)", flush=True)
    windows = (("tune", TUNE_V), ("confirm", CONFIRM_V), ("test", TEST_V),
               # held-out window split by roster proxy: 2022-24 strict is truly
               # preseason (first-10-game rosters); 2025-26 strict falls back to
               # season-V teams and is as leaky as NeurHL's own protocol
               ("test_first10_2022_24", [2022, 2023, 2024]),
               ("test_season_team_2025_26", [2025, 2026]))
    for name, vs in windows:
        rows = [res["seasons"][v] for v in vs if v in res["seasons"]]
        if not rows:
            continue
        res[f"pooled_{name}"] = {
            proto: {s: pooled([r[proto] for r in rows], s) for s in ("A", "B", "all")}
            for proto in ("matched", "strict")}
        res[f"pooled_{name}"]["marcel"] = {s: pooled([r["marcel"] for r in rows], s)
                                           for s in ("A", "B")}
        nA = [NEURHL_A[v] for v in vs if v in NEURHL_A]
        res[f"pooled_{name}"]["neurhl_A_mean"] = float(np.mean(nA)) if nA else None
        nw = [(NEURHL_A[v], NEURHL_A_N[v]) for v in vs if v in NEURHL_A]
        res[f"pooled_{name}"]["neurhl_A_pooled"] = (
            sum(a * n for a, n in nw) / sum(n for _, n in nw) if nw else None)
        nb = [(NEURHL_V2[v][2], res["seasons"][v]["matched"]["B"]["n"]) for v in vs if v in NEURHL_V2]
        res[f"pooled_{name}"]["neurhl_v2_blend_pooled"] = (
            sum(a * n for a, n in nb) / sum(n for _, n in nb) if nb else None)
    return res


def print_table(res: dict) -> None:
    print(f"\n{'V':>5} | {'NeurHL A':>8} {'ours M-A':>8} {'strict A':>8} {'Marcel A':>8} | "
          f"{'NeurHL bl':>9} {'ours M-B':>8} {'strict B':>8} | {'cov80 P':>7} {'proxy':>11}")
    for V, r in res["seasons"].items():
        nb = r["neurhl"]["v2_A_B_blend"]
        print(f"{V:>5} | {r['neurhl']['A_player_backtest'] or float('nan'):>8.2f} "
              f"{r['matched']['A']['mae']:>8.2f} {r['strict']['A']['mae']:>8.2f} "
              f"{r['marcel']['A']['mae']:>8.2f} | {nb[2] if nb else float('nan'):>9.2f} "
              f"{r['matched']['B']['mae']:>8.2f} {r['strict']['B']['mae']:>8.2f} | "
              f"{r['strict']['all'].get('cov80_p', float('nan')):>7.3f} {r['strict']['roster_proxy']:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--calibrate", action="store_true",
                    help="refit only the interval spread (tuning window)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.tune:
        best = tune()
        best, cal, usage_sd = calibrate_mc(best)
        prev = json.loads(PL.PARAMS_PATH.read_text()) if PL.PARAMS_PATH.exists() else {}
        prev.update({"hyper": asdict(best), "mc_calibration": cal,
                     "usage_sd_measured": usage_sd,
                     "tuned_on": TUNE_V, "objective": "strict protocol, sample A, pooled MAE"})
        D.write_json(prev, PL.PARAMS_PATH)
        print("tuned ->", PL.PARAMS_PATH)
    if args.calibrate:
        prm, cal, usage_sd = calibrate_mc(PL.load_params())
        prev = json.loads(PL.PARAMS_PATH.read_text())
        prev.update({"hyper": asdict(prm), "mc_calibration": cal,
                     "usage_sd_measured": usage_sd})
        D.write_json(prev, PL.PARAMS_PATH)
    prm = PL.load_params()
    res = report(prm, TUNE_V + CONFIRM_V + TEST_V, sims=args.sims)
    res["protocol_notes"] = __doc__
    D.write_json(res, OUT / "players_bt.json")
    print_table(res)
    for name in ("tune", "confirm", "test", "test_first10_2022_24", "test_season_team_2025_26"):
        if f"pooled_{name}" in res:
            pp = res[f"pooled_{name}"]
            print(f"POOLED {name:<26} matched A {pp['matched']['A']['mae']:.3f}  strict A "
                  f"{pp['strict']['A']['mae']:.3f}  Marcel A {pp['marcel']['A']['mae']:.3f}  "
                  f"NeurHL A {pp['neurhl_A_mean']}  | matched B {pp['matched']['B']['mae']:.3f} "
                  f"strict B {pp['strict']['B']['mae']:.3f} NeurHL blend "
                  f"{pp['neurhl_v2_blend_pooled']}")
    print("->", OUT / "players_bt.json")


if __name__ == "__main__":
    main()
