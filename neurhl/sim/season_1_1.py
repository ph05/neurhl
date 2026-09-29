"""NeurHL 1.1 C2: the 2026-27 season layer rerun with the adopted variant,
issued as a dated file set beside the NeurHL 1.0 freeze (PLAN_NeurHL_1_1 A1).

Reads the frozen games file, neurhl/output/neurhl_1_0/games_2027.csv. Each
game's stacked probability, outcome4 and rate sensitivity are used unchanged.
Only the season simulation changes:
  - team strength s(week) = s0 + random walk (weekly sd sw), s0 ~ N(0, sigma0);
  - each game's logit shrunk toward the league mean logit by rho ** week.
Playoff series use each team's end-of-season strength. Standings rules,
tiebreakers, bracket and Bradley-Terry ratings are as sim/unified_2027.py.

Writes neurhl/output/neurhl_1_0/season_<tag>/teams_2027.csv and
team_points_quantiles_2027.csv, plus run.json. Usage:
  season_1_1.py --sigma0 S --sw W --rho R --tag 20260929
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sim.unified_2027 as U  # noqa: E402

SRC = ROOT / "output" / "neurhl_1_0"


def season_mc_dyn(games, p, o4, kg, teams, div, conf, sims, seed, sigma0, sw, rho):
    rng = np.random.default_rng(seed)
    tie_rng = np.random.default_rng(seed + 7)
    po_rng = np.random.default_rng(seed + 11)
    T, N = len(teams), len(games)
    ti = {t: i for i, t in enumerate(teams)}
    hi = games.home.map(ti).to_numpy()
    ai = games.away.map(ti).to_numpy()
    wk = ((pd.to_datetime(games.date) - pd.to_datetime(games.date).min()).dt.days // 7).to_numpy()
    W = int(wk.max()) + 1
    H = np.zeros((N, T), np.float32)
    H[np.arange(N), hi] = 1
    Aw = np.zeros((N, T), np.float32)
    Aw[np.arange(N), ai] = 1
    lp = U.logit(p)
    c = lp.mean()
    base = c + (lp - c) * rho ** wk
    rh = o4[:, 0] / np.maximum(o4[:, 0] + o4[:, 2], 1e-9)
    ra = o4[:, 1] / np.maximum(o4[:, 1] + o4[:, 3], 1e-9)
    stats = {k: np.zeros((sims, T), np.int16) for k in ("pts", "w", "rw", "row", "otl")}
    end_s = np.zeros((sims, T), np.float32)
    C = 1000
    for c0 in range(0, sims, C):
        n = min(C, sims - c0)
        s0 = rng.normal(0.0, sigma0, (n, T)) if sigma0 > 0 else np.zeros((n, T))
        if sw > 0:
            walk = np.cumsum(rng.normal(0.0, sw, (n, W, T)), axis=1)
            walk = np.concatenate([np.zeros((n, 1, T)), walk[:, :-1]], axis=1)
            S = s0[:, None, :] + walk
            sh, sa = S[:, wk, hi], S[:, wk, ai]
            end_s[c0:c0 + n] = S[:, -1, :]
        else:
            sh, sa = s0[:, hi], s0[:, ai]
            end_s[c0:c0 + n] = s0
        pp = U.sigmoid(base[None, :] + kg[None, :] * (sh - sa))
        hw = rng.random((n, N)) < pp
        reg = np.where(hw, rng.random((n, N)) < rh[None, :], rng.random((n, N)) < ra[None, :])
        so = rng.random((n, N)) < U.P_SO_GIVEN_TIE
        f = lambda m_h, m_a: (m_h.astype(np.float32) @ H + m_a.astype(np.float32) @ Aw)  # noqa: E731
        stats["w"][c0:c0 + n] = f(hw, ~hw)
        stats["rw"][c0:c0 + n] = f(hw & reg, ~hw & reg)
        stats["row"][c0:c0 + n] = f(hw & (reg | ~so), ~hw & (reg | ~so))
        stats["otl"][c0:c0 + n] = f(~hw & ~reg, hw & ~reg)
    stats["pts"] = 2 * stats["w"] + stats["otl"]
    key = (stats["pts"].astype(np.float64) * 1e7 + stats["rw"] * 1e5 + stats["row"] * 1e3
           + stats["w"] * 10 + tie_rng.random((sims, T)))
    divs = sorted(set(div[t] for t in teams))
    res = {k: np.zeros(T) for k in ("playoff", "division", "presidents", "r2", "cf", "final", "cup")}
    r, hedge = U.bt_ratings(games, teams, lp)
    kbar = float(np.median(kg))
    res["presidents"] += np.bincount(key.argmax(1), minlength=T)
    teams_of_div = {d: np.array([ti[t] for t in teams if div[t] == d]) for d in divs}
    for s_i in range(sims):
        ks = key[s_i]
        bracket = {}
        for cf in ("E", "W"):
            dvs = [d for d in divs if conf[teams[teams_of_div[d][0]]] == cf]
            tops, rest = {}, []
            for d in dvs:
                ids = teams_of_div[d][np.argsort(-ks[teams_of_div[d]])]
                tops[d] = ids[:3]
                res["division"][ids[0]] += 1
                rest.extend(ids[3:])
            rest = np.array(rest)
            wc = rest[np.argsort(-ks[rest])][:2]
            q = np.r_[np.concatenate([tops[d] for d in dvs]), wc]
            res["playoff"][q] += 1
            d1, d2 = sorted(dvs, key=lambda d: -ks[tops[d][0]])
            bracket[cf] = [((tops[d1][0], wc[1]), (tops[d1][1], tops[d1][2])),
                           ((tops[d2][0], wc[0]), (tops[d2][1], tops[d2][2]))]
        sh_ = end_s[s_i]

        def play(a, b):
            hi_, lo_ = (a, b) if ks[a] > ks[b] else (b, a)
            d_ = r[hi_] - r[lo_] + kbar * (sh_[hi_] - sh_[lo_])
            pw = U.series_prob(U.sigmoid(hedge + d_), U.sigmoid(-hedge + d_))
            return hi_ if po_rng.random() < pw else lo_

        champs = []
        for cf in ("E", "W"):
            sides = []
            for (m1, m2) in bracket[cf]:
                a_, b_ = play(*m1), play(*m2)
                res["r2"][[a_, b_]] += 1
                sides.append(play(a_, b_))
            res["cf"][sides] += 1
            champs.append(play(*sides))
        res["final"][champs] += 1
        res["cup"][play(*champs)] += 1
    for k in res:
        res[k] = res[k] / sims
    return stats, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma0", type=float, required=True)
    ap.add_argument("--sw", type=float, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=711)
    a = ap.parse_args()
    g = pd.read_csv(SRC / "games_2027.csv")
    frozen = pd.read_csv(SRC / "teams_2027.csv")
    teams = sorted(set(g.home))
    div, conf = U.divisions()
    o4 = g[["p_home_reg", "p_away_reg", "p_home_ot", "p_away_ot"]].to_numpy()
    stats, res = season_mc_dyn(g, g.p_home_win.to_numpy(), o4, g.rate_sensitivity.to_numpy(), teams,
                               div, conf, a.sims, a.seed, a.sigma0, a.sw, a.rho)
    pts = stats["pts"].astype(float)
    tm = frozen.set_index("team").loc[teams].reset_index()
    tm["points"] = pts.mean(0)
    for q in (10, 50, 90):
        tm[f"points_p{q}"] = np.percentile(pts, q, axis=0)
    for k in ("w", "otl", "rw"):
        tm[k] = stats[k].mean(0)
    tm["l"] = tm.gp - tm.w - tm.otl
    for k, lab in (("playoff", "playoff_pct"), ("division", "division_pct"),
                   ("presidents", "presidents_pct"), ("r2", "round2_pct"), ("cf", "conf_final_pct"),
                   ("final", "cup_final_pct"), ("cup", "cup_pct")):
        tm[lab] = res[k] * 100
    tm = tm.sort_values("points", ascending=False)
    pq = pd.DataFrame(np.percentile(pts, np.arange(1, 100), axis=0).T,
                      columns=[f"q{q:02d}" for q in range(1, 100)])
    pq.insert(0, "team", teams)
    out = SRC / f"season_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    tm.to_csv(out / "teams_2027.csv", index=False, float_format="%.4f")
    pq.to_csv(out / "team_points_quantiles_2027.csv", index=False, float_format="%.2f")
    lp = 2 * len(g) + tm.otl.sum()
    (out / "run.json").write_text(json.dumps({"sigma0": a.sigma0, "sw": a.sw, "rho": a.rho, "sims": a.sims,
                                              "seed": a.seed, "source": "neurhl/output/neurhl_1_0/games_2027.csv",
                                              "league_points": float(tm.points.sum()),
                                              "identity_2N_plus_otl": float(lp)}, indent=1))
    for col, slots in (("playoff_pct", 16), ("division_pct", 4), ("cup_pct", 1)):
        assert abs(tm[col].sum() / 100 - slots) < 1e-6, col
    assert abs(tm.points.sum() - lp) < 0.05
    print(tm[["team", "points", "points_p10", "points_p90", "playoff_pct", "cup_pct"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
