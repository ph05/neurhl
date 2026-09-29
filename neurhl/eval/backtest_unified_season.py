"""NeurHL 1.1 C2: historical backtest of the season model in the preseason
convention (PLAN_NeurHL_1_1, C2).

NeurHL 1.0 evaluates every 2026-27 game with opening-night states and turns
the per-game probabilities into season distributions with one team-strength
shock (sd 0.07 on the log goal rate, carried over from NeurHL-2 and never
calibrated for this engine). This script rebuilds that situation for past
seasons V, walk-forward, so the season layer can be calibrated and judged:

  engine    NeurHL-G configuration g1 trained on seasons < V (5 seeds, the
            live bundle's recipe), cached under data/tensors/_season_bt/
  states    every game of V is assembled from opening-night rows: each team's
            first game of V gives its skaters, goalie and team row (the
            opening-night lineup, used for every game: no in-season
            availability information), while rest, back-to-back, travel and
            time zones stay the game's own; days-into-season is the opening
            value; the Elo term is the frozen v1 rating entering season V
  final p   the live bundle's stack over Elo and NeurHL-G (the fallback stack
            of checkpoints/g/g2027_v1, because NeurHL-H has no preseason
            projection in history)
  sensitivity  d logit / d log(lambda_h / lambda_a) from each snapshot's own
            hazard integration, as sim/unified_2027.py computes it

Seasons 2025 and 2026 are sealed and refused. 2013 and 2021 are never scored.
Writes data/tensors/_season_bt/pre_<V>.parquet (per-game preseason outputs);
eval/season_layer_c2.py does the season simulation and scoring.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402
from train.train_neurhl_g import Data, predict, train_snapshot  # noqa: E402

OUT = TENSORS / "_season_bt"
SEALED = {2025, 2026}
SEASONS = [2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024]
SCHED_COLS = ["rest", "b2b", "km3d", "dtz"]
ALIAS = {"PHX": "UTA", "ARI": "UTA", "ATL": "WPG"}   # franchise names in data/processed/games.csv


def preseason_elo(V: int) -> dict:
    """Frozen v1 Elo rating entering season V, by NHL abbreviation."""
    sys.path.insert(0, str(PROJ / "src"))
    import engine as E
    v1 = json.loads((PROJ / "output" / "params.json").read_text())
    g = pd.read_csv(PROJ / "data" / "processed" / "games.csv", keep_default_na=False,
                    parse_dates=["date"])
    g = g[g.season_end <= V].sort_values("date").reset_index(drop=True)
    _, _, pre = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"], expansion_init=v1["expansion_init"])
    r = dict(pre[V])
    # an expansion team (VGK 2018, SEA 2022) enters at the frozen v1 expansion rating
    for t in set(g[g.season_end == V].home) | set(g[g.season_end == V].away):
        r.setdefault(t, v1["expansion_init"])
    return r, v1["H"]


def preseason_arrays(D, P: dict, idx: np.ndarray, elo_pre_raw: np.ndarray, stats: dict) -> dict:
    """Copy of P's rows for games idx with every team's first-game rows."""
    meta = D.meta.iloc[idx].reset_index(drop=True)
    tf = D.names["tm_feat"]
    sch = [tf.index(c) for c in SCHED_COLS]
    ctx = D.names["ctx"]
    first = {}
    for j, r in enumerate(meta.itertuples()):
        for s, t in ((0, r.home_idx), (1, r.away_idx)):
            first.setdefault(int(t), (j, s))
    out = {k: v[idx].copy() for k, v in P.items()}
    per_side = ["SK", "SKB", "SKM", "SKP", "RAPM", "GK", "GKR", "TM", "TMR", "HR"]
    for j, r in enumerate(meta.itertuples()):
        for s, t in ((0, r.home_idx), (1, r.away_idx)):
            fj, fs = first[int(t)]
            for k in per_side:
                if k in out:
                    keep = out[k][j, s].copy() if k == "TM" else None
                    out[k][j, s] = P[k][idx[fj], fs]
                    if k == "TM":
                        out[k][j, s, sch] = keep[sch]
    mu, sd = stats["CTX"]
    ie, idn = ctx.index("elo_logit"), ctx.index("days_in")
    out["CTX"][:, ie] = np.clip((elo_pre_raw - mu[ie]) / sd[ie], -8, 8)
    out["CTX"][:, idn] = out["CTX"][int(np.argmin(meta.date.values)), idn]
    out["ELO"] = elo_pre_raw.astype(np.float32)
    return out


def sensitivity(model, lh, la, eps=0.05):
    lh, la = torch.as_tensor(lh, dtype=torch.float32), torch.as_tensor(la, dtype=torch.float32)
    up, dn = np.exp(eps / 2), np.exp(-eps / 2)
    with torch.no_grad():
        p1 = model.outcome(lh * up, la * dn)["p_home_win"].numpy()
        p0 = model.outcome(lh * dn, la * up)["p_home_win"].numpy()
    lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))  # noqa: E731
    return (lg(p1) - lg(p0)) / eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    assert not set(a.seasons) & SEALED, "sealed seasons requested"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs" / "neurhl_g" / "g1.json").read_text())
    bundle = json.loads((ROOT / "checkpoints" / "g" / "g2027_v1" / "bundle.json").read_text())
    fb = bundle["stack"]["fallback"]
    tidx = json.loads((TENSORS / "maps.json").read_text())["team"]
    abbr = {v: ALIAS.get(k, k) for k, v in tidx.items()}
    D = Data("train")
    s_all = D.meta.season_end.to_numpy()
    for V in a.seasons:
        dst = OUT / f"pre_{V}.parquet"
        if dst.exists():
            print(f"[c2] {V}: cached", flush=True)
            continue
        t0 = time.time()
        idx = np.where(s_all == V)[0]
        idx = idx[np.argsort(D.meta.date.values[idx], kind="stable")]
        meta = D.meta.iloc[idx].reset_index(drop=True)
        pre, H = preseason_elo(V)
        miss = sorted({abbr[int(t)] for t in np.r_[meta.home_idx, meta.away_idx]} - set(pre))
        assert not miss, f"{V}: no preseason Elo for {miss}"
        rh = meta.home_idx.map(lambda t: pre[abbr[int(t)]]).to_numpy()
        ra = meta.away_idx.map(lambda t: pre[abbr[int(t)]]).to_numpy()
        elo_pre = (rh + H - ra) * np.log(10) / 400.0
        acc = []
        for sd in range(a.seeds):
            ck = OUT / f"snap_{V}_{sd}.pt"
            model, P = train_snapshot(D, V, {**cfg, "seed": sd})
            torch.save(model.state_dict(), ck)
            stats = dict(D.stats)
            Q = preseason_arrays(D, P, idx, elo_pre, stats)
            o = predict(model, Q, np.arange(len(idx)))
            k = sensitivity(model, o["goals"][:, 0], o["goals"][:, 1])
            acc.append({"p_g": o["p_home_win"], "o4": o["o4"], "goals": o["goals"], "k": k})
            print(f"[c2] {V} seed {sd}: {time.time() - t0:.0f}s", flush=True)
        p_g = np.mean([x["p_g"] for x in acc], 0)
        o4 = np.mean([x["o4"] for x in acc], 0)
        goals = np.mean([x["goals"] for x in acc], 0)
        kk = np.mean([x["k"] for x in acc], 0)
        lg = np.log(p_g / (1 - p_g))
        z = fb["intercept"] + fb["coef"][0] * elo_pre + fb["coef"][1] * lg
        p = 1 / (1 + np.exp(-z))
        o4s = o4.copy()                                 # outcome4 rescaled to the stacked p (as 1.0)
        hw = o4[:, 0] + o4[:, 2]
        o4s[:, [0, 2]] *= (p / hw)[:, None]
        o4s[:, [1, 3]] *= ((1 - p) / (1 - hw))[:, None]
        df = pd.DataFrame({"game_id": meta.game_id, "date": meta.date, "season_end": V,
                           "home_idx": meta.home_idx, "away_idx": meta.away_idx,
                           "outcome4": meta.outcome4, "elo_pre": elo_pre, "p_g": p_g, "p": p,
                           "o4_hr": o4s[:, 0], "o4_ar": o4s[:, 1], "o4_ho": o4s[:, 2],
                           "o4_ao": o4s[:, 3], "goals_h": goals[:, 0], "goals_a": goals[:, 1],
                           "k": kk})
        df.to_parquet(dst, index=False)
        print(f"[c2] {V}: {len(df)} games, mean p {p.mean():.3f}, sd logit {np.std(np.log(p/(1-p))):.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
