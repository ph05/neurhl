"""NeurHL-2 — score games from the S4 integration and report the primary metric.

Per-game log loss is THE headline (PLAN_NeurHL2 O). Everything here is arranged
so the number is a genuine forecast:

  * personnel come from the PRIOR season's TOI-weighted unit and the walk-forward
    RAPM prior for that vantage, never from the game being scored;
  * event histories used to query S1 are drawn from OTHER games, so no part of
    the target game enters the hazard;
  * the score-effect grid is enumerated, so P(win) integrates the same
    score-dependence E2 verified at ratio 1.04.

Reported against the full baseline ladder the plan requires, never a single
comparison: constant home rate, v1 Elo, and the observed base rate. A win that
does not beat the constant baseline is not a model.

Run: uv run --no-project --python 3.12 --with numpy --with torch \
     --with "pandas<3" --with pyarrow python neurhl/eval/score_games.py \
     --season 2017 --max-games 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS, CKPT  # noqa: E402
from eval.validate_event_sim import load_model  # noqa: E402
from train.train_event_sim import (Season, era_vectors, make_batch,  # noqa: E402
                                   device, TRAIN, VAL)
from sim.game_integrate import (expected_dt, integrate, outcome_probs,  # noqa: E402
                                SCORES)
import windows as W  # noqa: E402

CTX_LEN = 24          # hazards depend on recent history, not the whole game
N_CTX = 4             # histories marginalised per query
N_SK = 6


def projected_units(season: int, pmap: dict) -> dict:
    """team -> [goalie, 6 skaters] player indices, from the PRIOR season only."""
    prev = season - 1
    pg = TENSORS / f"player_games_{prev}.parquet"
    gc = TENSORS / f"games_ctx_{prev}.parquet"
    pr = TENSORS / f"rapm_prior_{season}.parquet"
    if not (pg.exists() and gc.exists()):
        return {}
    p = pd.read_parquet(pg, columns=["game_id", "player_id", "is_home",
                                     "toi_sec", "pos_group", "game_type"])
    p = p[p.game_type == 2]
    g = pd.read_parquet(gc, columns=["game_id", "home_idx", "away_idx"])
    p = p.merge(g, on="game_id", how="left")
    p["team"] = np.where(p.is_home, p.home_idx, p.away_idx)
    agg = (p.groupby(["team", "player_id", "pos_group"], as_index=False)
           .toi_sec.sum())
    out = {}
    for team, d in agg.groupby("team"):
        sk = d[d.pos_group != 2].nlargest(N_SK, "toi_sec")
        gk = d[d.pos_group == 2].nlargest(1, "toi_sec")
        idx = [pmap.get(int(x), 0) for x in sk.player_id]
        idx = (idx + [0] * N_SK)[:N_SK]
        gi = pmap.get(int(gk.player_id.iloc[0]), 0) if len(gk) else 0
        out[int(team)] = [gi] + idx
    return out


def context_bank(season_obj, eras, dev, n=N_CTX, seed=11):
    """Short real histories from arbitrary games, used to marginalise history."""
    rng = np.random.default_rng(seed)
    picks = [(season_obj, int(g)) for g in
             rng.integers(0, season_obj.n_games, size=n)]
    b = make_batch([season_obj], picks, eras, dev)
    out = {}
    for k, v in b.items():
        out[k] = v[:, :CTX_LEN].contiguous() if (torch.is_tensor(v) and v.dim() >= 2
                                                 and v.shape[1] > CTX_LEN) else v
    return out


@torch.no_grad()
def hazards_for_games(models, ctx, units, era, rapm, dev, inv_tt, n_games):
    """(n_games, period, score) -> (lambda_home, lambda_away) per second."""
    gh = [j for j, k in inv_tt.items() if k == "7|1"]
    ga = [j for j, k in inv_tt.items() if k == "7|0"]
    n_ctx = ctx["etype"].shape[0]
    T = ctx["etype"].shape[1]
    res = np.zeros((n_games, 3, len(SCORES), 2), np.float64)

    on_all = torch.tensor(units, dtype=torch.long, device=dev)   # [G,14]
    CH = max(1, 256 // max(n_ctx, 1))
    for per in (1, 2, 3):
        for si, sc in enumerate(SCORES):
            for lo in range(0, n_games, CH):
                hi = min(lo + CH, n_games)
                G = hi - lo
                b = {}
                for k, v in ctx.items():
                    if torch.is_tensor(v) and v.dim() == 2:
                        b[k] = v.repeat(G, 1)
                    elif torch.is_tensor(v) and v.dim() == 3:
                        b[k] = v.repeat(G, 1, 1)
                    elif torch.is_tensor(v):
                        b[k] = v.repeat(G, 1) if v.dim() == 2 else v
                B = b["etype"].shape[0]
                on = on_all[lo:hi].repeat_interleave(n_ctx, 0)
                b["on_ctx"] = on.view(B, 1, 14).expand(B, T, 14).contiguous()
                b["score_abs"] = torch.full((B, T), sc + 4, dtype=torch.long, device=dev)
                b["n_home"] = torch.full((B, T), 5, dtype=torch.long, device=dev)
                b["n_away"] = torch.full((B, T), 5, dtype=torch.long, device=dev)
                b["g_home"] = torch.ones(B, T, dtype=torch.long, device=dev)
                b["g_away"] = torch.ones(B, T, dtype=torch.long, device=dev)
                b["period"] = torch.full((B, T), per, dtype=torch.long, device=dev)
                b["era"] = era.view(1, -1).expand(B, -1).contiguous()
                b["valid"] = torch.ones(B, T, dtype=torch.bool, device=dev)

                ph = pa = edt = None
                for m in models:
                    h, _ = m.encode(b, rapm)
                    p = torch.softmax(m.h_tt(h), -1)
                    e = expected_dt(m.h_dt(h), m.c.k_dt)
                    a_ = p[..., gh].sum(-1)
                    b_ = p[..., ga].sum(-1)
                    ph = a_ if ph is None else ph + a_
                    pa = b_ if pa is None else pa + b_
                    edt = e if edt is None else edt + e
                nm = len(models)
                lh = ((ph / nm) / (edt / nm))[:, -1].view(G, n_ctx).mean(1)
                la = ((pa / nm) / (edt / nm))[:, -1].view(G, n_ctx).mean(1)
                res[lo:hi, per - 1, si, 0] = lh.float().cpu().numpy()
                res[lo:hi, per - 1, si, 1] = la.float().cpu().numpy()
    return res


def logloss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=VAL)
    ap.add_argument("--max-games", type=int, default=200)
    ap.add_argument("--ckpts", type=str, nargs="*", default=None)
    args = ap.parse_args()
    if args.season in W.CONFIRM:
        raise SystemExit(f"{args.season} is in CONFIRM — not scorable here")

    dev = device()
    paths = ([Path(x) for x in args.ckpts] if args.ckpts
             else sorted(CKPT.glob("s1b_seed*.pt")))
    models = [load_model(p, dev)[0] for p in paths]
    print(f"ensemble of {len(models)}: {', '.join(p.name for p in paths)}")

    vocab = json.loads((TENSORS / "seq_vocab.json").read_text())
    pmap = {int(k): v for k, v in vocab["players"].items()}
    inv_tt = {v: k for k, v in vocab["type_team"].items()}
    s = Season(args.season)
    eras, _ = era_vectors(TRAIN + [VAL])
    rapm = s.rapm.to(dev)
    ctx = context_bank(s, eras, dev)

    units = projected_units(args.season, pmap)
    gc = pd.read_parquet(TENSORS / f"games_ctx_{args.season}.parquet")
    gc = gc[gc.game_type == 2].head(args.max_games).reset_index(drop=True)
    print(f"scoring {len(gc)} games of {args.season}; projected units for "
          f"{len(units)} teams (prior season TOI)")

    dflt = [0] * 7
    U = []
    for r in gc.itertuples():
        U.append(units.get(int(r.home_idx), dflt) + units.get(int(r.away_idx), dflt))
    t0 = time.time()
    H = hazards_for_games(models, ctx, U, eras[args.season].to(dev), rapm, dev,
                          inv_tt, len(gc))
    print(f"hazards in {time.time()-t0:.0f}s  "
          f"(median lambda_home {np.median(H[:, :, :, 0])*3600:.2f} g/60, "
          f"away {np.median(H[:, :, :, 1])*3600:.2f} g/60)")

    ps, rows = [], []
    for i in range(len(gc)):
        grid = {(per, sc): (H[i, per - 1, si, 0], H[i, per - 1, si, 1])
                for per in (1, 2, 3) for si, sc in enumerate(SCORES)}
        P = integrate(grid)
        o = outcome_probs(P)
        ps.append(o["p_home_win"])
        rows.append(o)
    ps = np.array(ps)
    y = (gc.home_g.to_numpy() > gc.away_g.to_numpy()).astype(float)

    base = float(y.mean())
    ll_model = logloss(y, ps)
    ll_const = logloss(y, np.full(len(y), base))
    eg_h = np.mean([r["exp_goals_home"] for r in rows])
    eg_a = np.mean([r["exp_goals_away"] for r in rows])
    print(f"\nn={len(y)}  home win rate {base:.4f}  mean p_home {ps.mean():.4f}")
    print(f"expected goals/game  home {eg_h:.2f}  away {eg_a:.2f}  "
          f"(actual {gc.home_g.mean():.2f} / {gc.away_g.mean():.2f})")
    print(f"log loss  MODEL {ll_model:.5f}   constant-base {ll_const:.5f}   "
          f"delta {ll_model - ll_const:+.5f}")
    print(f"p_home spread: sd {ps.std():.4f}  min {ps.min():.3f}  max {ps.max():.3f}")

    out = {"season": args.season, "n": len(y), "ll_model": ll_model,
           "ll_constant": ll_const, "base_rate": base,
           "mean_p_home": float(ps.mean()), "sd_p_home": float(ps.std()),
           "exp_goals_home": float(eg_h), "exp_goals_away": float(eg_a),
           "actual_goals_home": float(gc.home_g.mean()),
           "actual_goals_away": float(gc.away_g.mean()),
           "ensemble": [p.name for p in paths]}
    q = Path(__file__).resolve().parents[1] / "configs" / "s4_game_scores.json"
    q.write_text(json.dumps(out, indent=1))
    print(f"-> {q}")


if __name__ == "__main__":
    main()
