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
        # A real 5v5 unit is 3 forwards + 2 defencemen. Ranking all skaters by
        # TOI does NOT give that: defencemen play far more minutes, so the top 6
        # comes out at 3.2 D on average (measured across all 30 teams), which is
        # a defensively skewed unit that no team ever ices and which depresses
        # the offensive hazard.
        fw = d[d.pos_group == 0].nlargest(3, "toi_sec")
        df = d[d.pos_group == 1].nlargest(2, "toi_sec")
        gk = d[d.pos_group == 2].nlargest(1, "toi_sec")
        idx = [pmap.get(int(x), 0) for x in list(fw.player_id) + list(df.player_id)]
        idx = (idx + [0] * N_SK)[:N_SK]
        gi = pmap.get(int(gk.player_id.iloc[0]), 0) if len(gk) else 0
        out[int(team)] = [gi] + idx
    return out


def state_matched_contexts(season, eras, dev, n_per_cell=64, n_games=400,
                           seed=11):
    """Real histories INDEXED BY THE STATE THEY ALREADY HAVE.

    The first version of this fabricated state: it took arbitrary contexts and
    overwrote period/score/strength. That produces self-contradictory inputs --
    `period = 3` on a context whose clock still says early first period -- and
    the model, correctly, returns nonsense for them (measured: lambda_away 3.09
    against lambda_home 2.48 with mirrored identical units, and E[dt] collapsing
    from 12.2s to 7.8s).

    So nothing is overwritten except PERSONNEL. Contexts are selected because
    they already sit at the wanted (period, score, 5v5) state, which keeps every
    field mutually consistent and in-distribution. This is what "marginalise the
    history" should have meant all along: average over real histories that
    reached this state, not over invented ones.
    """
    rng = np.random.default_rng(seed)
    cells = {}
    n_games = min(n_games, season.n_games)
    d = season.d
    for g in range(n_games):
        a0, b0 = int(season.off[g]), int(season.off[g + 1])
        per = d["period"][a0:b0]
        sc = d["score_abs"][a0:b0].astype(int) - 4
        st = d["strength"][a0:b0]
        for pos in range(CTX_LEN, b0 - a0 - 1):
            if st[pos] != 0:
                continue
            key = (int(per[pos]), int(np.clip(sc[pos], -4, 4)))
            if key[0] < 1 or key[0] > 3:
                continue
            cells.setdefault(key, [])
            if len(cells[key]) < n_per_cell * 3:
                cells[key].append((g, pos))
    out = {}
    for key, lst in cells.items():
        if len(lst) < 2:
            continue
        pick = [lst[i] for i in rng.choice(len(lst),
                                           size=min(n_per_cell, len(lst)),
                                           replace=False)]
        out[key] = build_window(season, pick, eras, dev)
    return out


def build_window(season, picks, eras, dev):
    """A batch of CTX_LEN-token windows ENDING at each chosen position."""
    B = len(picks)
    T = CTX_LEN
    d = season.d
    I = ["etype", "team", "zone", "stype", "strength", "score", "period",
         "score_abs", "n_home", "n_away", "g_home", "g_away"]
    F = ["dt", "t_rem", "xa", "ya", "xg"]
    Bm = ["has_xy", "has_xg", "n_for", "n_against"]
    out = {k: np.zeros((B, T), np.int64) for k in I + Bm}
    out.update({k: np.zeros((B, T), np.float32) for k in F})
    on = np.zeros((B, T, 14), np.int64)
    pos_idx = np.zeros((B, T), np.int64)
    for i, (g, pos) in enumerate(picks):
        a0 = int(season.off[g])
        sl = slice(a0 + pos - T + 1, a0 + pos + 1)
        for k in I + Bm:
            out[k][i] = d[k][sl]
        for k in F:
            out[k][i] = d[k][sl]
        on[i] = d["on"][sl]
        pos_idx[i] = np.arange(pos - T + 1, pos + 1)
    r = {k: torch.from_numpy(v).to(dev) for k, v in out.items()}
    r["pos_idx"] = torch.from_numpy(pos_idx).to(dev)
    r["on_ctx"] = torch.from_numpy(on).to(dev)
    r["valid"] = torch.ones(B, T, dtype=torch.bool, device=dev)
    r["era"] = eras[season.se].to(dev).view(1, -1).expand(B, -1).contiguous()
    return r


@torch.no_grad()
def hazards_for_games(models, cells, units, rapm, dev, inv_tt, n_games):
    """(game, period, score) -> (lambda_home, lambda_away) goals per second.

    ONLY personnel are substituted into each state-matched context.
    """
    gh = [j for j, k in inv_tt.items() if k == "7|1"]
    ga = [j for j, k in inv_tt.items() if k == "7|0"]
    res = np.zeros((n_games, 3, len(SCORES), 2), np.float64)
    on_all = torch.tensor(units, dtype=torch.long, device=dev)

    for per in (1, 2, 3):
        for si, sc in enumerate(SCORES):
            ctx = cells.get((per, sc)) or cells.get((per, 0))
            if ctx is None:
                continue
            n_ctx, T = ctx["etype"].shape
            CH = max(1, 192 // max(n_ctx, 1))
            for lo in range(0, n_games, CH):
                hi = min(lo + CH, n_games)
                G = hi - lo
                b = {}
                for k, v in ctx.items():
                    if torch.is_tensor(v) and v.dim() == 3:
                        b[k] = v.repeat(G, 1, 1)
                    elif torch.is_tensor(v) and v.dim() == 2:
                        b[k] = v.repeat(G, 1)
                    else:
                        b[k] = v
                B = b["etype"].shape[0]
                sub = on_all[lo:hi].repeat_interleave(n_ctx, 0)
                b["on_ctx"] = sub.view(B, 1, 14).expand(B, T, 14).contiguous()
                ph = pa = edt = None
                for m in models:
                    h, _ = m.encode(b, rapm)
                    p = torch.softmax(m.h_tt(h), -1)
                    e = expected_dt(m.h_dt(h), m.c.k_dt)
                    a_, b_ = p[..., gh].sum(-1), p[..., ga].sum(-1)
                    ph = a_ if ph is None else ph + a_
                    pa = b_ if pa is None else pa + b_
                    edt = e if edt is None else edt + e
                nm = len(models)
                # rate = mean(p) / mean(E[dt]) -- a time-average. The mean of
                # ratios is not the rate, and with few contexts it is wildly
                # unstable: at 6 contexts the away estimate had sd 1.41 g/60.
                mp_h = (ph / nm)[:, -1].view(G, n_ctx).mean(1)
                mp_a = (pa / nm)[:, -1].view(G, n_ctx).mean(1)
                m_dt = (edt / nm)[:, -1].view(G, n_ctx).mean(1)
                lh, la = mp_h / m_dt, mp_a / m_dt
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
    cells = state_matched_contexts(s, eras, dev)

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
    print(f"state-matched context cells: {len(cells)} of 27 "
          f"(period x score, all 5v5)")
    H = hazards_for_games(models, cells, U, rapm, dev, inv_tt, len(gc))
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
