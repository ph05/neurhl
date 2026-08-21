"""NeurHL — evaluation of the player-game rate heads (H4).

The house models predict team outcomes only; per-player game rates are a
capability NeurHL adds rather than one it competes on. This scores head H4 for
one predict-season against the honest baseline: each player's OWN pre-game
EWMA rate, which the network also receives as input (pctx.ewma_toi) — so
beating it means the network learned something beyond "this player's recent
usage", e.g. opponent, teammates, rest, or role context.

Metrics per target:
  toi_share  MAE and Pearson r vs realised share of team skater TOI
  shots      MAE and Poisson deviance vs realised shots on goal
  P(goal)    log loss and Brier vs realised "scored >= 1"
  P(assist)  log loss and Brier vs realised "assisted >= 1"

Baselines: player pre-game EWMA (shifted, P1-clean) for toi/shots; for the
binary targets, the EWMA of the player's own historical rate. Writes
neurhl/output/player_eval_<T>.json and prints a table.

Usage: ... python neurhl/eval/backtest_player.py --predict-season 2017
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, NOUT, TENSORS  # noqa: E402
from models.game_model import from_config  # noqa: E402
from train.train_game import (EWMA_ALPHA, add_form, build_tensors,  # noqa: E402
                              load_frames, team_form)

SEEDS = [0, 1, 2, 3, 4]


def ewma_baselines(pg: pd.DataFrame, T: int) -> pd.DataFrame:
    """Per (game, player) pre-game EWMA of shots / goal-rate / assist-rate."""
    p = pg[pg.pos_group < 2].sort_values(["player_id", "date", "game_id"]).copy()
    grp = p.groupby("player_id")
    p["b_shots"] = grp.sog.transform(
        lambda s: s.ewm(alpha=EWMA_ALPHA).mean().shift(1))
    p["b_goal"] = grp.goals.transform(
        lambda s: (s > 0).astype(float).ewm(alpha=EWMA_ALPHA).mean().shift(1))
    p["b_ast"] = grp.assists.transform(
        lambda s: (s > 0).astype(float).ewm(alpha=EWMA_ALPHA).mean().shift(1))
    p["b_toi"] = grp.toi_sec.transform(
        lambda s: s.ewm(alpha=EWMA_ALPHA).mean().shift(1))
    return p[p.season_end == T][["game_id", "player_id", "is_home", "b_shots",
                                 "b_goal", "b_ast", "b_toi"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-season", type=int, required=True)
    args = ap.parse_args()
    T = args.predict_season
    cfg = json.loads((TENSORS.parents[1] / "configs" / "game_model.json")
                     .read_text())
    gc, pg = load_frames(list(range(cfg.get("train_from", 2009), T + 1)))
    pg = add_form(pg)
    form = team_form(gc) if cfg.get("use_team_form") else None
    npz = np.load(TENSORS / f"embeddings_v{T}.npz")
    te, te_ids, _ = build_tensors(gc[gc.season_end == T], pg, npz, False, form)

    # ---- ensemble forward, keep the player head
    acc = None
    for s in SEEDS:
        ck = CKPT / f"game_T{T}_s{s}.pt"
        if not ck.exists():
            continue
        m = from_config()
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        outs = {"home": [], "away": []}
        with torch.no_grad():
            for i in range(0, len(te_ids), 256):
                b = {k: v[i:i + 256] for k, v in te.items()}
                o = m(b)["p4"]
                for side in ("home", "away"):
                    outs[side].append(o[side])
        cur = {k: torch.cat(v) for k, v in outs.items()}
        acc = cur if acc is None else {k: acc[k] + cur[k] for k in cur}
    n_models = sum((CKPT / f"game_T{T}_s{s}.pt").exists() for s in SEEDS)
    if not n_models:
        print(f"no game_T{T} checkpoints found")
        sys.exit(1)
    pred = {k: v / n_models for k, v in acc.items()}

    base = ewma_baselines(pg, T).set_index(["game_id", "player_id"])
    rows = []
    for si, (side, gh) in enumerate((("home", "h"), ("away", "a"))):
        o = pred[side]                                   # N,20,4
        sk = te[f"skater_mask_{gh}"]
        toi = te[f"toi_{gh}"]
        share_true = (toi / toi.sum(-1, keepdim=True).clamp(min=1))
        logit = o[..., 0].masked_fill(~sk, -1e30)
        share_pred = torch.softmax(logit, -1)
        # honest baseline: the player's OWN pre-game EWMA ice time, renormalised
        # over the same dressed skaters (this is an input the network also sees)
        ew = te["pctx"][:, si, :, 3] * 20.0
        ew = ew.masked_fill(~sk, 0.0)
        share_ewma = ew / ew.sum(-1, keepdim=True).clamp(min=1e-6)
        rate = F.softplus(o[..., 1]) + 1e-4
        p_goal = torch.sigmoid(o[..., 2])
        p_ast = torch.sigmoid(o[..., 3])
        m = sk.numpy()
        rows.append(pd.DataFrame({
            "share_pred": share_pred.numpy()[m], "share_true": share_true.numpy()[m],
            "share_ewma": share_ewma.numpy()[m],
            "shots_pred": rate.numpy()[m], "shots_true": te[f"shots_{gh}"].numpy()[m],
            "pg": p_goal.numpy()[m], "goal": (te[f"goals_p_{gh}"].numpy()[m] > 0),
            "pa": p_ast.numpy()[m], "ast": (te[f"assists_p_{gh}"].numpy()[m] > 0),
            "game_id": np.repeat(te_ids, 20)[m.reshape(-1)],
            "player_id": te["pids"][:, si, :].numpy()[m],
        }))
    d = pd.concat(rows, ignore_index=True)
    d = d.join(base, on=["game_id", "player_id"])
    cov = float(d.b_shots.notna().mean())
    d = d.dropna(subset=["b_shots", "b_goal", "b_ast"])

    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    # every metric carries BOTH a naive baseline and the strong one (the
    # player's own pre-game EWMA, which the network receives as input)
    out = {
        "season": T, "n_player_games": int(len(d)), "n_models": n_models,
        "ewma_join_coverage": cov,
        "toi_share": {
            "mae_model": float(np.abs(d.share_pred - d.share_true).mean()),
            "mae_ewma_baseline": float(np.abs(d.share_ewma - d.share_true).mean()),
            "mae_equal_split": float(np.abs(1.0 / 18 - d.share_true).mean()),
            "r_model": float(np.corrcoef(d.share_pred, d.share_true)[0, 1]),
            "r_ewma": float(np.corrcoef(d.share_ewma, d.share_true)[0, 1])},
        "shots": {
            "mae_model": float(np.abs(d.shots_pred - d.shots_true).mean()),
            "mae_ewma_baseline": float(np.abs(d.b_shots - d.shots_true).mean()),
            "r_model": float(np.corrcoef(d.shots_pred, d.shots_true)[0, 1]),
            "r_ewma": float(np.corrcoef(d.b_shots, d.shots_true)[0, 1]),
            "mean_pred": float(d.shots_pred.mean()),
            "mean_true": float(d.shots_true.mean())},
        "p_goal": {
            "logloss_model": ll(d.pg.to_numpy(), d.goal.to_numpy()),
            "logloss_ewma_baseline": ll(d.b_goal.to_numpy(), d.goal.to_numpy()),
            "logloss_baserate": ll(np.full(len(d), d.goal.mean()),
                                   d.goal.to_numpy()),
            "base_rate": float(d.goal.mean())},
        "p_assist": {
            "logloss_model": ll(d.pa.to_numpy(), d.ast.to_numpy()),
            "logloss_ewma_baseline": ll(d.b_ast.to_numpy(), d.ast.to_numpy()),
            "logloss_baserate": ll(np.full(len(d), d.ast.mean()),
                                   d.ast.to_numpy()),
            "base_rate": float(d.ast.mean())}}
    out["verdict"] = {
        k: ("model beats own-EWMA baseline" if better else
            "NULL: no better than the player's own recent form")
        for k, better in (
            ("toi_share", out["toi_share"]["mae_model"]
             < out["toi_share"]["mae_ewma_baseline"]),
            ("shots", out["shots"]["mae_model"]
             < out["shots"]["mae_ewma_baseline"]),
            ("p_goal", out["p_goal"]["logloss_model"]
             < out["p_goal"]["logloss_ewma_baseline"]),
            ("p_assist", out["p_assist"]["logloss_model"]
             < out["p_assist"]["logloss_ewma_baseline"]))}
    (NOUT / f"player_eval_{T}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
