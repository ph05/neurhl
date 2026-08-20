"""NeurHL — walk-forward game-model training for one predict-season.

Usage:
  ... python neurhl/train/train_game.py --predict-season 2015 [--seeds 0 1 2 3 4]
      [--smoke]

Protocol (PLAN_NeurHL P1/P7): train on regular-season games with season_end in
[2009, T-1] (2008 serves as form/era history), early-stop on T-1, then predict
season T's games. Player embeddings from embeddings_v<T>.npz (players <= T-1);
per-player form features are EWMAs shifted strictly pre-game (within T this is
legitimate in-season h2-style information). Per-seed checkpoints + a per-game
ensemble prediction CSV: neurhl/output/preds/game_preds_<T>.csv.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, NOUT, TENSORS  # noqa: E402
from models.game_model import GameModel, game_loss  # noqa: E402

EWMA_ALPHA = 0.1
ERA_COLS = ["prior_gpg", "prior_ot_share", "prior_so_share", "prior_margin_abs",
            "prior_parity", "flag_3v3", "flag_covid", "season_scaled"]
ERA_SCALE = np.array([1 / 6, 1, 1, 1 / 2.3, 10, 1, 1, 1], dtype=np.float32)


def load_frames(seasons):
    gc, pg = [], []
    for se in seasons:
        g = pd.read_parquet(TENSORS / f"games_ctx_{se}.parquet")
        g["season_end"] = se
        gc.append(g[g.game_type == 2])
        p = pd.read_parquet(TENSORS / f"player_games_{se}.parquet")
        p = p[p.game_type == 2].copy()
        p["season_end"] = se
        pg.append(p)
    gc = pd.concat(gc, ignore_index=True)
    pg = pd.concat(pg, ignore_index=True)
    dates = gc.set_index("game_id").date.astype(str)
    pg["date"] = pg.game_id.map(dates)
    return gc, pg


def add_form(pg: pd.DataFrame) -> pd.DataFrame:
    pg = pg.sort_values(["player_id", "date", "game_id"]).copy()
    grp = pg.groupby("player_id")
    pg["ewma_toi"] = grp.toi_sec.transform(
        lambda s: s.ewm(alpha=EWMA_ALPHA).mean().shift(1)).fillna(0) / 60.0
    pg["gp_todate"] = grp.cumcount()
    return pg


def build_tensors(gc, pg, emb_npz, smoke=False):
    ids = emb_npz["ids"]
    emb = emb_npz["emb"]
    row = {int(p): i for i, p in enumerate(ids)}
    d = emb.shape[1]
    pos_mean = np.zeros((3, d), dtype=np.float32)
    bios = pd.read_parquet(TENSORS / "career_bios.parquet")
    for pgi in range(3):
        sel = [row[p] for p in bios[bios.pos_group == pgi].player_id
               if p in row]
        if sel:
            pos_mean[pgi] = emb[sel].mean(0)
    if smoke:
        gc = gc.head(400)
    pg = pg[pg.game_id.isin(set(gc.game_id))]
    by_game = {k: v for k, v in pg.groupby("game_id")}
    N = len(gc)
    out = {
        "emb": np.zeros((N, 2, 20, d), np.float32),
        "pctx": np.zeros((N, 2, 20, 6), np.float32),
        "pad": np.ones((N, 2, 20), bool),
        "ctx": np.zeros((N, 16), np.float32),
        "era": np.zeros((N, 8), np.float32),
        "outcome4": np.zeros(N, np.int64),
        "goals_h": np.zeros(N, np.int64), "goals_a": np.zeros(N, np.int64),
        "sat5_h": np.zeros(N, np.int64), "sat5_a": np.zeros(N, np.int64),
    }
    for side in ("h", "a"):
        out[f"skater_mask_{side}"] = np.zeros((N, 20), bool)
    for f in ("toi", "shots", "goals_p", "assists_p"):
        out[f"{f}_h"] = np.zeros((N, 20), np.float32)
        out[f"{f}_a"] = np.zeros((N, 20), np.float32)
    n_coldstart = 0
    game_ids = np.zeros(N, np.int64)
    for i, g in enumerate(gc.itertuples(index=False)):
        game_ids[i] = g.game_id
        rows = by_game.get(g.game_id)
        if rows is None:
            continue
        for t, is_home in ((0, True), (1, False)):
            side = rows[rows.is_home == is_home]
            gk = side[side.pos_group == 2].sort_values(
                ["goalie_start", "ewma_toi"], ascending=False)
            sk = side[side.pos_group < 2].sort_values("ewma_toi",
                                                     ascending=False)
            sel = pd.concat([gk.head(2), sk.head(18)])
            suf = "h" if is_home else "a"
            for j, r in enumerate(sel.itertuples(index=False)):
                if j >= 20:
                    break
                ridx = row.get(r.player_id)
                if ridx is None:
                    out["emb"][i, t, j] = pos_mean[r.pos_group]
                    n_coldstart += 1
                else:
                    out["emb"][i, t, j] = emb[ridx]
                pos1h = np.zeros(3, np.float32)
                pos1h[r.pos_group] = 1
                out["pctx"][i, t, j] = [*pos1h, r.ewma_toi / 20.0,
                                        np.log1p(r.gp_todate) / 5.0,
                                        r.goalie_start]
                out["pad"][i, t, j] = False
                if r.pos_group < 2 and j >= 2:
                    k = j
                    out[f"skater_mask_{suf}"][i, k] = True
                    out[f"toi_{suf}"][i, k] = r.toi_sec
                    out[f"shots_{suf}"][i, k] = r.sog
                    out[f"goals_p_{suf}"][i, k] = r.goals
                    out[f"assists_p_{suf}"][i, k] = r.assists
        b2b_h = float(getattr(g, "home_rest", 9) <= 1)
        b2b_a = float(getattr(g, "away_rest", 9) <= 1)
        out["ctx"][i, :9] = [min(g.home_rest, 7) / 7, min(g.away_rest, 7) / 7,
                             b2b_h, b2b_a, g.home_km3d / 1000,
                             g.away_km3d / 1000, g.home_dtz / 3,
                             g.away_dtz / 3, g.days_in / 200]
        out["era"][i] = np.array([getattr(g, c) for c in ERA_COLS],
                                 np.float32) * ERA_SCALE
        out["outcome4"][i] = g.outcome4
        out["goals_h"][i] = g.home_g
        out["goals_a"][i] = g.away_g
        out["sat5_h"][i] = g.sat5_h
        out["sat5_a"][i] = g.sat5_a
    out["ctx"] = np.nan_to_num(out["ctx"])
    out["era"] = np.nan_to_num(out["era"])
    tensors = {k: torch.as_tensor(v) for k, v in out.items()}
    return tensors, game_ids, n_coldstart


def slice_batch(tensors, idx):
    return {k: v[idx] for k, v in tensors.items()}


def eval_probs(model, tensors, device, bs=512):
    model.eval()
    ps = []
    with torch.no_grad():
        n = len(tensors["outcome4"])
        for i in range(0, n, bs):
            b = {k: v[i:i + bs].to(device) for k, v in tensors.items()}
            out = model(b)
            ps.append(torch.softmax(out["out4"], -1).cpu())
    return torch.cat(ps)


def logloss4(p, y):
    return float(-torch.log(p.gather(1, y.unsqueeze(1)).clamp(1e-9)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-season", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-from", type=int, default=2009)
    args = ap.parse_args()
    T = args.predict_season
    cfg = json.loads((TENSORS.parents[1] / "configs" / "game_model.json")
                     .read_text())
    seasons = list(range(args.train_from, T + 1))
    gc, pg = load_frames(seasons)
    pg = add_form(pg)
    emb_npz = np.load(TENSORS / f"embeddings_v{T}.npz")
    tr_gc = gc[gc.season_end < T - 1]
    va_gc = gc[gc.season_end == T - 1]
    te_gc = gc[gc.season_end == T]
    tens_tr, _, cold_tr = build_tensors(tr_gc, pg, emb_npz, args.smoke)
    tens_va, _, _ = build_tensors(va_gc, pg, emb_npz, args.smoke)
    tens_te, te_ids, cold_te = build_tensors(te_gc, pg, emb_npz, args.smoke)
    print(f"T={T}: train {len(tens_tr['outcome4'])}, val "
          f"{len(tens_va['outcome4'])}, test {len(tens_te['outcome4'])} "
          f"(cold-start slots: train {cold_tr}, test {cold_te})")
    device = "mps" if torch.backends.mps.is_available() and not args.smoke \
        else "cpu"
    W = cfg["loss_weights"]
    ens, ens_val = [], []
    for seed in (args.seeds[:1] if args.smoke else args.seeds):
        torch.manual_seed(1000 * T + seed)
        np.random.seed(1000 * T + seed)
        model = GameModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        n = len(tens_tr["outcome4"])
        best, best_ep, patience = float("inf"), -1, 10
        ck = CKPT / f"game_T{T}_s{seed}.pt"
        epochs = 2 if args.smoke else cfg["epochs_max"]
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                b = {k: v[idx].to(device) for k, v in tens_tr.items()}
                opt.zero_grad(set_to_none=True)
                losses = game_loss(model(b), b, W,
                                   cfg["label_smoothing_outcome4"])
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            pv = eval_probs(model, tens_va, device)
            vll = logloss4(pv, tens_va["outcome4"])
            if ep % 5 == 0 or args.smoke:
                print(f"  seed {seed} ep {ep}: val out4 ll {vll:.4f}")
                sys.stdout.flush()
            if vll < best - 1e-4:
                best, best_ep = vll, ep
                torch.save(model.state_dict(), ck)
            elif ep - best_ep >= patience:
                break
        model.load_state_dict(torch.load(ck, map_location=device))
        pt = eval_probs(model, tens_te, device)
        ens.append(pt)
        ens_val.append(eval_probs(model, tens_va, device))
        (ck.with_suffix(".sha256")).write_text(
            hashlib.sha256(ck.read_bytes()).hexdigest())
        print(f"  seed {seed}: best val {best:.4f}; test out4 ll "
              f"{logloss4(pt, tens_te['outcome4']):.4f}")
    P = torch.stack(ens).mean(0)
    y = tens_te["outcome4"]
    p_home = (P[:, 0] + P[:, 2]).clamp(1e-9, 1 - 1e-9)
    y_home = ((y == 0) | (y == 2)).float()
    ll_home = float(-(y_home * torch.log(p_home)
                      + (1 - y_home) * torch.log(1 - p_home)).mean())
    df = pd.DataFrame({"game_id": te_ids, "p_home_reg": P[:, 0],
                       "p_away_reg": P[:, 1], "p_home_extra": P[:, 2],
                       "p_away_extra": P[:, 3], "outcome4": y})
    (NOUT / "preds").mkdir(parents=True, exist_ok=True)
    df.to_csv(NOUT / "preds" / f"game_preds_{T}.csv", index=False)
    # val-season ensemble preds: the ONLY data calibrate.py may fit on (P1/P7)
    Pv = torch.stack(ens_val).mean(0)
    pd.DataFrame({"p_home_reg": Pv[:, 0], "p_away_reg": Pv[:, 1],
                  "p_home_extra": Pv[:, 2], "p_away_extra": Pv[:, 3],
                  "outcome4": tens_va["outcome4"]}).to_csv(
        NOUT / "preds" / f"game_val_{T}.csv", index=False)
    print(f"T={T} ENSEMBLE: out4 ll {logloss4(P, y):.5f}, "
          f"home-win ll {ll_home:.5f} -> preds/game_preds_{T}.csv")


if __name__ == "__main__":
    main()
