"""NeurHL — train Layer 1 (player performance) and emit Layer 2 team projections.

Walk-forward: for predict-season T, Layer 1 trains on player-games with
season_end < T and then predicts every player-game of season T. Predictions are
aggregated (TOI-weighted, over the players actually dressed) into per-game team
projections, which Layer 2 consumes as context features.

Career-context features are the point of putting the layer here: age and age^2
for aging curves, games-with-current-team plus a recent-move flag for trade and
call-up acclimation, and career games for experience. A team-level model cannot
express any of these; a player-level one trained on ~750k player-games can.

Output: neurhl/data/tensors/proj_team_<T>.parquet with, per game_id,
  proj_cf_h/a, proj_ca_h/a, proj_clf_h/a, proj_cfpct_h/a
Usage: ... python neurhl/train/train_player.py --predict-season 2017
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, TENSORS  # noqa: E402
from models.player_model import (EWMA_COLS, PlayerModel,  # noqa: E402
                                 STATIC_COLS, player_loss)
from train.train_game import load_frames  # noqa: E402

ALPHA = 0.1


def build_player_frame(seasons) -> pd.DataFrame:
    rates = pd.concat(
        [pd.read_parquet(TENSORS / f"onice_rates_{se}.parquet").assign(season_end=se)
         for se in seasons], ignore_index=True)
    pg = pd.concat(
        [pd.read_parquet(TENSORS / f"player_games_{se}.parquet",
                         columns=["game_id", "player_id", "is_home",
                                  "pos_group", "toi_sec"]).assign(season_end=se)
         for se in seasons], ignore_index=True)
    gc, _ = load_frames(seasons)
    meta = gc.set_index("game_id")[["date", "home_idx", "away_idx"]]
    d = pg.merge(rates.drop(columns=["season_end"]),
                 on=["game_id", "player_id", "is_home"], how="left")
    for c in ("cf", "ca", "cf5", "ca5", "clf", "cla"):
        d[c] = d[c].fillna(0.0)
    d = d.join(meta, on="game_id")
    d["team_idx"] = np.where(d.is_home, d.home_idx, d.away_idx)
    d["date"] = d.date.astype(str)
    d = d.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)

    g = d.groupby("player_id")
    for src, dst in (("cf", "e_cf"), ("ca", "e_ca"), ("clf", "e_clf"),
                     ("cla", "e_cla"), ("toi_sec", "e_toi")):
        d[dst] = g[src].transform(lambda s: s.ewm(alpha=ALPHA).mean().shift(1))
    d["e_cfpct"] = d.e_cf / (d.e_cf + d.e_ca).clip(lower=1e-6)
    d["career_gp"] = g.cumcount()
    # games with the CURRENT team: resets whenever team_idx changes
    same = d.team_idx.eq(g.team_idx.shift(1)) & d.player_id.eq(d.player_id.shift(1))
    grp_id = (~same).cumsum()
    d["team_gp"] = d.groupby(grp_id).cumcount()
    d["recent_move"] = (d.team_gp < 10).astype(np.float32)
    # rest days
    dt = pd.to_datetime(d.date)
    d["rest"] = (dt - g["date"].shift(1).pipe(pd.to_datetime)).dt.days.fillna(9)
    d["rest"] = d.rest.clip(0, 9)

    bios = pd.read_parquet(TENSORS / "career_bios.parquet").set_index("player_id")
    birth = d.player_id.map(bios.birth_year)
    d["age"] = (d.season_end - birth).clip(17, 45).fillna(27.0)
    d["pos_f"] = (d.pos_group == 0).astype(np.float32)
    d["pos_d"] = (d.pos_group == 1).astype(np.float32)
    d["pos_g"] = (d.pos_group == 2).astype(np.float32)
    d["home"] = d.is_home.astype(np.float32)
    return d.dropna(subset=EWMA_COLS)


def featurize(d: pd.DataFrame, emb_npz) -> tuple:
    ids = emb_npz["ids"]
    row = {int(p): i for i, p in enumerate(ids)}
    E = emb_npz["emb"]
    idx = d.player_id.map(row)
    have = idx.notna().to_numpy()
    emb = np.zeros((len(d), E.shape[1]), np.float32)
    emb[have] = E[idx[have].astype(int).to_numpy()]
    f = pd.DataFrame(index=d.index)
    f["pos_f"], f["pos_d"], f["pos_g"] = d.pos_f, d.pos_d, d.pos_g
    f["age"] = (d.age - 27.0) / 5.0
    f["age_sq"] = ((d.age - 27.0) / 5.0) ** 2
    f["career_gp"] = np.log1p(d.career_gp) / 6.0
    f["team_gp"] = np.log1p(d.team_gp) / 6.0
    f["recent_move"] = d.recent_move
    f["home"] = d.home
    f["rest"] = d.rest / 9.0
    f["e_cf"] = d.e_cf / 20.0
    f["e_ca"] = d.e_ca / 20.0
    f["e_clf"] = d.e_clf / 6.0
    f["e_cla"] = d.e_cla / 6.0
    f["e_toi"] = d.e_toi / 1200.0
    f["e_cfpct"] = d.e_cfpct
    X = f[STATIC_COLS + EWMA_COLS].to_numpy(np.float32)
    # baseline the residual model anchors on: the player's own EWMA per-60
    # rates and EWMA ice time (levels, not scaled features)
    eh = (d.e_toi.clip(lower=60.0) / 3600.0).to_numpy(np.float32)
    B = np.stack([(d.e_cf.to_numpy(np.float32) / eh),
                  (d.e_ca.to_numpy(np.float32) / eh),
                  (d.e_clf.to_numpy(np.float32) / eh),
                  (d.e_cla.to_numpy(np.float32) / eh),
                  d.e_toi.to_numpy(np.float32)], 1)
    hours = (d.toi_sec.clip(lower=1) / 3600.0).to_numpy(np.float32)
    Y = np.stack([d.cf.to_numpy(np.float32), d.ca.to_numpy(np.float32),
                  d.clf.to_numpy(np.float32), d.cla.to_numpy(np.float32),
                  d.toi_sec.to_numpy(np.float32)], 1)
    return emb, np.nan_to_num(X), Y, np.nan_to_num(B, posinf=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-season", type=int, required=True)
    ap.add_argument("--train-from", type=int, default=2008)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    T = args.predict_season
    torch.manual_seed(50000 + T)
    np.random.seed(50000 + T)

    seasons = list(range(args.train_from, T + 1))
    d = build_player_frame(seasons)
    emb_npz = np.load(TENSORS / f"embeddings_v{T}.npz")
    emb, X, Y, BASE = featurize(d, emb_npz)
    is_tr = (d.season_end < T).to_numpy()
    print(f"T={T}: Layer-1 train {int(is_tr.sum()):,} player-games, "
          f"predict {int((~is_tr).sum()):,}")

    dev = "cpu"
    model = PlayerModel(d_player=emb.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    Etr = torch.as_tensor(emb[is_tr]); Xtr = torch.as_tensor(X[is_tr])
    Ytr = torch.as_tensor(Y[is_tr]); Btr = torch.as_tensor(BASE[is_tr])
    n = len(Xtr)
    nv = max(n // 10, 1)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(7))
    vi, ti = perm[:nv], perm[nv:]
    best, best_ep = float("inf"), -1
    ck = CKPT / f"player_T{T}.pt"
    for ep in range(args.epochs):
        model.train()
        order = ti[torch.randperm(len(ti))]
        for i in range(0, len(order), 4096):
            b = order[i:i + 4096]
            opt.zero_grad(set_to_none=True)
            L = player_loss(model(Etr[b], Xtr[b], Btr[b]), Ytr[b])
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(player_loss(model(Etr[vi], Xtr[vi], Btr[vi]),
                                  Ytr[vi])["total"])
        if v < best - 1e-4:
            best, best_ep = v, ep
            torch.save(model.state_dict(), ck)
        print(f"  ep {ep}: val {v:.4f}")
        sys.stdout.flush()
        if ep - best_ep >= 3:
            break
    model.load_state_dict(torch.load(ck, map_location=dev))
    model.eval()

    # ---- Layer 2: predict season T and aggregate to team projections
    te = d[~is_tr]
    Ete = torch.as_tensor(emb[~is_tr]); Xte = torch.as_tensor(X[~is_tr])
    Bte = torch.as_tensor(BASE[~is_tr])
    with torch.no_grad():
        P = torch.cat([model(Ete[i:i + 8192], Xte[i:i + 8192], Bte[i:i + 8192])
                       for i in range(0, len(Xte), 8192)]).numpy()
    proj = te[["game_id", "is_home", "pos_group"]].copy()
    proj["p_toi"] = P[:, 4]
    for i, name in enumerate(("cf", "ca", "clf", "cla")):
        proj[f"p_{name}"] = P[:, i] * P[:, 4] / 3600.0    # expected counts
    sk = proj[proj.pos_group < 2]
    agg = sk.groupby(["game_id", "is_home"])[
        ["p_cf", "p_ca", "p_clf", "p_cla", "p_toi"]].sum()
    agg["cfpct"] = agg.p_cf / (agg.p_cf + agg.p_ca).clip(lower=1e-6)
    piv = agg.reset_index().pivot(index="game_id", columns="is_home")
    piv.columns = [f"{a}_{'h' if b else 'a'}" for a, b in piv.columns]
    piv.to_parquet(TENSORS / f"proj_team_{T}.parquet")
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    ck.with_suffix(".json").write_text(json.dumps(
        {"predict_season": T, "val": best, "n_train": int(is_tr.sum()),
         "sha256": sha}, indent=1))
    print(f"wrote proj_team_{T}.parquet ({len(piv)} games); Layer-1 val {best:.4f}")


if __name__ == "__main__":
    main()
