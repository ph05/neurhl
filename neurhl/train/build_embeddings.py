"""NeurHL — per-vantage final player-embedding matrix (rookie blend, P1).

For vantage V, produces neurhl/data/tensors/embeddings_v<V>.npz:
  ids  int64[N]    playerIds covered (vocab players + career-only players)
  emb  float32[N,64]
  gp   int32[N]    dressed games with season_end <= V-1 (blend input)
Blend: emb = w * event_lm_embedding + (1-w) * career_embedding,
w = gp/(gp+40) (PLAN_NeurHL A). Players with a career file but no corpus
appearance get pure career embeddings (w=0). Players with neither are absent —
the game-model dataloader maps them to the position-group mean and counts them
(cold-start null ledger).

Usage: ... python neurhl/train/build_embeddings.py --vantage 2017
Requires event_lm_v<V>.pt and career_v<V>.pt. CPU, deterministic.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, TENSORS  # noqa: E402
from models.career_encoder import CareerEncoder, career_features  # noqa: E402

N0 = 40


def dressed_gp(vantage: int) -> pd.Series:
    app = pd.read_parquet(TENSORS / "_edacache" / "player_appearances.parquet")
    app = app[(app.dressed > 0) & (app.season_end <= vantage - 1)][
        ["player_id", "dressed"]]
    parts = [app]
    for se in range(2008, min(2012, vantage)):
        p = TENSORS / f"player_games_{se}.parquet"
        if p.exists():
            pg = pd.read_parquet(p, columns=["player_id"])
            pg = pg.groupby("player_id").size().rename("dressed").reset_index()
            parts.append(pg)
    allp = pd.concat(parts, ignore_index=True)
    return allp.groupby("player_id").dressed.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", type=int, required=True)
    args = ap.parse_args()
    V = args.vantage
    torch.manual_seed(40140 + V)

    vocab = {int(k): v for k, v in json.loads(
        (TENSORS / f"vocab_{V}.json").read_text()).items()}
    state = torch.load(CKPT / f"event_lm_v{V}.pt", map_location="cpu")
    table = state["player_emb.weight"].numpy()          # (n_vocab+1, 64)
    enc = CareerEncoder()
    enc.load_state_dict(torch.load(CKPT / f"career_v{V}.pt",
                                   map_location="cpu"))
    enc.eval()
    lmap = json.loads((TENSORS / "league_map.json").read_text())
    nhl_idx = lmap.get("NHL", -1)
    careers = pd.read_parquet(TENSORS / "careers.parquet")
    careers = careers[careers.season_end <= V - 1].sort_values(
        ["player_id", "season_end"])
    bios = pd.read_parquet(TENSORS / "career_bios.parquet").set_index("player_id")
    gp = dressed_gp(V)

    # career embeddings for every player with pre-NHL rows <= V-1
    feats, cids = [], []
    for pid, grp in careers.groupby("player_id"):
        if pid not in bios.index:
            continue
        rows = list(zip(grp.league_idx, grp.age, grp.gp, grp.g, grp.a, grp.p))
        if not [r for r in rows if r[0] != nhl_idx]:
            continue
        feats.append(career_features(rows, bios.loc[pid], nhl_idx))
        cids.append(pid)
    with torch.no_grad():
        cemb = enc(torch.as_tensor(np.stack([f[0] for f in feats])),
                   torch.as_tensor(np.stack([f[1] for f in feats])),
                   torch.as_tensor([f[2] for f in feats]),
                   torch.as_tensor(np.stack([f[3] for f in feats]))).numpy()
    career_map = {pid: cemb[i] for i, pid in enumerate(cids)}

    ids = sorted(set(vocab) | set(career_map))
    emb = np.zeros((len(ids), table.shape[1]), dtype=np.float32)
    gps = np.zeros(len(ids), dtype=np.int32)
    for i, pid in enumerate(ids):
        g = int(gp.get(pid, 0))
        gps[i] = g
        w = g / (g + N0)
        pbp = table[vocab[pid]] if pid in vocab else 0.0
        car = career_map.get(pid)
        if car is None:
            emb[i] = pbp                          # vocab player, no career rows
        else:
            emb[i] = w * pbp + (1 - w) * car
    out = TENSORS / f"embeddings_v{V}.npz"
    np.savez_compressed(out, ids=np.array(ids, dtype=np.int64), emb=emb, gp=gps)
    print(f"embeddings_v{V}: {len(ids)} players "
          f"({len(vocab)} vocab, {len(career_map)} with careers)")


if __name__ == "__main__":
    main()
