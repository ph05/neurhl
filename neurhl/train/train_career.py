"""NeurHL — career-encoder distillation onto a frozen event-LM snapshot.

Usage:
  uv run --no-project --python 3.12 --with torch --with numpy --with "pandas<3" \
    --with pyarrow python neurhl/train/train_career.py --vantage 2017 [--smoke]

Targets: event-LM player embeddings (frozen) for vocab players that have >= 1
pre-NHL career row with season_end <= vantage-1 (P1). Sample weight =
sqrt(event mentions). Loss = MSE. Saves career_v<V>.pt + sha256.
Seed 30140 + vantage.
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
from models.career_encoder import CareerEncoder, career_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", type=int, required=True)
    ap.add_argument("--event-lm", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    V = args.vantage
    seed = 30140 + V
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab = json.loads((TENSORS / f"vocab_{V}.json").read_text())
    lmap = json.loads((TENSORS / "league_map.json").read_text())
    nhl_idx = lmap.get("NHL", -1)
    ck = Path(args.event_lm) if args.event_lm else CKPT / f"event_lm_v{V}.pt"
    state = torch.load(ck, map_location="cpu")
    table = state["player_emb.weight"]                    # (n+1, 64)

    careers = pd.read_parquet(TENSORS / "careers.parquet")
    careers = careers[careers.season_end <= V - 1].sort_values(
        ["player_id", "season_end"])
    bios = pd.read_parquet(TENSORS / "career_bios.parquet").set_index("player_id")
    app = pd.read_parquet(TENSORS / "_edacache" / "player_appearances.parquet")
    mentions = app[app.season_end <= V - 1].groupby("player_id") \
        .event_mentions.sum()

    X, ids, w = [], [], []
    for pid, grp in careers.groupby("player_id"):
        if pid not in bios.index or str(pid) not in vocab:
            continue
        rows = list(zip(grp.league_idx, grp.age, grp.gp, grp.g, grp.a, grp.p))
        pre = [r for r in rows if r[0] != nhl_idx]
        if not pre:
            continue
        X.append(career_features(rows, bios.loc[pid], nhl_idx))
        ids.append(vocab[str(pid)])
        w.append(float(np.sqrt(mentions.get(pid, 1.0))))
    if args.smoke:
        X, ids, w = X[:256], ids[:256], w[:256]
    league = torch.as_tensor(np.stack([x[0] for x in X]))
    feats = torch.as_tensor(np.stack([x[1] for x in X]))
    lens = torch.as_tensor([x[2] for x in X])
    static = torch.as_tensor(np.stack([x[3] for x in X]))
    tgt = table[torch.as_tensor(ids)]
    weight = torch.as_tensor(w, dtype=torch.float32)
    weight = weight / weight.mean()
    n = len(ids)
    print(f"distillation set: {n} players (vocab {len(vocab)})")

    perm = torch.randperm(n)
    n_val = max(n // 10, 1)
    va, tr = perm[:n_val], perm[n_val:]
    model = CareerEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best, best_ep = float("inf"), -1
    out_ck = CKPT / f"career_v{V}.pt"
    epochs = 3 if args.smoke else 60
    for ep in range(epochs):
        model.train()
        for i in range(0, len(tr), 512):
            idx = tr[i:i + 512]
            opt.zero_grad(set_to_none=True)
            pred = model(league[idx], feats[idx], lens[idx], static[idx])
            loss = (weight[idx].unsqueeze(1)
                    * (pred - tgt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(league[va], feats[va], lens[va], static[va])
            vloss = float(((pv - tgt[va]) ** 2).mean())
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep}: val mse {vloss:.5f}")
        if vloss < best - 1e-5:
            best, best_ep = vloss, ep
            torch.save(model.state_dict(), out_ck)
        elif ep - best_ep >= 8:
            print(f"early stop at {ep}")
            break
    # baseline: predicting the mean embedding
    base = float(((tgt[va] - tgt[tr].mean(0)) ** 2).mean())
    sha = hashlib.sha256(out_ck.read_bytes()).hexdigest()
    out_ck.with_suffix(".sha256").write_text(sha)
    out_ck.with_suffix(".json").write_text(json.dumps(
        {"vantage": V, "n_players": n, "val_mse": best,
         "mean_baseline_mse": base, "seed": seed, "sha256": sha,
         "event_lm": ck.name}, indent=1))
    print(f"saved {out_ck.name}: val mse {best:.5f} vs mean-baseline {base:.5f}")


if __name__ == "__main__":
    main()
