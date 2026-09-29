"""Exploratory (PLAN_NeurHL_1_1, deferred C4): in-season fine-tuning of the
NeurHL-G engine with a decoupled L2-SP anchor, on the iteration window only.

For each season T of G_ITER (2012, 2014-2018), the season's frozen snapshot is
the g1 engine trained on seasons < T. These are the snapshots
eval/backtest_unified_season.py saved: data/tensors/_season_bt/snap_T_seed.pt.
The season's games are then walked in blocks of BLOCK games, from game START:

  1. A copy of the frozen weights is trained on the season's games already
     played: EPOCHS passes, AdamW with no decay toward zero. After each step
     the weights are pulled toward the frozen ones: theta -= lr * lam *
     (theta - theta_frozen), decoupled from Adam's scaling (Loshchilov and
     Hutter 2019; Li et al. 2018).
  2. The next block is predicted with the tuned copy and with the frozen
     snapshot.

Reported: the per-game log loss of the engine's home-win probability, tuned
minus frozen, by season, for each anchor strength lam. This decides only
whether the idea deserves a declared G_GATE pre-gate. No window beyond
G_ITER is read.

Writes neurhl/output/neurhl_1_1/finetune_c4_iter.json.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402
import windows as W  # noqa: E402
from models.neurhl_g import NeurHLG  # noqa: E402
from train.train_neurhl_g import DEFAULT, Data, batch, losses, predict  # noqa: E402

SNAP = TENSORS / "_season_bt"
OUT = ROOT / "output" / "neurhl_1_1" / "finetune_c4_iter.json"
START, BLOCK, EPOCHS, LR = 300, 150, 3, 3e-4


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def build(D, cfg):
    n = D.names
    return NeurHLG(len(n["sk_feat"]), len(n["gk_feat"]), len(n["tm_feat"]), len(n["ctx"]),
                   d=cfg["d"], p=cfg["dropout"], attn=cfg["attn"], freeze_heads=cfg["freeze_heads"],
                   elo_anchor=cfg.get("elo_anchor", False), lineup_terms=cfg.get("lineup_terms", False),
                   h_terms=cfg.get("h_terms", False))


def tune(model, P, idx, lam, names, seed):
    m = copy.deepcopy(model)
    anchor = [q.detach().clone() for q in m.parameters()]
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    m.train()
    for _ in range(EPOCHS):
        for i in range(0, len(idx), 128):
            bt = batch(P, rng.permutation(idx)[i:i + 128])
            loss, _ = losses(m(bt), bt, names, 1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                for q, a in zip(m.parameters(), anchor):
                    q.sub_(LR * lam * (q - a))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=[s for s in W.G_ITER])
    ap.add_argument("--lams", type=float, nargs="*", default=[1.0, 10.0, 100.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    assert set(a.seasons) <= set(W.G_ITER), "iteration window only"
    torch.set_num_threads(a.threads)
    cfg = {**DEFAULT, **json.loads((ROOT / "configs" / "neurhl_g" / "g1.json").read_text())}
    D = Data("train")
    s_all = D.meta.season_end.to_numpy()
    y_all = np.isin(D.meta.outcome4.to_numpy(), [0, 2]).astype(float)
    out = {"start": START, "block": BLOCK, "epochs": EPOCHS, "lr": LR, "seed": a.seed, "by_season": {}}
    for T in a.seasons:
        ck = SNAP / f"snap_{T}_{a.seed}.pt"
        if not ck.exists():
            print(f"[c4] {T}: no snapshot yet, skipped", flush=True)
            continue
        P = D.prepare((s_all < T) & (s_all >= cfg["train_from"]))
        frozen = build(D, cfg)
        frozen.load_state_dict(torch.load(ck))
        frozen.eval()
        idx = np.where(s_all == T)[0]
        idx = idx[np.argsort(D.meta.date.values[idx], kind="stable")]
        base = predict(frozen, P, idx)["p_home_win"]
        res = {lam: [] for lam in a.lams}
        froz = []
        for c in range(START, len(idx), BLOCK):
            blk = idx[c:c + BLOCK]
            j = np.arange(c, min(c + BLOCK, len(idx)))
            froz.append(nll(base[j], y_all[blk]))
            for lam in a.lams:
                m = tune(frozen, P, idx[:c], lam, D.names, a.seed * 100 + c)
                res[lam].append(nll(predict(m, P, blk)["p_home_win"], y_all[blk]))
        f = np.concatenate(froz)
        out["by_season"][int(T)] = {"n": int(len(f)), "frozen": float(f.mean()),
                                    **{f"lam_{lam:g}": float(np.concatenate(res[lam]).mean() - f.mean())
                                       for lam in a.lams}}
        print(f"[c4] {T}: {json.dumps(out['by_season'][int(T)])}", flush=True)
    if out["by_season"]:
        tot = sum(v["n"] for v in out["by_season"].values())
        out["pooled_diff"] = {f"lam_{lam:g}": sum(v[f"lam_{lam:g}"] * v["n"] for v in out["by_season"].values()) / tot
                              for lam in a.lams}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out.get("pooled_diff"), indent=1))


if __name__ == "__main__":
    main()
