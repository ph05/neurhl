"""Exploratory (PLAN_NeurHL_1_1, C4): in-season fine-tuning, measured the way
the live model is used. The live model is a five-seed ensemble, stacked with
Elo and NeurHL-H. Iteration window only.

For each season T of G_ITER and each block, the five frozen seed snapshots
(data/tensors/_season_bt/snap_T_{0..4}.pt) and five tuned copies (as
eval/finetune_g_c4.py, anchor lam) each predict the block. Home-win
probabilities are averaged over seeds. The final probability is the live
bundle's stack (checkpoints/g/g2027_v1, Elo + G + H). Elo logits and NeurHL-H
come from neurhl/output/preds/g_iter_games_r11.csv.

Reported per season and pooled: raw-ensemble and stacked log loss, tuned minus
frozen, with per-game standard errors.

Writes neurhl/output/neurhl_1_1/finetune_c4_stack_iter.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import windows as W  # noqa: E402
from finetune_g_c4 import BLOCK, SNAP, START, build, nll, tune  # noqa: E402
from train.train_neurhl_g import DEFAULT, Data, predict  # noqa: E402

OUT = ROOT / "output" / "neurhl_1_1" / "finetune_c4_stack_iter.json"


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(W.G_ITER))
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    assert set(a.seasons) <= set(W.G_ITER), "iteration window only"
    torch.set_num_threads(a.threads)
    cfg = {**DEFAULT, **json.loads((ROOT / "configs" / "neurhl_g" / "g1.json").read_text())}
    st = json.loads((ROOT / "checkpoints" / "g" / "g2027_v1" / "bundle.json").read_text())["stack"]
    ref = pd.read_csv(ROOT / "output" / "preds" / "g_iter_games_r11.csv").set_index("game_id")
    D = Data("train")
    s_all = D.meta.season_end.to_numpy()
    gid = D.meta.game_id.to_numpy()
    y_all = np.isin(D.meta.outcome4.to_numpy(), [0, 2]).astype(float)
    rows = []
    for T in a.seasons:
        cks = [SNAP / f"snap_{T}_{sd}.pt" for sd in range(a.seeds)]
        if not all(c.exists() for c in cks):
            print(f"[c4s] {T}: snapshots missing, skipped", flush=True)
            continue
        P = D.prepare((s_all < T) & (s_all >= cfg["train_from"]))
        models = []
        for c in cks:
            m = build(D, cfg)
            m.load_state_dict(torch.load(c))
            m.eval()
            models.append(m)
        idx = np.where(s_all == T)[0]
        idx = idx[np.argsort(D.meta.date.values[idx], kind="stable")]
        for c in range(START, len(idx), BLOCK):
            blk = idx[c:c + BLOCK]
            pf = np.mean([predict(m, P, blk)["p_home_win"] for m in models], 0)
            pt = np.mean([predict(tune(m, P, idx[:c], a.lam, D.names, sd * 100 + c), P, blk)["p_home_win"]
                          for sd, m in enumerate(models)], 0)
            rows.append(pd.DataFrame({"season": T, "block": c, "game_id": gid[blk], "y": y_all[blk],
                                      "pf": pf, "pt": pt}))
        print(f"[c4s] {T} done", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d = d[d.game_id.isin(ref.index[ref.p_h.notna() & ref.elo_logit.notna()])].copy()
    e, h = ref.loc[d.game_id, "elo_logit"].to_numpy(), lg(ref.loc[d.game_id, "p_h"].to_numpy())
    b0, (ce, cg, ch) = st["intercept"], st["coef"]
    stack = lambda p: 1 / (1 + np.exp(-(b0 + ce * e + cg * lg(p) + ch * h)))  # noqa: E731
    d["sf"], d["stt"] = stack(d.pf.to_numpy()), stack(d.pt.to_numpy())
    out = {"lam": a.lam, "seeds": a.seeds, "by_season": {}}
    for key, grp in list(d.groupby("season")) + [("pooled", d)]:
        raw = nll(grp.pt, grp.y) - nll(grp.pf, grp.y)
        stk = nll(grp.stt, grp.y) - nll(grp.sf, grp.y)
        out["by_season"][str(key)] = {"n": int(len(grp)),
                                      "raw_frozen": float(nll(grp.pf, grp.y).mean()),
                                      "raw_diff": float(raw.mean()), "raw_se": float(raw.std() / np.sqrt(len(grp))),
                                      "stack_frozen": float(nll(grp.sf, grp.y).mean()),
                                      "stack_diff": float(stk.mean()), "stack_se": float(stk.std() / np.sqrt(len(grp)))}
        print(key, json.dumps(out["by_season"][str(key)]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    d.to_csv(OUT.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
