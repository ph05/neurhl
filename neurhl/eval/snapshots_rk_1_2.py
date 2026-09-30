"""NeurHL 1.2 fix 5: preseason snapshots of the shipped engine's recipe (config g1rk,
rookie inputs; tensors g_master_rk) for the rookie season test (PLAN_NeurHL_1_2 A1).

Walk-forward like eval/backtest_unified_season.py: for each season V, 5 seeds trained
on seasons < V. Writes data/tensors/_season_bt/snaprk_<V>_<seed>.pt (cached).
Run with NEURHL_MASTER_TAG=rk.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402
from train.train_neurhl_g import Data, train_snapshot  # noqa: E402

OUT = TENSORS / "_season_bt"
SEASONS = [2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    assert os.environ.get("NEURHL_MASTER_TAG") == "rk", "run with NEURHL_MASTER_TAG=rk"
    assert not set(a.seasons) & {2025, 2026}
    cfg = json.loads((ROOT / "configs" / "neurhl_g" / "g1rk.json").read_text())
    D = Data("train")
    assert "rk_flag" in D.names["sk_feat"], "rookie tensors not loaded"
    t0 = time.time()
    for V in a.seasons:
        for sd in range(a.seeds):
            ck = OUT / f"snaprk_{V}_{sd}.pt"
            if ck.exists():
                continue
            model, _ = train_snapshot(D, V, {**cfg, "seed": sd, "threads": a.threads})
            torch.save(model.state_dict(), ck)
            print(f"[rk] {V} seed {sd}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
