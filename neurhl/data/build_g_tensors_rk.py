"""Candidate g1rk tensors (PLAN_NeurHL_1_1 A15): the master tensor plus three
skater inputs, a rookie's translated goals and assists per game and a rookie
flag (data/tensors/rookie_features.parquet, walk-forward by season). Every
other array is copied unchanged. Writes g_master_rk.npz, g_meta_rk.parquet and
g_names_rk.json beside the originals, which are never modified; load them with
NEURHL_MASTER_TAG=rk.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402

NEW = ["rk_pred_g", "rk_pred_a", "rk_flag"]


def main():
    z = dict(np.load(TENSORS / "g_master.npz"))
    meta = pd.read_parquet(TENSORS / "g_meta.parquet")
    names = json.loads((TENSORS / "g_names.json").read_text())
    rk = pd.read_parquet(TENSORS / "rookie_features.parquet")
    key = {(int(p), int(s)): (g, a) for p, s, g, a in zip(rk.player_id, rk.season_end, rk.rk_pred_g, rk.rk_pred_a)}
    SKID, SKM = z["SKID"], z["SKM"]
    season = meta.season_end.to_numpy()[:, None, None] * np.ones(SKID.shape, np.int64)
    add = np.full(SKID.shape + (3,), np.nan, np.float32)
    hit = 0
    for i, j, k in np.argwhere(SKM > 0):
        v = key.get((int(SKID[i, j, k]), int(season[i, j, k])))
        if v is not None:
            add[i, j, k] = (v[0], v[1], 1.0)
            hit += 1
        else:
            add[i, j, k, 2] = 0.0
    z["SK"] = np.concatenate([z["SK"], add], axis=-1)
    np.savez_compressed(TENSORS / "g_master_rk.npz", **z)
    shutil.copy(TENSORS / "g_meta.parquet", TENSORS / "g_meta_rk.parquet")
    names["sk_feat"] = names["sk_feat"] + NEW
    (TENSORS / "g_names_rk.json").write_text(json.dumps(names, indent=1))
    print(f"rookie skater-games tagged: {hit:,} of {int((SKM > 0).sum()):,}; SK dims {z['SK'].shape}")


if __name__ == "__main__":
    main()
