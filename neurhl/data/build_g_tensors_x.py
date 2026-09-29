"""Candidate g1x tensors (PLAN_NeurHL_1_1 A19): the rookie-input tensors (g_master_rk)
plus three all-player inputs from non-NHL leagues over the previous two
seasons: translated goals and assists per game and the non-NHL games behind
them (data/tensors/prior_features.parquet, walk-forward). Writes g_master_x.npz,
g_meta_x.parquet and g_names_x.json; load with NEURHL_MASTER_TAG=x.
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

NEW = ["px_g", "px_a", "px_gp"]


def main():
    z = dict(np.load(TENSORS / "g_master_rk.npz"))
    meta = pd.read_parquet(TENSORS / "g_meta_rk.parquet")
    names = json.loads((TENSORS / "g_names_rk.json").read_text())
    t = pd.read_parquet(TENSORS / "prior_features.parquet")
    key = {(int(p), int(s)): (g, a, n) for p, s, g, a, n in zip(t.player_id, t.season_end, t.px_g, t.px_a, t.px_gp)}
    SKID, SKM = z["SKID"], z["SKM"]
    season = meta.season_end.to_numpy()
    add = np.full(SKID.shape + (3,), np.nan, np.float32)
    hit = 0
    for i, j, k in np.argwhere(SKM > 0):
        v = key.get((int(SKID[i, j, k]), int(season[i])))
        if v is not None:
            add[i, j, k] = v
            hit += 1
        else:
            add[i, j, k, 2] = 0.0                      # no non-NHL games in the two seasons before
    z["SK"] = np.concatenate([z["SK"], add], axis=-1)
    np.savez_compressed(TENSORS / "g_master_x.npz", **z)
    shutil.copy(TENSORS / "g_meta_rk.parquet", TENSORS / "g_meta_x.parquet")
    names["sk_feat"] = names["sk_feat"] + NEW
    (TENSORS / "g_names_x.json").write_text(json.dumps(names, indent=1))
    print(f"skater-games with non-NHL priors: {hit:,} of {int((SKM > 0).sum()):,}; SK dims {z['SK'].shape}")


if __name__ == "__main__":
    main()
