"""Rookie priors in the live state builder (PLAN_NeurHL_1_1 A15).

When configs/rookie_priors_live.json says {"enabled": true}, every skater in
configs/rookie_priors_2027.csv has his scoring baselines shrink toward his
rookie prior instead of his position's average. This is the same formula
eval/rookie_engine_test.py validated:
  shots, attempts and individual xG per 60 scale with the prior's goals per
  game; assists per 60 with its assists per game; both at the position's
  baseline ice time.
His own NHL rates take over as his games accumulate (b = (n x + k p)/(n + k)).
Disabled or missing files leave the state exactly as built.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "rookie_priors_live.json"
TABLE = ROOT / "configs" / "rookie_priors_2027.csv"


def enabled() -> bool:
    try:
        return bool(json.loads(CFG.read_text()).get("enabled"))
    except (OSError, ValueError):
        return False


def prior_rates(pos: int, pred_g: float, pred_a: float, pri: dict) -> dict:
    toi = pri["toi_ev"][pos] + pri["toi_pp"][pos] + pri["toi_sh"][pos]
    g60, a60 = pred_g * 60.0 / toi, pred_a * 60.0 / toi
    s = g60 / (pri["isog60"][pos] * pri["g_per_sog"][pos])
    return {"isog60": pri["isog60"][pos] * s, "iatt60": pri["iatt60"][pos] * s,
            "ixg60": pri["ixg60"][pos] * s, "a60": a60}


def apply(sk: pd.DataFrame, pri: dict, table: pd.DataFrame = None) -> pd.DataFrame:
    """sk: the live skater state with its b_* columns (after data.build_g_tensors.shrink)."""
    from data.build_g_tensors import SK_BASE
    if table is None:
        if not enabled() or not TABLE.exists():
            return sk
        table = pd.read_csv(TABLE)
    t = table.set_index("player_id")
    rows = sk.player_id.isin(t.index).to_numpy()
    if not rows.any():
        return sk
    sk = sk.copy()
    for i in np.where(rows)[0]:
        r = t.loc[int(sk.player_id.iloc[i])]
        pos = int(np.clip(r.pos, 0, 1))
        for name, val in prior_rates(pos, float(r.pred_g), float(r.pred_a), pri).items():
            col, _, k = SK_BASE[name]
            tag = col.rsplit("_", 1)[1]
            n = sk[f"neff_{tag}"].iloc[i]
            x = sk[col].iloc[i]
            n = 0.0 if not np.isfinite(n) else float(n)
            x = val if not np.isfinite(x) else float(x)
            sk.iloc[i, sk.columns.get_loc(f"b_{name}")] = (n * x + k * val) / (n + k)
    return sk
