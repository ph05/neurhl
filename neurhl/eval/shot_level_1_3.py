"""NeurHL 1.3 R1 evidence: the season's league shot and attempt levels, engine
preseason vs last season's actual (PLAN_NeurHL_1_3 A1; descriptive).

For each backtest season V (g1 snapshots trained on seasons < V, cached by
eval/backtest_unified_season.py), every game is assembled from opening-night
rows as in the season set (convention P, eval/season_convention_1_2.py). The
league mean of predicted team shots and attempts per team-game is compared with
what happened, and with the previous full season's actual level (for V = 2014
and 2022, the season before the 48- and 56-game seasons).
Writes output/neurhl_1_3/shot_level_history.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from backtest_unified_season import ALIAS, preseason_arrays, preseason_elo  # noqa: E402
from finetune_g_c4 import SNAP, build  # noqa: E402
from train.train_neurhl_g import DEFAULT, Data, predict  # noqa: E402
from common import TENSORS  # noqa: E402

FIT = [2012, 2014, 2015, 2016, 2017, 2018]
JUDGE = [2019, 2020, 2022, 2023, 2024]
PREV = {2014: 2012, 2022: 2020}              # skip the 48-game and 56-game seasons
OUT = ROOT / "output" / "neurhl_1_3" / "shot_level_history.json"


def main():
    torch.set_num_threads(4)
    cfg = {**DEFAULT, **json.loads((ROOT / "configs" / "neurhl_g" / "g1.json").read_text())}
    D = Data("train")
    s_all = D.meta.season_end.to_numpy()
    tidx = json.loads((TENSORS / "maps.json").read_text())["team"]
    abbr = {v: ALIAS.get(k, k) for k, v in tidx.items()}
    lvl = {s: pd.read_parquet(TENSORS / f"tgx_{s}.parquet", columns=["sogf", "attf"]).mean()
           for s in range(2011, 2025)}
    rows = []
    for V in FIT + JUDGE:
        idx = np.where(s_all == V)[0]
        idx = idx[np.argsort(D.meta.date.values[idx], kind="stable")]
        meta = D.meta.iloc[idx].reset_index(drop=True)
        pre, H = preseason_elo(V)
        rh = meta.home_idx.map(lambda t: pre[abbr[int(t)]]).to_numpy()
        ra = meta.away_idx.map(lambda t: pre[abbr[int(t)]]).to_numpy()
        elo = (rh + H - ra) * np.log(10) / 400.0
        P = D.prepare((s_all < V) & (s_all >= cfg["train_from"]))
        Q = preseason_arrays(D, P, idx, elo, dict(D.stats))
        outs = []
        for sd in range(5):
            m = build(D, cfg)
            m.load_state_dict(torch.load(SNAP / f"snap_{V}_{sd}.pt"))
            m.eval()
            with torch.no_grad():
                outs.append(predict(m, Q, np.arange(len(idx))))
        prev = PREV.get(V, V - 1)
        row = {"season": V, "prev": prev}
        for k in ("sogf", "attf"):
            act = float(lvl[V][k])
            pred = float(np.mean([o[k].mean() for o in outs]))
            row.update({f"{k}_act": act, f"{k}_engine": pred, f"{k}_last": float(lvl[prev][k]),
                        f"{k}_err_engine": pred / act - 1, f"{k}_err_last": float(lvl[prev][k]) / act - 1})
        rows.append(row)
        print(V, {k: round(v, 4) for k, v in row.items() if "err" in k}, flush=True)
    R = pd.DataFrame(rows)
    out = {"rows": rows}
    for part, ss in (("fit", FIT), ("judge", JUDGE)):
        x = R[R.season.isin(ss)]
        out[part] = {k: {"mean_abs_err_engine": float(x[f"{k}_err_engine"].abs().mean()),
                         "mean_abs_err_last": float(x[f"{k}_err_last"].abs().mean())} for k in ("sogf", "attf")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("fit", "judge")}, indent=1))


if __name__ == "__main__":
    main()
