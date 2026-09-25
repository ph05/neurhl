"""Player-game confirmation: standard errors clustered by player AND game.

The recorded PG gates (configs/player_game_confirm_2018_2020.json) cluster the
paired per-row differences by game only. A skater appears in about 80 games a
season, so his rows are correlated too. This post-hoc re-analysis of the same
rows reports one-way (game), one-way (player) and two-way (Cameron, Gelbach &
Miller 2011) cluster-robust standard errors, plus relative effect sizes. It
changes no gate decision; it tells a reader how much the recorded z-scores
overstate precision.

Differences are rebuilt with the gate script's own functions, and the
game-clustered z must reproduce the record.
Writes configs/player_game_twoway_2018_2020.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIGS, TENSORS  # noqa: E402
from eval.backtest_player_game import (logloss, pick_k, pois_dev,  # noqa: E402
                                       shrunk_baseline)

SEASONS = [2018, 2019, 2020]


def cl_var(d: np.ndarray, key: np.ndarray) -> float:
    """Cluster-robust variance of the mean of d, clusters given by key."""
    r = pd.Series(d - d.mean()).groupby(key).sum().to_numpy()
    return float((r ** 2).sum() / len(d) ** 2)


def main():
    rows = []
    for v in SEASONS:
        d = pd.read_parquet(TENSORS / f"player_game_preds_{v}.parquet")
        g = d[d.gp_todate >= 1].copy()
        tot = g.groupby(["game_id", "team"]).toi_share_ewa1.transform("sum")
        g["toi_base"] = g.toi_share_ewa1 / tot.clip(lower=1e-9)
        prev = pd.read_parquet(TENSORS / f"player_game_preds_{v - 1}.parquet")
        prev = prev[prev.gp_todate >= 1]
        for tgt, ew in (("goal1", "goal_rate_ewa1"), ("assist1", "assist_rate_ewa1")):
            base = float(prev[tgt].mean())
            g[f"{tgt}_base"] = shrunk_baseline(g, tgt, ew, base,
                                               pick_k(prev, tgt, ew, base))
        g["d_toi"] = (np.abs(g.toi_share_pred - g.toi_share)
                      - np.abs(g.toi_base - g.toi_share))
        g["d_shots"] = (pois_dev(g.shots.to_numpy(), g.shots_pred.to_numpy())
                        - pois_dev(g.shots.to_numpy(), g.shots_ewa1.to_numpy()))
        g["d_goal"] = (logloss(g.goal1.to_numpy(), g.p_goal.to_numpy())
                       - logloss(g.goal1.to_numpy(), g.goal1_base.to_numpy()))
        g["d_assist"] = (logloss(g.assist1.to_numpy(), g.p_assist.to_numpy())
                         - logloss(g.assist1.to_numpy(), g.assist1_base.to_numpy()))
        g["base_toi"] = np.abs(g.toi_base - g.toi_share)
        g["base_shots"] = pois_dev(g.shots.to_numpy(), g.shots_ewa1.to_numpy())
        g["base_goal"] = logloss(g.goal1.to_numpy(), g.goal1_base.to_numpy())
        g["base_assist"] = logloss(g.assist1.to_numpy(), g.assist1_base.to_numpy())
        rows.append(g)
    A = pd.concat(rows, ignore_index=True)
    rec = json.loads((CONFIGS / "player_game_confirm_2018_2020.json").read_text())
    names = {"toi": "toi_share", "shots": "shots", "goal": "goal1",
             "assist": "assist1"}
    out = {"seasons": SEASONS, "n_rows": int(len(A)),
           "n_players": int(A.player_id.nunique()),
           "n_games": int(A.game_id.nunique()), "heads": {}}
    gid, pid = A.game_id.to_numpy(), A.player_id.to_numpy()
    cell = (A.game_id.astype(str) + "_" + A.player_id.astype(str)).to_numpy()
    for k, rk in names.items():
        d = A[f"d_{k}"].to_numpy()
        vg, vp, vgp = cl_var(d, gid), cl_var(d, pid), cl_var(d, cell)
        v2 = max(vg + vp - vgp, max(vg, vp))
        m = float(d.mean())
        out["heads"][rk] = {
            "mean_diff": m,
            "relative_to_baseline": m / float(A[f"base_{k}"].mean()),
            "z_game": float(m / np.sqrt(vg)), "z_player": float(m / np.sqrt(vp)),
            "z_twoway": float(m / np.sqrt(v2)),
            "recorded_z_game": rec["gates"][rk]["z"],
            "reproduces_record": bool(abs(m / np.sqrt(vg) - rec["gates"][rk]["z"]) < 0.01)}
        h = out["heads"][rk]
        print(f"{rk:10s} diff {m:+.6f} ({h['relative_to_baseline']:+.1%})  "
              f"z game {h['z_game']:7.2f} (rec {h['recorded_z_game']:7.2f})  "
              f"player {h['z_player']:7.2f}  two-way {h['z_twoway']:7.2f}")
    (CONFIGS / "player_game_twoway_2018_2020.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
