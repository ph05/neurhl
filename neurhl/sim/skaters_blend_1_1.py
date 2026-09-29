"""NeurHL 1.1 C4: blended skater season points, issued as a dated file set
beside the NeurHL 1.0 freeze (PLAN_NeurHL_1_1 A3).

The earlier player-season work declared a 50/50 blend of two paths and found
it had the lowest points error over 11 vantages: 9.07 per 82 games, against
9.41 and 9.77 for the paths alone. The paths are:
  path A  the gradient-boosted season model;
  path B  the player-game layer summed over the schedule.
NeurHL 1.0's skater totals come from the engine alone, whose player heads beat
path B on every shared target (PLAN_NeurHL4 gate PG).

This set blends the two surviving paths, 50/50, per game played:
  ppg = 0.5 * ppg_engine + 0.5 * c * ppg_A
  ppg_engine  NeurHL 1.0 points per game (skaters_2027.csv)
  ppg_A       path A's points over its expected games
              (output/player_proj_2027_20260928.csv, proj_p_path_a / exp_gp)
  c           one constant that puts path A on the engine's league scoring
              level over the matched skaters, so the blend redistributes points
              and changes no total
The engine's games played (the availability model) are kept. Each skater's
goals and assists are scaled by the same factor as his points, so the team
sums change only by the redistribution. Skaters without a path-A row keep
their NeurHL 1.0 values.

Writes neurhl/output/neurhl_1_0/skaters_blend_20260929/skaters_2027.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output" / "neurhl_1_0"
PROJ_A = ROOT / "output" / "player_proj_2027_20260928.csv"
OUT = SRC / "skaters_blend_20260929"


def main():
    s = pd.read_csv(SRC / "skaters_2027.csv")
    a = pd.read_csv(PROJ_A)[["player_id", "exp_gp", "proj_p_path_a"]]
    a = a[a.exp_gp > 0]
    m = s.merge(a, on="player_id", how="left")
    have = m.proj_p_path_a.notna() & (m.gp > 0)
    ppg_e = m.points / m.gp.where(m.gp > 0)
    ppg_a = m.proj_p_path_a / m.exp_gp
    c = float((ppg_e[have] * m.gp[have]).sum() / (ppg_a[have] * m.gp[have]).sum())
    ppg = np.where(have, 0.5 * ppg_e + 0.5 * c * ppg_a, ppg_e)
    f = np.where(have, ppg / ppg_e.where(ppg_e > 0, 1.0), 1.0)
    f = np.where(np.isfinite(f), f, 1.0)
    out = s.copy()
    for col in ("goals", "assists", "points", "g_p10", "g_p90", "a_p10", "a_p90", "points_p10", "points_p90"):
        out[col] = s[col] * f
    out["points_per_gp"] = out.points / out.gp.where(out.gp > 0)
    out["shooting_pct"] = 100 * out.goals / out.sog.where(out.sog > 0)
    out["blend_factor"] = f
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "skaters_2027.csv", index=False, float_format="%.4f")
    info = {"c_level": c, "matched": int(have.sum()), "skaters": int(len(s)),
            "points_total": {"frozen": float(s.points.sum()), "blend": float(out.points.sum())},
            "factor_range": [float(np.min(f)), float(np.max(f))]}
    (OUT / "run.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    top = out.sort_values("points", ascending=False).head(12)
    print(top.assign(frozen=s.set_index("player_id").loc[top.player_id, "points"].values)[
        ["name", "team", "gp", "frozen", "points"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
