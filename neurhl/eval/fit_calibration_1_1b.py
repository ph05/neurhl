"""NeurHL 1.1 C1b: fit the stat sheet's team-xG dispersion (PLAN_NeurHL_1_1 A6).

The frozen stat sheet draws team xG from a gamma distribution around the model
mean with shape 9 (sim/boxscore_mc.py XG_SHAPE). On the G_GATE window its 80%
intervals cover 0.855. This fits the shape k by maximum likelihood on the same
out-of-sample team-games as C1 (config g1; seasons 2019, 2020, 2022-2024).
Rows from 2025 or 2026 stop the fit. Writes configs/calibration_1_1b.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CONFIGS, NOUT  # noqa: E402

SRC = NOUT / "preds" / "g_gate_games.csv"
OUT = CONFIGS / "calibration_1_1b.json"
K_FROZEN = 9.0


def coverage(y, mu, k):
    u = stats.gamma.cdf(y, k, scale=mu / k)
    return float(np.mean((u > 0.1) & (u < 0.9)))


def main():
    d = pd.read_csv(SRC)
    bad = set(d.season_end.unique()) & {2025, 2026}
    if bad:
        sys.exit(f"sealed seasons in the fit input: {sorted(bad)}")
    mu = np.r_[d.xgf_h, d.xgf_a].astype(float)
    y = np.r_[d.y_xgf_all_h, d.y_xgf_all_a].astype(float)
    ok = np.isfinite(mu) & np.isfinite(y) & (mu > 0) & (y > 0)
    mu, y = mu[ok], y[ok]
    res = optimize.minimize_scalar(lambda lk: -stats.gamma.logpdf(y, np.exp(lk), scale=mu / np.exp(lk)).sum(),
                                   bounds=(np.log(1), np.log(200)), method="bounded", options={"xatol": 1e-6})
    k = float(np.exp(res.x))
    gain = float(np.mean(stats.gamma.logpdf(y, k, scale=mu / k) - stats.gamma.logpdf(y, K_FROZEN, scale=mu / K_FROZEN)))
    out = {"plan": "PLAN_NeurHL_1_1 A6 (C1b)", "source": str(SRC.relative_to(ROOT.parent)),
           "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "seasons": [int(s) for s in sorted(d.season_end.unique())], "n_team_games": int(len(y)),
           "k_frozen": K_FROZEN, "k_hat": round(k, 2),
           "coverage_frozen": coverage(y, mu, K_FROZEN), "coverage_khat": coverage(y, mu, round(k, 2)),
           "logscore_gain_per_team_game": gain,
           "excluded_team_games": int((~ok).sum())}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
