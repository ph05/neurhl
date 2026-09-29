"""NeurHL 1.1 C1d: team-xG mean slope, fitted jointly with the gamma shape
(PLAN_NeurHL_1_1 A10).

Team xG ~ gamma(k_x, mean = M * (mu / M) ** b_x), where mu is the engine's
team-xG mean and M the season's mean forecast. The fit is by maximum
likelihood on the G_GATE out-of-sample team-games (config g1; seasons 2019,
2020, 2022-2024). Rows from 2025 or 2026 stop the fit.
Writes configs/calibration_1_1d.json.
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
OUT = CONFIGS / "calibration_1_1d.json"


def main():
    d = pd.read_csv(SRC)
    if set(d.season_end.unique()) & {2025, 2026}:
        sys.exit("sealed seasons in the fit input")
    s = np.r_[d.season_end, d.season_end]
    mu = np.r_[d.xgf_h, d.xgf_a].astype(float)
    y = np.r_[d.y_xgf_all_h, d.y_xgf_all_a].astype(float)
    ok = np.isfinite(mu) & np.isfinite(y) & (mu > 0) & (y > 0)
    s, mu, y = s[ok], mu[ok], y[ok]
    M = pd.Series(mu).groupby(s).transform("mean").to_numpy()

    def nll(th):
        k = np.exp(th[1])
        return -stats.gamma.logpdf(y, k, scale=M * (mu / M) ** th[0] / k).sum()

    r = optimize.minimize(nll, [1.0, np.log(11.86)], method="Nelder-Mead",
                          options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 2000})
    b, k = round(float(r.x[0]), 4), round(float(np.exp(r.x[1])), 2)
    m2 = M * (mu / M) ** b
    u = stats.gamma.cdf(y, k, scale=m2 / k)
    out = {"plan": "PLAN_NeurHL_1_1 A10 (C1d)", "source": str(SRC.relative_to(ROOT.parent)),
           "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "seasons": [int(x) for x in sorted(set(s))], "n_team_games": int(len(y)),
           "b_x": b, "k_x": k, "centre": "the season's mean frozen pregame team-xG forecast",
           "coverage80": float(np.mean((u > 0.1) & (u < 0.9))),
           "logscore_gain_vs_frozen_k9": float(np.mean(stats.gamma.logpdf(y, k, scale=m2 / k)
                                                       - stats.gamma.logpdf(y, 9.0, scale=mu / 9.0)))}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
