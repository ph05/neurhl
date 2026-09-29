"""NeurHL 1.1 C1e: power-play opportunities per team-game (PLAN_NeurHL_1_1 A12).

They are under-dispersed: variance/mean about 0.67 on G_GATE, so the frozen
Poisson intervals are too wide. The calibrated distribution is
binomial(n, mu / n), variance mu (1 - mu / n), with n fitted by maximum
likelihood over integers 6-40 on the G_GATE out-of-sample team-games (config
g1; seasons 2019, 2020, 2022-2024). Rows from 2025 or 2026 stop the fit.
Writes configs/calibration_1_1e.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CONFIGS, NOUT  # noqa: E402

SRC = NOUT / "preds" / "g_gate_games.csv"
OUT = CONFIGS / "calibration_1_1e.json"


def cov(y, F):
    f1, f0 = F(y), F(y - 1)
    u = f0 + np.random.default_rng(711).random(len(y)) * (f1 - f0)
    return float(np.mean((u > 0.1) & (u < 0.9)))


def main():
    d = pd.read_csv(SRC)
    if set(d.season_end.unique()) & {2025, 2026}:
        sys.exit("sealed seasons in the fit input")
    mu = np.r_[d.pp_opps_h, d.pp_opps_a].astype(float)
    y = np.r_[d.y_pp_opps_h, d.y_pp_opps_a].astype(float)
    ok = np.isfinite(mu) & np.isfinite(y) & (mu > 0)
    mu, y = mu[ok], y[ok]
    ll = {n: stats.binom.logpmf(y, n, np.clip(mu / n, 1e-9, 1 - 1e-9)).sum() for n in range(6, 41)}
    n = max(ll, key=ll.get)
    out = {"plan": "PLAN_NeurHL_1_1 A12 (C1e)", "source": str(SRC.relative_to(ROOT.parent)),
           "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "seasons": [int(s) for s in sorted(d.season_end.unique())], "n_team_games": int(len(y)),
           "n_hat": int(n), "var_over_mean": float(np.var(y - mu) / mu.mean()),
           "coverage_poisson": cov(y, lambda k: stats.poisson.cdf(k, mu)),
           "coverage_binomial": cov(y, lambda k: stats.binom.cdf(k, n, mu / n)),
           "logscore_gain_vs_poisson": float(np.mean(stats.binom.logpmf(y, n, mu / n) - stats.poisson.logpmf(y, mu)))}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
