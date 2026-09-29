"""NeurHL 1.1 C1c: fit a skater-shots dispersion (PLAN_NeurHL_1_1 A7).

Skater shots on goal per game ~ NB2(mu, r_s) around the engine's skater mean,
with r_s fitted by maximum likelihood on NeurHL-G's out-of-sample skater-games
of G_GATE (config g1, five seeds, the cached gate predictions that
eval/gate_g_record.py reads; seasons 2019, 2020, 2022-2024, dressed skaters).
The gate's Poisson check covers 0.783 at a nominal 0.80. Sealed seasons are
never loaded (data/g_loader removes them). Writes configs/calibration_1_1c.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CONFIGS  # noqa: E402
import windows as W  # noqa: E402
from eval.gate_g import cached_outputs  # noqa: E402
from train.train_neurhl_g import Data  # noqa: E402

OUT = CONFIGS / "calibration_1_1c.json"


def cov(y, cdf):
    f1, f0 = cdf(y), cdf(y - 1)
    u = f0 + np.random.default_rng(711).random(len(y)) * (f1 - f0)
    return float(np.mean((u > 0.1) & (u < 0.9)))


def main():
    cfg = json.loads((CONFIGS / "neurhl_g" / "g1.json").read_text())
    D = Data("train")
    meta, A = D.meta, D.A
    assert not set(meta.season_end.unique()) & {2025, 2026}
    iy = D.names["sk_tgt"].index("isog")
    mus, ys = [], []
    for T in W.G_GATE:
        te = np.where(meta.season_end.to_numpy() == T)[0]
        o = cached_outputs(cfg, T, list(range(cfg.get("seeds", 5))))
        m = A["SKM"][te] > 0
        mu, y = o["isog"][m], A["SKY"][te][..., iy][m]
        ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
        mus.append(mu[ok]); ys.append(y[ok])
    mu, y = np.concatenate(mus), np.concatenate(ys)
    res = optimize.minimize_scalar(lambda lr: -stats.nbinom.logpmf(y, np.exp(lr), np.exp(lr) / (np.exp(lr) + mu)).sum(),
                                   bounds=(0, np.log(1000)), method="bounded", options={"xatol": 1e-6})
    r = round(float(np.exp(res.x)), 2)
    out = {"plan": "PLAN_NeurHL_1_1 A7 (C1c)", "seasons": list(W.G_GATE), "n_skater_games": int(len(y)),
           "r_s": r, "coverage_poisson": cov(y, lambda k: stats.poisson.cdf(k, mu)),
           "coverage_nb": cov(y, lambda k: stats.nbinom.cdf(k, r, r / (r + mu))),
           "logscore_gain_vs_poisson": float(np.mean(stats.nbinom.logpmf(y, r, r / (r + mu)) - stats.poisson.logpmf(y, mu))),
           "inputs_sha256": hashlib.sha256(mu.tobytes() + y.tobytes()).hexdigest()}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
