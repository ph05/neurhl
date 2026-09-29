"""NeurHL 1.1 C1: fit the stat sheet's count calibration (PLAN_NeurHL_1_1).

Two parameters, each fitted by maximising a proper score on NeurHL-G's
out-of-sample team-game predictions from the G_GATE window (config g1, the
live bundle's configuration; seasons 2019, 2020, 2022-2024):

  r_hat   negative-binomial (NB2) dispersion of team shots on goal around the
          model mean: y ~ NB(mu, r), Var = mu + mu^2 / r. The frozen stat sheet
          uses r = 40, and its 80% intervals cover 86% (gate C fails).
  b_hat   slope of team regulation goals on the model's goal mean, in logs,
          about each season's mean: y ~ Poisson(exp(a_s + b (log mu - log M_s))),
          M_s = the season's mean predicted goals, a_s a free level per season
          (the live level is set by the in-season multiplier of
          live/goal_calibration.py, not by this fit). b = 1 is the frozen
          stat sheet.

Seasons 2025 and 2026 are sealed: any row from them stops the fit.
Writes configs/calibration_1_1.json; running it twice gives the same file.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with scipy \
       python neurhl/eval/fit_calibration_1_1.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CONFIGS, NOUT  # noqa: E402

SRC = NOUT / "preds" / "g_gate_games.csv"
OUT = CONFIGS / "calibration_1_1.json"
SEALED = {2025, 2026}
R_FROZEN = 40.0


def nb_logpmf(y, mu, r):
    p = r / (r + mu)
    return stats.nbinom.logpmf(y, r, p)


def pois_logpmf(y, mu):
    return y * np.log(mu) - mu - special.gammaln(y + 1)


def pit_coverage(y, mu, r, seed=711):
    """Share of randomised PIT values inside (0.1, 0.9): nominal 0.80."""
    p = r / (r + mu)
    f1, f0 = stats.nbinom.cdf(y, r, p), stats.nbinom.cdf(y - 1, r, p)
    u = f0 + np.random.default_rng(seed).random(len(y)) * (f1 - f0)
    return float(np.mean((u > 0.1) & (u < 0.9)))


def load():
    d = pd.read_csv(SRC)
    bad = set(d.season_end.unique()) & SEALED
    if bad:
        sys.exit(f"sealed seasons in the fit input: {sorted(bad)}")
    return d


def fit_dispersion(d):
    mu = np.r_[d.sogf_h, d.sogf_a].astype(float)
    y = np.r_[d.y_sogf_h, d.y_sogf_a].astype(float)
    ok = np.isfinite(mu) & np.isfinite(y) & (mu > 0)
    mu, y = mu[ok], y[ok]
    res = optimize.minimize_scalar(lambda lr: -nb_logpmf(y, mu, np.exp(lr)).sum(),
                                   bounds=(np.log(5), np.log(5000)), method="bounded",
                                   options={"xatol": 1e-6})
    r = float(np.exp(res.x))
    by = {}
    for s in sorted(d.season_end.unique()):
        m = np.r_[d.season_end == s, d.season_end == s][ok]
        by[int(s)] = {"n": int(m.sum()),
                      "coverage_r40": pit_coverage(y[m], mu[m], R_FROZEN),
                      "coverage_rhat": pit_coverage(y[m], mu[m], r),
                      "logscore_gain_per_team_game": float(np.mean(nb_logpmf(y[m], mu[m], r)
                                                                   - nb_logpmf(y[m], mu[m], R_FROZEN)))}
    return {"r_hat": round(r, 2), "n_team_games": int(len(y)),
            "coverage_r40": pit_coverage(y, mu, R_FROZEN), "coverage_rhat": pit_coverage(y, mu, r),
            "logscore_gain_per_team_game": float(np.mean(nb_logpmf(y, mu, r) - nb_logpmf(y, mu, R_FROZEN))),
            "by_season": by}


def fit_slope(d):
    seas = np.r_[d.season_end, d.season_end]
    mu = np.r_[d.goals_h, d.goals_a].astype(float)
    y = np.r_[d.y_gf_reg_h, d.y_gf_reg_a].astype(float)
    ok = np.isfinite(mu) & np.isfinite(y) & (mu > 0)
    seas, mu, y = seas[ok], mu[ok], y[ok]
    S = sorted(set(seas))
    si = np.searchsorted(S, seas)
    logM = np.log(np.array([mu[seas == s].mean() for s in S]))
    x = np.log(mu) - logM[si]

    def negll(th):
        a, b = th[:-1], th[-1]
        return -pois_logpmf(y, np.exp(a[si] + b * x)).sum()

    th0 = np.r_[logM, 1.0]
    res = optimize.minimize(negll, th0, method="L-BFGS-B")
    a, b = res.x[:-1], float(res.x[-1])
    mu_cal = np.exp(logM[si] + b * x)            # the live form: level kept, slope applied
    q = pd.qcut(mu, 10, labels=False)
    dec = pd.DataFrame({"q": q, "pred": mu, "pred_cal": mu_cal, "obs": y}).groupby("q").mean()
    return {"b_hat": round(b, 4), "n_team_games": int(len(y)),
            "level_by_season": {int(s): float(np.exp(a[i] - logM[i])) for i, s in enumerate(S)},
            "logscore_gain_per_team_game": float(np.mean(pois_logpmf(y, mu_cal) - pois_logpmf(y, mu))),
            "deciles": {k: [round(float(v), 3) for v in dec[k]] for k in ("pred", "pred_cal", "obs")}}


def main():
    d = load()
    out = {"plan": "PLAN_NeurHL_1_1.md C1",
           "source": str(SRC.relative_to(ROOT.parent)),
           "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "seasons": [int(s) for s in sorted(d.season_end.unique())],
           "r_frozen": R_FROZEN,
           "sog_dispersion": fit_dispersion(d),
           "goal_slope": fit_slope(d),
           "live_goal_form": "mu' = m_t * M * (mu_raw / M) ** b_hat; m_t and M from "
                             "configs/live_goal_calibration.json and each forecast's goal_mult"}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("seasons", "r_frozen")}, indent=1))
    sd, gs = out["sog_dispersion"], out["goal_slope"]
    print(f"r_hat {sd['r_hat']}  coverage {sd['coverage_r40']:.3f} -> {sd['coverage_rhat']:.3f}  "
          f"gain {sd['logscore_gain_per_team_game']:+.4f}")
    print(f"b_hat {gs['b_hat']}  gain {gs['logscore_gain_per_team_game']:+.4f}  deciles {gs['deciles']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
