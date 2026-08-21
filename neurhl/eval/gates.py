"""NeurHL — mechanical gate evaluation against PLAN_NeurHL.md (game-level).

Reads the committed per-season prediction artifacts (preds/game_preds_<T>.csv,
preds/game_val_<T>.csv), fits temperature ONLY on each T's val season (P1),
and evaluates G1 (vs v1), G3 (calibration), G4 (vs Tier-0) exactly as locked.
Season-level gates G2/G5 are appended by eval/backtest_season.py after the
Tier-3 sim exists. Writes/updates neurhl/output/params_neurhl.json with full
per-season records; every decision carries rule_eval so the battery can
re-assert rule == outcome mechanically.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT  # noqa: E402

TUNE_T = [2012, 2013, 2014, 2015, 2016, 2017]
G4_T = [2014, 2015, 2016, 2017]
P4 = ["p_home_reg", "p_away_reg", "p_home_extra", "p_away_extra"]


def fit_temperature(p4: np.ndarray, y: np.ndarray) -> float:
    """1-param temperature on log-probs, grid + golden refinement."""
    logp = np.log(np.clip(p4, 1e-12, 1))

    def nll(tau):
        z = logp / tau
        z = z - z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q = q / q.sum(axis=1, keepdims=True)
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()

    taus = np.linspace(0.5, 2.5, 41)
    best = min(taus, key=nll)
    for _ in range(30):
        step = 0.01
        cands = [best - step, best, best + step]
        best = min(cands, key=nll)
    return float(best)


def apply_temperature(p4: np.ndarray, tau: float) -> np.ndarray:
    z = np.log(np.clip(p4, 1e-12, 1)) / tau
    z = z - z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def home_ll(p4: np.ndarray, y: np.ndarray) -> float:
    ph = np.clip(p4[:, 0] + p4[:, 2], 1e-9, 1 - 1e-9)
    yh = ((y == 0) | (y == 2)).astype(float)
    return float(-(yh * np.log(ph) + (1 - yh) * np.log(1 - ph)).mean())


def reliability(p4: np.ndarray, y: np.ndarray, bins: int = 10):
    ph = p4[:, 0] + p4[:, 2]
    yh = ((y == 0) | (y == 2)).astype(float)
    edges = np.quantile(ph, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, ph, "right") - 1, 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() > 5:
            xs.append(ph[m].mean())
            ys.append(yh[m].mean())
            ns.append(m.sum())
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
    slope = float(np.polyfit(xs, ys, 1, w=ns)[0])
    ece = float(np.average(np.abs(xs - ys), weights=ns))
    return slope, ece


def main():
    base = json.loads((NOUT / "baselines_tune.json").read_text())
    v1_ll = base["game_logloss"]
    tier0 = base["tier0"]["logloss_pooled_2014_2017"]

    rows, taus = {}, {}
    all_p, all_y = [], []
    g4_p, g4_y = [], []
    for T in TUNE_T:
        pred = pd.read_csv(NOUT / "preds" / f"game_preds_{T}.csv")
        val = pd.read_csv(NOUT / "preds" / f"game_val_{T}.csv")
        tau = fit_temperature(val[P4].to_numpy(), val.outcome4.to_numpy())
        taus[str(T)] = tau
        p4 = apply_temperature(pred[P4].to_numpy(), tau)
        y = pred.outcome4.to_numpy()
        rows[str(T)] = {"n": len(y), "tau": tau,
                        "nn_home_ll": home_ll(p4, y),
                        "v1_home_ll": v1_ll[str(T)]}
        all_p.append(p4)
        all_y.append(y)
        if T in G4_T:
            g4_p.append(p4)
            g4_y.append(y)
    P = np.concatenate(all_p)
    Y = np.concatenate(all_y)
    pooled = home_ll(P, Y)
    pooled_g4 = home_ll(np.concatenate(g4_p), np.concatenate(g4_y))
    wins = sum(rows[str(T)]["nn_home_ll"] < rows[str(T)]["v1_home_ll"]
               for T in TUNE_T)
    slope, ece = reliability(P, Y)

    gates = {
        "G1": {"rule": "pooled 2012-2017 home ll <= 0.67385 AND wins >= 4/6",
               "nn_pooled": pooled, "v1_pooled": v1_ll["pooled_2012_2017"],
               "wins": int(wins),
               "rule_eval": bool(pooled <= 0.67385 and wins >= 4),
               "pass": bool(pooled <= 0.67385 and wins >= 4)},
        "G3": {"rule": "slope in [0.90,1.10] AND ECE <= 0.015",
               "slope": slope, "ece": ece,
               "rule_eval": bool(0.90 <= slope <= 1.10 and ece <= 0.015),
               "pass": bool(0.90 <= slope <= 1.10 and ece <= 0.015)},
        "G4": {"rule": "pooled 2014-2017 home ll <= 0.67392",
               "nn_pooled_2014_2017": pooled_g4, "tier0": tier0,
               "rule_eval": bool(pooled_g4 <= 0.67392),
               "pass": bool(pooled_g4 <= 0.67392)},
    }
    out_p = NOUT / "params_neurhl.json"
    blob = json.loads(out_p.read_text()) if out_p.exists() else {}
    blob.setdefault("prereg", "PLAN_NeurHL.md @ commit 8c8e6d8")
    blob["temperatures"] = taus
    blob["game_per_season"] = rows
    blob.setdefault("gates", {}).update(gates)
    out_p.write_text(json.dumps(blob, indent=1))
    print(json.dumps({"pooled": pooled, "pooled_2014_2017": pooled_g4,
                      "wins_vs_v1": wins, "slope": slope, "ece": ece,
                      "G1": gates["G1"]["pass"], "G3": gates["G3"]["pass"],
                      "G4": gates["G4"]["pass"]}, indent=1))


if __name__ == "__main__":
    main()
