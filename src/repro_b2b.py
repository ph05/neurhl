"""Committed reproduction of the batch-3 rest/back-to-back measurement (B2B_ELO=38).

The original analysis was run ad hoc (2026-08-14) and never committed — flagged by the
2026-08-16 review as reproducibility debt. This script IS the measurement now.

Spec (as documented in NOTES.md): regular-season games 2010-2026, logistic regression of
home win on Elo diff (per 100, HFA in intercept) plus second-of-back-to-back indicators
for each side, flags keyed by team+date (the alignment lesson). Reported: win-prob deltas
at an even matchup, t-stats, Elo equivalents.

Acceptance band (PLAN_V4.md B3): deltas within ±1.5% of -/+6%, |t| >= 4,
Elo equivalent in [28, 48]. B2B_ELO stays 38 if reproduced.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E

PROJ = Path(__file__).resolve().parents[1]
SEASONS = range(2010, 2027)


def logistic_with_se(X: np.ndarray, y: np.ndarray):
    """IRLS fit + asymptotic standard errors from the observed information."""
    b = E.fit_logistic(X, y)
    Xd = np.column_stack([np.ones(len(X)), X])
    p = 1.0 / (1.0 + np.exp(-(Xd @ b)))
    W = p * (1 - p)
    cov = np.linalg.inv(Xd.T @ (Xd * W[:, None]))
    return b, np.sqrt(np.diag(cov))


def main():
    g = pd.read_csv(PROJ / "data/processed/games.csv", keep_default_na=False,
                    parse_dates=["date"])
    v1 = json.loads((PROJ / "output/params.json").read_text())
    preds, _, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])

    # preds rows are emitted in g's iteration order; verify before positional attach
    assert (g.home.to_numpy() == preds.home.to_numpy()).all()
    assert (g.away.to_numpy() == preds.away.to_numpy()).all()
    assert (g.season_end.to_numpy() == preds.season_end.to_numpy()).all()
    preds = preds.assign(date=g.date.to_numpy())

    hb, ab = E.b2b_flags(preds[["date", "home", "away"]])
    preds = preds.assign(hb2b=hb, ab2b=ab)

    sub = preds[(preds.game_type == "R") & preds.season_end.isin(SEASONS)]
    X = np.column_stack([sub.d_ex_hfa.to_numpy() / 100.0, sub.hb2b.to_numpy(),
                         sub.ab2b.to_numpy()])
    y = sub.home_win.to_numpy().astype(float)
    b, se = logistic_with_se(X, y)
    # b indices: 0 intercept (HFA), 1 Elo slope per 100, 2 home-b2b, 3 away-b2b
    intercept, elo_slope, c_hb2b, c_ab2b = b
    t_hb2b, t_ab2b = c_hb2b / se[2], c_ab2b / se[3]
    elo_hb = c_hb2b / elo_slope * 100
    elo_ab = c_ab2b / elo_slope * 100
    p_even = 1 / (1 + np.exp(-intercept))
    dwin_hb = 1 / (1 + np.exp(-(intercept + c_hb2b))) - p_even
    dwin_ab = 1 / (1 + np.exp(-(intercept + c_ab2b))) - p_even

    n_hb = int(sub.hb2b.sum())
    n_ab = int(sub.ab2b.sum())
    print(f"n = {len(sub)} R games 2010-2026; home-on-b2b {n_hb}, away-on-b2b {n_ab}")
    print(f"Elo slope {elo_slope:.3f}/100; HFA intercept -> P(home|even, rested) "
          f"{p_even:.3f}")
    print(f"home-b2b: coef {c_hb2b:+.4f} (t {t_hb2b:+.1f}) -> dP(win) {dwin_hb:+.3f}, "
          f"Elo equiv {elo_hb:+.1f}")
    print(f"away-b2b: coef {c_ab2b:+.4f} (t {t_ab2b:+.1f}) -> dP(win) {dwin_ab:+.3f}, "
          f"Elo equiv {elo_ab:+.1f}")
    mean_elo = (abs(elo_hb) + abs(elo_ab)) / 2
    print(f"mean |Elo equivalent| = {mean_elo:.1f} (shipped constant: 38)")

    ok = (abs(dwin_hb - (-0.06)) <= 0.015 and abs(dwin_ab - 0.06) <= 0.015
          and abs(t_hb2b) >= 4 and abs(t_ab2b) >= 4 and 28 <= mean_elo <= 48)
    print(f"ACCEPTANCE (PLAN_V4 B3): {'REPRODUCED' if ok else 'NOT REPRODUCED'}")
    return {"dwin_hb2b": round(float(dwin_hb), 4), "dwin_ab2b": round(float(dwin_ab), 4),
            "t_hb2b": round(float(t_hb2b), 1), "t_ab2b": round(float(t_ab2b), 1),
            "elo_equiv_mean": round(float(mean_elo), 1), "n": int(len(sub)),
            "reproduced": bool(ok)}


if __name__ == "__main__":
    main()
