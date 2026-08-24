"""NeurHL-3 — M1 trainer: chained walk-forward player-game models.

Per vantage V (A7 window discipline, cost = one season of data):
  fit the chain on seasons < V-1, calibrate the binaries by isotonic on the
  SAME deployed models' out-of-sample predictions of season V-1, mean-match
  shots on V-1, then predict season V. Nothing from V is touched.

Chain (opportunity -> volume -> conversion): toi_share; shots (Poisson) with
toi_pred appended; goal1/assist1 with toi_pred+shots_pred appended.
`toi_share` predictions are renormalised within (game, team) to sum to 1 —
the conservation law is exact at every vantage.

Deterministic: fixed seed, CPU, NaN-native trees (sparse blocks enter under
availability masks).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models.player_game as PGM  # noqa: E402

SEED = 20260824
REG = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
           min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
           validation_fraction=0.1, random_state=SEED)


def _renorm(df: pd.DataFrame, col: str) -> pd.Series:
    tot = df.groupby(["game_id", "team"])[col].transform("sum")
    return df[col].clip(lower=1e-6) / tot.clip(lower=1e-6)


def fit_chain(frame: pd.DataFrame, v: int) -> dict:
    """Fit the walk-forward chain for vantage v; return everything needed to
    predict arbitrary feature rows (used by both the backtest and the season
    aggregation in sim/player_season_v2)."""
    fit = frame[frame.season_end < v - 1].copy()
    cal = frame[frame.season_end == v - 1].copy()
    if not len(fit):
        raise RuntimeError(f"vantage {v}: empty fit window")
    F = PGM.FEATURES
    drop = [c for c in F if fit[c].dropna().nunique() < 2]
    cols = [c for c in F if c not in drop]

    m_toi = HistGradientBoostingRegressor(**REG)
    m_toi.fit(fit[cols], fit["toi_share"])
    for d in (fit, cal):
        d.loc[:, "toi_pred_raw"] = m_toi.predict(d[cols])
    cols2 = cols + ["toi_pred_raw"]
    m_sh = HistGradientBoostingRegressor(loss="poisson", **REG)
    m_sh.fit(fit[cols2], fit["shots"])
    for d in (fit, cal):
        d.loc[:, "shots_pred_raw"] = m_sh.predict(d[cols2])
    cols3 = cols2 + ["shots_pred_raw"]
    isos = {}
    clfs = {}
    for tgt in ("goal1", "assist1"):
        m = HistGradientBoostingClassifier(**REG)
        m.fit(fit[cols3], fit[tgt])
        p_cal = m.predict_proba(cal[cols3])[:, 1]
        isos[tgt] = IsotonicRegression(out_of_bounds="clip").fit(
            p_cal, cal[tgt].to_numpy())
        clfs[tgt] = m
    scale = float(cal.shots.mean()
                  / max(m_sh.predict(cal[cols2]).mean(), 1e-9)) if len(cal) \
        else 1.0
    return {"cols": cols, "cols2": cols2, "cols3": cols3, "m_toi": m_toi,
            "m_sh": m_sh, "clfs": clfs, "isos": isos, "shots_scale": scale}


def predict_rows(ch: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted chain to arbitrary feature rows (returns a copy)."""
    d = rows.copy()
    d["toi_pred_raw"] = ch["m_toi"].predict(d[ch["cols"]])
    d["shots_pred_raw"] = ch["m_sh"].predict(d[ch["cols2"]])
    d["shots_pred"] = d.shots_pred_raw * ch["shots_scale"]
    for tgt, out in (("goal1", "p_goal"), ("assist1", "p_assist")):
        d[out] = ch["isos"][tgt].predict(
            ch["clfs"][tgt].predict_proba(d[ch["cols3"]])[:, 1])
    return d


def fit_predict_vantage(frame: pd.DataFrame, v: int) -> pd.DataFrame:
    """Return season-V rows with model predictions appended."""
    te = frame[frame.season_end == v].copy()
    if not len(te):
        raise RuntimeError(f"vantage {v}: empty test season")
    ch = fit_chain(frame, v)
    te = predict_rows(ch, te)
    te["toi_share_pred"] = _renorm(te, "toi_pred_raw")
    return te


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantages", type=int, nargs="*", required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    frame = PGM.build_frames()
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "data" / "tensors")
    for v in args.vantages:
        te = fit_predict_vantage(frame, v)
        keep = ["game_id", "player_id", "team", "opp", "season_end",
                "pos_group", "gp_todate", "toi_share", "shots", "goal1",
                "assist1", "toi_share_pred", "shots_pred", "p_goal",
                "p_assist", "toi_share_ewa1", "shots_ewa1", "goal_rate_ewa1",
                "assist_rate_ewa1"]
        te[keep].to_parquet(out_dir / f"player_game_preds_{v}.parquet",
                            index=False)
        print(f"{v}: {len(te):,} rows -> player_game_preds_{v}.parquet")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
