"""Exploratory: where does NeurHL-H's confirmed gain over Elo come from?

Splits the one-shot confirmation games (2018-2026, 2021 excluded; spent, so
exploratory only) by how far each team's dressed lineup departs from normal:
the ice time of absent regulars (absences_<s>: players with a recent regular
role who did not dress, weighted by their recent ice time), and by month of
the season. If the lineup is what NeurHL-H adds to Elo, the gain should be
largest where absences are large and lopsided, and early in the season, when
rosters have changed and Elo has not yet seen them play.

Writes output/eda_lineup_gain.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import NOUT, TENSORS  # noqa: E402


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summ(d):
    n = len(d)
    return {"n": int(n), "mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(n))}


def main():
    g = pd.read_csv(NOUT / "preds" / "hier_restatement_games.csv")
    g = g[g.season != 2021].copy()
    g["d"] = ll(g.p_neurhl_h, g.y) - ll(g.p_elo, g.y)
    ab = pd.concat([pd.read_parquet(TENSORS / f"absences_{s}.parquet").assign(season=s)
                    for s in sorted(g.season.unique())], ignore_index=True)
    ctx = pd.concat([pd.read_parquet(TENSORS / f"games_ctx_{s}.parquet",
                                     columns=["game_id", "home_idx", "away_idx"])
                     for s in sorted(g.season.unique())], ignore_index=True)
    sides = sorted(ab.team.unique())
    if set(sides) <= {0, 1}:
        ab["side"] = np.where(ab.team == 0, "home", "away")
    else:
        ab = ab.merge(ctx, on="game_id", how="left")
        ab["side"] = np.where(ab.team == ab.home_idx, "home",
                              np.where(ab.team == ab.away_idx, "away", None))
    load = ab.groupby(["game_id", "side"]).ewma_toi_sec.sum().unstack(fill_value=0.0) / 60.0
    load = load.reindex(columns=["home", "away"], fill_value=0.0)
    g = g.merge(load.rename(columns={"home": "abs_home_min", "away": "abs_away_min"}),
                left_on="game_id", right_index=True, how="left").fillna(
        {"abs_home_min": 0.0, "abs_away_min": 0.0})
    g["abs_total"] = g.abs_home_min + g.abs_away_min
    g["abs_gap"] = (g.abs_home_min - g.abs_away_min).abs()
    out = {"window": "2018-2026 excluding 2021 (spent; exploratory)", "n_games": int(len(g)),
           "overall": summ(g.d)}
    # lopsidedness of absences: minutes of regular ice time missing, home minus away
    bins = [-0.01, 0.0, 20.0, 40.0, 1e9]
    labels = ["none", "0-20 min", "20-40 min", "over 40 min"]
    g["gap_bin"] = pd.cut(g.abs_gap, bins=bins, labels=labels)
    out["by_absence_gap"] = {str(k): summ(v.d) for k, v in g.groupby("gap_bin", observed=True)}
    g["tot_bin"] = pd.qcut(g.abs_total.rank(method="first"), 4, labels=["Q1 (fewest)", "Q2", "Q3", "Q4 (most)"])
    out["by_absence_total_quartile"] = {str(k): summ(v.d) for k, v in g.groupby("tot_bin", observed=True)}
    # early vs late season
    g["date"] = pd.to_datetime(g.date)
    g["month"] = g.date.dt.month
    mname = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
             6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
    order = [10, 11, 12, 1, 2, 3, 4]
    out["by_month"] = {mname[m]: summ(g[g.month == m].d) for m in order if (g.month == m).sum() > 50}
    # does H move away from Elo more where absences are lopsided?
    g["shift"] = (g.p_neurhl_h - g.p_elo).abs()
    out["mean_abs_prob_shift_by_gap"] = {str(k): float(v["shift"].mean())
                                         for k, v in g.groupby("gap_bin", observed=True)}
    # linear association of the gain with the gap, per 10 minutes
    X = np.c_[np.ones(len(g)), g.abs_gap / 10.0]
    beta, *_ = np.linalg.lstsq(X, g.d.to_numpy(), rcond=None)
    resid = g.d.to_numpy() - X @ beta
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / (len(g) - 2)
    out["slope_per_10min_gap"] = {"beta": float(beta[1]), "se": float(np.sqrt(cov[1, 1]))}
    (NOUT / "eda_lineup_gain.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
