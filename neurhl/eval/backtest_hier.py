"""NeurHL-H — the hierarchical model, evaluated against gate S1.

Layer 1 (neural, residual, ~750k player-games) predicts each player's on-ice
shot rates and ice time; Layer 2 aggregates those TOI-weighted over the players
actually dressed into a projected team shot share; Layer 3 (a THIN head, a
handful of parameters) combines the two teams' projections with the incumbent
Elo logit and rest/travel into a game probability. Layer 3 is deliberately tiny
because team-games are the scarce resource (~7k) — all the capacity lives in
Layer 1 where the data is (~750k).

Evaluation is strictly walk-forward: for predict-season T the thin head is fit
only on seasons < T, and Layer 1's projections for T come from a model trained
only on seasons < T. Scored by S1, the paired significance test against v1.

Usage: ... python neurhl/eval/backtest_hier.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, TENSORS  # noqa: E402
from train.train_game import elo_features, load_frames  # noqa: E402

TUNE = [2012, 2013, 2014, 2015, 2016, 2017]
# season_end 2013 = the 2012-13 LOCKOUT season: 48 games/team, conference-only
# scheduling, no preseason after a four-month layoff, compressed calendar.
# Prereg P6 already treats COVID (2020/2021) as structurally anomalous and
# always breaks it out; the lockout was an omission from that rule and is
# handled the same way here (amendment A4). Reported BOTH ways, never silently
# dropped.
SHORTENED = {2013}
CS = (0.01, 0.03, 0.1, 0.3, 1.0)


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    seasons = list(range(2008, max(TUNE) + 1))
    gc, _ = load_frames(seasons)
    elo = elo_features(gc)
    # every season with a Layer-1 projection is available for TRAINING the
    # thin head; only TUNE seasons are ever evaluated. (Filtering to TUNE
    # before the walk-forward silently left 2012 with no training data.)
    have = sorted(int(p.stem.split("_")[-1])
                  for p in TENSORS.glob("proj_team_*.parquet"))
    proj = pd.concat([pd.read_parquet(TENSORS / f"proj_team_{s}.parquet")
                      for s in have])
    d = gc.set_index("game_id").join(elo).join(proj)
    d = d.dropna(subset=["elo_logit", "cfpct_h", "cfpct_a"])
    d["y"] = ((d.outcome4 == 0) | (d.outcome4 == 2)).astype(int)
    d["proj_diff"] = d.cfpct_h - d.cfpct_a
    cl_h = d.p_clf_h / (d.p_clf_h + d.p_cla_h).clip(lower=1e-6)
    cl_a = d.p_clf_a / (d.p_clf_a + d.p_cla_a).clip(lower=1e-6)
    d["proj_cl_diff"] = cl_h - cl_a
    d["rest_diff"] = (d.home_rest.clip(upper=7) - d.away_rest.clip(upper=7)) / 7
    d = d.fillna({"rest_diff": 0.0})

    COLS = ["elo_logit", "proj_diff", "proj_cl_diff", "rest_diff"]
    rows, per_season = [], {}
    for T in TUNE:
        tr = d[d.season_end < T]
        te = d[d.season_end == T]
        if len(tr) < 900 or not len(te):
            continue
        sc = StandardScaler().fit(tr[COLS])
        best = (9.0, None)
        for C in CS:                       # chosen on TRAIN seasons only
            m = LogisticRegression(C=C, max_iter=4000).fit(
                sc.transform(tr[COLS]), tr.y)
            l_tr = nll(m.predict_proba(sc.transform(tr[COLS]))[:, 1],
                       tr.y.to_numpy()).mean()
            if l_tr < best[0]:
                best = (l_tr, m)
        p = best[1].predict_proba(sc.transform(te[COLS]))[:, 1]
        p_elo = 1 / (1 + np.exp(-te.elo_logit.to_numpy()))
        rows.append(pd.DataFrame({"season": T, "ll_h": nll(p, te.y.to_numpy()),
                                  "ll_v1": nll(p_elo, te.y.to_numpy())}))
        per_season[str(T)] = {
            "n": int(len(te)), "neurhl_h": float(nll(p, te.y.to_numpy()).mean()),
            "v1": float(nll(p_elo, te.y.to_numpy()).mean())}
    R = pd.concat(rows, ignore_index=True)
    diff = (R.ll_h - R.ll_v1).to_numpy()
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    t = diff.mean() / se
    p_two = float(stats.norm.sf(abs(t)) * 2)
    cl = R.groupby("season").apply(lambda g: (g.ll_h - g.ll_v1).mean(),
                                   include_groups=False)
    cl_t = float(cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl))))
    # clustered test uses the t-distribution with (seasons - 1) df, NOT the
    # normal: with a handful of seasons the tails are much heavier
    cl_p = float(stats.t.sf(abs(cl_t), df=len(cl) - 1) * 2)
    wins = int((cl < 0).sum())
    sign_p = float(stats.binomtest(wins, len(cl), 0.5,
                                   alternative="two-sided").pvalue)
    # same statistics with structurally anomalous seasons excluded
    Rx = R[~R.season.isin(SHORTENED)]
    dx = (Rx.ll_h - Rx.ll_v1).to_numpy()
    sex = dx.std(ddof=1) / np.sqrt(len(dx))
    tx = dx.mean() / sex
    clx = Rx.groupby("season").apply(lambda g: (g.ll_h - g.ll_v1).mean(),
                                     include_groups=False)
    ex = {"n_games": int(len(dx)), "neurhl_h": float(Rx.ll_h.mean()),
          "v1": float(Rx.ll_v1.mean()), "diff": float(dx.mean()),
          "t": float(tx), "p_two_sided": float(stats.norm.sf(abs(tx)) * 2),
          "clustered_t": float(clx.mean() / (clx.std(ddof=1) / np.sqrt(len(clx)))),
          "seasons_won": int((clx < 0).sum()),
          "seasons_evaluated": int(len(clx)),
          "S1_pass": bool(dx.mean() < 0 and abs(tx) > 1.96)}

    res = {
        "model": "NeurHL-H (Layer1 neural residual -> Layer2 aggregate -> thin head)",
        "excluding_shortened_seasons": ex,
        "shortened_excluded": sorted(SHORTENED),
        "n_games": int(n), "neurhl_h": float(R.ll_h.mean()),
        "v1": float(R.ll_v1.mean()), "diff": float(diff.mean()),
        "se": float(se), "t": float(t), "p_two_sided": p_two,
        "clustered_t_by_season": cl_t,
        "clustered_p_t_dist": cl_p, "seasons_won": wins,
        "seasons_evaluated": int(len(cl)), "sign_test_p": sign_p,
        "per_season": per_season,
        "S1_pass": bool(diff.mean() < 0 and abs(t) > 1.96 and cl.mean() < 0),
        "S1_note": ("S1 as locked = per-game paired p<0.05. Season-clustered "
                    "and sign tests are reported alongside for completeness, "
                    "not as substitutes."),
        "G1_pass": bool(R.ll_h.mean() <= 0.67385)}
    (NOUT / "hier_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_season"},
                     indent=1))
    print("\nper season (NeurHL-H vs v1):")
    for k, v in per_season.items():
        print(f"  {k}: {v['neurhl_h']:.5f} vs {v['v1']:.5f}  "
              f"{'WIN' if v['neurhl_h'] < v['v1'] else 'loss'}")


if __name__ == "__main__":
    main()
