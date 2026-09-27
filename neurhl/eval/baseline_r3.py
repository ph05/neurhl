"""Rung R3 (PLAN_NeurHL4 L): gradient-boosted trees and a logistic model on
game-level features from the same master tensor, walk-forward by season.

Does the network earn its place over trees? Features per game (home minus away
unless noted): Elo logit (home), team state (every tm_* column), starting
goalie state, lineup aggregates (TOI-baseline-weighted sums of skater shrunk
baselines: xGF60, xGA60, ixG60, SOG60), rest and travel. For predict-season T
the models train on seasons < T. Scored on the window given; appends ledger rows.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIGS  # noqa: E402
import windows as W  # noqa: E402
from data.g_loader import load_master  # noqa: E402
from eval.run_g import append_ledger, fit_stack, h_preds, logit, nll  # noqa: E402


def features(A, names):
    tm, gk, sk = A["TM"], A["GK"], A["SKB"]
    m = A["SKM"]
    bi = {c: i for i, c in enumerate(names["sk_base"])}
    w = np.nan_to_num(sk[..., bi["b_toi_ev"]]) * m
    w = w / np.clip(w.sum(-1, keepdims=True), 1e-6, None)
    lin = np.stack([(w * np.nan_to_num(sk[..., bi[k]])).sum(-1)
                    for k in ("b_xgf_ev60", "b_xga_ev60", "b_ixg60", "b_isog60")], -1)
    parts = [A["CTX"][:, :1],
             tm[:, 0] - tm[:, 1],
             gk[:, 0] - gk[:, 1],
             lin[:, 0] - lin[:, 1]]
    return np.concatenate(parts, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=("iter", "gate"), default="iter")
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    win = W.G_ITER if a.window == "iter" else W.G_GATE
    W.assert_scorable_g(win, a.window)
    A, meta, names = load_master("train")
    X = features(A, names)
    X = np.where(np.isfinite(X), X, np.nan)
    y = meta.outcome4.isin([0, 2]).to_numpy().astype(float)
    s = meta.season_end.to_numpy()
    out = []
    for T in sorted(set(range(2011, max(win) + 1)) - W.NO_SCORE):
        tr, te = (s < T), (s == T)
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03,
                                             max_leaf_nodes=15, l2_regularization=1.0,
                                             random_state=T).fit(X[tr], y[tr])
        lr = make_pipeline(SimpleImputer(), StandardScaler(),
                           LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y[tr])
        out.append(pd.DataFrame({"game_id": meta.game_id[te], "season": T, "y": y[te],
                                 "p_gbm": gbm.predict_proba(X[te])[:, 1],
                                 "p_lr": lr.predict_proba(X[te])[:, 1],
                                 "p_elo": 1 / (1 + np.exp(-A["CTX"][te, 0]))}))
    P = pd.concat(out, ignore_index=True)
    elo_l = logit(P.p_elo)
    for k in ("gbm", "lr"):                 # same walk-forward Elo stack as NeurHL-G
        P[f"p_{k}_st"] = np.nan
        for T in win:
            tr, te = P.season < T, P.season == T
            Xs = lambda d: np.column_stack([logit(d.p_elo), logit(d[f"p_{k}"])])
            mdl, use = fit_stack(Xs(P[tr]), P.y[tr].to_numpy(), P.season[tr].to_numpy())
            P.loc[te, f"p_{k}_st"] = mdl.predict_proba(Xs(P[te])[:, use])[:, 1]
    P = P[P.season.isin(win)].copy()
    P["p_gbm"], P["p_lr"] = P.p_gbm_st, P.p_lr_st
    P["p_h"] = P.game_id.map(h_preds(win))
    res = {k: float(nll(P[k], P.y).mean()) for k in ("p_gbm", "p_lr", "p_elo", "p_h")}
    res["n"] = int(len(P))
    print(json.dumps(res, indent=1))
    for k in ("gbm", "lr"):
        d = nll(P[f"p_{k}"], P.y) - nll(P.p_elo, P.y)
        row = {"run_id": f"r3_{k}:full:{a.window}",
               "date": dt.datetime.now().isoformat(timespec="minutes"),
               "parent": "", "delta": f"R3 {k} on game-level features", "seeds": 1,
               "n": len(P), f"ll_stack": res[f"p_{k}"], "ll_elo": res["p_elo"],
               "ll_h": res["p_h"], "d_vs_elo": float(d.mean()),
               "z_vs_elo": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))}
        append_ledger(row)


if __name__ == "__main__":
    main()
