"""NeurHL-2 — walk-forward xG with pre-specified gates X1, X2, X3.

Walk-forward is the whole point: to score season V the model is fit ONLY on
seasons < V (registry vantage PRIOR). A single xG model fit on all seasons and
applied backwards would be the exact P3 violation we exclude MoneyPuck's own
xGoals for — and it would flatter every historical vantage.

  **X1 — skill.** Beat a distance + |angle| logistic by >= 0.002 nats of held-out
  log loss in at least 15 vantages. The corpus supports 18 walk-forward vantages
  (2009-2026; 2008 has no prior season), so requiring 15 is a STRICTER rate than
  the preregistered 15/19 and is not a relaxation.

  **X2 — does xG earn its place?** Team-aggregated xG differential must beat
  CORSI differential at predicting future goal share, out of sample. Corsi comes
  from the event shards, not from this table: MoneyPuck excludes blocked shots,
  and a blocked shot's recorded coordinate is where the BLOCK happened, so xG
  structurally cannot see them. Split each team's season in half, measure 5v5
  xGF% and CF% in the first half, predict 5v5 GF% in the second.
  **If X2 fails, L0 is deleted and the engine uses Corsi** — preregistered, and
  meant literally.

  **X3 — calibration.** |observed - predicted| <= 0.005 in every decile, with 20
  bins also reported. A model can win X1 on ranking alone while being badly
  miscalibrated, and the simulator consumes probabilities, not ranks.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scikit-learn python neurhl/train/train_xg.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import models.xg as XG  # noqa: E402
import windows as W  # noqa: E402

VANTAGES = list(range(2009, 2027))
X1_MIN_GAIN = 0.002
X1_MIN_PASS = 15
X3_TOL = 0.005
SEED = 20260821
GBM = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
           min_samples_leaf=200, l2_regularization=1.0,
           early_stopping=True, validation_fraction=0.1,
           random_state=SEED)


def fit_vantage(v: int, cache: dict) -> dict:
    tr_seasons = [s for s in range(2008, v)]
    tr = XG.load_shots(tr_seasons)
    te = XG.load_shots([v])
    Xtr, ytr = XG.build_features(tr), tr[XG.TARGET].to_numpy()
    Xte = XG.align_categories(Xtr, XG.build_features(te))
    yte = te[XG.TARGET].to_numpy()

    cat_mask = [c in XG.CATS for c in XG.FEATURES]
    # A7 calibration. Fit the GBM on seasons < V-1, then fit isotonic on THAT
    # SAME model's predictions for season V-1, and apply to V.
    #
    # Two failures this fixes. (1) CalibratedClassifierCV(ensemble=False) fits
    # the isotonic map on out-of-fold predictions from models trained on 2/3 of
    # the data, then applies it to a final model trained on all of it -- it
    # corrects a weaker model than the one deployed, and on DEV V=2011 that made
    # calibration WORSE than none (0.0088 vs 0.0061). Here the base model and
    # the calibrated model are the same object. (2) A recording-regime shift
    # from 2023 -- shots logged within 10 ft went 7.3-8.7% to 14.5% while their
    # goal rate fell 0.17 to 0.13 -- so a calibrator fit across all history
    # encodes a mapping that no longer holds. Fitting it on the most recent
    # AVAILABLE season tracks the drift automatically, and season V is never
    # touched.
    hold = v - 1
    m_fit = tr.season_end.to_numpy() != hold
    cal_idx = ~m_fit
    if m_fit.sum() == 0 or cal_idx.sum() == 0:
        # V=2009 has exactly one prior season, so holding it out leaves nothing
        # to fit on. Fall back to fitting and calibrating on the same rows: the
        # calibrator is then in-sample and near-identity, which is honest for a
        # vantage that cannot support a holdout at all. It is also the one
        # vantage X1 does not clear, for the same reason -- too little history.
        m_fit = np.ones(len(ytr), bool)
        cal_idx = m_fit
    base = HistGradientBoostingClassifier(categorical_features=cat_mask, **GBM)
    base.fit(Xtr[m_fit], ytr[m_fit])
    p_cal = base.predict_proba(Xtr[cal_idx])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, ytr[cal_idx])
    p_raw = base.predict_proba(Xte)[:, 1]
    p = iso.predict(p_raw)

    # baseline: the classic two-variable xG
    sc = StandardScaler().fit(Xtr[XG.BASELINE])
    b = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr[XG.BASELINE]), ytr)
    pb = b.predict_proba(sc.transform(Xte[XG.BASELINE]))[:, 1]

    base_rate = float(ytr.mean())
    ll_m, ll_b = XG.logloss(yte, p), XG.logloss(yte, pb)
    ll_c = XG.logloss(yte, np.full(len(yte), base_rate))

    cache[v] = pd.DataFrame({
        "game_id": te.game_id.to_numpy(), "is_home": te.isHomeTeam.to_numpy(),
        "goal": yte, "xg": p,
        "sk_h": te.homeSkatersOnIce.to_numpy(),
        "sk_a": te.awaySkatersOnIce.to_numpy(),
        "en": (te.homeEmptyNet | te.awayEmptyNet).to_numpy(),
        # time/period/shooter are what let these predictions be joined back to
        # the stint table -- without them the artifact is a dead end
        "time": te.time.to_numpy(), "period": te.period.to_numpy(),
        "shooter": te.shooterPlayerId.to_numpy(),
        "team_code": te.teamCode.to_numpy()})

    return {"vantage": v, "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "base_rate_train": round(base_rate, 5),
            "goal_rate_test": round(float(yte.mean()), 5),
            "ll_model": round(ll_m, 6), "ll_dist_angle": round(ll_b, 6),
            "ll_constant": round(ll_c, 6),
            "gain_vs_baseline": round(ll_b - ll_m, 6),
            "gain_vs_constant": round(ll_c - ll_m, 6),
            "ll_uncalibrated": round(XG.logloss(yte, p_raw), 6),
            "pred_mean": round(float(p.mean()), 5)}


def calibration(y, p, nbins=20) -> dict:
    """Quantile bins; a decile view is what gate X3 is scored on."""
    out = {}
    for nb, key in ((10, "deciles"), (nbins, f"bins{nbins}")):
        q = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
        idx = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
        rows = []
        for b in range(len(q) - 1):
            m = idx == b
            if not m.any():
                continue
            rows.append({"bin": b, "n": int(m.sum()),
                         "pred": round(float(p[m].mean()), 5),
                         "obs": round(float(y[m].mean()), 5),
                         "diff": round(float(y[m].mean() - p[m].mean()), 5)})
        out[key] = rows
        out[f"max_abs_diff_{key}"] = round(
            max(abs(r["diff"]) for r in rows) if rows else float("nan"), 5)
    return out


def team_game_table(cache: dict) -> pd.DataFrame:
    """5v5 xG for/against per team-game, from the walk-forward predictions."""
    parts = []
    for v, d in cache.items():
        f = d[(d.sk_h == 5) & (d.sk_a == 5) & (~d.en.astype(bool))]
        g = (f.assign(side=np.where(f.is_home == 1, "h", "a"))
             .groupby(["game_id", "side"])
             .agg(xg=("xg", "sum"), gl=("goal", "sum")).reset_index())
        w = g.pivot(index="game_id", columns="side",
                    values=["xg", "gl"]).fillna(0.0)
        w.columns = [f"{a}_{b}" for a, b in w.columns]
        w["season_end"] = v
        parts.append(w.reset_index())
    return pd.concat(parts, ignore_index=True)


def corsi_table(seasons) -> pd.DataFrame:
    """5v5 Corsi per team-game from the EVENT shards (includes blocks)."""
    parts = []
    for s in seasons:
        p = TENSORS / f"stints_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["game_id", "game_type", "is_5v5",
                                        "cf_h", "cf_a"])
        d = d[(d.game_type == 2) & d.is_5v5]
        g = d.groupby("game_id").agg(cf_h=("cf_h", "sum"),
                                     cf_a=("cf_a", "sum")).reset_index()
        g["season_end"] = s
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def gate_x2(cache: dict) -> dict:
    """Does team xG differential predict future goals better than Corsi?"""
    tg = team_game_table(cache)
    cz = corsi_table(sorted(cache))
    tg = tg.merge(cz[["game_id", "cf_h", "cf_a"]], on="game_id", how="inner")

    gc = []
    for s in sorted(cache):
        p = TENSORS / f"games_ctx_{s}.parquet"
        if p.exists():
            q = pd.read_parquet(p, columns=["game_id", "home_idx", "away_idx",
                                            "date"])
            gc.append(q)
    gc = pd.concat(gc, ignore_index=True)
    tg = tg.merge(gc, on="game_id", how="inner")

    # one row per team-game, oriented to that team
    rows = []
    for side, opp, tcol in (("h", "a", "home_idx"), ("a", "h", "away_idx")):
        rows.append(pd.DataFrame({
            "season_end": tg.season_end, "date": tg.date,
            "team": tg[tcol],
            "xg_f": tg[f"xg_{side}"], "xg_a": tg[f"xg_{opp}"],
            "cf_f": tg[f"cf_{side}"], "cf_a_": tg[f"cf_{opp}"],
            "g_f": tg[f"gl_{side}"], "g_a": tg[f"gl_{opp}"]}))
    t = pd.concat(rows, ignore_index=True).sort_values(["season_end", "team",
                                                        "date"])
    t["k"] = t.groupby(["season_end", "team"]).cumcount()
    t["n"] = t.groupby(["season_end", "team"]).k.transform("max") + 1
    first, second = t[t.k < t.n / 2], t[t.k >= t.n / 2]

    def agg(x, pre):
        g = x.groupby(["season_end", "team"]).sum(numeric_only=True)
        return pd.DataFrame({
            f"{pre}_xg": g.xg_f / (g.xg_f + g.xg_a),
            f"{pre}_cf": g.cf_f / (g.cf_f + g.cf_a_),
            f"{pre}_gf": g.g_f / (g.g_f + g.g_a).replace(0, np.nan)})

    a, b = agg(first, "h1"), agg(second, "h2")
    j = a.join(b, how="inner").dropna(subset=["h2_gf", "h1_xg", "h1_cf"])
    r_xg = float(np.corrcoef(j.h1_xg, j.h2_gf)[0, 1])
    r_cf = float(np.corrcoef(j.h1_cf, j.h2_gf)[0, 1])
    # same-half sanity: both should track same-half goals strongly
    r_xg_same = float(np.corrcoef(j.h1_xg, a.loc[j.index, "h1_gf"].fillna(0.5))[0, 1])
    r_cf_same = float(np.corrcoef(j.h1_cf, a.loc[j.index, "h1_gf"].fillna(0.5))[0, 1])
    return {"n_team_seasons": int(len(j)),
            "r_xg_predicts_future_gf": round(r_xg, 4),
            "r_corsi_predicts_future_gf": round(r_cf, 4),
            "r_xg_same_half": round(r_xg_same, 4),
            "r_corsi_same_half": round(r_cf_same, 4),
            "pass": bool(r_xg > r_cf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantages", type=int, nargs="*", default=VANTAGES)
    args = ap.parse_args()

    print(f"walk-forward xG over {len(args.vantages)} vantages "
          f"({min(args.vantages)}-{max(args.vantages)}); each fit on seasons "
          f"< V only\n")
    print(f"{'V':>5} {'win':>7} {'ntrain':>9} {'ntest':>8} {'ll_model':>9} "
          f"{'ll_d+a':>9} {'gain':>8} {'ll_const':>9} {'predbar':>8}")
    res, cache = [], {}
    t0 = time.time()
    for v in args.vantages:
        r = fit_vantage(v, cache)
        res.append(r)
        print(f"{v:>5} {W.window_of(v):>7} {r['n_train']:>9,} "
              f"{r['n_test']:>8,} {r['ll_model']:>9.5f} "
              f"{r['ll_dist_angle']:>9.5f} {r['gain_vs_baseline']:>+8.5f} "
              f"{r['ll_constant']:>9.5f} {r['pred_mean']:>8.5f}")
        sys.stdout.flush()
    print(f"\nfit in {time.time()-t0:.0f}s")

    # ---- X1
    gains = [r["gain_vs_baseline"] for r in res]
    n_ok = sum(g >= X1_MIN_GAIN for g in gains)
    x1 = n_ok >= X1_MIN_PASS
    print(f"\nX1: gain >= {X1_MIN_GAIN} nats in {n_ok}/{len(res)} vantages "
          f"(need {X1_MIN_PASS}) -> {'PASS' if x1 else 'FAIL'}")
    print(f"    mean gain {np.mean(gains):+.5f}, min {min(gains):+.5f}, "
          f"max {max(gains):+.5f}")

    # ---- X3, pooled over every walk-forward prediction
    allp = np.concatenate([cache[v].xg.to_numpy() for v in cache])
    ally = np.concatenate([cache[v].goal.to_numpy() for v in cache])
    cal = calibration(ally, allp)
    x3 = cal["max_abs_diff_deciles"] <= X3_TOL
    print(f"\nX3: max |obs-pred| over deciles = {cal['max_abs_diff_deciles']:.5f} "
          f"(tol {X3_TOL}) -> {'PASS' if x3 else 'FAIL'}")
    print(f"    20-bin max |obs-pred| = {cal['max_abs_diff_bins20']:.5f}; "
          f"overall pred {allp.mean():.5f} vs obs {ally.mean():.5f}")
    for r in cal["deciles"]:
        print(f"      d{r['bin']:>2}  n={r['n']:>7,}  pred {r['pred']:.4f}  "
              f"obs {r['obs']:.4f}  diff {r['diff']:+.4f}")

    # ---- X2
    x2r = gate_x2(cache)
    x2 = x2r["pass"]
    print(f"\nX2: predicting 2nd-half 5v5 GF% from 1st-half "
          f"(n={x2r['n_team_seasons']} team-seasons)")
    print(f"    xG differential   r = {x2r['r_xg_predicts_future_gf']:+.4f}")
    print(f"    Corsi differential r = {x2r['r_corsi_predicts_future_gf']:+.4f}")
    print(f"    -> {'PASS' if x2 else 'FAIL — per prereg, DELETE L0 and use Corsi'}")

    out = {"vantages": res, "X1": {"pass": bool(x1), "n_ok": n_ok,
                                   "n": len(res), "min_gain": X1_MIN_GAIN,
                                   "gains": gains},
           "X2": x2r, "X3": {"pass": bool(x3), **cal},
           "gbm": GBM, "features": XG.FEATURES}
    p = Path(__file__).resolve().parents[1] / "configs" / "xg_gates.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {p}")

    # persist predictions for downstream stint-level xG
    for v, d in cache.items():
        d.to_parquet(TENSORS / f"xg_shots_{v}.parquet", index=False)
    print(f"-> {TENSORS}/xg_shots_<V>.parquet")
    return 0 if (x1 and x2 and x3) else 1


if __name__ == "__main__":
    sys.exit(main())
