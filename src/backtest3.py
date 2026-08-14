"""v3 train-only gates (pre-registered below, written before results are computed).

GATE F (features): each candidate in {finishing, st_pp, st_pk} enters production iff
  add-one-in LOSO dMAE(h1, train) <= +0.02 AND sign stability >= 0.67; the final set must
  satisfy LOSO MAE <= v2-set MAE + 0.05, else fall back to the v2 set.
GATE A (availability noise): enters sims iff, after refitting sigma_c with the noise ON,
  train 80% coverage stays in [0.72, 0.90] AND coverage spread across fragility tertiles
  does not exceed the uniform-sigma spread by more than 0.02.
The 2018-2026 windows remain untouched. The live 2026-27 season is the out-of-sample test.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import availability as A
import engine as E
import players as P
import ridge as R
from features import FEATURES, FEATURES_V3, FeatureBuilder
from overlay import skater_value_goals, points_per_goal

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
TRAIN_H1 = list(range(2012, 2018))
NEW_FEATS = ["finishing", "st_pp", "st_pk"]

PREREG = {
    "gate_features": "add-one-in LOSO dMAE(h1,train) <= +0.02 and sign_stability >= 0.67 "
                     "per candidate; final set LOSO <= v2 set + 0.05 else fallback",
    "gate_availability": "80% train coverage in [0.72,0.90] with noise on and refit sigma_c; "
                         "fragility-tertile spread <= uniform spread + 0.02",
    "windows": "train <= 2017 only; 2018-2026 untouched; 2026-27 live = true holdout",
}


def main():
    t0 = time.time()
    p2 = json.loads((OUT / "params_v2.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    p3 = {"prereg_v3": PREREG}
    (OUT / "params_v3.json").write_text(json.dumps(p3, indent=2))  # prereg on disk first

    g, ts = E.load()
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"], skater_delta=p2["skater_delta"])
    sk, go, skt, got, bios = P.load_panels()

    # ---------- GATE F ----------
    X, y, meta = fb.feature_matrix(TRAIN_H1, 1, feats=FEATURES_V3)
    seasons = meta["T"].to_numpy()

    def cols(feats):
        return X[:, [FEATURES_V3.index(f) for f in feats]]

    lam_v2, tab_v2 = R.loso_cv(cols(FEATURES), y, seasons)
    mae_v2 = tab_v2.mae.min()
    keep = []
    ledger_rows = []
    for f in NEW_FEATS:
        set_f = FEATURES + [f]
        lam_f, tab_f = R.loso_cv(cols(set_f), y, seasons)
        dmae = tab_f.mae.min() - mae_v2
        signs = R.loso_coef_signs(cols(set_f), y, seasons, lam_f)
        stab = float(signs[-1])
        passed = (dmae <= 0.02) and (stab >= 0.67)
        if passed:
            keep.append(f)
        ledger_rows.append({"feature": f, "horizon": 1, "add_one_in_dMAE": round(dmae, 4),
                            "sign_stability": round(stab, 2),
                            "status": "in model (v3)" if passed else "dropped (v3 gate)"})
        print(f"GATE F {f}: dMAE {dmae:+.4f}, stability {stab:.2f} -> "
              f"{'KEEP' if passed else 'DROP'}")
    final_set = FEATURES + keep
    lam_f, tab_f = R.loso_cv(cols(final_set), y, seasons)
    mae_f = tab_f.mae.min()
    if mae_f > mae_v2 + 0.05:
        print(f"final set {mae_f:.3f} vs v2 {mae_v2:.3f}: fallback to v2 set")
        final_set, lam_f, mae_f = list(FEATURES), float(tab_v2.sort_values(['mae','lam']).iloc[0].lam), mae_v2
    print(f"GATE F result: set={len(final_set)} features (kept: {keep}), "
          f"LOSO {mae_f:.3f} vs v2-set {mae_v2:.3f}, lam {lam_f}")

    # h2 lambda for the chosen set (for the bonus sheet)
    Xh2, yh2, meta2 = fb.feature_matrix(list(range(2013, 2018)), 2, feats=FEATURES_V3)
    lam_f2, tab_f2 = R.loso_cv(Xh2[:, [FEATURES_V3.index(f) for f in final_set]],
                               yh2, meta2["T"].to_numpy())

    # ---------- GATE A ----------
    c, k = p2["c"], p2["k"]
    ppg = points_per_goal(sk, ts, 2017)
    rng = np.random.default_rng(77)
    lam = lam_f
    Xf = cols(final_set)
    resid2, preds_by_T, noise_by_T = [], {}, {}
    for T in TRAIN_H1:
        tr, te = seasons != T, seasons == T
        b = R.fit_ridge(Xf[tr], y[tr], lam)
        pr = pd.Series(Xf[te] @ b, index=meta.loc[te, "team"].to_numpy())
        pr -= pr.mean()
        preds_by_T[T] = pr
        resid2 += list((pr.to_numpy() - (y[te] - y[te].mean())) ** 2)
        V = T - 1
        par = A.fit_availability(sk, bios, V)
        marcel = fb.marcel(V, 1)
        repl = P.replacement_rates(sk, V)
        vals = skater_value_goals(marcel, repl, ppg)
        roster = P.majority_team(skt[skt.season_end == V])
        ages = P.age_of(bios, roster.playerId, V + 1)
        rosters = {}
        for team, d in roster.groupby("team"):
            vv = vals.reindex(d.playerId).fillna(0.0)
            aa = P.age_of(bios, d.playerId, V + 1)
            top = sorted(zip(aa.to_numpy(), vv.to_numpy()),
                         key=lambda t_: -t_[1])[:A.TOP_N]
            rosters[team] = [(float(a_) if np.isfinite(a_) else 27.0, float(v_))
                             for a_, v_ in top]
        noise_by_T[T] = (par, rosters)
    target = float(np.mean(resid2))

    def run_cov(sigma, with_noise):
        cover, frag = [], []
        for T in TRAIN_H1:
            pr = preds_by_T[T]
            ratings = {t: 1505 + v / c for t, v in pr.items()}
            sched = E.actual_schedule(g, T)
            ratings = E.fill_missing(ratings, set(sched.home) | set(sched.away), 1505.0)
            om_c = E.fit_outcome(preds, list(range(2006, T)))
            teams_sorted = sorted(ratings)
            par, rosters = noise_by_T[T]
            noise_fn = A.make_extra_noise(teams_sorted, rosters, par, k)
            probe = noise_fn(400, np.random.default_rng(1))
            frag_sd = dict(zip(teams_sorted, probe.std(axis=0)))  # fragility: team property
            sim = E.simulate_season(ratings, sigma, sched, om_c, E.divisions_for(T),
                                    1200, rng, playoffs=False,
                                    extra_noise=noise_fn if with_noise else None)
            lo = np.percentile(sim["pts"], 10, axis=0)
            hi = np.percentile(sim["pts"], 90, axis=0)
            act = pd.read_csv(PROJ / "data/processed/team_seasons.csv")
            act = act[act.season_end == T].set_index("team")
            for i, t in enumerate(sim["teams"]):
                if t in act.index:
                    cover.append(int(lo[i] <= act.loc[t].pts <= hi[i]))
                    frag.append(frag_sd.get(t, 0.0))
        return np.array(cover), np.array(frag)

    # variance-match sigma_c with noise ON
    best = None
    for sigma in (20.0, 25.0, 30.0, 35.0, 40.0):
        vars_ = []
        for T in TRAIN_H1:
            par, rosters = noise_by_T[T]
            teams_sorted = sorted(preds_by_T[T].index)
            extra = A.make_extra_noise(teams_sorted, rosters, par, k)
            ratings = {t: 1505 + v / c for t, v in preds_by_T[T].items()}
            sched = E.actual_schedule(g, T)
            ratings = E.fill_missing(ratings, set(sched.home) | set(sched.away), 1505.0)
            om_c = E.fit_outcome(preds, list(range(2006, T)))
            sim = E.simulate_season(ratings, sigma, sched, om_c, E.divisions_for(T),
                                    800, rng, playoffs=False, extra_noise=extra)
            gp = 82.0
            vars_.append((sim["pts"].std(axis=0, ddof=1)).mean() ** 2)
        gap = abs(np.mean(vars_) - target)
        if best is None or gap < best[1]:
            best = (sigma, gap)
    sigma_c3 = best[0]
    print(f"sigma_c refit with availability noise: {sigma_c3} (target var {target:.0f})")

    cov_u, frag_u = run_cov(p2["sigma_c"]["h1"], with_noise=False)
    cov_n, frag = run_cov(sigma_c3, with_noise=True)
    tert = pd.qcut(pd.Series(frag), 3, labels=False, duplicates="drop")
    by_t_n = pd.Series(cov_n).groupby(tert).mean()
    by_t_u = pd.Series(cov_u).groupby(pd.qcut(pd.Series(frag_u), 3, labels=False,
                                              duplicates="drop")).mean()
    spread_n = float((by_t_n - 0.8).abs().max())
    spread_u = float((by_t_u - 0.8).abs().max())
    keep_avail = (0.72 <= cov_n.mean() <= 0.90) and (spread_n <= spread_u + 0.02)
    print(f"GATE A: coverage with noise {cov_n.mean():.3f} (uniform {cov_u.mean():.3f}), "
          f"tertile spread {spread_n:.3f} -> {'KEEP' if keep_avail else 'DROP'}")

    p3.update({
        "final_feature_set": final_set, "kept_new_features": keep,
        "lam_h1": lam_f, "lam_h2": float(tab_f2.sort_values(['mae', 'lam']).iloc[0].lam),
        "train_mae_h1": {"v2_set": round(mae_v2, 3), "v3_set": round(mae_f, 3)},
        "availability": {"keep": bool(keep_avail), "sigma_c3": sigma_c3,
                         "coverage": round(float(cov_n.mean()), 3),
                         "tertile_spread": round(spread_n, 3)},
        "ledger_v3": ledger_rows,
        "seconds": round(time.time() - t0, 1),
    })
    (OUT / "params_v3.json").write_text(json.dumps(p3, indent=2))
    print(f"params_v3.json written ({p3['seconds']}s)")


if __name__ == "__main__":
    main()
