"""v4 train-only gates (PLAN_V4, prereg committed 2026-08-17 BEFORE this ran).

GATE G  (goalie game layer):   coverage in [0.72,0.90] with sigma refit; tandem-gap
                               tertile spread <= no-layer spread + 0.02; max |mean pts
                               shift| < 0.4 (zero-mean verification).
SCREENS S1/S2 + n0_a tuning:   skater persistence corr >= 0.10; the tuned EB must also
                               beat the nested bucket-only baseline on its own MSE
                               metric, else persistence is dropped (spirit of the
                               ledger: nothing ships without beating its baseline).
GATE A2 (availability 2.0):    coverage in band AND fragility-tertile spread <=
                               v3-availability spread + 0.02.
GATE F2 (prospect_pipeline):   per horizon: add-one-in train LOSO dMAE <= +0.02 AND
                               sign stability >= 0.67; final set <= incumbent + 0.05.

2018-2026 untouched. All replays: train predict-seasons 2012-2017, h1. b2b d_adj is NOT
applied in replays (matches the v3 gate baseline for comparability); b2b flags are used
only for the goalie layer's start probabilities.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import availability as A3
import availability2 as A2
import engine as E
import goalie_game as GG
import players as P
import ridge as R
import scoring as SC
from features import FEATURES_V3, FEATURES_V4, FeatureBuilder
from overlay import points_per_goal, skater_value_goals

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
TRAIN_H1 = list(range(2012, 2018))
SIGMA_GRID = (20.0, 25.0, 30.0, 35.0, 40.0)
N_SIMS_FIT = 1200
N_SIMS_COV = 2000

PREREG = {
    "gate_G": "goalie layer: 80% coverage in [0.72,0.90] w/ sigma refit; tandem-gap "
              "tertile spread <= no-layer + 0.02; max |mean shift| < 0.4 pts",
    "gate_A2": "availability2: coverage in [0.72,0.90]; fragility-tertile spread <= "
               "v3-availability spread + 0.02; failed screens drop components",
    "screens": "S1 skater persistence r>=0.10 AND tuned EB beats bucket-only MSE; "
               "S2 goalie starter overdispersion >= 5x",
    "gate_F2": "prospect_pipeline per horizon: add-one-in LOSO dMAE <= +0.02, sign "
               "stability >= 0.67; final set <= incumbent + 0.05",
    "windows": "train <= 2017 only; 2018-2026 untouched; live 2026-27 = holdout",
    "committed_before_results": "PLAN_V4.md @ git 4bb92d9",
}


def main():
    t0 = time.time()
    (OUT / "params_v4_gates.json").write_text(json.dumps(PREREG, indent=2))

    p2 = json.loads((OUT / "params_v2.json").read_text())
    p3 = json.loads((OUT / "params_v3.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    c, k = p2["c"], p2["k"]
    g, ts = E.load()
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"], skater_delta=p2["skater_delta"])
    sk, go, skt, got, bios = P.load_panels()
    feats3 = p3["final_feature_set"]
    lam3 = p3["lam_h1"]
    p4 = {"prereg_ref": "params_v4_gates.json"}

    # ---------------- LOSO ridge predictions per train season (v3 final set)
    X, y, meta = fb.feature_matrix(TRAIN_H1, 1, feats=FEATURES_V4)
    seasons = meta["T"].to_numpy()
    idx3 = [FEATURES_V4.index(f) for f in feats3]
    X3 = X[:, idx3]
    preds_by_T, resid2 = {}, []
    for T in TRAIN_H1:
        tr, te = seasons != T, seasons == T
        b = R.fit_ridge(X3[tr], y[tr], lam3)
        pr = pd.Series(X3[te] @ b, index=meta.loc[te, "team"].to_numpy())
        pr -= pr.mean()
        preds_by_T[T] = pr
        resid2 += list((pr.to_numpy() - (y[te] - y[te].mean())) ** 2)
    target_var = float(np.mean(resid2))
    print(f"replay target resid var {target_var:.1f} (sd {np.sqrt(target_var):.2f})")

    # ---------------- per-season fixtures (schedules, flags, noise ingredients)
    n0_g, tab_g = GG.tune_n0_g(got, ts)
    print(f"n0_g grid:\n{tab_g.round(5).to_string(index=False)}\nchosen n0_g = {n0_g}")
    ppg = points_per_goal(sk, ts, 2017)
    fix = {}
    for T in TRAIN_H1:
        V = T - 1
        sched = g[(g.season_end == T) & (g.game_type == "R")][
            ["date", "home", "away"]].reset_index(drop=True)
        hb, ab = E.b2b_flags(sched)
        sched["hb2b"], sched["ab2b"] = hb, ab
        om_c = E.fit_outcome(preds, list(range(2006, T)))
        ratings = {t: 1505 + v / c for t, v in preds_by_T[T].items()}
        ratings = E.fill_missing(ratings, set(sched.home) | set(sched.away), 1505.0)

        # v3 availability rosters (age, value) — identical machinery to backtest3
        par3 = A3.fit_availability(sk, bios, V)
        marcel = fb.marcel(V, 1)
        repl = P.replacement_rates(sk, V)
        vals = skater_value_goals(marcel, repl, ppg)
        roster = P.majority_team(skt[skt.season_end == V])
        rosters3, rosters2 = {}, {}
        par2 = A2.fit_availability2(sk, bios, V)
        for team, d in roster.groupby("team"):
            vv = vals.reindex(d.playerId).fillna(0.0)
            aa = P.age_of(bios, d.playerId, V + 1)
            top = sorted(zip(aa.to_numpy(), vv.to_numpy()), key=lambda t_: -t_[1])[:A3.TOP_N]
            rosters3[team] = [(float(a_) if np.isfinite(a_) else 27.0, float(v_))
                              for a_, v_ in top]
            b_idx = A2._bucket_idx(np.array([a if np.isfinite(a) else 27.0
                                             for a, _ in top]))
            rosters2[team] = [(int(bi), par2["buckets"][int(bi)]["mean"], float(v_))
                              for bi, (_, v_) in zip(b_idx, top)]
        tand = GG.team_tandems(got, ts, fb.goalie_proj(V), V, n0_g)
        gpar = A2.fit_goalie_availability(got, ts, V)
        grow = {t: (gpar["mean"], max((tand.loc[t].theta1 - tand.loc[t].theta2), 0.0)
                    * A2.GOALIE_SHOTS)
                for t in tand.index}
        act = ts[ts.season_end == T].set_index("team")
        fix[T] = dict(sched=sched, om=om_c, ratings=ratings, rosters3=rosters3,
                      rosters2=rosters2, par3=par3, par2=par2, gpar=gpar, grow=grow,
                      tand=tand, act=act)

    def noise_for(T, mode):
        f = fix[T]
        teams_sorted = sorted(f["ratings"])
        if mode == "v3":
            return A3.make_extra_noise(teams_sorted, f["rosters3"], f["par3"], k)
        if mode == "v2.0":
            return A2.make_extra_noise2(teams_sorted, f["rosters2"], f["par2"],
                                        f["grow"], f["gpar"], k)
        return None

    def run_season(T, sigma, avail, goalie, n_sims, seed, beta_b2b=GG.BETA_B2B):
        f = fix[T]
        gn = (GG.make_game_noise(f["sched"], f["tand"], k, beta_b2b)
              if goalie else None)
        return E.simulate_season(f["ratings"], sigma, f["sched"][["home", "away"]],
                                 f["om"], E.divisions_for(T), n_sims,
                                 np.random.default_rng(seed), playoffs=False,
                                 extra_noise=noise_for(T, avail), game_noise=gn)

    def refit_sigma(avail, goalie, beta_b2b=GG.BETA_B2B):
        best = None
        for sigma in SIGMA_GRID:
            vs = []
            for T in TRAIN_H1:
                sim = run_season(T, sigma, avail, goalie, N_SIMS_FIT, 100 + int(sigma))
                vs.append((sim["pts"].std(axis=0, ddof=1)).mean() ** 2)
            gap = abs(np.mean(vs) - target_var)
            if best is None or gap < best[1]:
                best = (sigma, gap)
        return best[0]

    def coverage(avail, goalie, sigma, seed=7, beta_b2b=GG.BETA_B2B):
        cov, mean_pts = [], {}
        for T in TRAIN_H1:
            sim = run_season(T, sigma, avail, goalie, N_SIMS_COV, seed,
                             beta_b2b=beta_b2b)
            lo = np.percentile(sim["pts"], 10, axis=0)
            hi = np.percentile(sim["pts"], 90, axis=0)
            mu = sim["pts"].mean(axis=0)
            act = fix[T]["act"]
            for i, t in enumerate(sim["teams"]):
                if t in act.index:
                    cov.append({"T": T, "team": t,
                                "cover": int(lo[i] <= act.loc[t].pts <= hi[i])})
                    mean_pts[(T, t)] = float(mu[i])
        return pd.DataFrame(cov), mean_pts

    def tertile_spread(covdf, group_vals):
        gv = covdf.apply(lambda r: group_vals.get((r["T"], r["team"]), 0.0), axis=1)
        tert = pd.qcut(gv.rank(method="first"), 3, labels=False)
        by_t = covdf.cover.groupby(tert).mean()
        return float((by_t - 0.8).abs().max()), [round(float(x), 3) for x in by_t]

    # ================= GATE G (goalie layer; availability = v3 as shipped)
    print("\n=== GATE G: goalie game layer ===")
    gap_elo = {(T, t): float(fix[T]["tand"].gap_gpg.get(t, 0.0)) * k
               for T in TRAIN_H1 for t in fix[T]["tand"].index}
    cov_base, mean_base = coverage("v3", goalie=False, sigma=p3["availability"]["sigma_c3"])
    spread_base, tert_base = tertile_spread(cov_base, gap_elo)
    sigma_g = refit_sigma("v3", goalie=True)
    cov_g, mean_g = coverage("v3", goalie=True, sigma=sigma_g)
    spread_g, tert_g = tertile_spread(cov_g, gap_elo)
    # (c) zero-mean verification measured on the closure itself at high precision.
    # First formulation diffed two independent 2000-sim runs: per-team MC SE ~0.29,
    # max over 180 teams ~0.9 — cannot resolve the 0.4 criterion (first run showed
    # 0.809 of pure MC noise). Criterion unchanged; measurement corrected & documented.
    shift = 0.0
    for T in TRAIN_H1:
        f = fix[T]
        gn = GG.make_game_noise(f["sched"], f["tand"], k, GG.BETA_B2B)
        per_game = gn(20000, np.random.default_rng(11)).mean(axis=0)
        acc: dict = {}
        for i, gm in enumerate(f["sched"].itertuples(index=False)):
            acc[gm.home] = acc.get(gm.home, 0.0) + per_game[i]
            acc[gm.away] = acc.get(gm.away, 0.0) - per_game[i]
        worst_team = max(abs(v) for v in acc.values()) / 82.0  # season-mean Elo
        shift = max(shift, worst_team * c)                      # -> pts/82
    ok_g = ((0.72 <= cov_g.cover.mean() <= 0.90) and (spread_g <= spread_base + 0.02)
            and (shift < 0.4))
    print(f"baseline (no layer): coverage {cov_base.cover.mean():.3f}, "
          f"tandem-gap tertiles {tert_base} (spread {spread_base:.3f})")
    print(f"with layer: sigma refit {sigma_g}, coverage {cov_g.cover.mean():.3f}, "
          f"tertiles {tert_g} (spread {spread_g:.3f}), max mean shift {shift:.3f}")
    for bb in (0.3, 0.7):  # declared-constant sensitivity, reported not gated
        cov_s, _ = coverage("v3", goalie=True, sigma=sigma_g, beta_b2b=bb)
        print(f"  sensitivity beta_b2b={bb}: coverage {cov_s.cover.mean():.3f}")
    print(f"GATE G: {'PASS — SHIP' if ok_g else 'FAIL — DROP'}")
    p4["gate_G"] = {"pass": bool(ok_g), "sigma_with_layer": sigma_g,
                    "coverage": round(float(cov_g.cover.mean()), 3),
                    "spread_with": round(spread_g, 3),
                    "spread_without": round(spread_base, 3),
                    "max_mean_shift": round(float(shift), 3), "n0_g": n0_g}

    # ================= SCREENS + n0_a (persistence decision)
    print("\n=== SCREENS S1/S2 + persistence tuning ===")
    s1 = A2.screen_persistence(sk, bios, 2017)
    n0_a, tab_a = A2.tune_n0_a(sk, bios)
    eb_best = float(tab_a.mse.min())
    bucket_mse = float(tab_a.attrs["bucket_only_mse"])
    persist_ok = s1["pass"] and (eb_best < bucket_mse)
    print(f"S1 corr {s1['r_within_bucket']} (n {s1['n']}) pass={s1['pass']}; "
          f"EB best MSE {eb_best:.5f} vs bucket-only {bucket_mse:.5f} -> "
          f"persistence {'KEPT' if persist_ok else 'DROPPED (EB loses to nested baseline; '
                         'grid scaled for binomial precision but shares are ~24x '
                         'overdispersed -> a season is ~3 effective obs)'}")
    s2 = A2.screen_goalie_overdispersion(got, ts, 2017)
    print(f"S2 goalie overdispersion {s2['overdispersion']}x (n {s2['n']}) "
          f"pass={s2['pass']}")
    p4["screens"] = {"S1": s1, "S2": s2, "n0_a_table": tab_a.round(5).to_dict("records"),
                     "bucket_only_mse": round(bucket_mse, 5),
                     "persistence_kept": bool(persist_ok)}
    if persist_ok:
        # fixtures above were built with bucket-mean mus; a kept persistence component
        # would require rebuilding rosters2 with player_mu — fail loudly, never silently
        raise NotImplementedError("persistence passed its screen+baseline test; "
                                  "rebuild rosters2 with player_mu before gating A2")

    # ================= GATE A2 (availability 2.0 vs v3 availability)
    print("\n=== GATE A2: availability 2.0 (bucket + P0 + goalie slot) ===")
    goalie_on = bool(ok_g)
    frag2 = {}
    for T in TRAIN_H1:
        teams_sorted = sorted(fix[T]["ratings"])
        probe = noise_for(T, "v2.0")(400, np.random.default_rng(1))
        for j, t in enumerate(teams_sorted):
            frag2[(T, t)] = float(probe[:, j].std())
    sigma_a2 = refit_sigma("v2.0", goalie=goalie_on)
    cov_a2, _ = coverage("v2.0", goalie=goalie_on, sigma=sigma_a2)
    spread_a2, tert_a2 = tertile_spread(cov_a2, frag2)
    # v3-availability comparison spread under the same goalie-layer setting
    if goalie_on:
        cov_v3g, _ = coverage("v3", goalie=True, sigma=sigma_g)
        spread_v3, tert_v3 = tertile_spread(cov_v3g, frag2)
    else:
        spread_v3, tert_v3 = tertile_spread(cov_base, frag2)
    ok_a2 = (0.72 <= cov_a2.cover.mean() <= 0.90) and (spread_a2 <= spread_v3 + 0.02)
    print(f"availability2: sigma refit {sigma_a2}, coverage {cov_a2.cover.mean():.3f}, "
          f"fragility tertiles {tert_a2} (spread {spread_a2:.3f}) vs v3 spread "
          f"{spread_v3:.3f}")
    print(f"GATE A2: {'PASS — SHIP' if ok_a2 else 'FAIL — KEEP v3 availability'}")
    p4["gate_A2"] = {"pass": bool(ok_a2), "sigma": sigma_a2,
                     "coverage": round(float(cov_a2.cover.mean()), 3),
                     "spread": round(spread_a2, 3), "spread_v3": round(spread_v3, 3),
                     "goalie_slot": s2["pass"], "persistence": bool(persist_ok)}

    # ================= GATE F2 (prospect_pipeline)
    print("\n=== GATE F2: prospect_pipeline ===")
    p4["gate_F2"] = {}
    final_sets = {}
    for h, train_pred in ((1, TRAIN_H1), (2, list(range(2013, 2018)))):
        Xh, yh, mh = (X, y, meta) if h == 1 else fb.feature_matrix(train_pred, 2,
                                                                  feats=FEATURES_V4)
        sh = mh["T"].to_numpy()
        cols3 = [FEATURES_V4.index(f) for f in feats3]
        lam_i, tab_i = R.loso_cv(Xh[:, cols3], yh, sh)
        mae_i = tab_i.mae.min()
        cand = feats3 + ["prospect_pipeline"]
        colsc = [FEATURES_V4.index(f) for f in cand]
        lam_c, tab_c = R.loso_cv(Xh[:, colsc], yh, sh)
        mae_c = tab_c.mae.min()
        signs = R.loso_coef_signs(Xh[:, colsc], yh, sh, lam_c)
        stab = float(signs[-1])
        dmae = mae_c - mae_i
        passed = (dmae <= 0.02) and (stab >= 0.67)
        final_sets[h] = cand if passed else feats3
        lam_h = lam_c if passed else lam_i
        print(f"h{h}: incumbent LOSO {mae_i:.3f} | +prospect {mae_c:.3f} "
              f"(dMAE {dmae:+.4f}, stability {stab:.2f}, lam {lam_c}) -> "
              f"{'ENTER' if passed else 'STAY OUT'}")
        p4["gate_F2"][f"h{h}"] = {"pass": bool(passed), "dmae": round(float(dmae), 4),
                                  "stability": round(stab, 2), "lam": float(lam_h),
                                  "mae_incumbent": round(float(mae_i), 3),
                                  "mae_with": round(float(mae_c), 3)}

    # ================= assemble shipped config
    p4["shipped"] = {
        "goalie_layer": bool(ok_g),
        "availability": "v2.0" if ok_a2 else "v3",
        "sigma_c4": sigma_a2 if ok_a2 else (sigma_g if ok_g else
                                            p3["availability"]["sigma_c3"]),
        "features_h1": final_sets[1], "features_h2": final_sets[2],
        "lam_h1": p4["gate_F2"]["h1"]["lam"], "lam_h2": p4["gate_F2"]["h2"]["lam"],
        "n0_g": n0_g, "beta_b2b": GG.BETA_B2B, "b2b_elo": 38.0,
        "overlay": "overlay_team_deltas_v4.csv (rho*=0.421, production-consistent)",
        "ensemble": "ENS = 0.5*v1 + 0.5*v4 in Elo space (locked in PLAN_V4 I3)",
    }
    # train CRPS context (report-only, never a gate)
    crps_rows = []
    mode = "v2.0" if ok_a2 else "v3"
    for T in TRAIN_H1:
        sim = run_season(T, p4["shipped"]["sigma_c4"], mode, goalie_on, 1500, 55)
        act = fix[T]["act"].pts
        crps_rows.append(SC.crps_matrix(sim["pts"], sim["teams"], act).mean())
    p4["train_crps_context"] = round(float(np.mean(crps_rows)), 3)
    print(f"\ntrain CRPS (context only): {p4['train_crps_context']}")

    p4["seconds"] = round(time.time() - t0, 1)
    (OUT / "params_v4.json").write_text(json.dumps(p4, indent=2))
    print(f"params_v4.json written ({p4['seconds']}s)")


if __name__ == "__main__":
    main()
