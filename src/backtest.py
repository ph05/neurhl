"""Walk-forward tuning (train <= 2017) and frozen validation (2018-2026).

Protocol (anti-overfitting):
  Stage A: Elo params (K, H, phi_s) by in-season log loss, train regular-season games 2010-2017.
  Stage B: projection params (w, phi1) one-ahead on predict-seasons 2011-2017;
           phi2 two-ahead on 2012-2017. Causal beta/outcome fits per predicted season.
  Stage C: strength-draw sigma1/sigma2 matching residual dispersion on train.
  Freeze -> validate ONCE on 2018-2026 vs baselines. No parameter is revisited after this.

Baselines (Tango/Silver): uniform (causal trailing league mean), prior-points regressed by
train-fitted shrinkage, raw prior points.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E

PROJ = Path(__file__).resolve().parents[1]
TRAIN_LL_SEASONS = list(range(2010, 2018))
TRAIN_ONE = list(range(2011, 2018))   # predict these one-ahead in train
TRAIN_TWO = list(range(2012, 2018))
VALID_ONE = list(range(2018, 2027))
VALID_TWO = list(range(2018, 2027))
PLAYOFF_CAL_SEASONS = [2018, 2019, 2022, 2023, 2024, 2025, 2026]  # normal format only

g, ts = E.load()
ACT = ts.set_index(["season_end", "team"])


def logloss(preds, seasons):
    sub = preds[(preds.game_type == "R") & preds.season_end.isin(seasons)]
    y = sub.home_win.to_numpy().astype(float)
    p = np.clip(sub.e_home.to_numpy(), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ---------------------------------------------------------------- Stage A
def stage_a():
    results = []
    for K in (4, 6, 8, 10, 12):
        for H in (20, 28, 35, 45):
            for phi_s in (0.6, 0.7, 0.8):
                preds, _, _ = E.run_elo(g, K=K, H=H, phi_s=phi_s)
                results.append((K, H, phi_s, logloss(preds, TRAIN_LL_SEASONS)))
    df = pd.DataFrame(results, columns=["K", "H", "phi_s", "ll"]).sort_values("ll")
    print("Stage A top-8 (train in-season log loss):")
    print(df.head(8).to_string(index=False))
    best = df.iloc[0]
    # plateau check: spread of top decile
    print(f"plateau: best {df.ll.min():.5f}, 10th pct {df.ll.quantile(0.1):.5f}")
    return float(best.K), float(best.H), float(best.phi_s), df


# ------------------------------------------------- causal per-season inputs
def causal_inputs(end_r, preds, predict_season, horizon):
    from_season = predict_season - horizon
    hist = [s for s in range(2008, predict_season) if s in end_r]
    beta = E.fit_xg_beta(end_r, ts, [s for s in hist])
    om = E.fit_outcome(preds, [s for s in range(2006, predict_season)])
    return from_season, beta, om


def project_season(end_r, preds, predict_season, horizon, w, phi):
    from_season, beta, om = causal_inputs(end_r, preds, predict_season, horizon)
    ratings = E.project_ratings(end_r, ts, beta, from_season, w=w, phi=phi)
    sched = E.actual_schedule(g, predict_season)
    ratings = E.fill_missing(ratings, set(sched.home) | set(sched.away))
    xp = E.analytic_xpts(ratings, sched, om)
    return ratings, xp, om, sched


def norm_mae(xp, predict_season):
    act = ACT.loc[predict_season]
    rows = []
    for team, v in xp.items():
        a = act.loc[team]
        rows.append(abs(v - a.pts) / a.gp * 82)
    return float(np.mean(rows))


# ---------------------------------------------------------------- Stage B
def stage_b(end_r, preds):
    grid_results = []
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        for phi in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
            maes = [norm_mae(project_season(end_r, preds, T, 1, w, phi)[1], T)
                    for T in TRAIN_ONE]
            grid_results.append((w, phi, np.mean(maes)))
    df = pd.DataFrame(grid_results, columns=["w", "phi1", "mae"]).sort_values("mae")
    print("\nStage B one-ahead top-8 (train preseason MAE/82):")
    print(df.head(8).round(3).to_string(index=False))
    w, phi1 = float(df.iloc[0].w), float(df.iloc[0].phi1)

    two = []
    for phi2 in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        maes = [norm_mae(project_season(end_r, preds, T, 2, w, phi2)[1], T)
                for T in TRAIN_TWO]
        two.append((phi2, np.mean(maes)))
    df2 = pd.DataFrame(two, columns=["phi2", "mae"]).sort_values("mae")
    print("\nStage B two-ahead (train):")
    print(df2.round(3).to_string(index=False))
    phi2 = float(df2.iloc[0].phi2)
    return w, phi1, phi2, df, df2


# ---------------------------------------------------------------- Stage C
def stage_c(end_r, preds, w, phi1, phi2):
    rng = np.random.default_rng(42)
    out = {}
    for horizon, phi, seasons in ((1, phi1, TRAIN_ONE), (2, phi2, TRAIN_TWO)):
        resid2 = []
        cache = {}
        for T in seasons:
            ratings, xp, om, sched = project_season(end_r, preds, T, horizon, w, phi)
            act = ACT.loc[T]
            for team, v in xp.items():
                a = act.loc[team]
                resid2.append(((v - a.pts) / a.gp * 82) ** 2)
            cache[T] = (ratings, om, sched)
        target = float(np.mean(resid2))
        rows = []
        for sigma in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
            vars_ = []
            for T in seasons:
                ratings, om, sched = cache[T]
                sim = E.simulate_season(ratings, sigma, sched, om,
                                        E.divisions_for(T), 1500, rng, playoffs=False)
                gp = ACT.loc[T].gp.mean()
                vars_.append(((sim["pts"].std(axis=0, ddof=1)) * 82 / gp).mean() ** 2)
            rows.append((sigma, np.mean(vars_)))
        dfc = pd.DataFrame(rows, columns=["sigma", "sim_var"])
        dfc["gap"] = (dfc.sim_var - target).abs()
        # interpolate sigma: sim_var rises monotonically in sigma^2; linear interp on variance
        best = dfc.sort_values("gap").iloc[0]
        print(f"\nStage C horizon {horizon}: target resid var {target:.1f} (sd {np.sqrt(target):.2f})")
        print(dfc.round(1).to_string(index=False))
        out[horizon] = float(best.sigma)
    return out[1], out[2]


# ---------------------------------------------------------------- baselines
def baselines(predict_season, horizon, slope1, slope2):
    """Returns dict name -> Series of predicted points for predict_season."""
    prior_season = predict_season - horizon
    m_hist = ts[ts.season_end < predict_season].pts_pct.mean()
    act = ACT.loc[predict_season]
    prior = ts[ts.season_end == prior_season].set_index("team").pts_pct
    slope = slope1 if horizon == 1 else slope2
    out = {}
    teams = act.index
    gp = act.gp
    out["uniform"] = pd.Series(m_hist * 2 * gp, index=teams)
    reg = {t: (m_hist + slope * (prior.get(t, m_hist) - m_hist)) * 2 * gp[t] for t in teams}
    out["regressed_prior"] = pd.Series(reg)
    raw = {t: prior.get(t, m_hist) * 2 * gp[t] for t in teams}
    out["raw_prior"] = pd.Series(raw)
    return out


def fit_yoy_slope(max_season, lag):
    tss = ts[ts.season_end <= max_season].sort_values(["team", "season_end"])
    cur, nxt = tss.copy(), tss.copy()
    nxt["season_end"] -= lag
    m = cur.merge(nxt, on=["team", "season_end"], suffixes=("_t", "_n"))
    return float(np.polyfit(m.pts_pct_t, m.pts_pct_n, 1)[0])


# ---------------------------------------------------------------- validation
def validate(end_r, preds, params):
    w, phi1, phi2, s1, s2 = (params["w"], params["phi1"], params["phi2"],
                             params["sigma1"], params["sigma2"])
    slope1 = fit_yoy_slope(2017, 1)
    slope2 = fit_yoy_slope(2017, 2)
    print(f"\nbaseline shrinkage slopes (train-fitted): lag1 {slope1:.3f}, lag2 {slope2:.3f}")
    rng = np.random.default_rng(2027)
    all_rows = []
    po_rows = []
    for horizon, phi, sigma, seasons in ((1, phi1, s1, VALID_ONE), (2, phi2, s2, VALID_TWO)):
        for T in seasons:
            ratings, xp, om, sched = project_season(end_r, preds, T, horizon, w, phi)
            act = ACT.loc[T]
            base = baselines(T, horizon, slope1, slope2)
            row = {"horizon": horizon, "season": T,
                   "mae_model": norm_mae(xp, T)}
            for name, bxp in base.items():
                row[f"mae_{name}"] = float(np.mean(
                    [abs(bxp[t] - act.loc[t].pts) / act.loc[t].gp * 82 for t in bxp.index]))
            # rank correlation
            order = xp.reindex(act.index)
            row["spearman"] = float(order.rank().corr(act.pts_pct.rank()))
            # expansion-debut flag
            row["has_expansion_debut"] = int(T in (2018, 2022))
            all_rows.append(row)
            # playoff odds calibration
            if T in PLAYOFF_CAL_SEASONS:
                sim = E.simulate_season(ratings, sigma, sched, om, E.divisions_for(T),
                                        2000, rng, playoffs=True)
                po = pd.Series(sim["made_po"].mean(0), index=sim["teams"])
                # actual playoff qualifiers = top by same seeding on real standings
                made = actual_playoff_teams(T)
                for t in po.index:
                    po_rows.append({"horizon": horizon, "season": T, "team": t,
                                    "p": float(po[t]), "made": int(t in made)})
    res = pd.DataFrame(all_rows)
    po = pd.DataFrame(po_rows)
    return res, po


def actual_playoff_teams(T):
    """Derive the 16 qualifiers from real standings by seeding rules (approx tiebreak)."""
    act = ACT.loc[T].reset_index()
    key = act.set_index("team").apply(
        lambda r: r.pts * 1e8 + r.reg_w * 1e4 + (r.w), axis=1)
    div = E.divisions_for(T)
    made = set()
    for conf, dvs in E.CONFS.items():
        d23 = []
        rest = []
        for dv in dvs:
            order = sorted(div[dv], key=lambda t: -key[t])
            made.add(order[0])
            d23 += order[1:3]
            rest += order[3:]
        made.update(d23)
        wc = sorted(rest, key=lambda t: -key[t])[:2]
        made.update(wc)
    return made


def main():
    t0 = time.time()
    K, H, phi_s, dfa = stage_a()
    preds, end_r, _ = E.run_elo(g, K=K, H=H, phi_s=phi_s)
    w, phi1, phi2, dfb1, dfb2 = stage_b(end_r, preds)
    sigma1, sigma2 = stage_c(end_r, preds, w, phi1, phi2)
    params = {"K": K, "H": H, "phi_s": phi_s, "w": w, "phi1": phi1, "phi2": phi2,
              "sigma1": sigma1, "sigma2": sigma2,
              "expansion_init": E.EXPANSION_INIT,
              "train_through": 2017, "frozen": True}
    print(f"\nFROZEN PARAMS: {params}")

    res, po = validate(end_r, preds, params)
    pd.set_option("display.width", 200)
    print("\n=== VALIDATION (2018-2026, frozen params) ===")
    for h in (1, 2):
        sub = res[res.horizon == h]
        cols = ["season", "mae_model", "mae_uniform", "mae_regressed_prior", "mae_raw_prior",
                "spearman"]
        print(f"\n-- horizon {h} --")
        print(sub[cols].round(2).to_string(index=False))
        core = sub[~sub.season.isin([2020, 2021])]
        for name, d in (("all seasons", sub), ("excl. 2020/2021", core)):
            print(f"mean MAE ({name}): model {d.mae_model.mean():.2f} | uniform "
                  f"{d.mae_uniform.mean():.2f} | regressed {d.mae_regressed_prior.mean():.2f} | "
                  f"raw {d.mae_raw_prior.mean():.2f} | spearman {d.spearman.mean():.3f}")

    # playoff calibration
    print("\n-- playoff odds calibration (validation, normal-format seasons) --")
    for h in (1, 2):
        sub = po[po.horizon == h]
        brier = ((sub.p - sub.made) ** 2).mean()
        base_rate = sub.made.mean()
        brier_const = (base_rate * (1 - base_rate))
        print(f"horizon {h}: Brier {brier:.4f} vs constant-rate {brier_const:.4f} (n={len(sub)})")
        bins = pd.cut(sub.p, [0, .2, .4, .6, .8, 1.0])
        cal = sub.groupby(bins, observed=True).agg(pred=("p", "mean"), actual=("made", "mean"),
                                                   n=("made", "size"))
        print(cal.round(3).to_string())

    # in-season log loss, validation
    llv = logloss(preds, VALID_ONE)
    subv = preds[(preds.game_type == "R") & preds.season_end.isin(VALID_ONE)]
    yh = subv.home_win.mean()
    llb = float(-(yh * np.log(yh) + (1 - yh) * np.log(1 - yh)))
    print(f"\nin-season log loss validation: elo {llv:.4f} vs constant-home {llb:.4f}")

    (PROJ / "output").mkdir(exist_ok=True)
    with open(PROJ / "output/params.json", "w") as f:
        json.dump(params, f, indent=2)
    res.to_csv(PROJ / "output/backtest_seasons.csv", index=False)
    po.to_csv(PROJ / "output/backtest_playoff_calibration.csv", index=False)
    print(f"\nsaved params.json + backtest CSVs ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
