"""v2 backtest: Stage T (train tuning + Feature Ledger), Confirm (2018-2021), Final (2022-2026).

Stage T tunes everything on predict-seasons <= 2017. Confirm applies the pre-registered rung
rule. Final reports once. Pre-registration text is written into params_v2.json by overlay.py
BEFORE confirm runs (enforced by an assertion here).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import players as P
import ridge as R
from features import FeatureBuilder, FEATURES, elo_per_gpg, pts_per_elo

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"

TRAIN_PRED = {1: list(range(2012, 2018)), 2: list(range(2013, 2018))}
CONFIRM = list(range(2018, 2022))
FINAL = list(range(2022, 2027))
BASE_FEATS = ["elo_dev", "xg_dev"]
R2_FEATS = ["elo_dev", "xg_dev", "gsax_1yr", "gsax_marcel", "tandem_gsax",
            "goalie_consistency", "goalie_age", "goalie_trend", "toi_age", "share_u23",
            "share_32p", "prod_age_exp"]
PLAYOFF_CAL = [2018, 2019, 2022, 2023, 2024, 2025, 2026]

V1 = json.loads((OUT / "params.json").read_text())
g, ts = E.load()
ACT = ts.set_index(["season_end", "team"])


def sub_matrix(X, feats):
    idx = [FEATURES.index(f) for f in feats]
    return X[:, idx]


# ------------------------------------------------------------------ stage T
def tune_goalie_grid(go):
    rates = P.goalie_rates(go)
    nxt = rates.set_index(["playerId", "season_end"])
    results = []
    for delta in (0.5, 0.6, 0.7, 0.8):
        for nu0 in (2.0, 4.0, 8.0):
            for window in (3, 4):
                se, wt = [], []
                for V in range(2011, 2017):
                    proj = P.goalie_project(go, V, delta=delta, nu0=nu0, window=window)
                    for r_ in proj.itertuples():
                        key = (r_.playerId, V + 1)
                        if key in nxt.index:
                            row = nxt.loc[key]
                            if row.shots >= 300:
                                se.append((r_.theta - row.r) ** 2 * row.shots)
                                wt.append(row.shots)
                results.append((delta, nu0, window, sum(se) / sum(wt)))
    df = pd.DataFrame(results, columns=["delta", "nu0", "window", "wmse"]).sort_values("wmse")
    # baseline: theta=0 for everyone
    se0, wt0 = [], []
    for V in range(2011, 2017):
        d = rates[(rates.season_end == V + 1) & (rates.shots >= 300)]
        se0 += (d.r ** 2 * d.shots).tolist()
        wt0 += d.shots.tolist()
    base = sum(se0) / sum(wt0)
    print("goalie grid top-5:")
    print(df.head(5).to_string(index=False))
    print(f"predict-zero baseline wMSE {base:.3e} (skill ratio best/base "
          f"{df.wmse.min()/base:.3f})")
    b = df.iloc[0]
    return {"delta": float(b.delta), "nu0": float(b.nu0), "window": int(b.window)}


def tune_skater_delta(sk, bios):
    nxt = sk.set_index(["playerId", "season_end"])
    results = []
    for delta in (0.6, 0.7, 0.8, 0.9):
        se, wt = [], []
        for V in range(2011, 2017):
            m = P.skater_marcel(sk, bios, V, delta=delta, fe_coefs=None)
            for r_ in m.itertuples():
                key = (r_.playerId, V + 1)
                if key in nxt.index:
                    row = nxt.loc[key]
                    if row.toi_min >= 300:
                        se.append((r_.theta_pts60 - row.pts60) ** 2 * row.toi_min)
                        wt.append(row.toi_min)
        results.append((delta, sum(se) / sum(wt)))
    df = pd.DataFrame(results, columns=["delta", "wmse"]).sort_values("wmse")
    print("skater delta grid:")
    print(df.round(4).to_string(index=False))
    return float(df.iloc[0].delta)


def ledger_row(name, X, y, seasons, feats_all, base_mae, full_mae, lam_full):
    """add-one-in and leave-one-out MAEs for one candidate."""
    add_feats = BASE_FEATS + ([name] if name not in BASE_FEATS else [])
    lam_a, tab_a = R.loso_cv(sub_matrix(X, add_feats), y, seasons)
    add_mae = tab_a.mae.min()
    loo_feats = [f for f in feats_all if f != name]
    lam_l, tab_l = R.loso_cv(sub_matrix(X, loo_feats), y, seasons)
    loo_mae = tab_l.mae.min()
    return {"feature": name, "add_one_in_dMAE": round(add_mae - base_mae, 4),
            "leave_one_out_dMAE": round(loo_mae - full_mae, 4)}


def stage_t():
    t0 = time.time()
    sk, go, skt, got, bios = P.load_panels()
    print("=== Stage T (train <= 2017) ===")
    ghp = tune_goalie_grid(go)
    sdelta = tune_skater_delta(sk, bios)

    preds, end_r, _ = E.run_elo(g, K=V1["K"], H=V1["H"], phi_s=V1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=ghp, skater_delta=sdelta)

    out = {"goalie_hp": ghp, "skater_delta": sdelta, "lams": {}, "sigma_c": {},
           "hetero": {}, "ledger_note": "see output/feature_ledger.csv"}
    ledgers = []
    om_train = E.fit_outcome(preds, list(range(2006, 2018)))
    c = pts_per_elo(om_train)
    k = elo_per_gpg(end_r, ts, 2017)
    chain = k * c / 82
    print(f"value chain: k={k:.1f} Elo per GD/g, c={c:.3f} pts82/Elo, "
          f"k*c/82={chain:.3f} pts per season-goal")
    # ~0.24 is right for the modern NHL: the loser point + OT coin-flip compress points
    # sensitivity, and k regresses on shrunken end-season Elo. Band guards unit errors only.
    assert 0.20 <= chain <= 0.50, "value-chain invariant violated"
    out["k"], out["c"] = k, c

    for h in (1, 2):
        X, y, meta = fb.feature_matrix(TRAIN_PRED[h], h)
        seasons = meta["T"].to_numpy()
        lam_base, tab_base = R.loso_cv(sub_matrix(X, BASE_FEATS), y, seasons)
        base_mae = tab_base.mae.min()
        lam_full, tab_full = R.loso_cv(X, y, seasons)
        full_mae = tab_full.mae.min()
        signs = R.loso_coef_signs(X, y, seasons, lam_full)
        beta = R.fit_ridge(X, y, lam_full)
        edof = R.effective_dof(X, lam_full)
        print(f"\nh{h}: base(elo+xg) LOSO MAE {base_mae:.3f} | full {full_mae:.3f} "
              f"(lam {lam_full}, edof {edof:.1f})")
        lam_r2, tab_r2 = R.loso_cv(sub_matrix(X, R2_FEATS), y, seasons)
        print(f"h{h}: R2 (goalie+age blocks) LOSO MAE {tab_r2.mae.min():.3f} (lam {lam_r2})")
        out["lams"][f"h{h}"] = {"base": lam_base, "full": lam_full, "r2": lam_r2}
        out.setdefault("train_mae", {})[f"h{h}"] = {
            "base": round(base_mae, 3), "r2": round(tab_r2.mae.min(), 3),
            "full": round(full_mae, 3)}
        for i, f in enumerate(FEATURES):
            if f in BASE_FEATS:
                row = {"feature": f, "add_one_in_dMAE": 0.0, "leave_one_out_dMAE": np.nan}
            else:
                row = ledger_row(f, X, y, seasons, FEATURES, base_mae, full_mae, lam_full)
            row.update({"horizon": h, "ridge_coef": round(float(beta[i]), 3),
                        "sign_stability": round(float(signs[i]), 2)})
            ledgers.append(row)

    # ---- sigma_c calibration + heteroscedastic decision (train, LOSO residuals, h1&h2)
    rng = np.random.default_rng(42)
    for h in (1, 2):
        X, y, meta = fb.feature_matrix(TRAIN_PRED[h], h)
        seasons = meta["T"].to_numpy()
        lam = out["lams"][f"h{h}"]["full"]
        resid2, preds_by_T = [], {}
        for T in TRAIN_PRED[h]:
            tr = seasons != T
            te = seasons == T
            b = R.fit_ridge(X[tr], y[tr], lam)
            pr = X[te] @ b
            pr = pr - pr.mean()
            preds_by_T[T] = pd.Series(pr, index=meta.loc[te, "team"].to_numpy())
            resid2 += list((pr - (y[te] - y[te].mean())) ** 2)
        target = float(np.mean(resid2))
        rows = []
        for sigma in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
            vars_ = []
            for T in TRAIN_PRED[h]:
                ratings = {t: 1505 + v / c for t, v in preds_by_T[T].items()}
                sched = E.actual_schedule(g, T)
                ratings = E.fill_missing(ratings, set(sched.home) | set(sched.away), 1505.0)
                om_c = E.fit_outcome(preds, list(range(2006, T)))
                sim = E.simulate_season(ratings, sigma, sched, om_c, E.divisions_for(T),
                                        1000, rng, playoffs=False)
                gp = ACT.loc[T].gp.mean()
                vars_.append(((sim["pts"].std(axis=0, ddof=1)) * 82 / gp).mean() ** 2)
            rows.append((sigma, float(np.mean(vars_)), abs(np.mean(vars_) - target)))
        dfc = pd.DataFrame(rows, columns=["sigma", "sim_var", "gap"]).sort_values("gap")
        out["sigma_c"][f"h{h}"] = float(dfc.iloc[0].sigma)
        print(f"h{h} sigma_c: target resid var {target:.1f} -> sigma {dfc.iloc[0].sigma}")

    # ---- heteroscedastic sigma decision (h1, pre-registered coverage-by-tertile test)
    h = 1
    X, y, meta = fb.feature_matrix(TRAIN_PRED[h], h)
    seasons = meta["T"].to_numpy()
    lam = out["lams"]["h1"]["full"]
    sigma_c = out["sigma_c"]["h1"]
    cover = {"uniform": [], "hetero": []}
    tert_of = []
    for T in TRAIN_PRED[h]:
        V = T - h
        tr, te = seasons != T, seasons == T
        b = R.fit_ridge(X[tr], y[tr], lam)
        pr = pd.Series(X[te] @ b, index=meta.loc[te, "team"].to_numpy())
        pr -= pr.mean()
        tg = fb.team_goalies(V)
        shots_pg = 30.0
        g_elo = (k * shots_pg * np.sqrt(tg.tandem_var)).reindex(pr.index).fillna(
            float((k * shots_pg * np.sqrt(tg.tandem_var)).mean()))
        het = np.sqrt(np.maximum(sigma_c ** 2 + (g_elo ** 2 - (g_elo ** 2).mean()), 25.0))
        ratings = {t: 1505 + v / c for t, v in pr.items()}
        sched = E.actual_schedule(g, T)
        ratings = E.fill_missing(ratings, set(sched.home) | set(sched.away), 1505.0)
        om_c = E.fit_outcome(preds, list(range(2006, T)))
        act = ACT.loc[T]
        for scheme, sg in (("uniform", sigma_c), ("hetero", het.to_dict())):
            sim = E.simulate_season(ratings, sg, sched, om_c, E.divisions_for(T),
                                    1500, rng, playoffs=False)
            lo = np.percentile(sim["pts"], 10, axis=0)
            hi = np.percentile(sim["pts"], 90, axis=0)
            for i, t in enumerate(sim["teams"]):
                if t in act.index:
                    cover[scheme].append(int(lo[i] <= act.loc[t].pts <= hi[i]))
                    if scheme == "uniform":
                        tert_of.append(float(g_elo.get(t, g_elo.mean())))
    tert = pd.qcut(pd.Series(tert_of), 3, labels=False)
    spread = {}
    for scheme in ("uniform", "hetero"):
        cv = pd.Series(cover[scheme])
        by_t = cv.groupby(tert).mean()
        spread[scheme] = float((by_t - 0.8).abs().max())
        print(f"hetero test [{scheme}]: overall 80% coverage {cv.mean():.3f}, "
              f"by-tertile {by_t.round(3).tolist()}, max dev {spread[scheme]:.3f}")
    out["hetero"] = {"keep": bool(spread["hetero"] < spread["uniform"]),
                     "spread": spread, "shots_pg": 30.0}
    print(f"heteroscedastic sigma kept: {out['hetero']['keep']}")

    ledger = pd.DataFrame(ledgers)
    ledger.to_csv(OUT / "feature_ledger.csv", index=False)
    out["stage_t_seconds"] = round(time.time() - t0, 1)
    (OUT / "params_v2.json").write_text(json.dumps(out, indent=2))
    print(f"\nStage T done in {out['stage_t_seconds']}s; params_v2.json + feature_ledger.csv "
          f"written")
    return out


# ------------------------------------------------------- confirm/final machinery
def expanding_predictions(fb, h, predict_seasons, feats, lam):
    """Walk-forward: for each T, fit ridge on all pairs with target season <= T-h."""
    all_pred = list(range(2012 + (h - 1), 2027))
    X, y, meta = fb.feature_matrix(all_pred, h)
    seasons = meta["T"].to_numpy()
    res = {}
    for T in predict_seasons:
        tr = seasons <= T - h
        assert tr.sum() >= 100, f"too few training pairs for T={T}"
        b = R.fit_ridge(sub_matrix(X, feats)[tr], y[tr], lam)
        te = seasons == T
        pr = sub_matrix(X, feats)[te] @ b
        pr = pr - pr.mean()
        res[T] = pd.Series(pr, index=meta[te].team.to_numpy())
    return res


def v1_predictions(end_r, preds, h, predict_seasons):
    """v1 model in deviation terms (per-82), using frozen v1 params."""
    from backtest import causal_inputs, project_season
    res = {}
    for T in predict_seasons:
        phi = V1["phi1"] if h == 1 else V1["phi2"]
        _, xp, _, _ = project_season(end_r, preds, T, h, V1["w"], phi)
        act = ACT.loc[T]
        dev = (xp / (2 * act.gp.reindex(xp.index)) - (xp / (2 * act.gp.reindex(xp.index))).mean()) * 164
        res[T] = dev
    return res


def eval_predictions(pred_by_T, h):
    rows = []
    for T, pr in pred_by_T.items():
        act = ACT.loc[T]
        ydev = ((act.pts_pct - act.pts_pct.mean()) * 164).reindex(pr.index)
        rows.append({"horizon": h, "season": T,
                     "mae": float((pr - ydev).abs().mean()),
                     "spearman": float(pr.rank().corr(ydev.rank()))})
    return pd.DataFrame(rows)


def run_confirm():
    p2 = json.loads((OUT / "params_v2.json").read_text())
    assert "prereg" in p2, "pre-registration missing: run overlay.py first"
    assert "confirm_result" not in p2, "confirm already run (single-use)"
    sk, go, skt, got, bios = P.load_panels()
    preds, end_r, _ = E.run_elo(g, K=V1["K"], H=V1["H"], phi_s=V1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"], skater_delta=p2["skater_delta"])
    report = {}
    for h in (1, 2):
        lamf, lamr2 = p2["lams"][f"h{h}"]["full"], p2["lams"][f"h{h}"]["r2"]
        r3 = eval_predictions(expanding_predictions(fb, h, CONFIRM, FEATURES, lamf), h)
        r2 = eval_predictions(expanding_predictions(fb, h, CONFIRM, R2_FEATS, lamr2), h)
        r1 = eval_predictions(v1_predictions(end_r, preds, h, CONFIRM), h)
        report[h] = {"R3": r3, "R2": r2, "R1": r1}
    # pre-registered gate: h1, excl 2021, deviation MAE and spearman
    gate = {}
    for name in ("R3", "R2", "R1"):
        d = report[1][name]
        d = d[d.season != 2021]
        gate[name] = (d.mae.mean(), d.spearman.mean())
    v1_mae, v1_rho = gate["R1"]
    choice = "R1"
    if gate["R2"][0] <= v1_mae + 0.10 and gate["R2"][1] >= v1_rho - 0.02:
        choice = "R2"
    if gate["R3"][0] <= v1_mae + 0.10 and gate["R3"][1] >= v1_rho - 0.02:
        choice = "R3"
    print("confirm gate (h1, excl 2021): " +
          " | ".join(f"{n}: MAE {m:.3f} rho {r:.3f}" for n, (m, r) in gate.items()))
    print(f"RUNG CHOSEN: {choice}")
    p2["confirm_result"] = {
        "gate": {n: [round(m, 3), round(r, 3)] for n, (m, r) in gate.items()},
        "choice": choice,
        "detail": {f"h{h}": {n: report[h][n].round(3).to_dict("records")
                             for n in ("R3", "R2", "R1")} for h in (1, 2)}}
    (OUT / "params_v2.json").write_text(json.dumps(p2, indent=2))
    return choice


def run_final():
    p2 = json.loads((OUT / "params_v2.json").read_text())
    assert "confirm_result" in p2, "run confirm first"
    assert "final_result" not in p2, "final already run (single-use)"
    choice = p2["confirm_result"]["choice"]
    feats = {"R3": FEATURES, "R2": R2_FEATS, "R1": None}[choice]
    sk, go, skt, got, bios = P.load_panels()
    preds, end_r, _ = E.run_elo(g, K=V1["K"], H=V1["H"], phi_s=V1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"], skater_delta=p2["skater_delta"])
    from backtest import fit_yoy_slope
    slope = {1: fit_yoy_slope(2017, 1), 2: fit_yoy_slope(2017, 2)}
    tables = {}
    pred_store = {}
    for h in (1, 2):
        lam = p2["lams"][f"h{h}"]["full" if choice == "R3" else "r2"]
        model = (v1_predictions(end_r, preds, h, FINAL) if choice == "R1"
                 else expanding_predictions(fb, h, FINAL, feats, lam))
        pred_store[h] = model
        v1p = v1_predictions(end_r, preds, h, FINAL)
        rows = []
        for T in FINAL:
            act = ACT.loc[T]
            ydev = (act.pts_pct - act.pts_pct.mean()) * 164
            prior = ts[ts.season_end == T - h].set_index("team").pts_pct
            reg = ((prior - prior.mean()) * 164 * slope[h]).reindex(ydev.index).fillna(0)
            rows.append({
                "season": T,
                "mae_v2": float((model[T].reindex(ydev.index).fillna(0) - ydev).abs().mean()),
                "mae_v1": float((v1p[T].reindex(ydev.index).fillna(0) - ydev).abs().mean()),
                "mae_regressed": float((reg - ydev).abs().mean()),
                "mae_uniform": float(ydev.abs().mean()),
                "spearman_v2": float(model[T].reindex(ydev.index).rank().corr(ydev.rank())),
                "spearman_v1": float(v1p[T].reindex(ydev.index).rank().corr(ydev.rank())),
            })
        tables[h] = pd.DataFrame(rows)
        print(f"\n=== FINAL h{h} ({choice}) ===")
        print(tables[h].round(2).to_string(index=False))
        print(tables[h].drop(columns="season").mean().round(3).to_string())
    p2["final_result"] = {f"h{h}": tables[h].round(3).to_dict("records") for h in (1, 2)}
    (OUT / "params_v2.json").write_text(json.dumps(p2, indent=2))
    pd.concat([tables[1].assign(horizon=1), tables[2].assign(horizon=2)]).to_csv(
        OUT / "backtest_v2.csv", index=False)
    return pred_store, fb, end_r, preds, p2


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage_t"
    if cmd == "stage_t":
        stage_t()
    elif cmd == "confirm":
        run_confirm()
    elif cmd == "final":
        run_final()
