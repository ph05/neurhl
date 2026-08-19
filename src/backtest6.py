"""v6 train-only gates (PLAN_V6 G, prereg committed at b719257 BEFORE this ran).

GATE F4 (ridge candidates line_cont/toi_hhi_f/coach/prospect_prod): identical
protocol to v5's F3 — add-one-in train LOSO dMAE < 0.00 AND stability >= 0.67;
coach_new/coach_tenure -> single best entrant; prospect_prod vs prospect_pipeline
collinearity (add + replacement forms); greedy joint must beat the v5 incumbent.
GATE T1 (travel d_adj): logistic home win ~ Elo + b2b + net |tz| change (rest<=2)
+ net km/1000 over prior 3 days, 2010-2017 only; ship needs |t|>=4 AND mean |Elo
equivalent| >= 10 AND same sign in 2010-2013 / 2014-2017 halves.
SCREEN S3 (recorded only): spell-based injury propensity EB vs bucket-only MSE.

Writes output/params_v6.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import ridge as R
from features_v6 import FEATURES_V6_ALL, FeatureBuilderV6

PROJ = Path(__file__).resolve().parents[1]
PROC = PROJ / "data" / "processed"
OUT = PROJ / "output"
TRAIN = {1: list(range(2012, 2018)), 2: list(range(2013, 2018))}
COACH_PAIR = ("coach_new", "coach_tenure")
CANDS = ["line_cont", "toi_hhi_f", "coach_new", "coach_tenure", "prospect_prod"]


def gate_f4(fb, ship5, p6):
    final_sets, final_lams = {}, {}
    for h in (1, 2):
        X, y, meta = fb.feature_matrix(TRAIN[h], h, feats=FEATURES_V6_ALL)
        seasons = meta["T"].to_numpy()
        inc = list(ship5[f"features_h{h}"])
        cols_i = [FEATURES_V6_ALL.index(f) for f in inc]
        lam_i, tab_i = R.loso_cv(X[:, cols_i], y, seasons)
        mae_i = tab_i.mae.min()
        print(f"\n=== h{h}: v5 incumbent ({len(inc)}) LOSO MAE {mae_i:.4f} ===")
        rec = {"mae_incumbent": round(float(mae_i), 4), "candidates": {}}

        results = []
        for cand in CANDS:
            cols_c = cols_i + [FEATURES_V6_ALL.index(cand)]
            lam_c, tab_c = R.loso_cv(X[:, cols_c], y, seasons)
            mae_c = tab_c.mae.min()
            stab = float(R.loso_coef_signs(X[:, cols_c], y, seasons, lam_c)[-1])
            dmae = float(mae_c - mae_i)
            passed = (dmae < 0.0) and (stab >= 0.67)
            results.append((cand, dmae, stab, passed, mae_c))
            print(f"  {cand:15s} dMAE {dmae:+.4f}  stab {stab:.2f}  "
                  f"-> {'PASS' if passed else 'fail'}")
            rec["candidates"][cand] = {"dmae": round(dmae, 4),
                                       "stability": round(stab, 2),
                                       "pass": bool(passed)}
        passers = [r for r in results if r[3]]
        # coach pair: single best entrant
        cp = [r for r in passers if r[0] in COACH_PAIR]
        if len(cp) > 1:
            best = min(cp, key=lambda r: r[1])
            passers = [r for r in passers if r[0] not in COACH_PAIR or r is best]
        # prospect_prod vs prospect_pipeline: replacement form
        pp = next((r for r in passers if r[0] == "prospect_prod"), None)
        repl_note = None
        if pp and "prospect_pipeline" in inc:
            swap = [f for f in inc if f != "prospect_pipeline"] + ["prospect_prod"]
            cols_s = [FEATURES_V6_ALL.index(f) for f in swap]
            lam_s, tab_s = R.loso_cv(X[:, cols_s], y, seasons)
            mae_s = tab_s.mae.min()
            repl_note = {"mae_replace": round(float(mae_s), 4),
                         "mae_add": round(float(pp[4]), 4),
                         "chosen": "replace" if mae_s < pp[4] else "add"}
            print(f"  prospect_prod replacement: {mae_s:.4f} vs add {pp[4]:.4f} "
                  f"-> {repl_note['chosen']}")
            if mae_s < pp[4]:
                inc = [f for f in inc if f != "prospect_pipeline"]
                cols_i = [FEATURES_V6_ALL.index(f) for f in inc]
        rec["prospect_family"] = repl_note

        cur, cur_cols = list(inc), list(cols_i)
        lam_j, tab_j = R.loso_cv(X[:, cur_cols], y, seasons)
        mae_j = tab_j.mae.min()
        for cand, dmae, stab, _, _ in sorted(passers, key=lambda r: r[1]):
            trial = cur_cols + [FEATURES_V6_ALL.index(cand)]
            lam_t, tab_t = R.loso_cv(X[:, trial], y, seasons)
            if tab_t.mae.min() < mae_j:
                cur, cur_cols = cur + [cand], trial
                lam_j, mae_j = lam_t, tab_t.mae.min()
                print(f"  joint: +{cand} -> {mae_j:.4f} (kept)")
            else:
                print(f"  joint: +{cand} -> {tab_t.mae.min():.4f} (rejected)")
        ships = (mae_j < mae_i) and (set(cur) != set(ship5[f"features_h{h}"]))
        rec["joint"] = {"new": sorted(set(cur) - set(ship5[f"features_h{h}"])),
                        "removed": sorted(set(ship5[f"features_h{h}"]) - set(cur)),
                        "mae_final": round(float(mae_j), 4),
                        "dmae_vs_incumbent": round(float(mae_j - mae_i), 4),
                        "lam": float(lam_j), "ships": bool(ships)}
        print(f"  h{h} FINAL: {mae_j:.4f} vs {mae_i:.4f} -> "
              f"{'SHIPS' if ships else 'NULL (v5 set stands)'}")
        final_sets[h] = cur if ships else list(ship5[f"features_h{h}"])
        final_lams[h] = float(lam_j if ships else ship5[f"lam_h{h}"])
        p6["gate_F4"][f"h{h}"] = rec
    return final_sets, final_lams


def gate_t1(preds, g, p6):
    tv = pd.read_csv(PROC / "travel_games.csv", parse_dates=["date"]) \
        .drop(columns=["season_end"])
    assert len(preds) == len(g)          # run_elo emits one row per game, in order
    preds = preds.assign(date=g.date.to_numpy())
    pr = preds[(preds.game_type == "R") & preds.season_end.between(2010, 2017)]
    m = pr.merge(tv, on=["date", "home", "away"])
    m = m[~(m.went_ot | m.went_so)]      # regulation results, as in B3
    for s in ("home", "away"):
        m[f"{s}_tz_eff"] = np.where(m[f"{s}_rest"] <= 2,
                                    m[f"{s}_dtz"].abs(), 0.0)
        m[f"{s}_b2b"] = (m[f"{s}_rest"] == 1).astype(float)
    m["elo_d"] = (m.rh - m.ra) / 100.0
    m["net_tz"] = m.away_tz_eff - m.home_tz_eff
    m["net_km3"] = (m.away_km3d - m.home_km3d) / 1000.0
    cols = ["elo_d", "home_b2b", "away_b2b", "net_tz", "net_km3"]

    def fit(sub):
        X = sub[cols].to_numpy(float)
        b = E.fit_logistic(X, sub.home_win.to_numpy(float))
        # crude t-stats via IRLS covariance
        p = 1 / (1 + np.exp(-(b[0] + X @ b[1:])))
        W = p * (1 - p)
        Xd = np.column_stack([np.ones(len(X)), X])
        cov = np.linalg.inv(Xd.T @ (Xd * W[:, None]))
        se = np.sqrt(np.diag(cov))
        return b, b / se

    b, t = fit(m)
    b1, _ = fit(m[m.season_end <= 2013])
    b2, _ = fit(m[m.season_end >= 2014])
    elo_slope = b[1] / 100.0             # per Elo point
    rec = {"n": len(m)}
    ships_terms = []
    for term, i in (("net_tz", 4), ("net_km3", 5)):
        elo_eq = float(b[i] / elo_slope)
        mean_x = float(np.abs(m[cols[i - 1]]).mean())
        ok = (abs(t[i]) >= 4.0 and abs(elo_eq) * mean_x >= 10.0
              and np.sign(b1[i]) == np.sign(b2[i]))
        rec[term] = {"coef": round(float(b[i]), 4), "t": round(float(t[i]), 2),
                     "elo_equiv_per_unit": round(elo_eq, 1),
                     "mean_abs_unit": round(mean_x, 3),
                     "mean_elo_effect": round(abs(elo_eq) * mean_x, 1),
                     "sign_stable_halves": bool(np.sign(b1[i]) == np.sign(b2[i])),
                     "ships": bool(ok)}
        if ok:
            ships_terms.append(term)
        print(f"  T1 {term}: coef {b[i]:+.4f} t {t[i]:+.2f} "
              f"Elo/unit {elo_eq:+.1f} mean-effect {abs(elo_eq) * mean_x:.1f} "
              f"-> {'SHIPS' if ok else 'no ship'}")
    rec["ships"] = ships_terms
    p6["gate_T1"] = rec


def screen_s3(p6):
    ab = pd.read_csv(PROC / "player_absences.csv")
    bios = pd.read_csv(PROC / "panel_bios.csv", parse_dates=["birthDate"])
    ab = ab[ab.window_games >= 20].merge(bios, on="playerId", how="left")
    ab["age"] = ab.season_end - ab.birthDate.dt.year
    ab["avail"] = ab.dressed / ab.window_games
    ab["bucket"] = pd.cut(ab.age, [0, 22, 26, 30, 34, 60], labels=False)
    rows = []
    for T in range(2013, 2018):
        cur = ab[ab.season_end == T].copy()
        hist = ab[ab.season_end < T]
        bmean = hist.groupby("bucket").avail.mean()
        ph = hist.groupby("playerId").agg(n=("window_games", "sum"),
                                          a=("dressed", "sum"))
        ph["pavail"] = ph.a / ph.n
        cur = cur.merge(ph, on="playerId", how="left")
        cur["bpred"] = cur.bucket.map(bmean).fillna(hist.avail.mean())
        for n0 in (10.0, 40.0, 80.0, 160.0):
            w = cur.n.fillna(0) / (cur.n.fillna(0) + n0)
            eb = w * cur.pavail.fillna(cur.bpred) + (1 - w) * cur.bpred
            rows.append({"T": T, "n0": n0,
                         "mse_eb": float(((eb - cur.avail) ** 2).mean()),
                         "mse_bucket": float(((cur.bpred - cur.avail) ** 2).mean())})
    df = pd.DataFrame(rows)
    agg = df.groupby("n0")[["mse_eb", "mse_bucket"]].mean()
    best = agg.mse_eb.idxmin()
    rec = {"mse_bucket": round(float(agg.mse_bucket.iloc[0]), 5),
           "mse_eb_best": round(float(agg.mse_eb.min()), 5),
           "n0_best": float(best),
           "eb_beats_bucket": bool(agg.mse_eb.min() < agg.mse_bucket.iloc[0])}
    print(f"  S3: bucket MSE {rec['mse_bucket']} vs EB {rec['mse_eb_best']} "
          f"(n0={best}) -> {'EB WINS (future plan may act)' if rec['eb_beats_bucket'] else 'bucket stands'}")
    p6["screen_S3"] = rec


def main():
    t0 = time.time()
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p5 = json.loads((OUT / "params_v5.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    g, ts = E.load()
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilderV6(end_r, ts, goalie_hp=p2["goalie_hp"],
                          skater_delta=p2["skater_delta"])
    p6 = {"prereg": "PLAN_V6.md @ git b719257 (committed before fetch/gates)",
          "gate_F4": {}}
    final_sets, final_lams = gate_f4(fb, p5["shipped"], p6)
    print("\n=== GATE T1 (travel) ===")
    gate_t1(preds, g, p6)
    print("\n=== SCREEN S3 (spell propensity; recorded only) ===")
    screen_s3(p6)
    p6["shipped"] = {
        "features_h1": final_sets[1], "features_h2": final_sets[2],
        "lam_h1": final_lams[1], "lam_h2": final_lams[2],
        "travel_terms": p6["gate_T1"]["ships"],
        "v6_is_null": not (p6["gate_F4"]["h1"]["joint"]["ships"]
                           or p6["gate_F4"]["h2"]["joint"]["ships"]
                           or p6["gate_T1"]["ships"]),
        "inherits": "everything else from params_v4/params_v5 shipped",
    }
    p6["seconds"] = round(time.time() - t0, 1)
    (OUT / "params_v6.json").write_text(json.dumps(p6, indent=2))
    print(f"\nparams_v6.json written ({p6['seconds']}s)")


if __name__ == "__main__":
    main()
