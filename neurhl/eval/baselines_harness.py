"""NeurHL — incumbent baselines on the TUNE window (PLAN_NeurHL gate inputs).

Computes, on tune seasons only (game level 2012-2017, season level h1 predict
seasons 2012-2017), the numbers that PLAN_NeurHL.md locks before any NeurHL
gate runs:

  - v1 walk-forward in-season game log loss (frozen params.json Elo, the exact
    src/backtest.logloss protocol) — G1/G4 opponent
  - v1 walk-forward h1 season MAE/82 (house norm_mae), deviation-space MAE, and
    Spearman via src/backtest.project_season — G2/G5 opponent
  - uniform / regressed-prior / raw-prior baselines (causal per-season slopes)
  - Tier-0 slot (null until models/baseline_gbm.py writes its tune numbers)

Reuses src/backtest.py machinery verbatim — nothing here re-tunes anything and
no season > 2017 is touched. Output: neurhl/output/baselines_tune.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _missing_scipy_guard  # noqa: F401  (fail fast if absent)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, PROJ  # noqa: E402

import backtest as B  # noqa: E402  (src/ module; loads g, ts, ACT)
import engine as E  # noqa: E402

GAME_SEASONS = list(range(2012, 2018))
SEASON_T = list(range(2012, 2018))     # h1 predict seasons in the tune window


def spearman(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import spearmanr
    j = pd.concat([a, b], axis=1).dropna()
    return float(spearmanr(j.iloc[:, 0], j.iloc[:, 1]).statistic)


def main():
    v1 = json.loads((PROJ / "output" / "params.json").read_text())
    preds, end_r, _ = E.run_elo(B.g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])

    out = {"protocol": "tune-window only; v1 frozen params; backtest.py machinery",
           "v1_params": {k: v1[k] for k in ("K", "H", "phi_s", "w", "phi1")}}

    # ---- game-level log loss (v1 Elo expectation, house protocol)
    ll = {str(T): B.logloss(preds, [T]) for T in GAME_SEASONS}
    ll["pooled_2012_2017"] = B.logloss(preds, GAME_SEASONS)
    ll["pooled_2014_2017"] = B.logloss(preds, [t for t in GAME_SEASONS if t >= 2014])
    # constant-home baseline, per protocol: p = causal home-win rate to date
    sub = B.g[(B.g.game_type == "R") & B.g.season_end.isin(GAME_SEASONS)]
    hw_prior = B.g[(B.g.game_type == "R") & (B.g.season_end < 2012)]
    p_const = float((hw_prior.home_g > hw_prior.away_g).mean())
    y = (sub.home_g > sub.away_g).to_numpy().astype(float)
    ll["constant_home"] = float(-(y * np.log(p_const)
                                  + (1 - y) * np.log(1 - p_const)).mean())
    out["game_logloss"] = ll

    # ---- season-level h1 (v1 projection) + naive baselines
    rows = {}
    for T in SEASON_T:
        _, xp, _, _ = B.project_season(end_r, preds, T, 1, w=v1["w"], phi=v1["phi1"])
        xp = pd.Series(xp)
        act = B.ACT.loc[T]
        per82 = act.pts / act.gp * 82
        xp82 = (xp / act.gp * 82).reindex(per82.index)
        dev_mae = float((xp82 - xp82.mean() - (per82 - per82.mean())).abs().mean())
        slope1 = B.fit_yoy_slope(T - 1, 1)
        base = B.baselines(T, 1, slope1, slope1)
        rows[str(T)] = {
            "v1_mae82": B.norm_mae(xp.to_dict(), T),
            "v1_dev_mae82": dev_mae,
            "v1_spearman": spearman(xp82, per82),
            "uniform_mae82": B.norm_mae(base["uniform"].to_dict(), T),
            "regressed_mae82": B.norm_mae(base["regressed_prior"].to_dict(), T),
            "raw_prior_mae82": B.norm_mae(base["raw_prior"].to_dict(), T),
        }
    agg = {k: float(np.mean([rows[str(T)][k] for T in SEASON_T]))
           for k in next(iter(rows.values()))}
    out["season_h1_per_season"] = rows
    out["season_h1_mean_2012_2017"] = agg
    agg46 = {k: float(np.mean([rows[str(T)][k] for T in SEASON_T if T >= 2014]))
             for k in next(iter(rows.values()))}
    out["season_h1_mean_2014_2017"] = agg46

    # ---- season CRPS via the v1 simulator (G5 anchor), tune seasons only
    import scoring as SC
    crps_rows = {}
    rng_seed = 711
    for T in SEASON_T:
        ratings, _, om, sched = B.project_season(end_r, preds, T, 1,
                                                 w=v1["w"], phi=v1["phi1"])
        ratings = E.fill_missing(ratings, set(sched.home) | set(sched.away))
        sim = E.simulate_season(ratings, v1["sigma1"],
                                sched[["home", "away"]].assign(d_adj=0.0), om,
                                E.divisions_for(T), 4000,
                                np.random.default_rng(rng_seed + T),
                                playoffs=False)
        act = B.ACT.loc[T]
        teams = list(sim["teams"])
        pts = sim["pts"]
        act82 = (act.pts / act.gp * 82).reindex(teams).to_numpy()
        gp = act.gp.reindex(teams).to_numpy()
        pts82 = pts / gp[None, :] * 82
        crps_rows[str(T)] = float(np.mean(
            [SC.crps_draws(pts82[:, i], act82[i]) for i in range(len(teams))]))
    out["season_crps_v1"] = crps_rows
    out["season_crps_v1_mean"] = float(np.mean(list(crps_rows.values())))

    # preserve tier0 results across harness reruns (baseline_gbm.py owns them)
    prev = NOUT / "baselines_tune.json"
    out["tier0"] = (json.loads(prev.read_text()).get("tier0")
                    if prev.exists() else None)

    NOUT.mkdir(parents=True, exist_ok=True)
    (NOUT / "baselines_tune.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"game_logloss_pooled": ll["pooled_2012_2017"],
                      "constant_home": ll["constant_home"],
                      "season_mean": agg}, indent=1))
    print("wrote neurhl/output/baselines_tune.json")


if __name__ == "__main__":
    main()
