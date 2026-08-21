"""NeurHL — ONE-SHOT frozen restatement on 2018-2026 (PLAN_NeurHL P7/P8).

Run exactly once, after gates. For each season T in 2018-2026: preseason (h1)
inputs at vantage T (P1-clean), the T-specific walk-forward game-model
ensemble, temperature from T's val season, bridge -> sim (seed 711+T, 4000
draws), scored as deviation MAE/82, Spearman, CRPS. Compared per P8 against
the house restatement incumbents: HOWE per-season MAE/Spearman
(output/report_only_howe.csv h1) and v1 sim CRPS
(output/report_only_v4.json crps_restatement h1).

P6: 2020/2021 metrics are computed (per-82 normalized) but also reported with
the COVID seasons excluded; the P8 decision uses all-season means (house
convention) with the no-single-era-carry clause checked on pre-2022 seasons.

P8 decision (pre-committed): (i) standalone report track iff NeurHL beats HOWE
on >=2 of 3 {MAE lower, Spearman higher, CRPS lower vs v1-sim} AND does not
lose >=2 consecutive pre-2022 seasons while winning overall; (ii) else the
50/50 Elo-space blend ships iff it beats HOWE (evaluated only if reached —
requires recomputing HOWE per-team restatement predictions, implemented then);
(iii) else documented null. 2026-27 production stays v1/v4/HOWE regardless.

Writes params_neurhl.json["restatement"] + committed preds/restate_<T>.csv.
Refuses to run twice (single-use assertion, house precedent).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, PROJ  # noqa: E402
from eval.backtest_season import ensemble_probs, spearman  # noqa: E402
from eval.gates import apply_temperature, fit_temperature  # noqa: E402
from sim.preseason_inputs import build_preseason  # noqa: E402
from sim.ratings_bridge import simulate  # noqa: E402

import backtest as B  # noqa: E402
import engine as E  # noqa: E402
import scoring as SC  # noqa: E402

RESTATE_T = list(range(2018, 2027))
N_SIMS = 4000


def main():
    params_p = NOUT / "params_neurhl.json"
    blob = json.loads(params_p.read_text())
    assert "restatement" not in blob, \
        "restatement already run (single-use, P7)"
    assert all(blob.get("gates", {}).get(g, {}).get("pass") is not None
               for g in ("G1", "G2", "G3", "G4", "G5")), \
        "gates must be recorded before the restatement"

    howe = pd.read_csv(PROJ / "output" / "report_only_howe.csv")
    howe = howe[(howe.horizon == 1) & (howe.model == "howe")].set_index("season")
    v4j = json.loads((PROJ / "output" / "report_only_v4.json").read_text())
    v1crps = {r["season"]: r["crps_model"] for r in v4j["crps_restatement"]
              if r["horizon"] == 1}
    v1p = json.loads((PROJ / "output" / "params.json").read_text())
    preds_elo, _, _ = E.run_elo(B.g, K=v1p["K"], H=v1p["H"], phi_s=v1p["phi_s"])

    rows = {}
    for T in RESTATE_T:
        tensors, sched = build_preseason(T)
        # temperature: fit on T's own val-season artifact (train window of T)
        val = pd.read_csv(NOUT / "preds" / f"game_val_{T}.csv")
        tau = fit_temperature(val[["p_home_reg", "p_away_reg", "p_home_extra",
                                   "p_away_extra"]].to_numpy(),
                              val.outcome4.to_numpy())
        p4 = apply_temperature(ensemble_probs(T, tensors), tau)
        games = sched.assign(p_home=p4[:, 0] + p4[:, 2])
        om = E.fit_outcome(preds_elo, list(range(2006, T)))
        sim = simulate(games, om, N_SIMS, 711 + T, T)
        teams = list(sim["teams"])
        act = B.ACT.loc[T]
        gp = act.gp.reindex(teams).to_numpy()
        act82 = (act.pts / act.gp * 82).reindex(teams).to_numpy()
        pts82 = sim["pts"] / gp[None, :] * 82
        xp82 = pts82.mean(0)
        rows[str(T)] = {
            "dev_mae82": float(np.abs((xp82 - xp82.mean())
                                      - (act82 - act82.mean())).mean()),
            "spearman": spearman(xp82, act82),
            "crps": float(np.mean([SC.crps_draws(pts82[:, i], act82[i])
                                   for i in range(len(teams))])),
            "howe_mae": float(howe.loc[T, "mae"]),
            "howe_spearman": float(howe.loc[T, "spearman"]),
            "v1_crps": float(v1crps[T]),
            "tau": tau,
        }
        pd.DataFrame({"team": teams, "xp82": xp82, "sd82": pts82.std(0)}
                     ).to_csv(NOUT / "preds" / f"restate_{T}.csv", index=False)
        print(f"{T}: MAE {rows[str(T)]['dev_mae82']:.3f} "
              f"(HOWE {rows[str(T)]['howe_mae']:.3f})  "
              f"rho {rows[str(T)]['spearman']:.3f} "
              f"(HOWE {rows[str(T)]['howe_spearman']:.3f})  "
              f"CRPS {rows[str(T)]['crps']:.3f} (v1 {rows[str(T)]['v1_crps']:.3f})")
        sys.stdout.flush()

    def mean(key):
        return float(np.mean([r[key] for r in rows.values()]))

    mae, rho, crps = mean("dev_mae82"), mean("spearman"), mean("crps")
    h_mae, h_rho, v_crps = mean("howe_mae"), mean("howe_spearman"), mean("v1_crps")
    wins = int(mae < h_mae) + int(rho > h_rho) + int(crps < v_crps)
    pre22 = [T for T in RESTATE_T if T < 2022]
    lose_streak = 0
    max_streak = 0
    for T in pre22:
        if rows[str(T)]["dev_mae82"] > rows[str(T)]["howe_mae"]:
            lose_streak += 1
            max_streak = max(max_streak, lose_streak)
        else:
            lose_streak = 0
    era_carry = max_streak >= 2 and wins >= 2
    if wins >= 2 and not era_carry:
        decision = "standalone_report_track"
    else:
        decision = "blend_evaluation_required" if wins >= 1 else "null"
    blob["restatement"] = {
        "per_season": rows,
        "means": {"dev_mae82": mae, "spearman": rho, "crps": crps,
                  "howe_mae": h_mae, "howe_spearman": h_rho,
                  "v1_crps": v_crps},
        "no_covid_means": {
            k: float(np.mean([rows[str(T)][k] for T in RESTATE_T
                              if T not in (2020, 2021)]))
            for k in ("dev_mae82", "spearman", "crps")},
        "p8_wins": wins, "p8_pre2022_max_lose_streak": max_streak,
        "p8_decision": decision,
    }
    params_p.write_text(json.dumps(blob, indent=1))
    print(json.dumps({"means": blob["restatement"]["means"],
                      "p8_wins": wins, "decision": decision}, indent=1))


if __name__ == "__main__":
    main()
