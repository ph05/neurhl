"""NeurHL — season-level (h1) backtest on tune seasons -> gates G2/G5.

Per tune season T (2012-2017): build preseason inputs (prior-season rosters,
P1-clean), forward through the SAME 5 walk-forward game-model checkpoints
train_game.py produced for T (trained on < T), temperature from
preds/game_val_<T> tau (params_neurhl.json), bridge to Elo space, simulate the
real schedule (seed 711+T, 4000 sims), and score deviation MAE/82, Spearman,
and CRPS vs actuals. Appends G2/G5 records to params_neurhl.json. CPU-only
(P9); prediction artifacts committed as preds/season_sim_<T>.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, NOUT, PROC, TENSORS  # noqa: E402
from eval.gates import apply_temperature  # noqa: E402
from models.game_model import from_config  # noqa: E402
from sim.preseason_inputs import build_preseason  # noqa: E402
from sim.ratings_bridge import simulate  # noqa: E402

import backtest as B  # noqa: E402
import engine as E  # noqa: E402
import scoring as SC  # noqa: E402

TUNE_T = [2012, 2013, 2014, 2015, 2016, 2017]
SEEDS = [0, 1, 2, 3, 4]
N_SIMS = 4000


def ensemble_probs(T: int, tensors: dict) -> np.ndarray:
    ps = []
    for s in SEEDS:
        model = from_config()
        model.load_state_dict(torch.load(CKPT / f"game_T{T}_s{s}.pt",
                                         map_location="cpu"))
        model.eval()
        outs = []
        with torch.no_grad():
            n = len(tensors["ctx"])
            for i in range(0, n, 512):
                b = {k: v[i:i + 512] for k, v in tensors.items()}
                outs.append(torch.softmax(model(b)["out4"], -1))
        ps.append(torch.cat(outs).numpy())
    return np.mean(ps, axis=0)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def main():
    params_p = NOUT / "params_neurhl.json"
    blob = json.loads(params_p.read_text()) if params_p.exists() else {}
    taus = blob.get("temperatures", {})
    v1p = json.loads((TENSORS.parents[2] / "output" / "params.json")
                     .read_text())
    preds_elo, end_r, _ = E.run_elo(B.g, K=v1p["K"], H=v1p["H"],
                                    phi_s=v1p["phi_s"])
    rows = {}
    for T in TUNE_T:
        tensors, sched = build_preseason(T)
        p4 = ensemble_probs(T, tensors)
        p4 = apply_temperature(p4, float(taus.get(str(T), 1.0)))
        games = sched.assign(p_home=p4[:, 0] + p4[:, 2])
        om = E.fit_outcome(preds_elo, list(range(2006, T)))   # causal, < T
        sim = simulate(games, om, N_SIMS, 711 + T, T)
        teams = list(sim["teams"])
        act = B.ACT.loc[T]
        gp = act.gp.reindex(teams).to_numpy()
        act82 = (act.pts / act.gp * 82).reindex(teams).to_numpy()
        pts82 = sim["pts"] / gp[None, :] * 82
        xp82 = pts82.mean(0)
        dev_mae = float(np.abs((xp82 - xp82.mean()) - (act82 - act82.mean()))
                        .mean())
        rho = spearman(xp82, act82)
        crps = float(np.mean([SC.crps_draws(pts82[:, i], act82[i])
                              for i in range(len(teams))]))
        rows[str(T)] = {"dev_mae82": dev_mae, "spearman": rho, "crps": crps}
        pd.DataFrame({"team": teams, "xp82": xp82,
                      "sd82": pts82.std(0)}).to_csv(
            NOUT / "preds" / f"season_sim_{T}.csv", index=False)
        print(f"{T}: dev MAE {dev_mae:.3f}  rho {rho:.3f}  CRPS {crps:.3f}")
        sys.stdout.flush()
    mae = float(np.mean([r["dev_mae82"] for r in rows.values()]))
    rho = float(np.mean([r["spearman"] for r in rows.values()]))
    crps = float(np.mean([r["crps"] for r in rows.values()]))
    gates = {
        "G2": {"rule": "dev MAE mean <= 9.42 AND spearman >= 0.545",
               "dev_mae": mae, "spearman": rho,
               "rule_eval": bool(mae <= 9.42 and rho >= 0.545),
               "pass": bool(mae <= 9.42 and rho >= 0.545)},
        "G5": {"rule": "CRPS mean <= 6.674", "crps": crps,
               "rule_eval": bool(crps <= 6.674),
               "pass": bool(crps <= 6.674)},
    }
    blob["season_per_season"] = rows
    blob.setdefault("gates", {}).update(gates)
    params_p.write_text(json.dumps(blob, indent=1))
    print(json.dumps({"mae": mae, "spearman": rho, "crps": crps,
                      "G2": gates["G2"]["pass"], "G5": gates["G5"]["pass"]},
                     indent=1))


if __name__ == "__main__":
    main()
