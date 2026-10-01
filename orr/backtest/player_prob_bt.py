"""Calibration of per-game player probabilities (ORR 1.3).

For every skater-game in the box scores, P(goal) = 1 - exp(-g_rate) and
P(point) = 1 - exp(-(g_rate + a_rate)), where the rates are ORR 1.2's: the
strict preseason projection updated with the player's games so far this
season (orr/player_update.py, n0 from params/player_update.json). The
probabilities are conditional on the player dressing, as the live player
lines are before their dress-probability factor.

A Platt correction (logistic regression of the outcome on logit p) is fitted
on 2012 and 2014-2017 only and scored once on 2021-22 and 2022-23.
Pre-declared rule: the correction goes live (params/player_prob.json
"adopted": true) only if it improves test log loss for BOTH P(goal) and
P(point); otherwise it ships switched off.

Run: python3 -m orr.backtest.player_prob_bt  -> orr/output/backtest/player_prob_bt.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from orr import config as C
from orr import players as PL
from orr import player_update as PU
from orr.backtest import players_bt as PB

TUNE, TEST = [2012, 2014, 2015, 2016, 2017], [2022, 2023]
OUT = C.OUT / "backtest" / "player_prob_bt.json"
PARAMS = C.PARAMS / "player_prob.json"


def season_rows(V: int, prm, n0: dict) -> pd.DataFrame:
    tot = PB.run_season(V, prm, "strict")
    tot = tot[tot.gp > 0]
    prior = pd.DataFrame({"player_id": tot.player_id.astype(int), "g_pg": tot.g / tot.gp, "a_pg": tot.a / tot.gp})
    b = PU.season_boxes(V).sort_values(["player_id", "game_id"])
    b = b.merge(prior, on="player_id", how="inner")
    grp = b.groupby("player_id")
    b["n_b"] = grp.cumcount()
    b["g_b"] = grp.g.cumsum() - b.g
    b["a_b"] = grp.a.cumsum() - b.a
    lg = (n0["g"] * b.g_pg + b.g_b) / (n0["g"] + b.n_b)
    la = (n0["a"] * b.a_pg + b.a_b) / (n0["a"] + b.n_b)
    return pd.DataFrame({"season": V, "p_goal": 1 - np.exp(-lg), "p_point": 1 - np.exp(-(lg + la)),
                         "y_goal": (b.g >= 1).astype(int), "y_point": ((b.g + b.a) >= 1).astype(int)})


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def platt_fit(p, y):
    x = logit(p)
    f = lambda w: ll(1 / (1 + np.exp(-(w[0] + w[1] * x))), y)
    return minimize(f, [0.0, 1.0], method="Nelder-Mead").x


def platt(p, w):
    return 1 / (1 + np.exp(-(w[0] + w[1] * logit(p))))


def main():
    prm, n0 = PL.load_params(), PU.load_n0()
    data = {V: season_rows(V, prm, n0) for V in TUNE + TEST}
    tune, test = pd.concat([data[V] for V in TUNE]), pd.concat([data[V] for V in TEST])
    res = {"protocol": __doc__, "n_tune": int(len(tune)), "n_test": int(len(test))}
    adopt = True
    for k in ("goal", "point"):
        w = platt_fit(tune[f"p_{k}"].to_numpy(), tune[f"y_{k}"].to_numpy())
        slope_test = platt_fit(test[f"p_{k}"].to_numpy(), test[f"y_{k}"].to_numpy())   # diagnostic only
        raw, cal = ll(test[f"p_{k}"], test[f"y_{k}"]), ll(platt(test[f"p_{k}"].to_numpy(), w), test[f"y_{k}"])
        res[k] = {"platt_tune": [float(w[0]), float(w[1])], "test_logloss_raw": raw, "test_logloss_platt": cal,
                  "test_mean_p": float(test[f"p_{k}"].mean()), "test_rate": float(test[f"y_{k}"].mean()),
                  "test_calibration_diag": [float(slope_test[0]), float(slope_test[1])]}
        adopt = adopt and cal < raw
    res["adopted"] = bool(adopt)
    OUT.write_text(json.dumps(res, indent=1))
    PARAMS.write_text(json.dumps({"goal": res["goal"]["platt_tune"], "point": res["point"]["platt_tune"],
                                  "adopted": res["adopted"], "source": "orr/backtest/player_prob_bt.py"}, indent=1))
    for k in ("goal", "point"):
        r = res[k]
        print(f"P({k}): test mean p {r['test_mean_p']:.4f} vs rate {r['test_rate']:.4f}; log loss raw {r['test_logloss_raw']:.5f}"
              f" Platt {r['test_logloss_platt']:.5f}; tune Platt {np.round(r['platt_tune'], 3)}; test diag {np.round(r['test_calibration_diag'], 3)}")
    print("adopted:", res["adopted"], " n_tune", res["n_tune"], " n_test", res["n_test"])


if __name__ == "__main__":
    main()
