"""Shots-on-goal distributions for player lines (ORR 1.5).

Per skater-game, the expected shots are ORR 1.2's updated rate (the strict
preseason shots per game updated with the player's games so far, prior weight
from params/player_update.json). P(shots >= k) for k = 2, 3, 4 comes from a
Poisson or a negative binomial with size r (variance mu + mu^2 / r); r is
fitted by maximum likelihood on 2012 and 2014-2017. The test (2022, 2023)
runs once. Pre-declared rule: the negative binomial is adopted
(params/sog_dist.json "adopted") only if its mean log loss over the three
thresholds is lower than the Poisson's on test.

Run: python3 -m orr.backtest.sog_dist_bt
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from orr import config as C
from orr import players as PL
from orr import player_update as PU
from orr.backtest import players_bt as PB

TUNE, TEST = [2012, 2014, 2015, 2016, 2017], [2022, 2023]
KS = (2, 3, 4)
OUT = C.OUT / "backtest" / "sog_dist_bt.json"
PARAMS = C.PARAMS / "sog_dist.json"


def season_rows(V, prm, n0):
    tot = PB.run_season(V, prm, "strict")
    tot = tot[tot.gp > 0]
    prior = pd.DataFrame({"player_id": tot.player_id.astype(int), "s_pg": tot.sog / tot.gp})
    b = PU.season_boxes(V).sort_values(["player_id", "game_id"]).merge(prior, on="player_id", how="inner")
    grp = b.groupby("player_id")
    n_b, s_b = grp.cumcount(), grp.sog.cumsum() - b.sog
    mu = (n0["sog"] * b.s_pg + s_b) / (n0["sog"] + n_b)
    return pd.DataFrame({"mu": mu.clip(lower=0.02).to_numpy(), "y": b.sog.astype(int).to_numpy()})


def p_ge(mu, k, r=None):
    if r is None:
        return stats.poisson.sf(k - 1, mu)
    p = r / (r + mu)
    return stats.nbinom.sf(k - 1, r, p)


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    prm, n0 = PL.load_params(), PU.load_n0()
    tune = pd.concat([season_rows(V, prm, n0) for V in TUNE])
    test = pd.concat([season_rows(V, prm, n0) for V in TEST])
    nll = lambda lr: -stats.nbinom.logpmf(tune.y, np.exp(lr), np.exp(lr) / (np.exp(lr) + tune.mu)).mean()
    r = float(np.exp(minimize_scalar(nll, bounds=(-3, 6), method="bounded").x))
    res = {"protocol": __doc__, "r": r, "n_tune": int(len(tune)), "n_test": int(len(test)),
           "tune_var_ratio": float(((tune.y - tune.mu) ** 2).mean() / tune.mu.mean()), "test": {}}
    tot_p = tot_nb = 0.0
    for k in KS:
        y = (test.y >= k).astype(int).to_numpy()
        pp, pn = p_ge(test.mu.to_numpy(), k), p_ge(test.mu.to_numpy(), k, r)
        res["test"][str(k)] = {"rate": float(y.mean()), "mean_p_poisson": float(pp.mean()), "mean_p_nb": float(pn.mean()),
                               "ll_poisson": ll(pp, y), "ll_nb": ll(pn, y)}
        tot_p += ll(pp, y) / len(KS)
        tot_nb += ll(pn, y) / len(KS)
    res["test"]["mean_ll"] = {"poisson": tot_p, "nb": tot_nb}
    res["adopted"] = bool(tot_nb < tot_p)
    OUT.write_text(json.dumps(res, indent=1))
    PARAMS.write_text(json.dumps({"r": r, "adopted": res["adopted"], "source": "orr/backtest/sog_dist_bt.py"}, indent=1))
    print(json.dumps(res["test"], indent=1)); print("r", round(r, 3), "adopted", res["adopted"], "var ratio", round(res["tune_var_ratio"], 3))


if __name__ == "__main__":
    main()
