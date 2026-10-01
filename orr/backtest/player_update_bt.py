"""Walk-forward backtest of in-season skater rate updating (ORR 1.2, orr/player_update.py).

For each season V, the prior is ORR's strict preseason projection
(players_bt.run_season(V, 'strict'): first-10-game rosters, nothing else
from V). At checkpoints after 25%, 50% and 75% of the season's games (in
game-id order), each skater's goals and assists per game are predicted for
the REST of the season, and the prediction is scored against what he did
after the checkpoint, scaled by his actual remaining games so that only the
rates are judged:

    error = |(g_pg + a_pg) * n_after - points_after|, skaters with n_after >= 10.

n0 (pseudo-games, per statistic) is chosen on 2012 and 2014-2017 only from a
fixed grid; 2018-19 confirm; the test is 2022 and 2023 (the last seasons with
complete box scores), run once. Baseline: the preseason rate alone (n0 = inf).

Run: python3 -m orr.backtest.player_update_bt   -> orr/output/backtest/player_update_bt.json
and orr/output/params/player_update.json (the chosen n0).
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from orr import config as C
from orr import players as PL
from orr import player_update as PU
from orr.backtest import players_bt as PB
from orr.backtest.games_bt import paired

TUNE, CONFIRM, TEST = [2012, 2014, 2015, 2016, 2017], [2018, 2019], [2022, 2023]
CHECKPOINTS = (0.25, 0.5, 0.75)
GRID = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, np.inf]
OUT = C.OUT / "backtest" / "player_update_bt.json"


def season_rows(V: int, prm) -> pd.DataFrame:
    """One row per (checkpoint, skater): prior rates, before-counts, after-counts."""
    tot = PB.run_season(V, prm, "strict")
    tot = tot[tot.gp > 0]
    prior = pd.DataFrame({"player_id": tot.player_id.astype(int), "g_pg": tot.g / tot.gp,
                          "a_pg": tot.a / tot.gp, "sog_pg": tot.sog / tot.gp})
    b = PU.season_boxes(V)
    gids = np.sort(b.game_id.unique())
    rows = []
    for c in CHECKPOINTS:
        cut = gids[int(len(gids) * c)]
        before, after = PU.totals(b[b.game_id < cut]), PU.totals(b[b.game_id >= cut])
        m = prior.merge(before.add_suffix("_b").rename(columns={"player_id_b": "player_id"}), on="player_id", how="left") \
                 .merge(after.add_suffix("_f").rename(columns={"player_id_f": "player_id"}), on="player_id", how="inner")
        m = m.fillna({"n_b": 0, "g_b": 0, "a_b": 0, "sog_b": 0})
        rows.append(m[m.n_f >= 10].assign(season=V, checkpoint=c))
    return pd.concat(rows, ignore_index=True)


def predict(d: pd.DataFrame, n0g: float, n0a: float) -> np.ndarray:
    def upd(prior, cnt, n, k):
        return prior if not np.isfinite(k) else (k * prior + cnt) / (k + n)
    rate = upd(d.g_pg, d.g_b, d.n_b, n0g) + upd(d.a_pg, d.a_b, d.n_b, n0a)
    return (rate * d.n_f).to_numpy()


def mae(d, n0g, n0a):
    return float(np.abs(predict(d, n0g, n0a) - (d.g_f + d.a_f).to_numpy()).mean())


def main():
    prm = PL.load_params()
    data = {V: season_rows(V, prm) for V in TUNE + CONFIRM + TEST}
    tune = pd.concat([data[V] for V in TUNE])
    ledger = [{"n0_g": g, "n0_a": a, "tune_mae": mae(tune, g, a)} for g, a in itertools.product(GRID, GRID)]
    best = min(ledger, key=lambda r: r["tune_mae"])
    n0g, n0a = best["n0_g"], best["n0_a"]
    # shots: chosen the same way on its own error (shots after the checkpoint)
    def sog_mae(d, k):
        r = d.sog_pg if not np.isfinite(k) else (k * d.sog_pg + d.sog_b) / (k + d.n_b)
        return float(np.abs(r * d.n_f - d.sog_f).mean())
    n0s = min(GRID, key=lambda k: sog_mae(tune, k))
    res = {"protocol": __doc__, "grid": [str(x) for x in GRID], "ledger": ledger,
           "chosen": {"g": n0g, "a": n0a, "sog": n0s}, "tune": {}, "confirm": {}, "test": {}}
    for name, seasons in (("tune", TUNE), ("confirm", CONFIRM), ("test", TEST)):
        d = pd.concat([data[V] for V in seasons])
        y = (d.g_f + d.a_f).to_numpy()
        p1, p0 = predict(d, n0g, n0a), predict(d, np.inf, np.inf)
        e1, e0 = np.abs(p1 - y), np.abs(p0 - y)
        res[name] = {"n": int(len(d)), "mae_updated": float(e1.mean()), "mae_preseason": float(e0.mean()),
                     "by_checkpoint": {str(c): {"updated": float(e1[d.checkpoint == c].mean()),
                                                "preseason": float(e0[d.checkpoint == c].mean())}
                                       for c in CHECKPOINTS},
                     "sog": {"updated": sog_mae(d, n0s), "preseason": sog_mae(d, np.inf)}}
        if name == "test":     # paired bootstrap over rows, as games_bt.paired (on absolute errors)
            res[name]["d"] = paired_abs(e1, e0)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    PU.PARAMS.write_text(json.dumps({"n0": {"g": n0g, "a": n0a, "sog": n0s},
                                     "source": "orr/backtest/player_update_bt.py (tuned 2012, 2014-2017)"}, indent=1,
                                    default=lambda x: None if not np.isfinite(x) else x))
    for k in ("tune", "confirm", "test"):
        r = res[k]
        print(f"{k:8s} n={r['n']:6d}  updated {r['mae_updated']:.3f}  preseason {r['mae_preseason']:.3f}  "
              f"sog {r['sog']['updated']:.2f} vs {r['sog']['preseason']:.2f}")
    print("chosen n0:", res["chosen"], " test diff:", res["test"].get("d"))


def paired_abs(e1, e0, nb: int = 2000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    d = e1 - e0
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)])
    return {"diff": float(d.mean()), "se": float(bs.std()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                     float(np.percentile(bs, 97.5))]}


if __name__ == "__main__":
    main()
