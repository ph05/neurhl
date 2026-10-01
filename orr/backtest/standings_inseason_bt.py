"""In-season standings backtest (ORR 1.4): are the in-season playoff odds calibrated?

For each season and each checkpoint (25%, 50%, 75% of its games, by date),
the in-season filter (ratings.InSeasonFilter, ORR 1.0 configuration, from the
team-history prior) is updated with the games so far and the rest of the
season is simulated (season.simulate, completed games fixed). Final points
are scored with sample CRPS and 80% interval coverage, against:
  * preseason:  the same simulation from the un-updated prior;
  * pace:       points so far scaled to the full schedule (MAE only).

Tuned on 2012 and 2014-2017 only:
  (1) k, the multiplier on the rest-of-season drift sd
      (drift_full * sqrt(share of games left) * k), grid K;
  (2) N for ORR 1.4's long-term-absence offsets (lineups.absence_offsets),
      grid NS, at the chosen k.
Test, run once: 2022 and 2023 (the last box-score seasons with lineups).

Pre-declared rules: k is adopted as tuned (it is a calibration constant).
Absence offsets go live (params/standings_inseason.json "absence": true)
only if they lower test CRPS at the chosen k; otherwise they ship off.

Run: python3 -m orr.backtest.standings_inseason_bt
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from orr import config as C
from orr import lineups as LU
from orr import ratings as R
from orr import season as SS
from orr import structural as S
from orr.backtest import gamefile_bt as GB

TUNE, TEST = [2012, 2014, 2015, 2016, 2017], [2022, 2023]
CHECKPOINTS = (0.25, 0.5, 0.75)
K = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
NS = [5, 10, 20]
N_SIMS = 2000
OUT = C.OUT / "backtest" / "standings_inseason_bt.json"
PARAMS = C.PARAMS / "standings_inseason.json"


def crps_sample(x, y):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / n ** 2)


def final_points(gv: pd.DataFrame) -> pd.Series:
    win_h = gv.home_g > gv.away_g
    ot = gv.extra != "REG"
    pts = pd.concat([pd.Series(np.where(win_h, 2, np.where(ot, 1, 0)), index=gv.home.to_numpy()),
                     pd.Series(np.where(~win_h, 2, np.where(ot, 1, 0)), index=gv.away.to_numpy())])
    return pts.groupby(level=0).sum()


class Season:
    def __init__(self, V: int):
        self.V = V
        hp = self.hp = R.load_hp()
        self.P = R.fit_gamemodel_params(V, hp, write=False)
        self.params = {"hp": hp, "struct": S.structural(V, hp.window, hp.h_halflife, goalie_key=hp.goalie),
                       "P": self.P}
        self.pre = R.preseason_table(V, hp.ridge, use_roster=hp.use_roster, sp=hp.pre_sp)
        g = S.game_frame()
        self.gv = g[g.season_end == V].sort_values(["date", "gid"]).reset_index(drop=True)
        self.sch = GB.schedule(V)                                   # game_id = gid
        self.adj = SS.FittedModel(self.P, V).game_adjustments(self.sch)
        self.actual = final_points(self.gv)
        self.n_games = pd.concat([self.gv.home, self.gv.away]).value_counts()
        self.drift_full = float(np.sqrt(190 * (hp.q_s + hp.q_f)))
        self.states = {}

    def state(self, c: float):
        if c in self.states:
            return self.states[c]
        cut = self.gv.date.iloc[int(len(self.gv) * c)]
        done = self.gv[self.gv.date < cut]
        filt = R.InSeasonFilter(self.pre, self.params, season_end=self.V, start_date=self.gv.date.min())
        prior = filt.state()
        for _, day in done.groupby("date", sort=True):
            filt.update_day(day)
        P = dict(self.P)
        P["h"] = filt.levels()["h"]
        st = {"done": done, "cur": filt.state(), "prior": prior, "model": SS.FittedModel(P, self.V),
              "model0": SS.FittedModel(dict(self.P), self.V), "left": 1 - len(done) / len(self.gv)}
        self.states[c] = st
        return st

    def sim(self, c: float, k: float, which: str = "cur", adj=None) -> pd.DataFrame:
        st = self.state(c)
        r = st[which]
        rho = float(np.clip((r.od_cov / (r.o_sd * r.d_sd)).mean(), -0.95, 0.95))
        done = st["done"][["gid", "home_g", "away_g", "extra"]].rename(columns={"gid": "game_id"})
        res = SS.simulate(self.sch, r[["team", "o", "d", "o_sd", "d_sd"]],
                          st["model" if which == "cur" else "model0"], n_sims=N_SIMS, seed=C.SEED,
                          drift_sd=self.drift_full * np.sqrt(st["left"]) * k, completed=done,
                          game_adj=self.adj if adj is None else adj, playoffs=False, rho_od=rho)
        rows = []
        for j, t in enumerate(res.teams):
            x, y = res.points[:, j], float(self.actual[t])
            p10, p90 = np.percentile(x, [10, 90])
            rows.append({"season": self.V, "checkpoint": c, "team": t, "crps": crps_sample(x, y),
                         "abs_err": abs(float(x.mean()) - y), "cover80": bool(p10 <= y <= p90)})
        return pd.DataFrame(rows)

    def pace(self, c: float) -> pd.DataFrame:
        done = self.state(c)["done"]
        so_far = final_points(done)
        gp = pd.concat([done.home, done.away]).value_counts()
        pred = so_far / gp * self.n_games.reindex(so_far.index)
        return pd.DataFrame({"season": self.V, "checkpoint": c, "team": pred.index,
                             "abs_err": (pred - self.actual.reindex(pred.index)).abs().to_numpy()})

    def absence_adj(self, c: float, n_missed: int):
        done = self.state(c)["done"]
        d = LU.dressed()
        d = d[(d.season_end == self.V) & d.gid.isin(set(done.gid))]
        if not len(d):
            return None, 0
        tg = {}
        for side, col in (("h", "home"), ("a", "away")):
            for t, x in done.groupby(col):
                tg.setdefault(t, []).extend(x.gid.tolist())
        tg = {t: sorted(v) for t, v in tg.items()}
        off = LU.absence_offsets(d, tg, LU.player_values(self.V), LU.league_xg_pg(self.V), n_missed)
        return LU.apply_team_offsets(self.adj, self.sch, off), int(off.n_out.sum()) if len(off) else 0


def summarise(df: pd.DataFrame) -> dict:
    return {"n": int(len(df)), "crps": float(df.crps.mean()), "mae": float(df.abs_err.mean()),
            "cover80": float(df.cover80.mean())}


def main():
    t0 = time.time()
    seasons = {V: Season(V) for V in TUNE + TEST}
    ledger, res = [], {"protocol": __doc__}
    # (1) drift multiplier on the tuning seasons
    tune_rows = {k: [] for k in K}
    for V in TUNE:
        for c in CHECKPOINTS:
            for k in K:
                tune_rows[k].append(seasons[V].sim(c, k))
        print(f"tune {V} done ({time.time() - t0:.0f}s)", flush=True)
    by_k = {k: summarise(pd.concat(v)) for k, v in tune_rows.items()}
    k_best = min(K, key=lambda k: by_k[k]["crps"])
    ledger += [{"stage": "drift", "k": k, **by_k[k]} for k in K]
    # (2) absence offsets at k_best on the tuning seasons
    by_n = {}
    for n in NS:
        rows = []
        for V in TUNE:
            for c in CHECKPOINTS:
                adj, n_out = seasons[V].absence_adj(c, n)
                rows.append(seasons[V].sim(c, k_best, adj=adj))
        by_n[n] = summarise(pd.concat(rows))
        ledger.append({"stage": "absence", "N": n, "k": k_best, **by_n[n]})
    n_best = min(NS, key=lambda n: by_n[n]["crps"])
    print(f"tuned k={k_best} N={n_best} ({time.time() - t0:.0f}s)", flush=True)
    # test, once
    rows = {"filter_k1": [], "filter_kbest": [], "preseason": [], "absence": [], "pace": []}
    abs_counts = []
    for V in TEST:
        for c in CHECKPOINTS:
            s = seasons[V]
            rows["filter_k1"].append(s.sim(c, 1.0))
            rows["filter_kbest"].append(s.sim(c, k_best))
            rows["preseason"].append(s.sim(c, 1.0, which="prior"))
            adj, n_out = s.absence_adj(c, n_best)
            abs_counts.append(n_out)
            rows["absence"].append(s.sim(c, k_best, adj=adj))
            rows["pace"].append(s.pace(c))
    test = {k: pd.concat(v) for k, v in rows.items()}
    res["tuning"] = {"by_k": {str(k): v for k, v in by_k.items()}, "by_n": {str(n): v for n, v in by_n.items()},
                     "k_best": k_best, "n_best": n_best}
    res["test"] = {k: (summarise(v) if k != "pace" else {"n": int(len(v)), "mae": float(v.abs_err.mean())})
                   for k, v in test.items()}
    res["test"]["by_checkpoint"] = {
        str(c): {k: float(v[v.checkpoint == c].crps.mean()) for k, v in test.items() if k != "pace"}
        | {"pace_mae": float(test["pace"][test["pace"].checkpoint == c].abs_err.mean())} for c in CHECKPOINTS}
    res["test"]["absent_players_flagged"] = abs_counts
    a, b = test["absence"].crps.to_numpy(), test["filter_kbest"].crps.to_numpy()
    rng = np.random.default_rng(7)
    bs = [np.mean((a - b)[rng.integers(0, len(a), len(a))]) for _ in range(2000)]
    res["test"]["absence_minus_filter"] = {"diff": float(np.mean(a - b)),
                                           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["absence_adopted"] = bool(np.mean(a - b) < 0)
    res["ledger"] = ledger
    res["secs"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1))
    PARAMS.write_text(json.dumps({"drift_k": k_best, "absence": res["absence_adopted"], "absence_n": n_best,
                                  "source": "orr/backtest/standings_inseason_bt.py"}, indent=1))
    for k, v in res["test"].items():
        if isinstance(v, dict) and "n" in v:
            print(f"test {k:13s} {v}")
    print("absence - filter:", res["test"]["absence_minus_filter"], " adopted:", res["absence_adopted"])


if __name__ == "__main__":
    main()
