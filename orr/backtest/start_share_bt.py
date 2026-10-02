"""In-season goalie start shares (ORR 1.4).

Preseason start shares come from goalies.deploy_goalies on the opening-roster
proxy (as goalies_bt.starts_check). At checkpoints after 25%, 50% and 75% of
a season (by date), each team's shares are updated with its box-score starts
so far, as a Dirichlet posterior mean:

    share = (alpha * preseason_share + starts_so_far) / (alpha + team_games_so_far)

and scored against each goalie's share of the team's REMAINING starts:
error = |share * games_left - starts_left| per team-goalie (any goalie with a
preseason share or a start). alpha is chosen on 2012 and 2014-2017 from GRID;
the test (2022, 2023) runs once. Baseline: the preseason share (alpha = inf).

Run: python3 -m orr.backtest.start_share_bt   -> orr/output/backtest/start_share_bt.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from orr import config as C
from orr import data as D
from orr import goalies as GL

TUNE, TEST = [2012, 2014, 2015, 2016, 2017], [2022, 2023]
CHECKPOINTS = (0.25, 0.5, 0.75)
GRID = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
OUT = C.OUT / "backtest" / "start_share_bt.json"
PARAMS = C.PARAMS / "start_share.json"


def update_shares(pre: pd.DataFrame, starts: pd.DataFrame, games: pd.Series, alpha: float) -> pd.DataFrame:
    """pre: team, player_id, share; starts: team, player_id, n (starts so far);
    games: team -> games so far. Returns team, player_id, share (sums to 1 per team)."""
    m = pre.merge(starts, on=["team", "player_id"], how="outer").fillna({"share": 0.0, "n": 0.0})
    g = m.team.map(games).fillna(0.0)
    if np.isfinite(alpha):
        m["share"] = (alpha * m.share + m.n) / (alpha + g)
    m["share"] = m.share / m.groupby("team").share.transform("sum").replace(0, 1)
    return m[["team", "player_id", "share"]]


def season_rows(V: int) -> list[pd.DataFrame]:
    r = D.opening_rosters(V)
    r = r[r.grp == "G"][["player_id", "team"]]
    dep = GL.deploy_goalies(V, r, 82)
    pre = dep[["team", "player_id", "start_share"]].rename(columns={"start_share": "share"})
    gg = D.goalie_games()
    gg = gg[gg.season_end == V]
    sch = D.fr_schedule()[["game_id", "date", "home", "away"]]
    gg = gg.merge(sch, on="game_id")
    gg["team"] = np.where(gg.home_away.str.lower() == "home", gg.home, gg.away)
    dates = np.sort(gg.date.unique())
    out = []
    for c in CHECKPOINTS:
        cut = dates[int(len(dates) * c)]
        b, a = gg[gg.date < cut], gg[gg.date >= cut]
        so_far = b.groupby(["team", "goalie_id"]).size().rename("n").reset_index().rename(columns={"goalie_id": "player_id"})
        after = a.groupby(["team", "goalie_id"]).size().rename("n_after").reset_index().rename(columns={"goalie_id": "player_id"})
        out.append({"pre": pre, "so_far": so_far, "after": after,
                    "g_before": b.groupby("team").size(), "g_after": a.groupby("team").size(), "c": c, "V": V})
    return out


def score(rows: list, alpha: float) -> np.ndarray:
    errs = []
    for x in rows:
        sh = update_shares(x["pre"], x["so_far"], x["g_before"], alpha)
        m = sh.merge(x["after"], on=["team", "player_id"], how="outer").fillna({"share": 0.0, "n_after": 0.0})
        m["pred"] = m.share * m.team.map(x["g_after"]).fillna(0.0)
        errs.append(np.abs(m.pred - m.n_after).to_numpy())
    return np.concatenate(errs)


def main():
    data = {V: season_rows(V) for V in TUNE + TEST}
    tune = [x for V in TUNE for x in data[V]]
    test = [x for V in TEST for x in data[V]]
    ledger = [{"alpha": str(a), "tune_mae": float(score(tune, a).mean())} for a in GRID]
    best = min(GRID, key=lambda a: score(tune, a).mean())
    e1, e0 = score(test, best), score(test, np.inf)
    rng = np.random.default_rng(7)
    d = e1 - e0
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    res = {"protocol": __doc__, "ledger": ledger, "alpha": best,
           "test": {"n": int(len(e1)), "mae_updated": float(e1.mean()), "mae_preseason": float(e0.mean()),
                    "diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}}
    OUT.write_text(json.dumps(res, indent=1))
    PARAMS.write_text(json.dumps({"alpha": best, "source": "orr/backtest/start_share_bt.py"}, indent=1))
    print(ledger)
    print("alpha", best, "test", res["test"])


if __name__ == "__main__":
    main()
