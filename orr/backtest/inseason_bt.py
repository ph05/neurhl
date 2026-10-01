"""Backtest of the SHIPPED in-season loop: the preseason pipeline's ratings
(market-anchored, as frozen for 2026-27) updated game by game by the filter.

Live, orr/inseason.py starts ratings.InSeasonFilter from the freeze's
ratings and league level. games_bt.py validated the filter started from the
team-history table instead (ratings.preseason_table). This script rebuilds,
for each clean market season (2019, 2020, 2022-24), the preseason ratings
exactly as gamefile_bt.py does (shipped views mkt+td, convex weights fitted
leave-one-season-out, ratings solved on the actual schedule) and runs the
walk-forward filter from them (ratings.run_filter with pre_override). Every
prediction uses only games before its date. OT/SO parameters are taken from
the team-history run so that the starting prior is the only difference.

Compared on the same games with: the team-history start (games_bt), NeurHL-G
(its gate games, the in-season stack) and NeurHL's Elo, with paired
bootstrap CIs. Without starters and with known starters (use_goalie).

Run: python3 -m orr.backtest.inseason_bt
Writes orr/output/backtest/inseason_bt.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from orr import config as C
from orr import ratings as R
from orr.backtest import gamefile_bt as GB
from orr.backtest.games_bt import PREDS, paired

OUT = C.OUT / "backtest"
VIEWS = ["mkt_rel82", "td_rel82"]


def main():
    hp = R.load_hp()
    hist = GB.hist_frame()
    over, logs = {}, {}
    for V in GB.SEASONS:
        P_V = R.fit_gamemodel_params(V, hp, write=False)
        _, lg, r = GB.game_file_season(V, hist, VIEWS, P_V, with_ratings=True)
        over[V] = (r, lg["mu"])
        logs[V] = {k: lg[k] for k in ("weights", "mu", "rating_sd_net_mean")}
    gate = pd.read_csv(PREDS / "g_gate_games.csv")
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocol": __doc__, "seasons": GB.SEASONS, "season_logs": logs, "modes": {}}
    for tag, gk in (("no_starters", False), ("known_starters", True)):
        base = R.run_filter(hp, GB.SEASONS, use_goalie=gk)
        ot = {}
        pb = R.predict_probs(base, hp, ot_params=ot)
        ship = R.run_filter(hp, GB.SEASONS, use_goalie=gk, pre_override=over)
        ps = R.predict_probs(ship, hp, ot_params=dict(ot))
        g = R.S.game_frame()[["gid", "game_id", "season_end", "home_win"]]
        m = (g.merge(pb[["gid", "p_home_win"]].rename(columns={"p_home_win": "p_teamhist"}), on="gid")
              .merge(ps[["gid", "p_home_win"]].rename(columns={"p_home_win": "p_shipped"}), on="gid")
              .merge(gate[["game_id", "p_stack", "p_elo"]], on="game_id"))
        m = m[m.season_end.isin(GB.SEASONS)]
        rows = []
        for V, x in list(m.groupby("season_end")) + [("all", m)]:
            y = x.home_win.to_numpy()
            rows.append({"season": V if V == "all" else int(V), "n": int(len(x)),
                         "shipped": R.logloss(x.p_shipped, y), "teamhist": R.logloss(x.p_teamhist, y),
                         "neurhl_g": R.logloss(x.p_stack, y), "neurhl_elo": R.logloss(x.p_elo, y),
                         "d_shipped_vs_teamhist": paired(x.p_shipped, x.p_teamhist, y),
                         "d_shipped_vs_neurhl_g": paired(x.p_shipped, x.p_stack, y),
                         "d_shipped_vs_neurhl_elo": paired(x.p_shipped, x.p_elo, y)})
        res["modes"][tag] = rows
        print(f"\n[{tag}]")
        for r_ in rows:
            d1, d2 = r_["d_shipped_vs_teamhist"], r_["d_shipped_vs_neurhl_g"]
            print(f"{str(r_['season']):>5} n={r_['n']:>5} shipped {r_['shipped']:.4f} teamhist {r_['teamhist']:.4f}"
                  f" NeurHL-G {r_['neurhl_g']:.4f} Elo {r_['neurhl_elo']:.4f}"
                  f"  ship-th {d1['diff']:+.4f} [{d1['ci95'][0]:+.4f},{d1['ci95'][1]:+.4f}]"
                  f"  ship-G {d2['diff']:+.4f} [{d2['ci95'][0]:+.4f},{d2['ci95'][1]:+.4f}]")
    (OUT / "inseason_bt.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
