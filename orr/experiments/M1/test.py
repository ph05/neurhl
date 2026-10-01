"""M1 single test run: the tuned goals-only filter on NeurHL-G's gate games 2019-24.

Mirrors the ``goals_only_live`` mode of orr/backtest/inseason_bt.py exactly:
for each clean market season (2019, 2020, 2022-24) the preseason pipeline's
ratings are rebuilt with the FROZEN HP (as gamefile_bt / inseason_bt do; the
published preseason pipeline is not part of M1) and passed to
ratings.run_filter as pre_override; OT/SO parameters come from the same arm's
team-history run. No starters, use_shots=False.

Arms: 'current' = frozen HP with use_shots=False (the live goals-only loop),
'candidate' = orr/experiments/M1/config.json. NeurHL-G (p_stack) and NeurHL
Elo (p_elo) from neurhl/output/preds/g_gate_games.csv. Paired bootstrap CIs
from orr.backtest.games_bt.paired.

Runs once: refuses to run when test_output.json exists.
Run: python3 -m orr.experiments.M1.test
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from orr import ratings as R
from orr.backtest import gamefile_bt as GB
from orr.backtest.games_bt import PREDS, paired

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
OUT = HERE / "test_output.json"
PRED_OUT = HERE / "test_preds.csv.gz"
VIEWS = ["mkt_rel82", "td_rel82"]
MIN_GAIN = 0.0010


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the test has already been run once")
    t0 = time.time()
    cfg = json.loads(CONFIG.read_text())
    c = dict(cfg["hp"]); c["goalie"] = tuple(c["goalie"])
    cand = R.HP(**c)
    frozen = R.load_hp()
    cur = replace(frozen, use_shots=False)
    assert not cand.use_shots and not cur.use_shots
    # shipped preseason pipeline, exactly as inseason_bt.main (frozen HP)
    hist = GB.hist_frame()
    over = {}
    for V in GB.SEASONS:
        P_V = R.fit_gamemodel_params(V, frozen, write=False)
        _, lg, r = GB.game_file_season(V, hist, VIEWS, P_V, with_ratings=True)
        over[V] = (r, lg["mu"])
    print(f"preseason pipeline rebuilt ({time.time() - t0:.0f}s)", flush=True)
    gate = pd.read_csv(PREDS / "g_gate_games.csv")
    g = R.S.game_frame()[["gid", "game_id", "season_end", "home_win"]]
    m = g.merge(gate[["game_id", "p_stack", "p_elo"]], on="game_id")
    for tag, hp in (("current", cur), ("candidate", cand)):
        base = R.run_filter(hp, GB.SEASONS)
        ot = {}
        pb = R.predict_probs(base, hp, ot_params=ot)
        ship = R.run_filter(hp, GB.SEASONS, pre_override=over)
        ps = R.predict_probs(ship, hp, ot_params=dict(ot))
        m = (m.merge(pb[["gid", "p_home_win"]].rename(columns={"p_home_win": f"th_{tag}"}), on="gid")
              .merge(ps[["gid", "p_home_win"]].rename(columns={"p_home_win": f"p_{tag}"}), on="gid"))
        print(f"{tag} done ({time.time() - t0:.0f}s)", flush=True)
    m = m[m.season_end.isin(GB.SEASONS)].reset_index(drop=True)
    m.to_csv(PRED_OUT, index=False)
    rows = []
    for V, x in list(m.groupby("season_end")) + [("all", m)]:
        y = x.home_win.to_numpy()
        rows.append({
            "season": V if V == "all" else int(V), "n": int(len(x)),
            "candidate": R.logloss(x.p_candidate, y), "current": R.logloss(x.p_current, y),
            "neurhl_g": R.logloss(x.p_stack, y), "neurhl_elo": R.logloss(x.p_elo, y),
            "teamhist_candidate": R.logloss(x.th_candidate, y),
            "teamhist_current": R.logloss(x.th_current, y),
            "d_candidate_vs_current": paired(x.p_candidate, x.p_current, y),
            "d_candidate_vs_neurhl_g": paired(x.p_candidate, x.p_stack, y),
            "d_current_vs_neurhl_g": paired(x.p_current, x.p_stack, y),
            "d_candidate_vs_neurhl_elo": paired(x.p_candidate, x.p_elo, y),
            "d_teamhist_candidate_vs_current": paired(x.th_candidate, x.th_current, y)})
    pooled = rows[-1]
    d = pooled["d_candidate_vs_current"]
    gap = pooled["current"] - pooled["neurhl_g"]
    accepted = bool(d["ci95"][1] < 0 and -d["diff"] >= MIN_GAIN)
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocol": __doc__, "config": cfg, "seasons": GB.SEASONS, "rows": rows,
           "gain": -d["diff"], "gap_to_neurhl_g": gap,
           "gap_closed_fraction": (-d["diff"] / gap) if gap > 0 else None,
           "accept_rule": "pooled gate log loss improves on the current goals-only filter with a "
                          "95% CI that excludes 0 and the gain is at least 0.0010",
           "accepted": accepted, "secs": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    for r_ in rows:
        a, b = r_["d_candidate_vs_current"], r_["d_candidate_vs_neurhl_g"]
        print(f"{str(r_['season']):>5} n={r_['n']:>5} cand {r_['candidate']:.5f} cur {r_['current']:.5f} "
              f"G {r_['neurhl_g']:.5f} Elo {r_['neurhl_elo']:.5f}  "
              f"cand-cur {a['diff']:+.5f} [{a['ci95'][0]:+.5f},{a['ci95'][1]:+.5f}]  "
              f"cand-G {b['diff']:+.5f} [{b['ci95'][0]:+.5f},{b['ci95'][1]:+.5f}]")
    print(f"gain {res['gain']:+.5f}, gap closed {res['gap_closed_fraction']}, accepted {accepted}")


if __name__ == "__main__":
    main()
