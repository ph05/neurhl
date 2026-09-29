"""NeurHL 1.0: the unified simulator's game assembly equals the live path.

sim/unified_2027.py builds every game from opening-night state rows
(Canonical) instead of calling sim/g_live.build per game. On opening night
the two must agree exactly: same lineups, same state date, same schedule
context. This test assembles the opening-night games both ways with the same
lineups, runs the engine on both, and requires the outputs to match.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow \
       --with numba --with torch --with scipy --with scikit-learn==1.9.1 --with requests \
       python neurhl/tests/test_unified_canonical.py [--rosters-date 2026-09-27]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CONFIGS  # noqa: E402
import sim.g_live as GL  # noqa: E402
import sim.unified_2027 as U  # noqa: E402
from sim.g_forecast_core import raw_outputs  # noqa: E402
from sim.project_2027 import load_schedule  # noqa: E402
from sim.schedule_context import build as sched_ctx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rosters-date", default="2026-09-27")
    a = ap.parse_args()
    bundle = json.loads((CONFIGS / "live_models.json").read_text())["neurhl_g"]
    games = load_schedule()[["game_id", "date", "home", "away"]].sort_values(["date", "game_id"]).reset_index(drop=True)
    games["date"] = games.date.astype(str)
    opening = games[games.date == games.date.min()].reset_index(drop=True)
    av, src = U.availability(games, a.rosters_date)
    draw = av.draws(games, 1, 711)[0]
    lineups = {g: draw[g] for g in opening.game_id}
    sc = sched_ctx(load_schedule(), U.SEASON).set_index("game_id")
    days0 = float(sc.days_in.min())
    elo_all = GL.elo_logits(games)
    elo = elo_all[:len(opening)]
    teams = sorted(set(opening.home) | set(opening.away))
    cand = {t: {"skaters": [], "goalies": []} for t in sorted(set(games.home))}
    for r in opening.itertuples():
        for team, key in ((r.home, "home"), (r.away, "away")):
            cand[team]["skaters"] = [int(p) for p in lineups[r.game_id][key]["skaters"]]
            cand[team]["goalies"] = [int(lineups[r.game_id][key]["goalie"])]
    # teams not playing on opening night still need a row for the canonical slates
    ex = av.expected(games)
    for t, c in cand.items():
        if not c["skaters"]:
            c["skaters"] = [int(p) for p in ex[t]["skaters"]]
            c["goalies"] = [int(ex[t]["goalie"])] if ex[t].get("goalie") else []
    canon = U.Canonical(games, cand)
    A_c, missing = canon.assemble(opening, lineups, sc, elo, days0)
    A_l, _, _ = GL.build(opening, lineups)
    o_c, _, _ = raw_outputs(bundle, A_c)
    o_l, _, _ = raw_outputs(bundle, A_l)
    res = {"availability": src, "games": len(opening), "missing_state_rows": int(missing)}
    ok = missing == 0
    for k in ("p_home_win", "goals", "sogf", "xgf", "pp_opps"):
        d = float(np.nanmax(np.abs(o_c[k] - o_l[k])))
        res[k] = d
        ok &= d < 1e-4
    for k in ("g", "a", "isog", "toi_ev"):
        # player order within a side can differ; compare sorted per side
        d = float(np.nanmax(np.abs(np.sort(np.nan_to_num(o_c[k]), -1) - np.sort(np.nan_to_num(o_l[k]), -1))))
        res[k] = d
        ok &= d < 1e-4
    for k in ("CTX",):
        d = float(np.nanmax(np.abs(A_c[k] - A_l[k])))
        res[f"input_{k}"] = d
        ok &= d < 1e-5
    # later games, including back-to-backs: the per-game schedule context the
    # unified path writes into the team rows must equal the live builder's
    b2b = sc[(sc.home_rest <= 1) | (sc.away_rest <= 1)].index
    later = games[games.game_id.isin(b2b)].head(6).reset_index(drop=True)
    lu_l = {g: draw[g] for g in later.game_id}
    for r in later.itertuples():
        for team, key in ((r.home, "home"), (r.away, "away")):
            for pid in lu_l[r.game_id][key]["skaters"]:
                if int(pid) not in cand[team]["skaters"]:
                    cand[team]["skaters"].append(int(pid))
            gk_ = int(lu_l[r.game_id][key]["goalie"])
            if gk_ not in cand[team]["goalies"]:
                cand[team]["goalies"].append(gk_)
    canon2 = U.Canonical(games, cand)
    idx = [int(np.where(games.game_id == g)[0][0]) for g in later.game_id]
    A_c2, _ = canon2.assemble(later, lu_l, sc, elo_all[idx], days0)
    tf = canon2.names["tm_feat"]
    cols = [tf.index(c) for c in ("rest", "b2b", "km3d", "dtz")]
    worst = 0.0
    for i, g in enumerate(later.game_id):
        A_one, _, _ = GL.build(later.iloc[[i]].reset_index(drop=True), {g: lu_l[g]})
        worst = max(worst, float(np.nanmax(np.abs(A_c2["TM"][i][:, cols] - A_one["TM"][0][:, cols]))))
    res["b2b_games_checked"] = len(later)
    res["schedule_context_max_diff"] = worst
    ok &= worst < 1e-6
    print(json.dumps(res, indent=1))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
