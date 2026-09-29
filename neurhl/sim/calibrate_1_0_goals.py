"""NeurHL 1.1 C1: the goal-slope correction applied to the frozen NeurHL 1.0
files, issued as a new dated file set beside them (PLAN_NeurHL_1_1).

The frozen files are read, never written. Each team-game's goal mean g (the
A1-scaled engine mean averaged over the lineup draws) becomes

  g' = k * m0 * M * (g / m0 / M) ** b_hat

that is, a factor f = k * (g / m0 / M) ** (b_hat - 1) on that team-game, with k
one league-wide constant that keeps the league's goals equal to the frozen
total (the correction changes the spread, not A1's level). The same f
scales every goal-derived quantity of that team-game: the skaters' goals and
assists (so points), power-play goals, and the opposing goalie's goals
against. Win probabilities, standings, shots, ice time and every other column
are unchanged, and every sum rule of the consistency check still holds,
because each team-game is scaled as a unit. The skaters' 10th/90th percentile
goal and point columns are scaled by the player's season ratio (approximate).

b_hat was fitted on game-day predictions (configs/calibration_1_1.json); these
are preseason-convention predictions, so the correction is an approximation,
recorded as such in the plan.

Writes neurhl/output/neurhl_1_0/cal_20260929/{games,teams,skaters,goalies,
player_games}_2027.csv(.gz) and checks.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output" / "neurhl_1_0"
OUT = SRC / "cal_20260929"


def main():
    cal = json.loads((ROOT / "configs" / "calibration_1_1.json").read_text())
    gc = json.loads((ROOT / "configs" / "live_goal_calibration.json").read_text())
    run = json.loads((SRC / "run_2027.json").read_text())
    b, M, m0 = cal["goal_slope"]["b_hat"], gc["M"], run["goal_mult_m0"]
    assert abs(m0 - gc["m0"]) < 1e-12, "m0 of the run and of the goal calibration differ"
    g = pd.read_csv(SRC / "games_2027.csv")
    t = pd.read_csv(SRC / "teams_2027.csv")
    sk = pd.read_csv(SRC / "skaters_2027.csv")
    gl = pd.read_csv(SRC / "goalies_2027.csv")
    pg = pd.read_csv(SRC / "player_games_2027.csv.gz")

    f = {0: (g.goals_home / m0 / M) ** (b - 1), 1: (g.goals_away / m0 / M) ** (b - 1)}
    # slope only: renormalise so league goals are unchanged (the frozen level is A1's)
    tot = g.goals_home.sum() + g.goals_away.sum()
    k = tot / ((g.goals_home * f[0]).sum() + (g.goals_away * f[1]).sum())
    f = {s_: v * k for s_, v in f.items()}
    fk = pd.concat([pd.DataFrame({"game_id": g.game_id, "side": s, "f": f[s]}) for s in (0, 1)])
    g2 = g.copy()
    for s, side in ((0, "home"), (1, "away")):
        g2[f"goals_{side}"] = g[f"goals_{side}"] * f[s]
        g2[f"pp_goals_{side}"] = g[f"pp_goals_{side}"] * f[s]
    g2["goal_factor_home"], g2["goal_factor_away"] = f[0], f[1]

    pg2 = pg.merge(fk, on=["game_id", "side"], how="left")
    assert pg2.f.notna().all()
    pg2["g"], pg2["a"] = pg2.g * pg2.f, pg2.a * pg2.f
    pg2 = pg2.drop(columns="f")

    # teams: goals for/against and PP goals from the scaled games
    long = pd.concat([pd.DataFrame({"team": g2.home, "gf": g2.goals_home, "ga": g2.goals_away,
                                    "ppgf": g2.pp_goals_home, "ppga": g2.pp_goals_away}),
                      pd.DataFrame({"team": g2.away, "gf": g2.goals_away, "ga": g2.goals_home,
                                    "ppgf": g2.pp_goals_away, "ppga": g2.pp_goals_home})])
    agg = long.groupby("team").sum()
    t2 = t.set_index("team").copy()
    t2["goals_for"], t2["goals_against"] = agg.gf, agg.ga
    t2["pp_goals_for"], t2["pp_goals_against"] = agg.ppgf, agg.ppga
    t2["pp_pct"] = 100 * t2.pp_goals_for / t2.pp_opps_for
    t2["pk_pct"] = 100 * (1 - t2.pp_goals_against / t2.pp_opps_against)
    t2["shooting_pct"] = 100 * t2.goals_for / t2.sog_for
    t2["save_pct"] = 100 * (1 - t2.goals_against / t2.sog_against)
    t2 = t2.reset_index()

    # skaters: sums of the scaled player-games
    s = pg2.groupby("player_id")[["g", "a"]].sum()
    sk2 = sk.set_index("player_id").copy()
    rg = (s.g / sk2.goals).replace([np.inf, -np.inf], 1).fillna(1)
    rp = ((s.g + s.a) / sk2.points).replace([np.inf, -np.inf], 1).fillna(1)
    ra = (s.a / sk2.assists).replace([np.inf, -np.inf], 1).fillna(1)
    sk2["goals"], sk2["assists"] = s.g, s.a
    sk2["points"] = sk2.goals + sk2.assists
    for c in ("g_p10", "g_p90"):
        sk2[c] = sk2[c] * rg
    for c in ("a_p10", "a_p90"):
        sk2[c] = sk2[c] * ra
    for c in ("points_p10", "points_p90"):
        sk2[c] = sk2[c] * rp
    sk2["points_per_gp"] = sk2.points / sk2.gp.where(sk2.gp > 0)
    sk2["shooting_pct"] = 100 * sk2.goals / sk2.sog.where(sk2.sog > 0)
    sk2 = sk2.reset_index()

    # goalies: goals against scaled by the team's goals-against ratio
    ratio = agg.ga / t.set_index("team").goals_against
    gl2 = gl.copy()
    gl2["ga"] = gl.ga * gl.team.map(ratio)
    gl2["saves"] = gl2.sa - gl2.ga
    gl2["sv_pct"] = gl2.saves / gl2.sa.where(gl2.sa > 0)
    gl2["gaa"] = gl2.ga / gl2.starts.where(gl2.starts > 0)

    # the consistency check's goal rules, re-derived on the calibrated set
    tg = pg2.groupby(["game_id", "side"]).g.sum().unstack()
    gi = g2.set_index("game_id")
    chk = {
        "skater goals = game goals, per team-game": float(max((tg[0] - gi.goals_home).abs().max(),
                                                              (tg[1] - gi.goals_away).abs().max())),
        "team goals_for = sum of skater goals": float((sk2.groupby("team").goals.sum()
                                                       - t2.set_index("team").goals_for).abs().max()),
        "goalie goals against = team goals against": float((gl2.groupby("team").ga.sum()
                                                            - t2.set_index("team").goals_against).abs().max()),
        "league goals for = goals against": float(abs(t2.goals_for.sum() - t2.goals_against.sum())),
    }
    # tolerances of tests/check_neurhl_1_0.py: the frozen files carry 4 decimals
    tol = {"skater goals = game goals, per team-game": 1e-3, "team goals_for = sum of skater goals": 0.02,
           "goalie goals against = team goals against": 0.02, "league goals for = goals against": 0.05}
    ok = all(chk[c] < tol[c] for c in chk)
    summary = {"b_hat": b, "M": M, "m0": m0, "k": float(k), "pass": ok, "max_abs_diff": chk,
               "league_goals_per_team_game": {"frozen": float(g[["goals_home", "goals_away"]].to_numpy().mean()),
                                              "calibrated": float(g2[["goals_home", "goals_away"]].to_numpy().mean())},
               "team_goals_for_sd_per_game": {"frozen": float((t.goals_for / t.gp).std()),
                                              "calibrated": float((t2.goals_for / t2.gp).std())}}
    OUT.mkdir(parents=True, exist_ok=True)
    g2.to_csv(OUT / "games_2027.csv", index=False, float_format="%.4f")
    t2.to_csv(OUT / "teams_2027.csv", index=False, float_format="%.4f")
    sk2.to_csv(OUT / "skaters_2027.csv", index=False, float_format="%.4f")
    gl2.to_csv(OUT / "goalies_2027.csv", index=False, float_format="%.4f")
    pg2.to_csv(OUT / "player_games_2027.csv.gz", index=False, float_format="%.5f",
               compression={"method": "gzip", "mtime": 0})
    (OUT / "checks.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
