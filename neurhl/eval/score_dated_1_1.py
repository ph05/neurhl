"""Score the NeurHL 1.1 dated file sets beside the frozen NeurHL 1.0 files
(PLAN_NeurHL_1_1), with the frozen files' own measures and the same results:

  cal_20260929            goal-slope-corrected goal totals: team regulation
                          goals for and against (MAE per team, final; pace
                          interim), and skater goals and points MAE (>= 40 games)
  skaters_blend_20260929  blended skater points: points MAE (>= 40 games)
  season_<tag>            (only if C2 was adopted) points MAE, CRPS from
                          percentiles 1-99, and 10th-90th coverage

Interim scorecards are descriptive; the comparison is made once, after the
last regular-season game. Writes neurhl/output/live/scorecard_dated_1_1_2027.json.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT  # noqa: E402
from eval.score_neurhl_1_0 import crps_from_quantiles, team_points  # noqa: E402

LIVE = NOUT / "live"
UNI = NOUT / "neurhl_1_0"
N_GAMES = 1344


def reg_goals(res):
    rows = []
    for r in res.itertuples():
        extra = r.last_period in ("OT", "SO")
        hw = r.home_g > r.away_g
        rh = r.home_g - (1 if extra and hw else 0)
        ra = r.away_g - (1 if extra and not hw else 0)
        rows += [(r.home, rh, ra), (r.away, ra, rh)]
    return pd.DataFrame(rows, columns=["team", "gf", "ga"]).groupby("team").agg(
        gf=("gf", "sum"), ga=("ga", "sum"), gp=("gf", "size"))


def team_goals(res, final):
    act = reg_goals(res)
    out = {}
    for name, path in (("frozen", UNI / "teams_2027.csv"), ("cal", UNI / "cal_20260929" / "teams_2027.csv")):
        t = pd.read_csv(path).set_index("team")[["goals_for", "goals_against"]].join(act, how="inner")
        if final:
            pred_f, pred_a, af, aa = t.goals_for, t.goals_against, t.gf, t.ga
        else:                                   # pace: per game, projected over 84 against actual so far
            pred_f, pred_a = t.goals_for / 84, t.goals_against / 84
            af, aa = t.gf / t.gp, t.ga / t.gp
        out[name] = {"mae_goals_for": float((pred_f - af).abs().mean()),
                     "mae_goals_against": float((pred_a - aa).abs().mean())}
    out["unit"] = "goals per team-season" if final else "regulation goals per game (pace)"
    return out


def skaters():
    act_p = LIVE / "skaters_2027.csv"
    if not act_p.exists():
        return {"n": 0}
    act = pd.read_csv(act_p)
    f = pd.read_csv(UNI / "skaters_2027.csv")[["player_id", "goals", "points"]]
    c = pd.read_csv(UNI / "cal_20260929" / "skaters_2027.csv")[["player_id", "goals", "points"]]
    b = pd.read_csv(UNI / "skaters_blend_20260929" / "skaters_2027.csv")[["player_id", "points"]]
    m = act.merge(f, on="player_id").merge(c, on="player_id", suffixes=("", "_cal")).merge(
        b.rename(columns={"points": "points_blend"}), on="player_id")
    m = m[m.gp >= 40]
    out = {"n": int(len(m)),
           "points_mae": {"frozen": float((m.points - m.pts).abs().mean()),
                          "cal": float((m.points_cal - m.pts).abs().mean()),
                          "blend": float((m.points_blend - m.pts).abs().mean())}}
    if "g" in m:
        out["goals_mae"] = {"frozen": float((m.goals - m.g).abs().mean()),
                            "cal": float((m.goals_cal - m.g).abs().mean())}
    return out


def seasons(res, final):
    out = {}
    tp = team_points(res)
    for d in sorted(UNI.glob("season_*")):
        t = tp.join(pd.read_csv(d / "teams_2027.csv").set_index("team")[["points", "points_p10", "points_p90"]])
        if not final:
            out[d.name] = {"pace_mae": float((t.pts / t.gp * 84 - t.points).abs().mean())}
            continue
        q = pd.read_csv(d / "team_points_quantiles_2027.csv").set_index("team")
        out[d.name] = {"mae": float((t.pts - t.points).abs().mean()),
                       "crps": float(np.mean([crps_from_quantiles(q.loc[k].to_numpy(), v) for k, v in t.pts.items()])),
                       "coverage80": float(((t.pts >= t.points_p10) & (t.pts <= t.points_p90)).mean())}
    return out


def main():
    res_p = LIVE / "results_2027.csv"
    res = pd.read_csv(res_p) if res_p.exists() else pd.DataFrame()
    if len(res):
        res = res[(res.game_id // 1_000_000 == 2026) & (res.game_id // 10_000 % 100 == 2)].drop_duplicates("game_id")
    final = len(res) >= N_GAMES
    card = {"as_of": dt.date.today().isoformat(), "games_played": int(len(res)), "final": final,
            "note": "interim: descriptive only" if not final else "final: PLAN_NeurHL_1_1 dated sets"}
    if len(res):
        card["team_goals"] = team_goals(res, final)
        card["seasons"] = seasons(res, final)
    if final:
        card["skaters"] = skaters()
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "scorecard_dated_1_1_2027.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
