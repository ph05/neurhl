"""Score the frozen NeurHL 1.0 predictions for 2026-27 as the season is played
(PLAN_NeurHL_1_0.md, Scoring).

Reads the results cached by eval/score_live_2027.py (neurhl/output/live/
results_2027.csv; skaters_2027.csv once the season is over) and the frozen
files in neurhl/output/neurhl_1_0/:

  games      per-game log loss of NeurHL 1.0's home-win probability, paired
             on the same games against the preseason Elo reference and against
             the 2026-09-25 freeze, each with a week-block bootstrap interval
             (9,999 draws, seed 711)
  standings  interim: points pace (restated to 84 games) against the
             projection; final: MAE, CRPS from the frozen percentiles, and
             coverage of the 10th-90th percentile interval
  players    final: skater points MAE (>= 40 games) against the 2026-09-25 blend
Interim scorecards are descriptive; inference is made once, after the last
regular-season game. Writes neurhl/output/live/scorecard_1_0_2027.json.

Usage: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
         python neurhl/eval/score_neurhl_1_0.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT  # noqa: E402

LIVE = NOUT / "live"
UNI = NOUT / "neurhl_1_0"
N_GAMES = 1344


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def week_boot(d, dates):
    wk = pd.to_datetime(dates).dt.isocalendar()
    wk = (wk.year * 100 + wk.week).to_numpy()
    weeks = np.unique(wk)
    rng = np.random.default_rng(711)
    sums = pd.Series(d).groupby(wk).sum().reindex(weeks).to_numpy()
    cnts = pd.Series(d).groupby(wk).size().reindex(weeks).to_numpy()
    draws = rng.integers(0, len(weeks), size=(9999, len(weeks)))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def score_games(res):
    g = pd.read_csv(UNI / "games_2027.csv").merge(res[["game_id", "home_g", "away_g"]], on="game_id")
    if not len(g):
        return {"n_games": 0}
    y = (g.home_g > g.away_g).astype(float).to_numpy()
    l1, le, l0 = nll(g.p_home_win, y), nll(g.p_home_win_elo, y), nll(g.p_home_win_0925, y)
    out = {"n_games": int(len(g)), "neurhl_1_0": float(l1.mean()), "elo": float(le.mean()),
           "freeze_0925": float(l0.mean()), "diff_vs_elo": float((l1 - le).mean()),
           "diff_vs_0925": float((l1 - l0).mean()), "home_win_rate": float(y.mean())}
    if len(g) > 30 and pd.to_datetime(g.date).dt.isocalendar().week.nunique() >= 2:
        out["ci95_vs_elo"] = week_boot((l1 - le).to_numpy(), g.date)
        out["ci95_vs_0925"] = week_boot((l1 - l0).to_numpy(), g.date)
    return out


def team_points(res):
    rows = []
    for r in res.itertuples():
        extra = r.last_period in ("OT", "SO")
        hw = r.home_g > r.away_g
        rows += [(r.home, 2 if hw else (1 if extra else 0)),
                 (r.away, 0 if hw and not extra else (1 if hw else 2))]
    return pd.DataFrame(rows, columns=["team", "pts"]).groupby("team").pts.agg(
        ["sum", "size"]).rename(columns={"sum": "pts", "size": "gp"})


def crps_from_quantiles(q, x):
    """CRPS from percentiles 1-99: twice the mean pinball loss over the levels."""
    lv = np.arange(1, 100) / 100.0
    return float(2 * np.mean(np.where(x < q, (1 - lv) * (q - x), lv * (x - q))))


def score_standings(res, final):
    tp = team_points(res)
    t = tp.join(pd.read_csv(UNI / "teams_2027.csv").set_index("team")[["points", "points_p10", "points_p90"]])
    if not final:
        t["pace84"] = t.pts / t.gp * 84
        return {"final": False, "teams": int(len(t)),
                "pace_mae_neurhl_1_0": float((t.pace84 - t.points).abs().mean())}
    q = pd.read_csv(UNI / "team_points_quantiles_2027.csv").set_index("team")
    return {"final": True, "teams": int(len(t)),
            "mae_neurhl_1_0": float((t.pts - t.points).abs().mean()),
            "crps_neurhl_1_0": float(np.mean([crps_from_quantiles(q.loc[k].to_numpy(), v)
                                              for k, v in t.pts.items()])),
            "coverage80_neurhl_1_0": float(((t.pts >= t.points_p10) & (t.pts <= t.points_p90)).mean())}


def score_players():
    act_p = LIVE / "skaters_2027.csv"
    if not act_p.exists():
        return {"n": 0}
    act = pd.read_csv(act_p)
    sk = pd.read_csv(UNI / "skaters_2027.csv")[["player_id", "points"]].merge(act, on="player_id")
    old = pd.read_csv(NOUT / "player_proj_2027.csv")[["player_id", "proj_p"]]
    sk = sk.merge(old, on="player_id", how="left")
    sk = sk[sk.gp >= 40]
    return {"n": int(len(sk)), "mae_neurhl_1_0": float((sk.points - sk.pts).abs().mean()),
            "mae_freeze_0925": float((sk.proj_p - sk.pts).abs().mean())}


def main():
    res_p = LIVE / "results_2027.csv"
    res = pd.read_csv(res_p) if res_p.exists() else pd.DataFrame(
        columns=["game_id", "home", "away", "home_g", "away_g", "last_period"])
    final = len(res) >= N_GAMES
    card = {"as_of": dt.date.today().isoformat(), "games_played": int(len(res)), "final": final,
            "games": score_games(res) if len(res) else {"n_games": 0}}
    if len(res):
        card["standings"] = score_standings(res, final)
    if final:
        card["players"] = score_players()
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "scorecard_1_0_2027.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
