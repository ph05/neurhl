"""Score the frozen 2026-27 NeurHL predictions as the season is played.

Implements the scoring protocol in PLAN_NeurHL_LIVE.md. Every prediction it
scores was committed before opening night; this script only reads results.

  L1 games      per-game log loss of NeurHL's frozen home-win probability
                against the frozen preseason Elo reference, paired by game.
  L2 standings  team points against NeurHL's projection and HOWE's, plus
                80% interval coverage (NeurHL p10-p90). Scored at season end.
  L3 players    skater points (>= 40 GP) for the shipped blend and its two
                paths. Scored at season end.

Results come from the public NHL API (api-web.nhle.com, api.nhle.com) and are
cached as committed CSVs under output/live/, so every scorecard can be rebuilt
offline. Interim scorecards are descriptive; inference is made once, at the
end of the regular season (PLAN_NeurHL_LIVE.md, section 3).

Usage: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
         --with requests python neurhl/eval/score_live_2027.py [--offline]
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, PROJ, UA  # noqa: E402

LIVE = NOUT / "live"
FIRST, LAST = dt.date(2026, 9, 29), dt.date(2027, 4, 10)
SCORE_URL = "https://api-web.nhle.com/v1/score/{d}"
SKATER_URL = ("https://api.nhle.com/stats/rest/en/skater/summary?limit=-1"
              "&cayenneExp=seasonId=20262027%20and%20gameTypeId=2")


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fetch_results(today: dt.date) -> pd.DataFrame:
    """Completed regular-season games through `today`, cached day by day."""
    import requests
    path = LIVE / "results_2027.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(have.date) if len(have) else set()
    rows = [] if not len(have) else have.to_dict("records")
    d = FIRST
    while d <= min(today - dt.timedelta(days=1), LAST):
        ds = d.isoformat()
        if ds not in done:
            js = requests.get(SCORE_URL.format(d=ds), headers=UA, timeout=30).json()
            day = [g for g in js.get("games", [])
                   if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL")]
            for g in day:
                rows.append({"game_id": g["id"], "date": ds,
                             "home": g["homeTeam"]["abbrev"],
                             "away": g["awayTeam"]["abbrev"],
                             "home_g": g["homeTeam"]["score"],
                             "away_g": g["awayTeam"]["score"],
                             "last_period": g.get("gameOutcome", {}).get(
                                 "lastPeriodType", "REG")})
            time.sleep(0.5)
        d += dt.timedelta(days=1)
    out = pd.DataFrame(rows).drop_duplicates("game_id").sort_values(
        ["date", "game_id"]) if rows else pd.DataFrame(
        columns=["game_id", "date", "home", "away", "home_g", "away_g",
                 "last_period"])
    LIVE.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def score_games(res: pd.DataFrame) -> dict:
    g = pd.read_csv(NOUT / "games_2027.csv").merge(
        res[["game_id", "home_g", "away_g"]], on="game_id")
    if not len(g):
        return {"n_games": 0}
    y = (g.home_g > g.away_g).astype(float).to_numpy()
    lh, le = nll(g.p_home_win.to_numpy(), y), nll(g.p_home_win_elo.to_numpy(), y)
    d = lh - le
    out = {"n_games": int(len(g)), "neurhl": float(lh.mean()),
           "elo": float(le.mean()), "diff": float(d.mean()),
           "home_win_rate": float(y.mean())}
    if len(g) > 30:
        # moving-block bootstrap by week of play (declared, section 3)
        wk = pd.to_datetime(g.date).dt.isocalendar().week.astype(int).to_numpy()
        weeks = np.unique(wk)
        rng = np.random.default_rng(711)
        sums = pd.Series(d).groupby(wk).sum().reindex(weeks).to_numpy()
        cnts = pd.Series(d).groupby(wk).size().reindex(weeks).to_numpy()
        draws = rng.integers(0, len(weeks), size=(9999, len(weeks)))
        boot = sums[draws].sum(1) / cnts[draws].sum(1)
        out["ci95_weekly_bootstrap"] = [float(np.quantile(boot, 0.025)),
                                        float(np.quantile(boot, 0.975))]
    return out


def team_points(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in res.itertuples():
        extra = r.last_period in ("OT", "SO")
        hw = r.home_g > r.away_g
        rows += [(r.home, 2 if hw else (1 if extra else 0)),
                 (r.away, 0 if hw and not extra else (1 if hw else 2))]
    return pd.DataFrame(rows, columns=["team", "pts"]).groupby("team").pts.agg(
        ["sum", "size"]).rename(columns={"sum": "pts", "size": "gp"})


def score_standings(res: pd.DataFrame, final: bool) -> dict:
    tp = team_points(res)
    ne = pd.read_csv(NOUT / "projection_2027.csv").set_index("team")
    ho = pd.read_csv(PROJ / "output" / "projections_2026_27_howe.csv").set_index("Abbr")
    t = tp.join(ne[["proj_points", "p10", "p90"]]).join(ho[["xPts"]])
    if not final:
        # descriptive only: points pace restated to 84 games
        t["pace84"] = t.pts / t.gp * 84
        return {"final": False, "teams": int(len(t)),
                "pace_mae_neurhl": float((t.pace84 - t.proj_points).abs().mean()),
                "pace_mae_howe": float((t.pace84 - t.xPts).abs().mean())}
    return {"final": True, "teams": int(len(t)),
            "mae_neurhl": float((t.pts - t.proj_points).abs().mean()),
            "mae_howe": float((t.pts - t.xPts).abs().mean()),
            "coverage80_neurhl": float(((t.pts >= t.p10) & (t.pts <= t.p90)).mean())}


def score_players() -> dict:
    import requests
    js = requests.get(SKATER_URL, headers=UA, timeout=60).json()
    act = pd.DataFrame(js.get("data", []))
    if not len(act):
        return {"n": 0}
    act = act.rename(columns={"playerId": "player_id", "gamesPlayed": "gp",
                              "points": "pts"})[["player_id", "gp", "pts"]]
    act.to_csv(LIVE / "skaters_2027.csv", index=False)
    pp = pd.read_csv(NOUT / "player_proj_2027.csv").merge(act, on="player_id")
    pp = pp[pp.gp >= 40]
    return {"n": int(len(pp)),
            **{f"mae_{k}": float((pp[c] - pp.pts).abs().mean())
               for k, c in (("blend", "proj_p"), ("path_a", "proj_p_path_a"),
                            ("path_b", "proj_p_path_b"))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="score the cached results only; no network")
    a = ap.parse_args()
    today = dt.date.today()
    res = (pd.read_csv(LIVE / "results_2027.csv") if a.offline
           else fetch_results(today))
    final = len(res) >= 1344
    card = {"as_of": today.isoformat(), "games_played": int(len(res)),
            "final": final, "L1_games": score_games(res)}
    if len(res):
        card["L2_standings"] = score_standings(res, final)
    if final and not a.offline:
        card["L3_players"] = score_players()
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "scorecard_2027.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
