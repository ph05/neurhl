"""In-season predictions: update on results, forecast a day's games, re-run the season.

    python3 -m hattrick.inseason --date 2026-09-30 \
        [--results neurhl/output/live/results_2027.csv] \
        [--goalies goalies.csv] [--sims 20000]

Inputs
- The preseason freeze (hattrick/output/freeze_2027/): ratings, scoring-model
  parameters, schedule, goalie talent table.
- Results for games BEFORE --date: game_id, date, home, away, home_g, away_g,
  last_period (REG/OT/SO), optional shots_home/shots_away, optional
  goalie_home/goalie_away (starting goalie NHL ids).
- Optional confirmed or projected starters for --date: game_id, goalie_home,
  goalie_away (NHL ids). Unknown starters fall back to each team's start-share
  mix, which is exactly what the preseason file assumed.

What happens
1. Team ratings are filtered game by game from the preseason prior
   (hattrick.ratings.InSeasonFilter), using goals and, when present, shots;
   a known starting goalie's talent is removed from the team's defence so the
   filter learns about the skaters, not about who happened to be in net.
2. Every game on --date is forecast with the current ratings, rest/travel
   context and starter information.
3. The rest of the season is simulated from the current standings with the
   current ratings and their uncertainty.

Outputs (hattrick/output/live/<date>/): games_<date>.csv, standings_<date>.csv,
ratings_<date>.csv, and run_<date>.json with the code commit, input hashes and
the UTC creation time. A forecast only counts if it is committed and pushed
before puck drop; the GitHub push time, not the file's own timestamp, is the
evidence (see hattrick/score.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import season as S

FREEZE = C.OUT / "freeze_2027"
LIVE = C.OUT / "live"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _code_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_freeze() -> dict:
    st = json.loads((FREEZE / "state_2027.json").read_text())
    st["ratings"] = pd.read_csv(FREEZE / "ratings_2027.csv")
    st["schedule"] = pd.read_csv(FREEZE / "schedule_2027.csv", parse_dates=["date"])
    gp = FREEZE / "goalie_rates_2027.csv"
    st["goalies"] = pd.read_csv(gp) if gp.exists() else None
    return st


def load_results(path, before: pd.Timestamp) -> pd.DataFrame:
    r = pd.read_csv(path, parse_dates=["date"])
    r = r[r.date < before].copy()
    r["extra"] = r.last_period.fillna("REG").map(lambda x: {"OT": "OT", "SO": "SO"}.get(x, "REG"))
    return r.sort_values(["date", "game_id"]).reset_index(drop=True)


def goalie_offsets(freeze: dict, games: pd.DataFrame, starters: pd.DataFrame | None, P: dict):
    """Log-rate offsets on goals AGAINST for known starters, relative to the
    team's expected start mix (the preseason ratings already contain the mix).
    Save talent is goals saved above expected per shot on goal: the goalie
    layer's per-unblocked-attempt figure x 1.40 attempts per shot on goal."""
    out = pd.DataFrame({"game_id": games.game_id.to_numpy(), "gadj_h": 0.0, "gadj_a": 0.0})
    g = freeze.get("goalies")
    if starters is None or g is None or not len(starters):
        return out
    from hattrick import gamemodel as GM
    g = g.assign(talent=g.gsax_per_fa * 1.40)
    talent = g.set_index("player_id")
    mix = (g.assign(w=g.start_share * g.talent).groupby("team").w.sum()
           / g.groupby("team").start_share.sum())
    s = games[["game_id", "home", "away"]].merge(starters, on="game_id", how="left")
    for side, col in (("home", "gadj_h"), ("away", "gadj_a")):
        gid = s[f"goalie_{side}"]
        t = gid.map(talent.talent)
        diff = (t - s[side].map(mix)).fillna(0.0)
        out[col] = np.asarray(GM.goalie_offset(P, diff.to_numpy()), float)
    return out.fillna(0.0)


def run(date: str, results_path: str, goalies_path: str | None, sims: int, seed: int):
    from hattrick import gamemodel as GM
    from hattrick import ratings as R

    day = pd.Timestamp(date)
    fz = load_freeze()
    P = GM.load_params()
    P["mu"] = fz["league_level"]["mu_used"]          # the freeze's league level
    model = S.FittedModel(P, C.TARGET_SEASON)
    sch = fz["schedule"]
    res = load_results(results_path, day)
    ctx = GM.schedule_features(sch, C.TARGET_SEASON)[["game_id", "rest_h", "rest_a", "km_h",
                                                       "km_a", "dtz_h", "dtz_a"]]

    # 1. filter ratings through the results so far, one date at a time
    fp = R.load_filter_params()
    fp["P"] = {**fp["P"], "mu": P["mu"]}
    filt = R.InSeasonFilter(fz["ratings"][["team", "o", "d", "o_sd", "d_sd"]], fp,
                            start_date=sch.date.min())
    rr = res.merge(ctx, on="game_id", how="left")
    for _, day_games in rr.groupby("date", sort=True):
        filt.update_day(day_games)
    cur = filt.state()[["team", "o", "d", "o_sd", "d_sd"]]

    # 2. forecast today's games
    today = sch[sch.date == day].copy()
    starters = pd.read_csv(goalies_path) if goalies_path else None
    adj = pd.read_csv(FREEZE / "game_adjustments_2027.csv")
    gadj = goalie_offsets(fz, today, starters, P)
    t = today.merge(adj, on="game_id", how="left").merge(gadj, on="game_id", how="left")
    r = cur.set_index("team")
    lh, la = model.rates(r.o.reindex(t.home).to_numpy(), r.d.reindex(t.home).to_numpy(),
                         r.o.reindex(t.away).to_numpy(), r.d.reindex(t.away).to_numpy(),
                         t.adj_h.fillna(0).to_numpy(), t.adj_a.fillna(0).to_numpy())
    # a goalie better than the team's mix lowers the OPPONENT's scoring rate
    la = la * np.exp(t.gadj_h.to_numpy())
    lh = lh * np.exp(t.gadj_a.to_numpy())
    p = integrate_rating_uncertainty(model, cur, t, lh, la)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    games_out = t[["game_id", "date", "home", "away"]].assign(
        p_home_win=p["p_home"], p_home_reg=p["hreg"], p_away_reg=p["areg"],
        p_ot=p["tie"], exp_goals_home=lh, exp_goals_away=la,
        goalie_home=(starters.set_index("game_id").goalie_home.reindex(t.game_id).to_numpy()
                     if starters is not None else np.nan),
        goalie_away=(starters.set_index("game_id").goalie_away.reindex(t.game_id).to_numpy()
                     if starters is not None else np.nan),
        created_utc=created)

    # 3. re-simulate the rest of the season from the current standings
    done = res[["game_id", "home_g", "away_g", "extra"]]
    left = 1.0 - len(done) / len(sch)
    sim = S.simulate(sch, cur, model, n_sims=sims, seed=seed, completed=done,
                     game_adj=adj, drift_sd=fz.get("drift_sd", 0.05) * np.sqrt(left))
    standings = S.summarise(sim)

    outdir = LIVE / date
    outdir.mkdir(parents=True, exist_ok=True)
    games_out.to_csv(outdir / f"games_{date}.csv", index=False, float_format="%.5f")
    standings.to_csv(outdir / f"standings_{date}.csv", index=False, float_format="%.4f")
    cur.to_csv(outdir / f"ratings_{date}.csv", index=False, float_format="%.5f")
    run_meta = {"date": date, "created_utc": created, "code": _code_commit(),
                "results_through": str(res.date.max()) if len(res) else None,
                "n_results": int(len(res)), "sims": sims, "seed": seed,
                "inputs": {"results": {"path": str(results_path), "sha256": _sha(results_path)},
                           **({"goalies": {"path": goalies_path, "sha256": _sha(goalies_path)}}
                              if goalies_path else {}),
                           "freeze_state": _sha(FREEZE / "state_2027.json")}}
    (outdir / f"run_{date}.json").write_text(json.dumps(run_meta, indent=1))
    return games_out, standings


def integrate_rating_uncertainty(model, cur, t, lh, la, n: int = 64, seed: int = 7):
    """Average outcome probabilities over the ratings' posterior uncertainty.

    Plugging posterior MEANS into a non-linear win-probability function makes
    forecasts too confident; averaging over draws of (o, d) does not."""
    rng = np.random.default_rng(seed)
    r = cur.set_index("team")
    sd_h = np.hypot(r.o_sd.reindex(t.home).to_numpy(), r.d_sd.reindex(t.away).to_numpy())
    sd_a = np.hypot(r.o_sd.reindex(t.away).to_numpy(), r.d_sd.reindex(t.home).to_numpy())
    acc = None
    for _ in range(n):
        eh = np.exp(rng.standard_normal(len(t)) * sd_h)
        ea = np.exp(rng.standard_normal(len(t)) * sd_a)
        p = model.probs(lh * eh, la * ea)
        acc = p if acc is None else {k: acc[k] + p[k] for k in acc}
    return {k: v / n for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--results", default=str(C.ROOT / "neurhl/output/live/results_2027.csv"))
    ap.add_argument("--goalies", default=None)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    g, s = run(a.date, a.results, a.goalies, a.sims, a.seed)
    print(g[["game_id", "home", "away", "p_home_win", "exp_goals_home",
             "exp_goals_away"]].to_string(index=False))
    print(s[["team", "points", "playoff_pct", "cup_pct"]].head(10).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
