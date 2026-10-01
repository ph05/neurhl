"""In-season predictions: update on results, forecast a day's games, re-run the season.

    python3 -m orr.inseason --date 2026-09-30 \
        [--results orr/output/live/results_2027.csv] \
        [--goalies goalies.csv] [--sims 20000] [--model 1.1|1.0] [--lineup-dir DIR]

Model versions (--model, recorded in run_<date>.json):
- 1.1 (default from 2026-10-01): ORR 1.0 plus the accepted pre-registered
  item X1 (orr/PLAN_1_1.md; orr/lineups.py): every game's dressed skaters
  (on-ice xG value and projected ice time, present minus the team's expected
  lineup) and its starting goalies enter both the filter's update (past
  games) and the forecast (today's games). Lineups and starters come from
  NeurHL's committed pregame lineup files (--lineup-dir, default
  neurhl/output/live/2027/<date>/: the latest pregame file per game, else
  the morning, else the preview file); --goalies overrides their starters.
- 1.0: the loop as first shipped (no past starters, no lineups; --goalies
  applies to the forecast date's games only).

Inputs
- The preseason freeze (orr/output/freeze_2027/): ratings, scoring-model
  parameters, schedule, goalie talent table.
- Results for games BEFORE --date: game_id, date, home, away, home_g, away_g,
  last_period (REG/OT/SO), optional shots_home/shots_away (shots on goal).
- Optional confirmed or projected starters: game_id, goalie_home,
  goalie_away (NHL ids); in 1.1 rows for past games are used too. Unknown
  starters fall back to each team's start-share mix, which is exactly what
  the preseason file assumed.

What happens
1. Team ratings are filtered game by game from the preseason prior
   (orr.ratings.InSeasonFilter), using goals and, when the results file
   carries them, shots. In 1.1 past games' starting goalies (where both are
   known) and dressed-lineup offsets enter the update, as backtested
   (orr/backtest/inseason_bt_1_1.py); 1.0 uses neither.
   The filter's home-ice estimate replaces the preseason one.
2. Every game on --date is forecast with the current ratings (plug-in means,
   as backtested), rest/travel context and starter information.
3. The rest of the season is simulated from the current standings with the
   current ratings and their uncertainty.

Outputs (orr/output/live/<date>/): games_<date>.csv, standings_<date>.csv,
ratings_<date>.csv, and run_<date>.json with the code commit, input hashes and
the UTC creation time. A forecast only counts if it is committed and pushed
before puck drop; the GitHub push time, not the file's own timestamp, is the
evidence (see orr/score.py).
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

from orr import config as C
from orr import player_update as PU
from orr import season as S

FREEZE = C.OUT / "freeze_2027"
LIVE = C.OUT / "live"

# model versions: ORR 1.1 = 1.0 + the accepted pre-registered item X1
MODELS = {
    "1.3": {"version": "ORR 1.3", "past_starters": True, "lineups": True, "player_update": True,
            "box_first": True, "goalie_update": True, "player_calibration": True,
            "accepted_items": ["X1", "1.2: in-season skater rates", "1.3: box-score lineups, goalie talent"]},
    "1.2": {"version": "ORR 1.2", "past_starters": True, "lineups": True, "player_update": True,
            "accepted_items": ["X1", "1.2: in-season skater rates"]},
    "1.1": {"version": "ORR 1.1", "past_starters": True, "lineups": True,
            "accepted_items": ["X1"]},
    "1.0": {"version": "ORR 1.0", "past_starters": False, "lineups": False,
            "accepted_items": []},
}
DEFAULT_MODEL = "1.3"


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


def starter_diffs(fz: dict, games: pd.DataFrame, starters: pd.DataFrame | None,
                  talent_fn=None) -> pd.DataFrame:
    """Known starters' save talent minus the team's expected start mix, in the
    scoring model's fitted goalie units (structural.goalie_talent_2027).
    ``talent_fn(date)`` (ORR 1.3) gives the talent as of a date, so each game
    uses only goalie evidence from before it."""
    out = pd.DataFrame({"game_id": games.game_id.to_numpy(), "diff_h": np.nan, "diff_a": np.nan})
    g = fz.get("goalies")
    if starters is None or g is None or not len(starters):
        return out
    from orr import structural as ST
    s = games[["game_id", "date", "home", "away"]].merge(starters, on="game_id", how="left")
    dates = pd.to_datetime(s.date)
    for d in sorted(dates.unique()) if talent_fn is not None else [None]:
        tal = talent_fn(d) if talent_fn is not None else ST.goalie_talent_2027()
        mix = ST.usual_starter_talent(g, tal)
        rows = (dates == d).to_numpy() if d is not None else np.ones(len(s), bool)
        for side, col in (("home", "diff_h"), ("away", "diff_a")):
            gid = s.loc[rows, f"goalie_{side}"]
            t = gid.map(tal).fillna(ST.goalie_talent_default()).where(gid.notna())
            out.loc[rows, col] = (t - s.loc[rows, side].map(mix)).to_numpy()
    return out


def merge_starters(file_st: pd.DataFrame | None, lineup_st: pd.DataFrame | None) -> pd.DataFrame | None:
    """game_id, goalie_home, goalie_away: the starters file where it names a
    goalie, else the lineup files' starter (NaN = unknown)."""
    cols = ["game_id", "goalie_home", "goalie_away"]
    frames = [f[cols] for f in (file_st, lineup_st) if f is not None and len(f)]
    if not frames:
        return None
    if len(frames) == 1:
        out = frames[0].copy()
    else:
        out = (frames[0].set_index("game_id").astype(float)
               .combine_first(frames[1].set_index("game_id").astype(float)).reset_index())
    out = out.drop_duplicates("game_id", keep="first")
    for c in ("goalie_home", "goalie_away"):
        out[c] = out[c].astype(float)
    return out[cols]


def run(date: str, results_path: str, goalies_path: str | None, sims: int, seed: int,
        model: str = DEFAULT_MODEL, lineup_dir: str | None = None):
    from orr import gamemodel as GM
    from orr import lineups as LU
    from orr import ratings as R

    cfg = MODELS[model]
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
    # shots on goal enter the update when the results file carries them
    rr = rr.rename(columns={"shots_home": "sh_h", "shots_away": "sh_a"})

    # ORR 1.1 (X1): lineups and starters of past games and of today's games
    file_st = pd.read_csv(goalies_path) if goalies_path else None
    lu_sk, lu_gk, lu_files = None, None, []
    if cfg["lineups"] or cfg["past_starters"]:
        lu_sk, lu_gk, lu_files = LU.live_lineups(day, lineup_dir)
        if cfg.get("box_first"):          # ORR 1.3: past games' actual lineups and starters
            b_sk, b_gk = LU.box_lineups(day, sch)
            lu_sk, lu_gk = LU.prefer_box(lu_sk, lu_gk, b_sk, b_gk)
    tal_fn = None
    if cfg.get("goalie_update"):          # ORR 1.3: in-season goalie talent, as of each game's date
        from orr import structural as ST
        tal_fn = ST.goalie_talent_live
    starters = merge_starters(file_st, lu_gk) if cfg["past_starters"] else file_st
    lo = (LU.live_offsets(lu_sk, LU.X1) if cfg["lineups"]
          else pd.DataFrame(columns=["game_id", "lo_h", "lo_a", "known_h", "known_a"]))
    if cfg["past_starters"] and starters is not None and len(rr):
        d = starter_diffs(fz, rr, starters, tal_fn).rename(columns={"diff_h": "gdiff_h", "diff_a": "gdiff_a"})
        rr = rr.merge(d, on="game_id", how="left")
    if cfg["lineups"] and len(rr):
        rr = rr.merge(lo[["game_id", "lo_h", "lo_a"]], on="game_id", how="left")
    for _, day_games in rr.groupby("date", sort=True):
        filt.update_day(day_games)
    cur = filt.state()[["team", "o", "d", "o_sd", "d_sd", "od_cov"]]
    # Anchor the uncertainty to the freeze: the filter's prior split inflates
    # the net-strength SD; scale it so that, before any result, the filter
    # reproduces the freeze's calibrated net SD (its own shrinkage then applies).
    st0 = R.InSeasonFilter(fz["ratings"][["team", "o", "d", "o_sd", "d_sd"]], fp,
                           start_date=sch.date.min()).state()
    net = lambda x: np.sqrt(np.maximum(x.o_sd ** 2 + x.d_sd ** 2 - 2 * x.od_cov, 1e-12)).mean()
    fr = fz["ratings"]
    k_sd = float(np.sqrt(fr.o_sd ** 2 + fr.d_sd ** 2).mean() / net(st0))
    cur = cur.assign(o_sd=cur.o_sd * k_sd, d_sd=cur.d_sd * k_sd, od_cov=cur.od_cov * k_sd ** 2)
    # the filter's home-ice estimate, as in the backtest (state() folds only
    # the league level into o and d)
    P["h"] = P["h"] + (filt.levels()["h"] - fp["P"]["h"])
    model = S.FittedModel(P, C.TARGET_SEASON)

    # 2. forecast today's games
    today = sch[sch.date == day].copy()
    adj = pd.read_csv(FREEZE / "game_adjustments_2027.csv")
    t = today.merge(adj, on="game_id", how="left").merge(
        starter_diffs(fz, today, starters, tal_fn), on="game_id", how="left")
    # Known starters: the preseason offsets carry the EXPECTED goalie (the
    # league-average backup effect on back-to-backs plus the team-specific
    # gap). When a starter is confirmed, that expectation is replaced by the
    # starter-known context and the starter's own talent offset.
    known = t.diff_h.notna() | t.diff_a.notna()
    if known.any():
        ctx = GM.schedule_features(sch, C.TARGET_SEASON).set_index("game_id").loc[t.game_id[known]]
        gap = fz["b2b_goalie_gap"]
        ah, aa = GM.known_starter_offsets(P, ctx.reset_index(), t.diff_h[known].to_numpy(float),
                                          t.diff_a[known].to_numpy(float),
                                          gap_h=t.home[known].map(gap).fillna(0).to_numpy(),
                                          gap_a=t.away[known].map(gap).fillna(0).to_numpy())
        t.loc[known, "adj_h"] = ah
        t.loc[known, "adj_a"] = aa
    # ORR 1.1 (X1): dressed-lineup offsets on top of the context/starter offsets
    t = t.merge(lo[["game_id", "lo_h", "lo_a"]], on="game_id", how="left")
    t["lo_h"], t["lo_a"] = t.lo_h.astype(float).fillna(0.0), t.lo_a.astype(float).fillna(0.0)
    r = cur.set_index("team")
    lh, la = model.rates(r.o.reindex(t.home).to_numpy(), r.d.reindex(t.home).to_numpy(),
                         r.o.reindex(t.away).to_numpy(), r.d.reindex(t.away).to_numpy(),
                         (t.adj_h.fillna(0) + t.lo_h).to_numpy(), (t.adj_a.fillna(0) + t.lo_a).to_numpy())
    # plug-in probabilities at the filtered means: the rule inseason_bt validated
    p = model.probs(lh, la)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    games_out = t[["game_id", "date", "home", "away"]].assign(
        p_home_win=p["p_home"], p_home_reg=p["hreg"], p_away_reg=p["areg"],
        p_ot=p["tie"], exp_goals_home=lh, exp_goals_away=la,
        goalie_home=(starters.set_index("game_id").goalie_home.reindex(t.game_id).to_numpy()
                     if starters is not None else np.nan),
        goalie_away=(starters.set_index("game_id").goalie_away.reindex(t.game_id).to_numpy()
                     if starters is not None else np.nan),
        lineup_adj_h=t.lo_h.to_numpy(), lineup_adj_a=t.lo_a.to_numpy(),
        model=cfg["version"], created_utc=created)

    players_out = player_lines(fz, games_out, day, update_rates=bool(cfg.get("player_update")),
                               calibrate=bool(cfg.get("player_calibration")))

    # 3. re-simulate the rest of the season from the current standings
    done = res[["game_id", "home_g", "away_g", "extra"]]
    left = 1.0 - len(done) / len(sch)
    rho = float(np.clip((cur.od_cov / (cur.o_sd * cur.d_sd)).mean(), -0.95, 0.95))
    sim = S.simulate(sch, cur, model, n_sims=sims, seed=seed, completed=done,
                     game_adj=adj, drift_sd=fz.get("drift_sd", 0.05) * np.sqrt(left), rho_od=rho)
    standings = S.summarise(sim)

    outdir = LIVE / date
    outdir.mkdir(parents=True, exist_ok=True)
    games_out.to_csv(outdir / f"games_{date}.csv", index=False, float_format="%.5f")
    players_out.to_csv(outdir / f"players_{date}.csv", index=False, float_format="%.4f")
    standings.to_csv(outdir / f"standings_{date}.csv", index=False, float_format="%.4f")
    cur.to_csv(outdir / f"ratings_{date}.csv", index=False, float_format="%.5f")
    run_meta = {"date": date, "created_utc": created, "code": _code_commit(),
                "results_through": str(res.date.max()) if len(res) else None,
                "n_results": int(len(res)), "sims": sims, "seed": seed,
                "missing_results_before_date": sorted(map(int, set(sch[sch.date < day].game_id) - set(res.game_id))),
                "rho_od": rho, "sd_anchor": k_sd, "home_ice": P["h"],
                "model": cfg["version"],
                "settings": {**cfg, "lineup_hp": LU.settings(LU.X1) if cfg["lineups"] else None},
                "orr_1_1": {"results_with_both_starters": int(rr.gdiff_h.notna().mul(rr.gdiff_a.notna()).sum())
                            if "gdiff_h" in rr else 0,
                            "results_with_lineup_offset": int((rr.get("lo_h", pd.Series(dtype=float))
                                                               .fillna(0) != 0).sum()),
                            "today_lineup_offsets": int((t.lo_h != 0).sum()),
                            "today_starters_known": int(known.sum())},
                "inputs": {"results": {"path": str(results_path), "sha256": _sha(results_path)},
                           **({"goalies": {"path": goalies_path, "sha256": _sha(goalies_path)}}
                              if goalies_path else {}),
                           **({"lineups": [{"path": str(Path(f).relative_to(C.ROOT))
                                            if str(f).startswith(str(C.ROOT)) else str(f),
                                            "sha256": _sha(f)} for f in lu_files]}
                              if lu_files else {}),
                           "freeze_state": _sha(FREEZE / "state_2027.json")}}
    (outdir / f"run_{date}.json").write_text(json.dumps(run_meta, indent=1))
    return games_out, standings


def player_lines(fz: dict, games: pd.DataFrame, day: pd.Timestamp, update_rates: bool = False,
                 calibrate: bool = False) -> pd.DataFrame:
    """Per-game lines for every skater likely to dress: expected goals, points
    and shots, P(at least one goal) and P(at least one point).

    A player's per-game rates are his preseason season line divided by his
    expected games; they are scaled by how many goals his team is expected
    to score in THIS game relative to its season average, and multiplied by
    the probability that he dresses (zero while a known absence lasts)."""
    sk = pd.read_csv(FREEZE / "skaters_2027.csv")
    # season-average expected REGULATION goals per game (the units of lam)
    gz = pd.read_csv(FREEZE / "games_2027.csv")
    base = pd.concat([pd.Series(gz.exp_reg_goals_home.to_numpy(), index=gz.home),
                      pd.Series(gz.exp_reg_goals_away.to_numpy(), index=gz.away)]).groupby(level=0).mean()
    # games already scheduled before today (whether or not their results are in)
    s0 = fz["schedule"][fz["schedule"].date < day]
    played = pd.concat([s0.home, s0.away]).value_counts()
    rows = []
    for g in games.itertuples(index=False):
        for team, lam in ((g.home, g.exp_goals_home), (g.away, g.exp_goals_away)):
            s = sk[sk.team == team].copy()
            if not len(s):
                continue
            gp_left = 84 - played.get(team, 0)
            out_now = s.games_out.fillna(0) > played.get(team, 0)
            p_dress = np.where(out_now, 0.0,
                               np.clip(s.gp / np.maximum(84 - s.games_out.fillna(0), 1), 0, 1))
            # exactly 12 forwards and 6 defencemen dress: scale the healthy
            # players' dress probabilities up to the slots (each capped at 1)
            p_dress = np.asarray(p_dress, float)
            for pos, slots in (("F", 12.0), ("D", 6.0)):
                m = (s.pos.to_numpy() == pos) & (p_dress > 0)
                need = min(slots, m.sum())
                for _ in range(20):
                    tot = p_dress[m].sum()
                    if tot <= 0 or abs(tot - need) < 1e-6:
                        break
                    p_dress[m] = np.minimum(1.0, p_dress[m] * need / tot)
            scale = lam / base.get(team, lam)
            rates = PU.live_rates(s, before=day) if update_rates else None   # ORR 1.2
            if rates is not None:
                r = rates.set_index("player_id").reindex(s.player_id)
                upd = {"g": r.g_pg.to_numpy(), "a": r.a_pg.to_numpy(), "sog": r.sog_pg.to_numpy()}
                per = lambda c: pd.Series(upd[c], index=s.index) if c in upd else s[c] / s.gp.clip(lower=1)
                per_p = lambda: per("g") + per("a")
            else:
                per = lambda c: s[c] / s.gp.clip(lower=1)
                per_p = lambda: per("p")
            eg, ep, es = per("g") * scale, per_p() * scale, per("sog") * scale
            s = s.assign(game_id=g.game_id, opponent=g.away if team == g.home else g.home,
                         p_dress=p_dress, exp_goals=eg * p_dress, exp_points=ep * p_dress,
                         exp_sog=es * p_dress,
                         p_goal=p_dress * (1 - np.exp(-eg)), p_point=p_dress * (1 - np.exp(-ep)),
                         toi_pg=s.toi / s.gp.clip(lower=1))
            rows.append(s[s.p_dress > 0.05][["game_id", "team", "opponent", "player_id", "name", "pos",
                                             "p_dress", "toi_pg", "exp_goals", "exp_points", "exp_sog",
                                             "p_goal", "p_point"]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cal = PU.load_calibration() if calibrate else None      # ORR 1.3: only if the backtest adopted it
    if cal is not None and len(out):
        for k, col in (("goal", "p_goal"), ("point", "p_point")):
            p = (out[col] / out.p_dress.clip(lower=1e-9)).clip(1e-6, 1 - 1e-6)
            out[col] = out.p_dress * PU.platt(p.to_numpy(), cal[k])
    return out.sort_values(["game_id", "team", "exp_points"], ascending=[True, True, False])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--results", default=str(C.OUT / "live" / "results_2027.csv"))
    ap.add_argument("--goalies", default=None)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
                    help="1.1 (default): lineups and past starters (X1); 1.0: the original loop")
    ap.add_argument("--lineup-dir", default=None,
                    help="NeurHL live folder with <date>/pregame_*_lineups.json "
                         "(default neurhl/output/live/2027)")
    a = ap.parse_args()
    g, s = run(a.date, a.results, a.goalies, a.sims, a.seed, model=a.model, lineup_dir=a.lineup_dir)
    print(g[["game_id", "home", "away", "p_home_win", "exp_goals_home",
             "exp_goals_away"]].to_string(index=False))
    print(s[["team", "points", "playoff_pct", "cup_pct"]].head(10).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
