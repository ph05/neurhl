"""Bottom-up team inputs for the team-strength layer.

For a season V and a roster, every component is an aggregate of the player
projections AFTER deployment (hattrick.deploy), so each player counts in
proportion to the ice time he is expected to get:

  ev_xgf_impact, ev_xga_impact
      Sum over the five skaters on the ice of their 5v5 on-ice xG/60
      relative to their team without them, per 60 team minutes
      (= sum_i TOI_ev_i * rel_i / team EV minutes). Positive xgf is good,
      positive xga is bad. Relative-to-team impacts are collinear within a
      line, so treat these as indices for the regression, not as additive
      goal counts.
  pp_xgf60      Team power-play xGF/60 implied by its shooters' projected PP
                individual xG (every PP shot belongs to one shooter).
  pp_onice_rel  TOI-weighted on-ice PP xGF/60 relative to league (+ = good).
  pk_xga60      TOI-weighted on-ice xGA/60 of the penalty killers (lower =
                better), and pk_onice_rel relative to league.
  finishing_pg  Projected goals minus league-calibrated projected ixG, per
                team game (shooting talent beyond shot quality).
  goalie_gsax60 Expected-start-weighted goalie GSAx/60 (call-up starts at the
                replacement prior); goalie_gsax_per_fa likewise.
  gf_pg_skaters Projected team goals per game from the summed player goal
                rates (rostered skaters), gf_pg_total adds the call-ups'
                share at replacement rates (conservation: the team's goals
                ARE its skaters' goals).

Historical table (walk-forward, V = 2011..2026):
    hattrick/output/backtest/team_components_hist.csv
roster proxy per season: 'first10' (skaters and goalies who dressed in a
team's first 10 games; 2011-2024) or 'season_team' (2025-2026: each player's
most-TOI team in V, which uses in-season information). Actual outcomes are
appended as act_* columns for the regression.

Run: python3 -m hattrick.team_components            (history + 2027)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import data as D
from hattrick import deploy as DP
from hattrick import goalies as GL
from hattrick import players as PL

SITS = PL.SITS


def replacement_rates(V: int, prm: PL.Params) -> dict:
    """Per-60 goal and ixG rates of a call-up (usage prior at fringe usage)."""
    pri = PL.rate_priors(V, prm.prior_window, prm.eb_window)
    fr = PL.usage_prior_tpg(V)
    tgt = PL.project_league(V)
    out = {}
    for pos in ("F", "D"):
        X = np.array([[1.0, fr[(pos, "ev")], fr[(pos, "pp")], fr[(pos, "sh")]]])
        for st in ("g", "ixg"):
            for k in SITS:
                c = f"{st}_{k}"
                out[(pos, c)] = float(np.clip(X @ pri[(c, pos)]["beta"], 1e-3, None)[0]) \
                    * tgt[c] * 60.0
    return out


def components(V: int, skaters: pd.DataFrame, goalies: pd.DataFrame, games: int,
               prm: PL.Params | None = None, roster_kind: str = "opening",
               games_out: pd.Series | None = None, games_out_range: dict | None = None,
               goalie_out: pd.Series | None = None,
               goalie_present: pd.Series | None = None,
               extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team components for season V. skaters/goalies: player_id, team.

    roster_kind: 'opening' (depth-chart games with call-up coverage) or
    'expost' (team-agnostic games; the leakier season-team rosters).
    Returns (team table, per-skater deployment table)."""
    prm = prm or PL.load_params()
    tot, proj = PL.pipeline(V, prm, skaters[["player_id", "team"]], games, roster_kind,
                            games_out=games_out, games_out_range=games_out_range,
                            extra=extra)
    cols = ["player_id", "rel_xgf60", "rel_xga60", "pp_onice_rel", "pp_onice_60",
            "sh_onice_rel", "sh_onice_60"]
    tot = tot.merge(proj[[c for c in cols if c in proj.columns]], on="player_id", how="left")
    lg = PL.project_league(V)
    rep = replacement_rates(V, prm)
    bud, rtpg = DP.budgets_for(V)
    rows = []
    for team, t in tot.groupby("team"):
        ev_min = games * lg["team_min_ev"]
        pp_min = games * lg["team_min_pp"]
        sh_min = games * lg["team_min_sh"]
        r = {"team": team, "n_skaters": len(t)}
        r["ev_xgf_impact"] = float((t.toi_ev * t.rel_xgf60).sum() / ev_min)
        r["ev_xga_impact"] = float((t.toi_ev * t.rel_xga60).sum() / ev_min)
        r["pp_xgf60"] = float((t.toi_pp * t.r60_ixg_pp).sum() / 60.0 / pp_min * 60.0)
        r["pp_onice_rel"] = float((t.toi_pp * t.pp_onice_rel.fillna(0)).sum() / max(t.toi_pp.sum(), 1e-9))
        r["pk_xga60"] = float((t.toi_sh * t.sh_onice_60.fillna(0)).sum() / max(t.toi_sh.sum(), 1e-9))
        r["pk_onice_rel"] = float((t.toi_sh * t.sh_onice_rel.fillna(0)).sum() / max(t.toi_sh.sum(), 1e-9))
        ixg_cal = sum(t[f"toi_{k}"] * t[f"r60_ixg_{k}"] * lg[f"rho_{k}"] / 60.0 for k in SITS)
        r["finishing_pg"] = float((t.g - ixg_cal).sum() / games)
        r["gf_pg_skaters"] = float(t.g.sum() / games)
        r["gf_ev_pg"] = float((t.toi_ev * t.r60_g_ev).sum() / 60 / games)
        r["gf_pp_pg"] = float((t.toi_pp * t.r60_g_pp).sum() / 60 / games)
        # call-ups: the dressed games the roster does not cover, at
        # replacement rates and replacement minutes per game
        rep_g = 0.0
        for pos in ("F", "D"):
            rg = float(t[f"rep_gp_{pos}"].iloc[0]) if f"rep_gp_{pos}" in t else 0.0
            rep_g += rg * sum(rtpg[(pos, k)] * rep[(pos, f"g_{k}")] / 60.0 for k in SITS)
            r[f"callup_gp_{pos}"] = rg
        r["gf_pg_total"] = r["gf_pg_skaters"] + rep_g / games
        r["skater_toi_share"] = float(t.toi.sum() / (games * sum(bud.values())))
        rows.append(r)
    comp = pd.DataFrame(rows)
    # goalies
    if goalies is not None and len(goalies):
        gd = GL.deploy_goalies(V, goalies, games, goalie_out, goalie_present, prm.q_scale)
        pr = GL.talent_prior(V)
        lgg = GL.league_goalie(V)
        rep_rate = pr["beta"][0] + pr["beta"][1] * np.log(0.05)
        g = gd.groupby("team").apply(lambda d: pd.Series({
            "goalie_starts": d.starts.sum(), "callup_starts": d.callup_starts.iloc[0],
            "goalie_gsax_per_fa": (d.starts * d.gsax_per_fa).sum() + d.callup_starts.iloc[0] * rep_rate}),
            include_groups=False).reset_index()
        g["goalie_gsax_per_fa"] /= games
        g["goalie_gsax60"] = g.goalie_gsax_per_fa * lgg["fa_per60"]
        comp = comp.merge(g, on="team", how="left")
    comp.insert(0, "season_end", V)
    return comp, tot


def historical_goalies(V: int, kind: str) -> pd.DataFrame:
    if kind == "first10":
        r = D.opening_rosters(V)
        return r[r.grp == "G"][["player_id", "team"]]
    gt = D.goalie_team_seasons()
    gt = gt[gt.season_end == V].sort_values("toi", ascending=False)
    return gt.drop_duplicates("player_id")[["player_id", "team"]]


def history(seasons=range(2011, 2027)) -> pd.DataFrame:
    """Walk-forward components per team-season with actual outcomes."""
    prm = PL.load_params()
    out = []
    t = D.team_seasons()
    for V in seasons:
        sk, flag = DP.historical_roster(V, "opening")
        gl = historical_goalies(V, flag)
        comp, _ = components(V, sk, gl, 82, prm,
                             roster_kind="opening" if flag == "first10" else "expost")
        comp["roster_proxy"] = flag
        comp["broken_season"] = V in C.BROKEN_SEASONS
        a = t[t.season_end == V].set_index("team")
        comp["act_gp"] = comp.team.map(a.gp)
        for c in ("gf_all", "ga_all", "xgf_all", "xga_all", "gf_ev", "ga_ev",
                  "xgf_ev", "xga_ev", "gf_pp", "xgf_pp", "ga_sh", "xga_sh"):
            if c in a.columns:
                comp[f"act_{c}_pg"] = comp.team.map(a[c] / a.gp)
        out.append(comp)
        print(f"{V}: {len(comp)} teams ({flag})  corr(gf_pg_total, actual GF/gp) = "
              f"{np.corrcoef(comp.gf_pg_total, comp.act_gf_all_pg)[0, 1]:+.2f}  "
              f"corr(goalie_gsax60, -GA/gp) = "
              f"{np.corrcoef(comp.goalie_gsax60.fillna(0), -comp.act_ga_all_pg)[0, 1]:+.2f}",
              flush=True)
    h = pd.concat(out, ignore_index=True)
    p = C.OUT / "backtest" / "team_components_hist.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    h.round(5).to_csv(p, index=False)
    print("->", p)
    return h


def run_2027() -> pd.DataFrame:
    V = C.TARGET_SEASON
    games = C.GAMES_PER_TEAM[V]
    ros = PL.roster_2027()
    sk = ros[ros.grp != "G"]
    gl = ros[ros.grp == "G"][["player_id", "team", "birth", "name"]]
    extra = sk.assign(pos=np.where(sk.grp == "D", "D", "F"))[["player_id", "pos", "birth", "name"]]
    go, rng_, _ = DP.games_out_2027(sk, games, with_range=True)
    ggo, gpres, _ = GL.goalie_status_2027(gl, games)
    comp, _ = components(V, sk[["player_id", "team", "name"]], gl, games, roster_kind="opening",
                         games_out=go, games_out_range=rng_, goalie_out=ggo,
                         goalie_present=gpres, extra=extra)
    comp["roster_proxy"] = "opening_2026-09-29"
    p = C.OUT / f"freeze_{V}" / f"team_components_{V}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    comp.round(5).to_csv(p, index=False)
    print(comp.round(3).to_string())
    print("->", p)
    return comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--season", type=int, default=None)
    a = ap.parse_args()
    if a.history or a.season is None:
        history()
    if a.season == C.TARGET_SEASON or a.season is None:
        run_2027()
