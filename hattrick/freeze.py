"""The 2026-27 preseason freeze: every game, team and player, before puck drop.

    python3 -m hattrick.freeze [--sims 40000]

Information: every repository input comes from the pre-cutoff git snapshot
(hattrick.snapshot); added inputs (box scores to 2023-24, historical market
lines, researched injury timelines dated <= 2026-09-29) are listed with their
provenance in the manifest. Nothing after 2026-09-29 21:00 UTC is read.

Steps
 1. Player layer: skater and goalie projections with availability and ice time
    (hattrick.players / goalies / deploy), and the roster components.
 2. Team views, each as points per 82 above the league mean:
      market      the 2026-08-17 points lines, de-vigged, PLUS the news the
                  line had not seen (roster moves, injuries, suspensions and
                  the Hellebuyck standoff between 08-17 and the cutoff),
                  priced by the roster-change model;
      top-down    regressed team history;
      roster      top-down corrected for the change from last season's
                  roster to this one.
    Blend weights are the non-negative least-squares weights fitted on every
    season with a preseason market line (2019-2026), whose leave-one-season-
    out accuracy is reported in the team backtest.
 3. Targets -> offence/defence ratings on the actual schedule (calibrate),
    rating SD from the blend's out-of-sample error net of game luck.
 4. Simulate the season (season.simulate) with the fitted scoring model,
    rest/travel effects, strength drawn per simulation and drifting in season.
 5. Game file: every game's probabilities averaged over rating uncertainty.
 6. Player lines reconciled to simulated team goals; goalie lines to team GA.
 7. Manifest: SHA-256 of every input and output, code commit, cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from hattrick import calibrate as K
from hattrick import config as C
from hattrick import data as D
from hattrick import roster_delta as RD
from hattrick import season as S
from hattrick import snapshot as SNAP
from hattrick import teams as T

V = C.TARGET_SEASON
GAMES = C.GAMES_PER_TEAM[V]
OUT = C.OUT / f"freeze_{V}"

# Hellebuyck (WPG, suspended 2026-09-17 after a public trade request on
# 08-27). Research (data_injuries_2027.csv, reports <= 09-29): a trade is
# expected within ~12 games; Carolina reported as the most aggressive bidder,
# Buffalo, Utah and San Jose linked. Expected share of the season he plays
# for each destination = P(destination) x (84 - 13)/84; as a starter he takes
# ~65% of starts. These are stated assumptions, not fitted parameters.
# Empty-net goals against per team-game, 2023-24..2025-26 (team GA minus the
# goalies' GA in MoneyPuck): 0.170, 0.200, 0.195.
EN_GA_PER_GAME = 0.188

HELLEBUYCK = {"player_id": 8476945, "p_dest": {"CAR": 0.35, "BUF": 0.12, "UTA": 0.12, "SJS": 0.08},
              "season_share": (84 - 13) / 84, "start_share": 0.65}


def _sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _commit() -> str:
    return subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# Rosters at the market date and at the cutoff
# ---------------------------------------------------------------------------
def august_roster(ros: pd.DataFrame) -> pd.DataFrame:
    """Undo every roster move recorded since the August snapshot (the market
    lines were recorded 2026-08-17): moves files are applied in reverse."""
    r = ros[["player_id", "team", "grp"]].copy()
    moves = [("2026-09-29", "moves_vs_2026-09-28.csv"), ("2026-09-28", "moves_vs_2026-09-27.csv"),
             ("2026-09-27", "moves_vs_2026-08.csv")]
    for d, f in moves:
        m = pd.read_csv(SNAP.path(f"data/raw/rosters/{d}/{f}"))
        m["grp"] = np.where(m.pos == "G", "G", np.where(m.pos == "D", "D", "F"))
        for x in m.itertuples(index=False):
            if x.change == "added":
                r = r[~((r.player_id == x.player_id) & (r.team == x.team))]
            elif x.change == "removed":
                if not ((r.player_id == x.player_id) & (r.team == x.team)).any():
                    r = pd.concat([r, pd.DataFrame([{"player_id": x.player_id, "team": x.team,
                                                     "grp": x.grp}])])
    return r.drop_duplicates("player_id").reset_index(drop=True)


def components_now_and_august():
    """Bottom-up components at the cutoff (with injuries, suspensions, the
    Hellebuyck mixture) and at the market date (August roster, all healthy)."""
    from hattrick import deploy as DP
    from hattrick import goalies as GL
    from hattrick import players as PL
    from hattrick import team_components as TC

    now = pd.read_csv(OUT / f"team_components_{V}.csv")
    ros = PL.roster_2027()
    aug = august_roster(ros)
    sk = aug[aug.grp != "G"][["player_id", "team"]]
    gl = aug[aug.grp == "G"][["player_id", "team"]]
    info = ros.set_index("player_id")
    extra = (sk.assign(pos="F").merge(ros[["player_id", "grp", "birth", "name"]], on="player_id", how="left"))
    extra["pos"] = np.where(extra.grp == "D", "D", "F")
    augc, _ = TC.components(V, sk, gl, GAMES, roster_kind="opening",
                            extra=extra[["player_id", "pos", "birth", "name"]].dropna(subset=["pos"]))
    return now, augc, aug


def hellebuyck_adjustment(now: pd.DataFrame, goalies: pd.DataFrame) -> pd.Series:
    """Expected change in team goalie GSAx/60 from a possible trade."""
    h = HELLEBUYCK
    row = goalies[goalies.player_id == h["player_id"]]
    if not len(row):
        return pd.Series(0.0, index=now.team)
    g60 = float(row.gsax60.iloc[0])
    cur = now.set_index("team").goalie_gsax60
    adj = pd.Series(0.0, index=cur.index)
    for t, p in h["p_dest"].items():
        adj[t] = p * h["season_share"] * h["start_share"] * (g60 - cur[t])
    return adj


# ---------------------------------------------------------------------------
# Team targets
# ---------------------------------------------------------------------------
def team_targets(log: dict) -> pd.DataFrame:
    from hattrick import players as PL  # noqa: F401  (ensures params exist)
    goalies = pd.read_csv(OUT / f"goalie_rates_{V}.csv")
    now, augc, aug = components_now_and_august()
    now = now.copy()
    now["goalie_gsax60"] = now.goalie_gsax60 + now.team.map(hellebuyck_adjustment(now, goalies))

    # roster-change model, fitted on history (walk-forward up to 2026)
    hist_delta = pd.read_csv(RD.OUT)
    rd_fit = RD.to_points(hist_delta, V)
    comps = [c[2:] for c in rd_fit["cols"]]

    def delta(a, b):
        a, b = a.set_index("team"), b.set_index("team")
        return pd.DataFrame({"team": a.index, **{f"d_{c}": (a[c] - b[c].reindex(a.index)).to_numpy()
                                                  for c in comps}})

    news = RD.apply(rd_fit, delta(now, augc))              # 08-17 -> cutoff
    prev = RD.previous_components(V, GAMES)
    rd = RD.apply(rd_fit, delta(now, prev))                 # 2025-26 roster -> now

    teams = sorted(C.TEAMS_2027)
    mk = T.market_rel82(V).set_index("team").mkt_rel82.reindex(teams)
    td = T.predict_topdown(T.fit_topdown(V, alpha=8.0), V, teams).set_index("team").td_rel82
    f = pd.DataFrame({"team": teams, "mkt_rel82": (mk + news.reindex(teams)).to_numpy(),
                      "mkt_line_rel82": mk.to_numpy(), "news_rel82": news.reindex(teams).to_numpy(),
                      "td_rel82": td.reindex(teams).to_numpy(),
                      "tdr_rel82": (td.reindex(teams) + rd.reindex(teams)).to_numpy(),
                      "rd_rel82": rd.reindex(teams).to_numpy()})

    # blend weights: all seasons with a market line
    hist = T.blend_frame([2019, 2020, 2022, 2023, 2024, 2025, 2026])
    wf = RD.walk_forward(hist_delta)
    hist = hist.merge(wf, on=["team", "season_end"], how="left")
    hist["tdr_rel82"] = hist.td_rel82 + hist.rd_rel82
    views = ["mkt_rel82", "td_rel82", "tdr_rel82"]
    w = T.fit_blend(hist, views)
    loso = T.loso_blend(hist, views)
    rmse = float(np.sqrt((loso.blend_rmse ** 2 * loso.n).sum() / loso.n.sum()))
    f["target_rel82"] = f[views].to_numpy() @ w
    league84 = T.league_points_per_team(V, GAMES)
    f["target84"] = league84 + f.target_rel82 * GAMES / 82.0
    log["blend"] = {"views": views, "weights": dict(zip(views, map(float, w))),
                    "loso_rmse_82": rmse, "loso_mae_82": float((loso.blend_mae * loso.n).sum() / loso.n.sum()),
                    "league_points_per_team_84": league84}
    log["talent_sd_82"] = float(np.sqrt(max(rmse ** 2 - T.LUCK_SD_82 ** 2, 1.0)))
    log["hellebuyck"] = HELLEBUYCK
    log["august_roster_size"] = int(len(aug))
    return f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log: dict = {"cutoff_utc": C.CUTOFF_UTC.isoformat(), "created_utc":
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), "code": _commit()}

    from hattrick import gamemodel as GM
    P = GM.load_params()
    model = S.FittedModel(P, V)
    sch = D.schedule_2027()
    adj = model.game_adjustments(sch)

    # 2-3: targets and ratings
    tg = team_targets(log)
    style = T.predict_style(T.fit_style(V), V, sorted(C.TEAMS_2027))
    targets = tg.set_index("team").target84
    r = K.solve_ratings(sch, targets, style[["team", "o_m", "d_m"]], model, adj)
    slope = K.points_per_rating(sch, r[["team", "o", "d"]], model, adj)
    net_sd = log["talent_sd_82"] * GAMES / 82.0 / slope.reindex(r.team).to_numpy()
    r["o_sd"] = net_sd / np.sqrt(2)
    r["d_sd"] = net_sd / np.sqrt(2)
    drift = float(json.loads((C.PARAMS / "ratings_hp.json").read_text()).get("season_drift_sd", 0.05)) \
        if (C.PARAMS / "ratings_hp.json").exists() else 0.05
    log["drift_sd"] = drift
    log["rating_sd_net_mean"] = float(np.mean(net_sd))

    # 4: simulate
    res = S.simulate(sch, r[["team", "o", "d", "o_sd", "d_sd"]], model, n_sims=a.sims,
                     seed=a.seed, game_adj=adj, drift_sd=drift)
    teams = S.summarise(res).merge(tg, on="team", how="left")
    teams = teams.merge(r[["team", "o", "d", "o_sd", "d_sd"]], on="team")

    # 5: game file, averaged over rating uncertainty
    games = game_file(sch, r, model, adj)

    # 6: player and goalie lines reconciled to team goals
    sk, gl = player_lines(teams)

    # write
    teams.to_csv(OUT / "teams_2027.csv", index=False, float_format="%.4f")
    games.to_csv(OUT / "games_2027.csv", index=False, float_format="%.5f")
    sk.to_csv(OUT / "skaters_2027.csv", index=False, float_format="%.3f")
    gl.to_csv(OUT / "goalies_2027.csv", index=False, float_format="%.4f")
    r.to_csv(OUT / "ratings_2027.csv", index=False, float_format="%.5f")
    sch.to_csv(OUT / "schedule_2027.csv", index=False)
    state = {**log, "sims": a.sims, "seed": a.seed}
    (OUT / "state_2027.json").write_text(json.dumps(state, indent=1, default=float))
    manifest = SNAP.manifest()
    manifest["added_inputs"] = {
        "hattrick/data_market_history.csv": _sha(C.PKG / "data_market_history.csv"),
        "hattrick/data_injuries_2027.csv": _sha(C.PKG / "data_injuries_2027.csv"),
        "fastRhockey box scores": "sportsdataverse/fastRhockey-data, seasons 2010-11..2023-24"}
    manifest["outputs"] = {p.name: _sha(p) for p in sorted(OUT.glob("*.csv"))}
    manifest["code"] = log["code"]
    (OUT / "manifest_2027.json").write_text(json.dumps(manifest, indent=1))
    print(teams[["team", "points", "points_p10", "points_p90", "playoff_pct", "cup_pct",
                 "target84", "news_rel82"]].round(1).to_string(index=False))


def game_file(sch, r, model, adj, n_draws: int = 400, seed: int = 11) -> pd.DataFrame:
    """P(outcomes) per game averaged over draws of every team's rating."""
    rng = np.random.default_rng(seed)
    rr = r.set_index("team")
    teams = rr.index.to_numpy()
    idx = {t: i for i, t in enumerate(teams)}
    h = sch.home.map(idx).to_numpy()
    aw = sch.away.map(idx).to_numpy()
    a2 = adj.set_index("game_id").reindex(sch.game_id)
    acc, lh_m, la_m = None, 0.0, 0.0
    for _ in range(n_draws):
        o = rr.o.to_numpy() + rng.standard_normal(len(teams)) * rr.o_sd.to_numpy()
        d = rr.d.to_numpy() + rng.standard_normal(len(teams)) * rr.d_sd.to_numpy()
        lh, la = model.rates(o[h], d[h], o[aw], d[aw], a2.adj_h.to_numpy(), a2.adj_a.to_numpy())
        p = model.probs(lh, la)
        acc = p if acc is None else {k: acc[k] + p[k] for k in acc}
        lh_m, la_m = lh_m + lh, la_m + la
    p = {k: v / n_draws for k, v in acc.items()}
    return pd.DataFrame({"game_id": sch.game_id, "date": sch.date.dt.date, "home": sch.home,
                         "away": sch.away, "p_home_win": p["p_home"], "p_home_reg": p["hreg"],
                         "p_away_reg": p["areg"], "p_ot": p["tie"],
                         "p_home_ot": p["h_ot"], "p_home_so": p["h_so"],
                         "exp_reg_goals_home": lh_m / n_draws, "exp_reg_goals_away": la_m / n_draws,
                         "rest_adj_home": a2.adj_h.to_numpy(), "rest_adj_away": a2.adj_a.to_numpy()})


def player_lines(teams: pd.DataFrame):
    """Scale skater goals/assists (and their bands) so that, per team, rostered
    skaters' goals + the call-ups' share equal the simulated team goals
    (excluding shootout 'goals'); goalies' GA to the simulated team GA."""
    sk = pd.read_csv(OUT / f"player_rates_{V}.csv")
    gl = pd.read_csv(OUT / f"goalie_rates_{V}.csv")
    comp = pd.read_csv(OUT / f"team_components_{V}.csv").set_index("team")
    t = teams.set_index("team")
    # simulated GF includes one goal per shootout win; remove the expected number
    so_wins = t.w - t.row
    gf_real = t.gf - so_wins
    share = (comp.gf_pg_skaters / comp.gf_pg_total).reindex(t.index)
    tgt = gf_real * share
    cur = sk.groupby("team").g.sum().reindex(t.index)
    k = (tgt / cur).rename("goal_scale")
    sk["goal_scale"] = sk.team.map(k).fillna(1.0)
    for c in ("g", "g_p10", "g_p90", "g_sd", "a", "a_p10", "a_p90", "a_sd", "a1", "a2",
              "ppg", "ppa", "p", "p_p10", "p_p50", "p_p90", "p_sd"):
        if c in sk:
            sk[c] = sk[c] * sk.goal_scale
    # goalies are charged with neither shootout "goals" nor empty-net goals
    gt = t.ga - t.so_losses - EN_GA_PER_GAME * GAMES
    cur_ga = gl.groupby("team").ga.sum().reindex(t.index)
    kg = (gt / cur_ga).rename("ga_scale")
    gl["ga_scale"] = gl.team.map(kg).fillna(1.0)
    for c in ("ga", "ga_p10", "ga_p90"):
        gl[c] = gl[c] * gl.ga_scale
    gl["sv_pct"] = 1 - gl.ga / gl.sa.clip(lower=1)
    return sk, gl


if __name__ == "__main__":
    main()
