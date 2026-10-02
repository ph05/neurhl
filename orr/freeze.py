"""The 2026-27 preseason forecast: every game, team and player.

    python3 -m orr.freeze [--sims 40000] [--allow-dirty]

What it is: a forecast built ONLY from information dated before the first puck
drop (2026-09-29 21:00 UTC). Every repository input is read from the git tree
of the last commit before that moment (orr.snapshot); the added inputs
(box scores to 2023-24, historical preseason lines, researched injury timelines
reported before 2026-09-29) are listed with their hashes in the manifest.
It was BUILT on 2026-09-30/10-01, after the opening games had been played; the
information set, not the build time, is what predates the season, and the
scorecard treats it accordingly (orr.score).

Steps
 1. Player layer: skater and goalie projections with availability and ice time
    (orr.players / goalies / deploy), and the bottom-up roster components.
 2. Team views, each as points per 82 above the league mean:
      market     the 2026-08-17 points lines (no-vig mean from the over/under
                 prices) PLUS the news the line had not seen (trades, signings
                 and departures after 08-17, injuries and suspensions reported
                 after it, the Hellebuyck standoff), priced by the player model;
      top-down   regressed team history;
      roster     team history corrected for the change from last season's
                 roster to this one (orr.roster_delta);
      bottom-up  the opening roster's summed player projections
                 (orr.team_points_map).
    A pre-declared list of view combinations is scored leave-one-season-out on
    the seasons that have both a preseason line and a clean (first-10-games)
    roster proxy: 2019, 2020, 2022, 2023, 2024. The combination with the lowest
    out-of-sample RMSE is used, its weights refitted on those seasons. Weights
    are convex (non-negative, summing to one; teams.fit_blend), so the blend
    is never more spread out than its inputs.
 3. Targets -> offence/defence ratings on the actual schedule (calibrate); the
    rating SD is the blend's out-of-sample error net of the game luck the
    scoring model itself implies.
 4. Simulate the season (season.simulate) with the fitted scoring model,
    rest/travel effects, strength drawn per simulation and drifting in season.
 5. Game file: every game's probabilities averaged over rating uncertainty.
 6. Player lines reconciled to simulated team goals; goalie lines to team GA.
 7. Manifest: SHA-256 of every input, code file and output; refuses to run on
    uncommitted code unless --allow-dirty (and then records the diff state).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from orr import calibrate as K
from orr import config as C
from orr import data as D
from orr import roster_delta as RD
from orr import season as S
from orr import snapshot as SNAP
from orr import teams as T

V = C.TARGET_SEASON
GAMES = C.GAMES_PER_TEAM[V]
OUT = C.OUT / f"freeze_{V}"

# Empty-net goals against per team-game, 2023-24..2025-26 (team GA minus the
# goalies' GA in MoneyPuck): 0.170, 0.200, 0.195.
EN_GA_PER_GAME = 0.188

# Hellebuyck (WPG) asked for a trade on 08-27 and was suspended on 09-17. The
# goalie layer gives him a 10% chance of playing for Winnipeg (orr.goalies
# HOLDOUTS). Where he would go if traded was not settled before the cutoff, so
# no destination is credited (spread over ~20 possible clubs it is negligible).
HELLEBUYCK_ID = 8476945

# Injuries that were public before the 2026-08-17 market line (surgery dates
# from orr/data_injuries_2027.csv); the line already priced them, so the
# news counterfactual keeps them out too.
PRE_MARKET_INJURIES = {8478873: "Terry, hip surgery June",
                       8484144: "Bedard, shoulder surgery July",
                       8482093: "Jarvis, shoulder surgery late June",
                       8477967: "Demko, hip surgery January",
                       8480873: "Sandin, ACL surgery April"}

# Seasons with a preseason line AND a first-10-games roster proxy. 2025 and
# 2026 have no box scores, so their rosters would come from each player's
# season team -- in-season information -- and are excluded from every fit.
CLEAN_MARKET_SEASONS = [2019, 2020, 2022, 2023, 2024]


def _sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _commit() -> str:
    return subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# Rosters at the market date and at the cutoff
# ---------------------------------------------------------------------------
MARKET_DATE = pd.Timestamp("2026-08-17")
MOVE_FILES = [("2026-09-27", "moves_vs_2026-08.csv"), ("2026-09-28", "moves_vs_2026-09-27.csv"),
              ("2026-09-29", "moves_vs_2026-09-28.csv")]


def news_moves(ros: pd.DataFrame) -> pd.DataFrame:
    """Roster news after the market date that the line could not have priced.

    Only moves that change an NHL team's real strength are kept:
      * explicit team switches of players with 20+ NHL games in 2025-26 (a
        player removed from one club and added to another in the roster
        diffs: trades, terminations and re-signings),
        e.g. Kreider ANA->MTL (09-12), Evangelista NSH->NJD (09-02), the
        Knies/Marchenko/Andrae trade (09-28), plus Merzlikins CBJ->TOR (09-28,
        per the injury research file);
      * veterans (age >= 25, 50+ NHL games in 2025-26) added from outside;
      * departures: players with 50+ NHL games in 2025-26 removed from a club
        and on no NHL roster (or injured list) at the cutoff.
    Camp churn (prospects added, depth players assigned or waived) is ignored:
    the line already assumed a typical opening lineup.
    Returns player_id, from_team (None = not on an NHL club), to_team.
    """
    ms = pd.concat([pd.read_csv(SNAP.path(f"data/raw/rosters/{d}/{f}")) for d, f in MOVE_FILES])
    sk = D.skater_seasons()
    gs = D.goalie_seasons()
    last = pd.concat([sk[sk.season_end == V - 1].groupby("player_id").gp.sum(),
                      gs[gs.season_end == V - 1].groupby("player_id").gp.sum()])
    last = last.groupby(level=0).sum()
    ages = D.age_on(ros.set_index("player_id").birth, V)
    out = []
    for pid, d in ms.groupby("player_id"):
        rem = d[d.change == "removed"].team.tolist()
        add = d[d.change == "added"].team.tolist()
        if rem and add and rem[0] != add[-1] and last.get(pid, 0) >= 20:
            out.append((pid, rem[0], add[-1], "switch"))
        elif add and not rem and last.get(pid, 0) >= 50 and ages.get(pid, 0) >= 25:
            out.append((pid, None, add[-1], "veteran addition"))
        elif rem and not add and last.get(pid, 0) >= 50 and pid not in set(ros.player_id):
            out.append((pid, rem[0], None, "departure"))
    out.append((8478007, "CBJ", "TOR", "switch (research file)"))   # Merzlikins
    m = pd.DataFrame(out, columns=["player_id", "from_team", "to_team", "kind"])
    return m.drop_duplicates("player_id", keep="last")


def news_components():
    """Bottom-up components at the cutoff and in the counterfactual 'market
    date' state: the same roster with the news moves undone (switches moved
    back, veteran additions removed, departures restored), every absence
    reported after the market date healed, and Hellebuyck in Winnipeg.
    LTIR placements and the injuries in PRE_MARKET_INJURIES keep their
    absences in both states. Both states use the same deterministic seeds."""
    from orr import deploy as DP
    from orr import goalies as GL
    from orr import players as PL
    from orr import team_components as TC

    ros = PL.roster_2027()
    moves = news_moves(ros)
    cf = ros.copy()
    for mv in moves.itertuples(index=False):
        if mv.kind == "departure":
            info = _player_info(mv.player_id)
            if info is not None:
                cf = pd.concat([cf, pd.DataFrame([{**info, "player_id": mv.player_id,
                                                   "team": mv.from_team}])], ignore_index=True)
        elif mv.player_id in set(cf.player_id):
            if mv.from_team is None:
                cf = cf[cf.player_id != mv.player_id]
            else:
                cf.loc[cf.player_id == mv.player_id, "team"] = mv.from_team
    status = D.availability_raw_2027().set_index("player_id").manual_status.dropna()
    ltir = [pid for pid, st in status.items() if st == "LTIR"]

    def comps(r, now: bool):
        sk = r[r.grp != "G"]
        gl = r[r.grp == "G"][["player_id", "team", "birth", "name"]]
        extra = sk.assign(pos=np.where(sk.grp == "D", "D", "F"))[["player_id", "pos", "birth", "name"]]
        go, rng_, _ = DP.games_out_2027(sk, GAMES, with_range=True)
        ggo, gpres, _ = GL.goalie_status_2027(gl, GAMES)
        if not now:
            keep = set(ltir) | set(PRE_MARKET_INJURIES)
            go = go.where(go.index.isin(keep), 0.0)
            rng_ = None
            ggo = ggo.where(ggo.index.isin(keep), 0.0)
            gpres = pd.Series(1.0, index=ggo.index)
        c, _ = TC.components(V, sk[["player_id", "team", "name"]], gl, GAMES,
                             roster_kind="opening", games_out=go, games_out_range=rng_,
                             goalie_out=ggo, goalie_present=gpres, extra=extra)
        return c

    return comps(ros, True), comps(cf, False), moves


def _player_info(pid: int) -> dict | None:
    """grp/name/birth for a player absent from the 2026-27 rosters."""
    sk = D.skater_seasons()
    row = sk[sk.player_id == pid].sort_values("season_end").tail(1)
    if not len(row):
        return None
    b = D.bios().set_index("player_id")
    return {"grp": "D" if row.pos.iloc[0] == "D" else "F", "name": row.name.iloc[0],
            "birth": b.birth.get(pid, pd.NaT), "pos_raw": row.position.iloc[0]}


def points_per_goal_diff() -> float:
    """Standings points per goal of differential (per 82), fitted on 2012-2026."""
    st = T.standings_all()
    st = st[(st.season_end >= 2012) & ~st.season_end.isin(C.BROKEN_SEASONS)]
    x = (st.gf - st.ga) * 82.0 / st.gp
    return float(np.polyfit(x, st.pts82, 1)[0])


def value_news(now: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    """Price the news as points per 82: change in projected goal differential
    per game (skater goals incl. call-ups; goaltending GSAx; half of the
    even-strength on-ice xGA index, since relative on-ice impacts overlap
    within a line) times points per goal of differential."""
    a, b = now.set_index("team"), cf.set_index("team").reindex(now.team)
    d_gf = a.gf_pg_total - b.gf_pg_total
    d_ga = -(a.goalie_gsax60 - b.goalie_gsax60) + 0.5 * (a.ev_xga_impact - b.ev_xga_impact) * 49.0 / 60.0
    ppg = points_per_goal_diff()
    return pd.DataFrame({"team": a.index, "news_d_gf_pg": d_gf.to_numpy(),
                         "news_d_ga_pg": d_ga.to_numpy(),
                         "news_rel82": (ppg * 82.0 * (d_gf - d_ga)).to_numpy()})


# ---------------------------------------------------------------------------
# Team targets
# ---------------------------------------------------------------------------
def team_targets(log: dict) -> pd.DataFrame:
    from orr import players as PL  # noqa: F401  (ensures params exist)
    now, cf, moves = news_components()
    news = value_news(now, cf).set_index("team").news_rel82

    # roster-change model (current roster vs last season's), fitted on history
    hist_delta = pd.read_csv(RD.OUT)
    hist_delta = hist_delta[hist_delta.season_end <= max(CLEAN_MARKET_SEASONS)]   # first-10 rosters only
    rd_fit = RD.to_points(hist_delta, V)
    comps = [c[2:] for c in rd_fit["cols"]]
    a_, b_ = now.set_index("team"), RD.previous_components(V, GAMES).set_index("team")
    rd = RD.apply(rd_fit, pd.DataFrame({"team": a_.index, **{f"d_{c}": (a_[c] - b_[c].reindex(a_.index)).to_numpy()
                                                          for c in comps}}))
    # bottom-up roster view (orr.team_points_map), fitted on the seasons
    # whose roster proxy is the first 10 games (2011-2024)
    from orr import team_points_map as TPM
    bu_hist = pd.read_csv(C.OUT / "backtest" / "team_components_hist.csv")
    bu_hist = bu_hist[bu_hist.roster_proxy == "first10"]
    bu_fit = TPM.fit(pd.concat([bu_hist, now.assign(season_end=V)], ignore_index=True), V)
    bu = TPM.predict(bu_fit, pd.concat([bu_hist, now.assign(season_end=V)], ignore_index=True), V)
    bu = bu.set_index("team").bu_rel82
    log["news_moves"] = moves.assign(player_id=moves.player_id.astype(int)).to_dict("records")
    log["points_per_goal_diff"] = points_per_goal_diff()

    teams = sorted(C.TEAMS_2027)
    mk = T.market_rel82(V).set_index("team").mkt_rel82.reindex(teams)
    td = T.predict_topdown(T.fit_topdown(V, alpha=8.0), V, teams).set_index("team").td_rel82
    f = pd.DataFrame({"team": teams, "mkt_rel82": (mk + news.reindex(teams)).to_numpy(),
                      "mkt_line_rel82": mk.to_numpy(), "news_rel82": news.reindex(teams).to_numpy(),
                      "td_rel82": td.reindex(teams).to_numpy(),
                      "tdr_rel82": (td.reindex(teams) + rd.reindex(teams)).to_numpy(),
                      "rd_rel82": rd.reindex(teams).to_numpy(),
                      "bu_rel82": bu.reindex(teams).to_numpy()})

    # blend: seasons with a preseason line and a clean roster proxy
    hist = T.blend_frame(CLEAN_MARKET_SEASONS)
    hist = hist.merge(TPM.walk_forward_points(bu_hist), on=["team", "season_end"], how="left")
    wf = RD.walk_forward(hist_delta)
    hist = hist.merge(wf, on=["team", "season_end"], how="left")
    hist["rd_rel82"] = hist.rd_rel82.fillna(0.0)          # expansion teams: no previous roster
    hist["tdr_rel82"] = hist.td_rel82 + hist.rd_rel82
    # pre-declared candidate view sets; the one with the lowest leave-one-
    # season-out RMSE is used, with weights refitted on the same seasons
    candidates = [["mkt_rel82"], ["mkt_rel82", "td_rel82"], ["mkt_rel82", "tdr_rel82"],
                  ["mkt_rel82", "td_rel82", "tdr_rel82"], ["mkt_rel82", "bu_rel82"],
                  ["mkt_rel82", "td_rel82", "bu_rel82"]]
    scored = []
    for cand in candidates:
        l = T.loso_blend(hist, cand)
        scored.append((float(np.sqrt((l.blend_rmse ** 2 * l.n).sum() / l.n.sum())), cand, l))
    rmse, views, loso = min(scored, key=lambda x: x[0])
    log["blend_candidates"] = {"+".join(c): r_ for r_, c, _ in scored}
    w = T.fit_blend(hist, views)
    f["target_rel82"] = f[views].to_numpy() @ w
    league84 = T.league_points_per_team(V, GAMES)
    f["target84"] = league84 + f.target_rel82 * GAMES / 82.0
    log["blend"] = {"views": views, "weights": dict(zip(views, map(float, w))),
                    "loso_rmse_82": rmse, "loso_mae_82": float((loso.blend_mae * loso.n).sum() / loso.n.sum()),
                    "league_points_per_team_84": league84}
    log["blend_seasons"] = CLEAN_MARKET_SEASONS
    log["pre_market_injuries"] = PRE_MARKET_INJURIES
    return f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run with uncommitted code (recorded in the manifest)")
    a = ap.parse_args()
    dirty = _dirty_code()
    if dirty and not a.allow_dirty:
        raise SystemExit("uncommitted code changes:\n" + "\n".join(dirty) +
                         "\ncommit first (or pass --allow-dirty)")
    OUT.mkdir(parents=True, exist_ok=True)
    log: dict = {"cutoff_utc": C.CUTOFF_UTC.isoformat(), "created_utc":
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), "code": _commit(),
                 "code_dirty": dirty,
                 "status": ("forecast from information dated before the cutoff; built after "
                            "the first puck drop (see created_utc)")}

    from orr import gamemodel as GM
    P = GM.load_params()
    log["mu_fitted"] = P["mu"]
    model = S.FittedModel(P, V)
    sch = D.schedule_2027()
    adj = model.game_adjustments(sch)
    adj = add_backup_b2b(adj, sch, P, GM, log)

    # 2-3: targets and ratings
    tg = team_targets(log)
    style = T.predict_style(T.fit_style(V), V, sorted(C.TEAMS_2027))
    targets = tg.set_index("team").target84
    r = K.solve_ratings(sch, targets, style[["team", "o_m", "d_m"]], model, adj)

    # rating uncertainty: out-of-sample blend error net of the game luck that
    # the scoring model itself implies over an 82-game schedule
    luck = luck_sd_82(sch, r, model, adj)
    talent = float(np.sqrt(max(log["blend"]["loso_rmse_82"] ** 2 - luck ** 2, 1.0)))
    hp = json.loads((C.PARAMS / "ratings_hp.json").read_text())["hp"]
    # within-season drift of each of o and d: the in-season filter's daily
    # process variance (shot-rate + finishing parts) over a ~190-day season
    drift = float(np.sqrt(190 * (hp["q_s"] + hp["q_f"])))
    slope = K.points_per_rating(sch, r[["team", "o", "d"]], model, adj)
    net_sd = talent * GAMES / 82.0 / slope.reindex(r.team).to_numpy()
    log.update(luck_sd_82=luck, talent_sd_82=talent, drift_sd=drift,
               rating_sd_net_mean=float(np.mean(net_sd)))

    # league scoring level (the player layer's rule: last intact season), with
    # the Jensen term for this rating dispersion and drift; then re-solve
    # Var(log lam) = o_sd^2 + d_sd^2 (= net_sd^2) + two centred random walks (drift^2/6 each)
    var = float(np.mean(net_sd ** 2) + 2 * drift ** 2 / 6)
    P["mu"] = league_level_mu(model, sch, adj, log, var)
    model = S.FittedModel(P, V)
    r = K.solve_ratings(sch, targets, style[["team", "o_m", "d_m"]], model, adj)
    r["o_sd"] = net_sd / np.sqrt(2)
    r["d_sd"] = net_sd / np.sqrt(2)

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
    adj.to_csv(OUT / "game_adjustments_2027.csv", index=False, float_format="%.6f")
    sch.to_csv(OUT / "schedule_2027.csv", index=False)
    state = {**log, "sims": a.sims, "seed": a.seed, "P_mu": P["mu"]}
    (OUT / "state_2027.json").write_text(json.dumps(state, indent=1, default=float))
    manifest = SNAP.manifest()
    fr = sorted((C.ROOT / "data" / "raw" / "fastrhockey").glob("*.parquet"))
    manifest["added_inputs"] = {
        "orr/data_market_history.csv": _sha(C.PKG / "data_market_history.csv"),
        "orr/data_injuries_2027.csv": _sha(C.PKG / "data_injuries_2027.csv"),
        "neurhl/output/preds (game ids, realised shots; unchanged since before the cutoff)":
            {p.name: _sha(p) for p in sorted((C.ROOT / "neurhl/output/preds").glob("*.csv"))},
        "data/raw/fastrhockey (sportsdataverse/fastRhockey-data, 2010-11..2023-24)":
            {p.name: _sha(p) for p in fr}}
    manifest["code"] = {"commit": log["code"], "dirty": dirty,
                        "files": {str(p.relative_to(C.ROOT)): _sha(p)
                                  for p in sorted(C.PKG.rglob("*.py")) if "cache" not in p.parts}}
    manifest["outputs"] = {p.name: _sha(p) for p in sorted(OUT.glob("*.csv"))}
    (OUT / "manifest_2027.json").write_text(json.dumps(manifest, indent=1))
    print(teams[["team", "points", "points_p10", "points_p90", "playoff_pct", "cup_pct",
                 "target84", "news_rel82"]].round(1).to_string(index=False))


def _dirty_code() -> list[str]:
    out = subprocess.run(["git", "-C", str(C.ROOT), "status", "--porcelain", "--", "orr"],
                         capture_output=True, text=True).stdout.splitlines()
    fitted = ("orr/output/params/", "orr/output/backtest/team_components_hist.csv",
              "orr/output/backtest/roster_delta_hist.csv", "orr/data_")
    return [l for l in out if l.strip().endswith(".py") or any(f in l for f in fitted)]


def luck_sd_82(sch, r, model, adj) -> float:
    """SD of a team's season points from game outcomes alone, at the solved
    ratings: per game, points are 2 / 1 / 0 with the model's probabilities."""
    rr = r.set_index("team")
    a2 = adj.set_index("game_id").reindex(sch.game_id)
    lh, la = model.rates(rr.o.reindex(sch.home).to_numpy(), rr.d.reindex(sch.home).to_numpy(),
                         rr.o.reindex(sch.away).to_numpy(), rr.d.reindex(sch.away).to_numpy(),
                         a2.adj_h.to_numpy(), a2.adj_a.to_numpy())
    p = model.probs(lh, la)
    otl_h = p["tie"] - p["h_ot"] - p["h_so"]
    otl_a = p["h_ot"] + p["h_so"]
    win_h, win_a = p["p_home"], 1 - p["p_home"]
    var_h = 4 * win_h + otl_h - (2 * win_h + otl_h) ** 2
    var_a = 4 * win_a + otl_a - (2 * win_a + otl_a) ** 2
    v = pd.concat([pd.Series(var_h, index=sch.home.to_numpy()),
                   pd.Series(var_a, index=sch.away.to_numpy())]).groupby(level=0).sum()
    return float(np.sqrt(v.mean() * 82.0 / GAMES))


def league_level_mu(model, sch, adj, log, var_log_rate: float) -> float:
    """mu such that expected goals per team-game (regulation + overtime
    winners, no shootout) equal last season's league level, allowing for the
    rating dispersion the simulation adds (Jensen: E[exp(x)] = exp(E x + var/2))."""
    t = D.team_seasons()
    last = t[t.season_end == V - 1]
    L = float((last.gf_all / last.gp).mean())
    P = model.P
    flat = np.zeros(len(sch))
    a2 = adj.set_index("game_id").reindex(sch.game_id)
    for _ in range(3):
        lh, la = model.rates(flat, flat, flat, flat, a2.adj_h.to_numpy(), a2.adj_a.to_numpy())
        p = model.probs(lh, la)
        ot = (p["h_ot"] + p["a_ot"]).mean() / 2.0
        reg = (lh.mean() + la.mean()) / 2.0 * np.exp(0.5 * var_log_rate)
        P["mu"] = P["mu"] + np.log((L - ot) / reg)
        model = S.FittedModel(P, V)
    log["league_level"] = {"gf_per_team_game_target": L, "mu_used": P["mu"],
                           "var_log_rate": var_log_rate}
    return P["mu"]


def add_backup_b2b(adj, sch, P, GM, log) -> pd.DataFrame:
    """Team-specific backup-on-back-to-back effect: a team whose backup is far
    below its starter loses more on the second night of a back-to-back.
    The starter-minus-backup gap is in the scoring model's own fitted goalie
    units (structural.team_gap_2027)."""
    from orr import structural as ST
    g = pd.read_csv(OUT / f"goalie_rates_{V}.csv")
    gap = ST.team_gap_2027(g)
    log["b2b_goalie_gap"] = {k: float(v) for k, v in gap.items()}
    sf = GM.schedule_features(sch, V)
    a = adj.copy()
    b2b_h = (sf.rest_h.to_numpy() == 1)
    b2b_a = (sf.rest_a.to_numpy() == 1)
    off_h = GM.b2b_goalie_offset(P, sch.home.map(gap).fillna(0).to_numpy())
    off_a = GM.b2b_goalie_offset(P, sch.away.map(gap).fillna(0).to_numpy())
    a["adj_a"] = a.adj_a + np.where(b2b_h, off_h, 0.0)   # home tired -> away scores more
    a["adj_h"] = a.adj_h + np.where(b2b_a, off_a, 0.0)
    return a


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
    """Reconcile the player layer to the simulated team totals.

    Skaters: goals, assists and their bands are scaled so that, per team, the
    rostered skaters' goals plus the call-ups' share equal the simulated team
    goals (no shootout 'goals'). Goalies: the rostered goalies' GA is scaled to
    the team's GA net of empty-net and shootout goals and of the call-ups'
    starts; save percentage and GSAx move with it; wins follow the team's
    expected win rate."""
    sk = pd.read_csv(OUT / f"player_rates_{V}.csv")
    gl = pd.read_csv(OUT / f"goalie_rates_{V}.csv")
    comp = pd.read_csv(OUT / f"team_components_{V}.csv").set_index("team")
    t = teams.set_index("team")
    gf_real = t.gf - (t.w - t.row)                      # remove shootout 'goals'
    share = (comp.gf_pg_skaters / comp.gf_pg_total).reindex(t.index)
    k = (gf_real * share / sk.groupby("team").g.sum().reindex(t.index)).rename("goal_scale")
    sk = sk.assign(goal_scale=sk.team.map(k).fillna(1.0))
    cols = [c for c in ("g", "g_p10", "g_p90", "g_sd", "a", "a_p10", "a_p90", "a_sd", "a1", "a2",
                        "ppg", "ppa", "p", "p_p10", "p_p50", "p_p90", "p_sd") if c in sk]
    sk[cols] = sk[cols].mul(sk.goal_scale, axis=0)

    starts = gl.groupby("team").starts.sum().reindex(t.index)
    callup = gl.groupby("team").callup_starts.first().reindex(t.index).fillna(0.0)
    team_goalie_ga = (t.ga - t.so_losses - EN_GA_PER_GAME * GAMES)
    tgt = team_goalie_ga * starts / (starts + callup)
    kg = (tgt / gl.groupby("team").ga.sum().reindex(t.index)).rename("ga_scale")
    gl = gl.assign(ga_scale=gl.team.map(kg).fillna(1.0))
    ga0 = gl.ga.copy()
    for c in ("ga", "ga_p10", "ga_p90"):
        gl[c] = gl[c] * gl.ga_scale
    shift = gl.ga - ga0                                   # GSAx = xGA - GA moves with GA
    for c in ("gsax", "gsax_p10", "gsax_p90"):
        if c in gl:
            gl[c] = gl[c] - shift
    sa = gl.sa.clip(lower=1)
    gl["sv_pct"] = 1 - gl.ga / sa
    gl["sv_pct_p10"] = 1 - gl.ga_p90 / sa
    gl["sv_pct_p90"] = 1 - gl.ga_p10 / sa
    gl["wins"] = gl.starts * gl.team.map(t.w / t.gp)
    gl = gl.drop(columns=[c for c in ("wins_placeholder",) if c in gl])
    return sk, gl


if __name__ == "__main__":
    main()
