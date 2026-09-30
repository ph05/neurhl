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
      * veterans (age >= 25, 50+ NHL games in 2025-26) added from outside.
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
    out.append((8478007, "CBJ", "TOR", "switch (research file)"))   # Merzlikins
    m = pd.DataFrame(out, columns=["player_id", "from_team", "to_team", "kind"])
    return m.drop_duplicates("player_id", keep="last")


def news_components():
    """Bottom-up components at the cutoff and in the counterfactual 'market
    date' state: same roster except the news moves undone, every injury,
    suspension and holdout reported after the market date healed, and
    Hellebuyck in Winnipeg. LTIR placements made before the market date
    (Pietrangelo, Ellis) stay in both states."""
    from hattrick import deploy as DP
    from hattrick import goalies as GL
    from hattrick import players as PL
    from hattrick import team_components as TC

    ros = PL.roster_2027()
    moves = news_moves(ros)
    cf = ros.copy()
    for mv in moves.itertuples(index=False):
        if mv.player_id in set(cf.player_id):
            if mv.from_team is None:
                cf = cf[cf.player_id != mv.player_id]
            else:
                cf.loc[cf.player_id == mv.player_id, "team"] = mv.from_team
    status = D.availability_raw_2027().set_index("player_id").manual_status.dropna()
    pre_market_out = [pid for pid, st in status.items() if st == "LTIR"]

    def comps(r, injuries: bool):
        sk = r[r.grp != "G"]
        gl = r[r.grp == "G"][["player_id", "team", "birth", "name"]]
        extra = sk.assign(pos=np.where(sk.grp == "D", "D", "F"))[["player_id", "pos", "birth", "name"]]
        if injuries:
            go, rng_, _ = DP.games_out_2027(sk, GAMES, with_range=True)
            ggo, gpres, _ = GL.goalie_status_2027(gl, GAMES)
        else:
            go = pd.Series(0.0, index=sk.player_id.to_numpy())
            go[go.index.isin(pre_market_out)] = GAMES
            rng_, ggo, gpres = None, None, None
        c, _ = TC.components(V, sk[["player_id", "team", "name"]], gl, GAMES,
                             roster_kind="opening", games_out=go, games_out_range=rng_,
                             goalie_out=ggo, goalie_present=gpres, extra=extra)
        return c

    now = pd.read_csv(OUT / f"team_components_{V}.csv")
    return now, comps(cf, injuries=False), moves


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
    now, cf, moves = news_components()
    now = now.copy()
    now["goalie_gsax60"] = now.goalie_gsax60 + now.team.map(hellebuyck_adjustment(now, goalies))
    news = value_news(now, cf).set_index("team").news_rel82

    # roster-change model (current roster vs last season's), fitted on history
    hist_delta = pd.read_csv(RD.OUT)
    rd_fit = RD.to_points(hist_delta, V)
    comps = [c[2:] for c in rd_fit["cols"]]
    a_, b_ = now.set_index("team"), RD.previous_components(V, GAMES).set_index("team")
    rd = RD.apply(rd_fit, pd.DataFrame({"team": a_.index, **{f"d_{c}": (a_[c] - b_[c].reindex(a_.index)).to_numpy()
                                                          for c in comps}}))
    # bottom-up roster view (hattrick.team_points_map), fitted on 2011-2026
    from hattrick import team_points_map as TPM
    bu_hist = pd.read_csv(C.OUT / "backtest" / "team_components_hist.csv")
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

    # blend weights: all seasons with a market line
    hist = T.blend_frame([2019, 2020, 2022, 2023, 2024, 2025, 2026])
    hist = hist.merge(TPM.walk_forward_points(bu_hist), on=["team", "season_end"], how="left")
    wf = RD.walk_forward(hist_delta)
    hist = hist.merge(wf, on=["team", "season_end"], how="left")
    hist["rd_rel82"] = hist.rd_rel82.fillna(0.0)          # expansion teams: no previous roster
    hist["tdr_rel82"] = hist.td_rel82 + hist.rd_rel82
    # pre-declared candidate view sets; the one with the lowest leave-one-
    # season-out RMSE is used, with weights refitted on all market seasons
    candidates = [["mkt_rel82"], ["mkt_rel82", "td_rel82"], ["mkt_rel82", "tdr_rel82"],
                  ["mkt_rel82", "td_rel82", "tdr_rel82"], ["mkt_rel82", "bu_rel82"]]
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
    log["talent_sd_82"] = float(np.sqrt(max(rmse ** 2 - T.LUCK_SD_82 ** 2, 1.0)))
    log["hellebuyck"] = HELLEBUYCK
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
    adj = add_backup_b2b(adj, sch, P, GM)

    # league scoring level: the same rule the player layer uses (last intact
    # season), so team goals and the sum of player goals agree
    lvl = league_level_mu(model, sch, adj, log)
    P["mu"] = lvl
    model = S.FittedModel(P, V)

    # 2-3: targets and ratings
    tg = team_targets(log)
    style = T.predict_style(T.fit_style(V), V, sorted(C.TEAMS_2027))
    targets = tg.set_index("team").target84
    r = K.solve_ratings(sch, targets, style[["team", "o_m", "d_m"]], model, adj)
    slope = K.points_per_rating(sch, r[["team", "o", "d"]], model, adj)
    net_sd = log["talent_sd_82"] * GAMES / 82.0 / slope.reindex(r.team).to_numpy()
    r["o_sd"] = net_sd / np.sqrt(2)
    r["d_sd"] = net_sd / np.sqrt(2)
    # within-season drift of each of o and d: the in-season filter's daily
    # process variance (shot-rate + finishing parts) over a ~190-day season
    hp = json.loads((C.PARAMS / "ratings_hp.json").read_text())["hp"]
    drift = float(np.sqrt(190 * (hp["q_s"] + hp["q_f"])))
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
    adj.to_csv(OUT / "game_adjustments_2027.csv", index=False, float_format="%.6f")
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


def league_level_mu(model, sch, adj, log, drift_var: float = 0.005) -> float:
    """mu such that expected goals per team-game (regulation + overtime
    winners, no shootout) equal last season's league level, allowing for the
    rating dispersion the simulation adds (Jensen: E[exp(x)] = exp(E x + var/2))."""
    t = D.team_seasons()
    last = t[t.season_end == V - 1]
    L = float((last.gf_all / last.gp).mean())
    P = model.P
    flat = np.zeros(len(sch))
    a2 = adj.set_index("game_id").reindex(sch.game_id)
    mu0 = P["mu"]
    for _ in range(3):
        lh, la = model.rates(flat, flat, flat, flat, a2.adj_h.to_numpy(), a2.adj_a.to_numpy())
        p = model.probs(lh, la)
        ot = (p["h_ot"] + p["a_ot"]).mean() / 2.0
        reg = (lh.mean() + la.mean()) / 2.0 * np.exp(0.5 * (2 * 0.0067 + drift_var))
        P["mu"] = P["mu"] + np.log((L - ot) / reg)
        model = S.FittedModel(P, V)
    log["league_level"] = {"gf_per_team_game_target": L, "mu_fitted": mu0, "mu_used": P["mu"]}
    return P["mu"]


def add_backup_b2b(adj, sch, P, GM) -> pd.DataFrame:
    """Team-specific backup-on-back-to-back effect: a team whose backup is far
    below its starter loses more on the second night of a back-to-back.
    Save talent per shot on goal from the goalie layer (per unblocked attempt
    x attempts per shot on goal)."""
    g = pd.read_csv(OUT / f"goalie_rates_{V}.csv")
    g = g[g.p_present > 0.5].sort_values("start_share", ascending=False)
    per_sog = g.gsax_per_fa * 1.40
    g = g.assign(t=per_sog)
    top2 = g.groupby("team").head(2)
    gap = top2.groupby("team").t.agg(lambda x: x.iloc[0] - x.iloc[1] if len(x) > 1 else 0.0)
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
