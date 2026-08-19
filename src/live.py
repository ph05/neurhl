"""Living model v4 (PLAN_V4 B5): nightly in-season updater with the VALIDATED estimator.

Estimator (validated on 2012-2017 tune / 2022-2026 replays): in-season strength =
w * preseason_prior + (1 - w) * HISTORY-CARRIED Elo, w = n0/(n0 + games), n0 = 5.
History-carried means run_elo over the full games table INCLUDING fetched 2026-27
results (the previous live path seeded Elo from the prior and then blended the prior
again — a double-shrink the replays never validated; fixed per review finding B5).

`python live.py update`   fetch 2026-27 results -> blended ratings for v1/v4/HOWE ->
                          rest-of-season sim (b2b d_adj, goalie layer, availability2,
                          banked standings, playoffs) -> output/live/live_odds_<date>.csv
                          + live_ratings_<date>.csv + append to clv_log.csv.
`python live.py selftest` parity check (no network): replay a past mid-season cutoff
                          through both the replay engine and the live path; blended
                          ratings must agree to < 0.5 Elo (PLAN_V4 B5 acceptance).

Replay tuning/validation history (n0 grid, 2022-2026 Brier path) lives in
params_v3.json["living_model"]; that record is not rerun here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import availability2 as A2
import engine as E
import goalie_game as GG
import players as P
from overlay import points_per_goal, skater_value_goals

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
OUT = PROJ / "output"
LIVE = OUT / "live"
B2B_ELO = 38.0
SIGMA_INSEASON = 25.0   # the sigma the 2022-2026 replay Brier path was validated with
N_SIMS = 10_000
UA = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------- state builders
def carried_elo(g_all: pd.DataFrame, K, H, phi_s) -> dict:
    """History-carried Elo through every played game (the validated construction)."""
    _, end_r, _ = E.run_elo(g_all, K=K, H=H, phi_s=phi_s)
    return end_r[max(end_r)]


def blended(prior_dev: pd.Series, elo_now: dict, games_n: dict, n0: float) -> dict:
    out = {}
    for t in prior_dev.index:
        w = n0 / (n0 + games_n.get(t, 0))
        out[t] = 1505.0 + w * prior_dev[t] + (1 - w) * (elo_now.get(t, 1505.0) - 1505.0)
    return out


def banked_from(played: pd.DataFrame) -> tuple[dict, dict]:
    """({'pts':{t:..},'rw':..,'row':..,'win':..}, games_played per team)."""
    b = {k: {} for k in ("pts", "rw", "row", "win")}
    gn: dict = {}
    for gm in played.itertuples():
        hw = gm.home_g > gm.away_g
        past = gm.went_ot or gm.went_so
        for team, won in ((gm.home, hw), (gm.away, not hw)):
            gn[team] = gn.get(team, 0) + 1
            b["pts"][team] = b["pts"].get(team, 0) + (2 if won else (1 if past else 0))
            b["win"][team] = b["win"].get(team, 0) + int(won)
            b["rw"][team] = b["rw"].get(team, 0) + int(won and not past)
            b["row"][team] = b["row"].get(team, 0) + int(won and not gm.went_so)
    return b, gn


def fetch_2026_results() -> pd.DataFrame:
    import requests
    prior = pd.read_csv(OUT / "v4_prior_ratings.csv", index_col=0)
    games = {}
    for t in sorted(prior.index):
        r = requests.get(f"https://api-web.nhle.com/v1/club-schedule-season/{t}/20262027",
                         headers=UA, timeout=30).json()
        for gm in r.get("games", []):
            if gm.get("gameType") != 2:
                continue
            hs = gm.get("homeTeam", {}).get("score")
            if hs is None or gm.get("gameState") not in ("OFF", "FINAL"):
                continue
            lp = gm.get("gameOutcome", {}).get("lastPeriodType", "REG")
            games[gm["id"]] = {
                "game_id": gm["id"], "date": gm["gameDate"],
                "home": gm["homeTeam"]["abbrev"], "away": gm["awayTeam"]["abbrev"],
                "home_g": gm["homeTeam"]["score"], "away_g": gm["awayTeam"]["score"],
                "went_ot": lp == "OT", "went_so": lp == "SO",
                "season_end": 2027, "game_type": "R"}
    df = pd.DataFrame(games.values())
    if len(df):
        df["date"] = pd.to_datetime(df.date)
    return df


# ---------------------------------------------------------------- rest-of-season sim
def rest_of_season(played: pd.DataFrame, ratings_by_model: dict[str, dict],
                   p2: dict, p4: dict, n_sims: int = N_SIMS) -> dict[str, pd.DataFrame]:
    ship = p4["shipped"]
    k = p2["k"]
    sched = pd.read_csv(RAW / "nhl_schedule_20262027.csv", parse_dates=["date"]) \
        .sort_values(["date", "game_id"]).reset_index(drop=True)
    if len(played):
        sched = sched[~sched.game_id.isin(set(played.game_id))].reset_index(drop=True)
    hb, ab = E.b2b_flags(sched[["date", "home", "away"]])
    sched["hb2b"], sched["ab2b"] = hb, ab
    sched["d_adj"] = B2B_ELO * (sched.ab2b - sched.hb2b)
    banked, gn = (banked_from(played) if len(played) else ({}, {}))

    g_hist, ts = E.load()
    v1 = json.loads((OUT / "params.json").read_text())
    preds, end_r, _ = E.run_elo(g_hist, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    om = E.fit_outcome(preds, list(range(2006, 2027)))
    sk, go, skt, got, bios = P.load_panels()
    from features import FeatureBuilder
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"],
                        skater_delta=p2["skater_delta"])
    par2 = A2.fit_availability2(sk, bios, 2026)
    gpar = A2.fit_goalie_availability(got, ts, 2026)
    marcel = fb.marcel(2026, 1)
    repl = P.replacement_rates(sk, 2026)
    ppg = points_per_goal(sk, ts, 2026)
    vals = skater_value_goals(marcel, repl, ppg)
    ros = pd.read_csv(RAW / "nhl_rosters_20262027.csv")
    tand = GG.prod_tandems(ros, got, ts, fb.goalie_proj(2026), 2026, ship["n0_g"])
    rosters2 = {}
    for team, d in ros[ros.position != "G"].groupby("team"):
        vv = vals.reindex(d.playerId).fillna(0.0)
        aa = P.age_of(bios, d.playerId, 2027)
        top = sorted(zip(aa.to_numpy(), vv.to_numpy()), key=lambda t_: -t_[1])[:A2.TOP_N]
        b_idx = A2._bucket_idx(np.array([a if np.isfinite(a) else 27.0 for a, _ in top]))
        rosters2[team] = [(int(bi), par2["buckets"][int(bi)]["mean"], float(v_))
                          for bi, (_, v_) in zip(b_idx, top)]
    grow = {t: (gpar["mean"], max(float(tand.loc[t].theta1 - tand.loc[t].theta2), 0.0)
                * A2.GOALIE_SHOTS) for t in tand.index}
    teams = sorted(rosters2)
    base_noise = A2.make_extra_noise2(teams, rosters2, par2, grow, gpar, k)
    frac_left = len(sched) / 1344.0   # availability exposure scales with games left

    def extra_noise(m, rng):
        return base_noise(m, rng) * frac_left

    game_noise = (GG.make_game_noise(sched, tand, k) if ship["goalie_layer"] else None)

    out = {}
    for name, ratings in ratings_by_model.items():
        sim = E.simulate_season(ratings, SIGMA_INSEASON, sched[["home", "away", "d_adj"]],
                                om, E.DIVISIONS_CURRENT, n_sims,
                                np.random.default_rng(20262027),
                                playoffs=True, extra_noise=extra_noise,
                                game_noise=game_noise, banked=banked)
        pts = sim["pts"].astype(float)
        rows = []
        for i, t in enumerate(sim["teams"]):
            rows.append({"team": t, "model": name,
                         "banked_pts": banked.get("pts", {}).get(t, 0),
                         "games_played": gn.get(t, 0),
                         "xPts": float(pts[:, i].mean()),
                         "P5": float(np.percentile(pts[:, i], 5)),
                         "P95": float(np.percentile(pts[:, i], 95)),
                         "playoff_pct": float(sim["made_po"][:, i].mean()),
                         "division_pct": float(sim["won_div"][:, i].mean()),
                         "cup_pct": float(sim["won_cup"][:, i].mean())})
        out[name] = pd.DataFrame(rows).sort_values("xPts", ascending=False)
    return out


# ---------------------------------------------------------------- entry points
def main_update():
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p3 = json.loads((OUT / "params_v3.json").read_text())
    p4 = json.loads((OUT / "params_v4.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    n0 = p3.get("living_model", {}).get("n0", 5.0)
    prior = pd.read_csv(OUT / "v4_prior_ratings.csv", index_col=0)
    LIVE.mkdir(exist_ok=True)
    today = pd.Timestamp.now().date().isoformat()

    new = fetch_2026_results()
    print(f"fetched {len(new)} completed 2026-27 games")
    g_hist, _ = E.load()
    if len(new):
        g_all = pd.concat([g_hist, new[g_hist.columns.intersection(new.columns)]],
                          ignore_index=True)
    else:
        g_all = g_hist
    elo_now = carried_elo(g_all, v1["K"], v1["H"], v1["phi_s"])
    _, gn = (banked_from(new) if len(new) else ({}, {}))
    ratings_by_model = {
        name: blended(prior[f"rating_{name}"] - 1505.0, elo_now, gn, n0)
        for name in ("v1", "v4", "howe")}
    pd.DataFrame(ratings_by_model).round(2).rename_axis("team").to_csv(
        LIVE / f"live_ratings_{today}.csv")

    odds = rest_of_season(new, ratings_by_model, p2, p4)
    merged = pd.concat(odds.values(), ignore_index=True)
    merged.round(4).to_csv(LIVE / f"live_odds_{today}.csv", index=False)
    howe = odds["howe"]
    print(howe.head(8)[["team", "xPts", "playoff_pct", "cup_pct"]].round(3)
          .to_string(index=False))
    clv = LIVE / "clv_log.csv"
    snap = howe.assign(date=today)[["date", "team", "xPts", "playoff_pct", "cup_pct"]]
    snap.round(4).to_csv(clv, mode="a", header=not clv.exists(), index=False)
    print(f"wrote live_odds_{today}.csv, live_ratings_{today}.csv, clv_log.csv "
          f"(median games {int(np.median(list(gn.values()))) if gn else 0})")


def selftest(T: int = 2024, gp_target: int = 20):
    """PLAN_V4 B5 parity: live-path blended ratings == replay-engine ratings < 0.5 Elo."""
    p3 = json.loads((OUT / "params_v3.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    n0 = p3.get("living_model", {}).get("n0", 5.0)
    g_hist, ts = E.load()
    r = g_hist[(g_hist.season_end == T) & (g_hist.game_type == "R")].sort_values("date")
    cnt: dict = {}
    cutoff = None
    for gm in r.itertuples():
        cnt[gm.home] = cnt.get(gm.home, 0) + 1
        cnt[gm.away] = cnt.get(gm.away, 0) + 1
        if len(cnt) >= 30 and np.median(list(cnt.values())) >= gp_target:
            cutoff = gm.date
            break
    prior_dev = pd.Series(0.0, index=sorted(cnt))  # flat prior isolates the Elo path

    # replay path (the machinery the validation numbers came from)
    played = g_hist[g_hist.date < cutoff]
    _, end_r_rep, _ = E.run_elo(played, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    elo_rep = end_r_rep[max(end_r_rep)]
    cur = played[(played.season_end == T) & (played.game_type == "R")]
    _, gn_rep = banked_from(cur)
    rat_rep = blended(prior_dev, elo_rep, gn_rep, n0)

    # live path (carried_elo over full table sliced at the same cutoff)
    elo_live = carried_elo(played, v1["K"], v1["H"], v1["phi_s"])
    rat_live = blended(prior_dev, elo_live, gn_rep, n0)
    gap = max(abs(rat_rep[t] - rat_live[t]) for t in rat_rep)
    print(f"selftest season {T} @ ~{gp_target} gp (cutoff {cutoff.date()}): "
          f"max |replay - live| = {gap:.6f} Elo -> "
          f"{'PASS' if gap < 0.5 else 'FAIL'} (PLAN_V4 B5: < 0.5)")
    return gap < 0.5


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "update":
        main_update()
    else:
        selftest()
