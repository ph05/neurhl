"""M3 (ORR 1.1 plan): calibrate the standings uncertainty.

Hypothesis (orr/PLAN_1_1.md, M3): the season simulator's team-strength SD
comes from a formula (the blend's out-of-sample RMSE net of game luck, plus
in-season drift). A multiplier on that SD, chosen leave-one-season-out by
team-points CRPS, gives better-calibrated season point distributions.

Protocol (fixed before any outcome was scored):
  * seasons: the five clean market seasons 2019, 2020, 2022, 2023, 2024;
  * per season, the preseason ratings are rebuilt exactly as
    orr/backtest/gamefile_bt.py (game_file_season) does with the shipped views
    mkt_rel82 + td_rel82: leave-one-season-out convex blend weights, style,
    calibrate.solve_ratings on the actual schedule with the game model fitted
    on seasons < V, talent SD = sqrt(LOSO RMSE^2 - luck^2), drift from the
    filter's process noise, league level with the Jensen term, ratings
    re-solved at that level;
  * the multiplier k scales the WHOLE team-strength SD: the season-start
    rating SD (o_sd, d_sd) AND the in-season drift SD. The league level is
    re-set for the scaled dispersion (the Jensen term uses k^2 * var), i.e.
    exactly what freeze.main would do with net_sd and drift multiplied by k;
  * k in {0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4}; season.simulate with
    N_SIMS seasons each, the same seed for every k (common random numbers),
    regular season only (playoffs=False);
  * scores per team-season on raw standings points over the games actually
    played (2020 included as played): sample CRPS of the simulated points
    distribution, and 80% coverage of the [p10, p90] band (inclusive, the
    band season.summarise publishes);
  * leave-one-season-out: for each season, k is the grid value with the
    lowest pooled CRPS on the other four seasons; the held-out season is
    scored at that k. Baseline = k = 1.0 (the shipped formula).
  * accept (plan rule): LOSO CRPS better than k = 1.0 by >= 0.05 points AND
    LOSO 80% coverage closer to 0.80 than k = 1.0's.

Run: python3 -m orr.experiments.M3.standings_sd [--smoke]
  --smoke builds one season and runs a small simulation, printing only
  simulated quantities (no actual points are read or scored).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from orr import calibrate as K
from orr import config as C
from orr import freeze as F
from orr import ratings as R
from orr import season as SS
from orr import teams as T
from orr.backtest import gamefile_bt as GB

HERE = Path(__file__).resolve().parent
SEASONS = [2019, 2020, 2022, 2023, 2024]
VIEWS = ["mkt_rel82", "td_rel82"]
MULTS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4]
N_SIMS = 20000
SEED = C.SEED
NB = 10000


# ---------------------------------------------------------------------------
# Ratings: copy of gamefile_bt.game_file_season steps 1-4, split so that the
# SD multiplier is applied before the league level (Jensen) and the re-solve.
# ---------------------------------------------------------------------------
def season_base(V: int, hist: pd.DataFrame, P_V: dict) -> dict:
    """Steps 1-4 up to (not including) the league-level step: schedule,
    adjustments, targets, style, net rating SD and drift at k = 1."""
    sch = GB.schedule(V)
    n_t = pd.concat([sch.home, sch.away]).value_counts()
    teams = sorted(n_t.index)
    P = json.loads(json.dumps(P_V))
    model = SS.FittedModel(P, V)
    adj = model.game_adjustments(sch)
    other = hist[hist.season_end != V]
    w = T.fit_blend(other, VIEWS)
    cur = hist[hist.season_end == V].set_index("team")
    rel82 = pd.Series(cur[VIEWS].to_numpy(float) @ w, index=cur.index).reindex(teams)
    if rel82.isna().any():
        raise ValueError(f"{V}: missing views for {list(rel82[rel82.isna()].index)}")
    targets = rel82 * n_t.reindex(teams) / 82.0
    lo = T.loso_blend(other, VIEWS)
    rmse = float(np.sqrt((lo.blend_rmse ** 2 * lo.n).sum() / lo.n.sum()))
    style = T.predict_style(T.fit_style(V), V, teams)[["team", "o_m", "d_m"]]
    old = (F.V, F.GAMES)
    try:
        F.V, F.GAMES = V, float(n_t.mean())
        r = K.solve_ratings(sch, targets, style, model, adj)
        luck = F.luck_sd_82(sch, r, model, adj)
        talent = float(np.sqrt(max(rmse ** 2 - luck ** 2, 1.0)))
        hp = R.load_hp()
        drift = float(np.sqrt(190 * (hp.q_s + hp.q_f)))
        slope = K.points_per_rating(sch, r[["team", "o", "d"]], model, adj)
        net_sd = talent * n_t.reindex(r.team).to_numpy() / 82.0 / slope.reindex(r.team).to_numpy()
        net_sd = pd.Series(net_sd, index=r.team.to_numpy())
    finally:
        F.V, F.GAMES = old
    return {"V": V, "sch": sch, "n_t": n_t, "teams": teams, "P": P, "adj": adj,
            "targets": targets, "style": style, "net_sd": net_sd, "drift": drift,
            "log": {"weights": dict(zip(VIEWS, map(float, w))), "loso_rmse_other_82": rmse,
                    "luck_sd_82": luck, "talent_sd_82": talent, "drift_sd": drift,
                    "rating_sd_net_mean": float(net_sd.mean()),
                    "games_per_team_mean": float(n_t.mean())}}


def season_ratings(b: dict, k: float) -> tuple:
    """Step 4 tail at multiplier k: league level with the Jensen term for the
    scaled dispersion, ratings re-solved, o_sd = d_sd = k * net_sd / sqrt(2),
    drift k * drift. Returns (model, ratings, drift_sd, mu)."""
    V, sch, adj = b["V"], b["sch"], b["adj"]
    P = json.loads(json.dumps(b["P"]))           # same starting mu for every k
    model = SS.FittedModel(P, V)
    net_sd = b["net_sd"].to_numpy() * k
    drift = b["drift"] * k
    var = float(np.mean(net_sd ** 2) / 2 * 2 + 2 * drift ** 2 / 6)   # as gamefile_bt / freeze
    old = (F.V, F.GAMES)
    try:
        F.V, F.GAMES = V, float(b["n_t"].mean())
        P["mu"] = F.league_level_mu(model, sch, adj, {}, var)
        model = SS.FittedModel(P, V)
        r = K.solve_ratings(sch, b["targets"], b["style"], model, adj)
    finally:
        F.V, F.GAMES = old
    r = r.set_index("team").reindex(b["net_sd"].index).reset_index().rename(columns={"index": "team"})
    r["o_sd"] = net_sd / np.sqrt(2)
    r["d_sd"] = net_sd / np.sqrt(2)
    return model, r[["team", "o", "d", "o_sd", "d_sd"]], drift, float(P["mu"])


def simulate_points(b: dict, k: float, n_sims: int) -> tuple:
    model, r, drift, mu = season_ratings(b, k)
    res = SS.simulate(b["sch"], r, model, n_sims=n_sims, seed=SEED, drift_sd=drift,
                      game_adj=b["adj"], playoffs=False)
    gp = res.gp[0]
    assert (res.gp == gp).all()
    return res.teams, res.points, gp, mu


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def crps_sample(x: np.ndarray, y: float) -> float:
    """CRPS of the empirical distribution of x at y: E|X-y| - E|X-X'|/2."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    i = np.arange(1, n + 1)
    e_xx = 2.0 * np.sum((2 * i - n - 1) * x) / n ** 2
    return float(np.mean(np.abs(x - y)) - 0.5 * e_xx)


def score_rows(teams, pts, actual: pd.Series, V: int, k: float) -> list:
    rows = []
    for j, t in enumerate(teams):
        x = pts[:, j]
        y = float(actual[t])
        p10, p90 = np.percentile(x, 10), np.percentile(x, 90)
        pit_mid = float((x < y).mean() + 0.5 * (x == y).mean())
        rows.append({"season": V, "k": k, "team": t, "actual": y,
                     "sim_mean": float(x.mean()), "sim_sd": float(x.std()),
                     "p10": float(p10), "p90": float(p90),
                     "crps": crps_sample(x, y), "cover80": bool(p10 <= y <= p90),
                     "pit_mid": pit_mid})
    return rows


def loso(scores: pd.DataFrame) -> dict:
    """Leave-one-season-out choice of k by pooled CRPS on the other seasons."""
    chosen, held = {}, []
    for V in SEASONS:
        tr = scores[scores.season != V].groupby("k").crps.mean()
        k = float(tr.idxmin())
        chosen[V] = {"k": k, "train_crps_by_k": {str(kk): float(v) for kk, v in tr.items()}}
        held.append(scores[(scores.season == V) & (scores.k == k)])
    h = pd.concat(held, ignore_index=True)
    return {"chosen": chosen, "rows": h}


def boot_diff(a: np.ndarray, b: np.ndarray, nb: int = NB, seed: int = 7) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (nb, len(d)))].mean(1)
    return {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def boot_cov_gap(ca: np.ndarray, cb: np.ndarray, nb: int = NB, seed: int = 8) -> dict:
    """|cov_a - 0.8| - |cov_b - 0.8| with a paired bootstrap CI."""
    ca, cb = np.asarray(ca, float), np.asarray(cb, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ca), (nb, len(ca)))
    g = np.abs(ca[idx].mean(1) - 0.8) - np.abs(cb[idx].mean(1) - 0.8)
    return {"diff": float(abs(ca.mean() - 0.8) - abs(cb.mean() - 0.8)),
            "ci95": [float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975))]}


# ---------------------------------------------------------------------------
def build_all(seasons) -> dict:
    hp = R.load_hp()
    hist = GB.hist_frame()
    out = {}
    for V in seasons:
        t0 = time.time()
        P_V = R.fit_gamemodel_params(V, hp, write=False)
        out[V] = season_base(V, hist, P_V)
        print(f"[{V}] base built in {time.time() - t0:.0f}s: {out[V]['log']}", flush=True)
    return out


def smoke():
    """Pipeline check on 2019 with 300 simulations. Reads no actual points."""
    b = build_all([2019])[2019]
    for k in (0.7, 1.0, 1.4):
        t0 = time.time()
        teams, pts, gp, mu = simulate_points(b, k, 300)
        print(f"k={k}: {time.time() - t0:.1f}s mu={mu:.4f} gp={sorted(set(gp))} "
              f"league mean pts {pts.mean():.2f}, mean team sd {pts.std(0).mean():.2f}, "
              f"sd of team means {pts.mean(0).std():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sims", type=int, default=N_SIMS)
    a = ap.parse_args()
    if a.smoke:
        smoke()
        return
    t_start = time.time()
    bases = build_all(SEASONS)
    st = T.standings_all()
    rows, sim_logs = [], {}
    for V in SEASONS:
        b = bases[V]
        act = st[st.season_end == V].set_index("team")
        # the simulated schedule must be exactly the games in the standings
        assert (act.gp.reindex(b["teams"]).to_numpy() == b["n_t"].reindex(b["teams"]).to_numpy()).all(), V
        sim_logs[V] = {"base": b["log"], "by_k": {}}
        for k in MULTS:
            t0 = time.time()
            teams, pts, gp, mu = simulate_points(b, k, a.sims)
            assert (gp == act.gp.reindex(teams).to_numpy()).all()
            rr = score_rows(teams, pts, act.pts, V, k)
            rows += rr
            d = pd.DataFrame(rr)
            sim_logs[V]["by_k"][str(k)] = {"mu": mu, "crps": float(d.crps.mean()),
                                           "cover80": float(d.cover80.mean()),
                                           "mean_sim_sd": float(d.sim_sd.mean()),
                                           "seconds": round(time.time() - t0, 1)}
            print(f"[{V}] k={k:<5} crps {d.crps.mean():.3f}  cover80 {d.cover80.mean():.3f}  "
                  f"sim sd {d.sim_sd.mean():.2f}  ({time.time() - t0:.0f}s)", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(HERE / "scores_by_team_k.csv.gz", index=False, float_format="%.5f")

    # pooled by k
    by_k = scores.groupby("k").agg(crps=("crps", "mean"), cover80=("cover80", "mean"),
                                   sim_sd=("sim_sd", "mean"),
                                   pit_mid_cover80=("pit_mid", lambda p: float(((p >= 0.1) & (p <= 0.9)).mean())))
    by_season_k = scores.pivot_table(index="season", columns="k", values="crps")
    cov_season_k = scores.pivot_table(index="season", columns="k", values="cover80")
    L = loso(scores)
    h = L["rows"].sort_values(["season", "team"]).reset_index(drop=True)
    base = scores[scores.k == 1.0].sort_values(["season", "team"]).reset_index(drop=True)
    assert (h.team.to_numpy() == base.team.to_numpy()).all()
    crps_loso, crps_base = float(h.crps.mean()), float(base.crps.mean())
    cov_loso, cov_base = float(h.cover80.mean()), float(base.cover80.mean())
    d_crps = boot_diff(h.crps.to_numpy(), base.crps.to_numpy())
    d_cov = boot_cov_gap(h.cover80.to_numpy(), base.cover80.to_numpy())
    gain = crps_base - crps_loso
    cond1 = gain >= 0.05
    cond2 = abs(cov_loso - 0.8) < abs(cov_base - 0.8)
    accepted = bool(cond1 and cond2)
    k_all = float(scores.groupby("k").crps.mean().idxmin())
    # dispersion diagnostics at k = 1: actual minus simulated mean, z-scores
    z = (base.actual - base.sim_mean) / base.sim_sd
    res = {
        "id": "M3", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": __doc__, "seasons": SEASONS, "views": VIEWS, "multipliers": MULTS,
        "n_sims": a.sims, "seed": SEED, "n_team_seasons": int(len(base)),
        "pooled_by_k": {str(k): {kk: float(v) for kk, v in r.items()} for k, r in by_k.iterrows()},
        "crps_by_season_k": {int(s): {str(k): float(v) for k, v in r.items()} for s, r in by_season_k.iterrows()},
        "cover80_by_season_k": {int(s): {str(k): float(v) for k, v in r.items()} for s, r in cov_season_k.iterrows()},
        "loso_choice": {int(V): c for V, c in L["chosen"].items()},
        "loso": {"crps": crps_loso, "cover80": cov_loso,
                 "by_season": {int(V): {"k": L["chosen"][V]["k"],
                                        "crps": float(h[h.season == V].crps.mean()),
                                        "cover80": float(h[h.season == V].cover80.mean())}
                               for V in SEASONS}},
        "baseline_k1": {"crps": crps_base, "cover80": cov_base,
                        "by_season": {int(V): {"crps": float(base[base.season == V].crps.mean()),
                                               "cover80": float(base[base.season == V].cover80.mean())}
                                      for V in SEASONS}},
        "crps_loso_minus_base": d_crps,
        "cover_gap_loso_minus_base": d_cov,
        "k_chosen_all_seasons": k_all,
        "diagnostics_k1": {"z_sd": float(z.std()), "z_mean": float(z.mean()),
                           "mae_mean": float((base.actual - base.sim_mean).abs().mean()),
                           "rmse_mean": float(np.sqrt(((base.actual - base.sim_mean) ** 2).mean())),
                           "mean_sim_sd": float(base.sim_sd.mean()),
                           "league_bias_by_season": {int(V): float((x.actual - x.sim_mean).mean())
                                                     for V, x in base.groupby("season")}},
        "rule": {"text": "Accept if the leave-one-season-out CRPS improves by at least 0.05 points "
                         "and 80% coverage moves closer to 0.80.",
                 "crps_gain": gain, "crps_gain_ge_0.05": bool(cond1),
                 "coverage_closer": bool(cond2), "accepted": accepted},
        "season_logs": {int(k): v for k, v in sim_logs.items()},
        "wall_seconds": round(time.time() - t_start),
    }
    (HERE / "result_raw.json").write_text(json.dumps(res, indent=1, default=float))
    print(by_k.round(4).to_string())
    print(by_season_k.round(3).to_string())
    print(json.dumps({k: res[k] for k in ("loso_choice", "loso", "baseline_k1", "crps_loso_minus_base",
                                          "cover_gap_loso_minus_base", "k_chosen_all_seasons",
                                          "diagnostics_k1", "rule")}, indent=1, default=float))


if __name__ == "__main__":
    main()
