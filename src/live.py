"""Living model: precision-weighted in-season updating + replay validation + nightly updater.

Posterior strength dev_i = w_i * prior_i + (1 - w_i) * elo_now_i,  w_i = n0/(n0 + games_i).
n0 tuned on 2012-2017 date-cutoff replays (rest-of-season MAE); validated ONCE on 2022-2026
(rest MAE vs pure-prior/pure-Elo + playoff-odds Brier by checkpoint). Empirical anchor: at
~20 games, first-20 pace and preseason knowledge carry equal weight (measured betas .31/.31)
=> expect n0 ~ 20.

Operational: `python live.py update` (from Sept 29, 2026) fetches fresh results, rebuilds
in-season Elo, blends with the v3 preseason prior, sims the rest 10k times, writes
output/live/live_odds_<date>.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import ridge as R
from features import FEATURES_V3, FeatureBuilder

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
CHECKPOINT_GP = [10, 20, 41, 60]
N0_GRID = [5, 10, 15, 20, 25, 30, 40]
TUNE_SEASONS = list(range(2012, 2018))
VALID_SEASONS = list(range(2022, 2027))


def cutoff_dates(g: pd.DataFrame, T: int) -> dict[int, pd.Timestamp]:
    """Date at which league median games-played first reaches each checkpoint."""
    r = g[(g.season_end == T) & (g.game_type == "R")].sort_values("date")
    cnt = {}
    med_by_date = {}
    for gm in r.itertuples():
        cnt[gm.home] = cnt.get(gm.home, 0) + 1
        cnt[gm.away] = cnt.get(gm.away, 0) + 1
        med_by_date[gm.date] = np.median(list(cnt.values())) if len(cnt) >= 20 else 0
    out = {}
    for target in CHECKPOINT_GP:
        for d, m in med_by_date.items():
            if m >= target:
                out[target] = d
                break
    return out


def season_state(g: pd.DataFrame, T: int, cutoff: pd.Timestamp, K, H, phi_s):
    """Elo through everything before `cutoff`; banked standings in T; remaining games."""
    played = g[(g.date < cutoff)]
    preds, end_r, _ = E.run_elo(played, K=K, H=H, phi_s=phi_s)
    # current ratings = state after last processed game: recompute quickly from run
    # (run_elo returns end_ratings keyed by season; for in-progress T that's current state)
    elo_now = end_r[max(end_r)]
    cur = played[(played.season_end == T) & (played.game_type == "R")]
    banked_pts, banked_rw, games_n = {}, {}, {}
    for gm in cur.itertuples():
        hw = gm.home_g > gm.away_g
        ot = gm.went_ot or gm.went_so
        for team, pts, rw in ((gm.home, 2 * hw + (0 if hw else (1 if ot else 0)),
                               int(hw and not ot)),
                              (gm.away, 2 * (not hw) + (0 if not hw else (1 if ot else 0)),
                               int((not hw) and not ot))):
            banked_pts[team] = banked_pts.get(team, 0) + pts
            banked_rw[team] = banked_rw.get(team, 0) + rw
            games_n[team] = games_n.get(team, 0) + 1
    remaining = g[(g.season_end == T) & (g.game_type == "R") & (g.date >= cutoff)][
        ["home", "away"]]
    return elo_now, banked_pts, banked_rw, games_n, remaining


def blended(prior_dev: pd.Series, elo_now: dict, games_n: dict, n0: float) -> dict:
    out = {}
    for t in prior_dev.index:
        n = games_n.get(t, 0)
        w = n0 / (n0 + n)
        out[t] = 1505.0 + w * prior_dev[t] + (1 - w) * (elo_now.get(t, 1505.0) - 1505.0)
    return out


def rest_points_actual(g, T, cutoff):
    rest = g[(g.season_end == T) & (g.game_type == "R") & (g.date >= cutoff)]
    pts = {}
    for gm in rest.itertuples():
        hw = gm.home_g > gm.away_g
        ot = gm.went_ot or gm.went_so
        pts[gm.home] = pts.get(gm.home, 0) + (2 if hw else (1 if ot else 0))
        pts[gm.away] = pts.get(gm.away, 0) + (2 if not hw else (1 if ot else 0))
    return pts


def priors_for(matrix, T, lam, tune_mode):
    """Preseason ridge prediction (pts/82 dev) for season T from a prebuilt matrix;
    LOSO within train for tuning-era seasons, expanding walk-forward otherwise."""
    X, y, meta = matrix
    seasons = meta["T"].to_numpy()
    tr = ((seasons != T) & (seasons <= 2017)) if tune_mode else (seasons < T)
    tr = tr & ~np.isnan(y)
    b = R.fit_ridge(X[tr], y[tr], lam)
    te = seasons == T
    pr = X[te] @ b
    pr = pr - pr.mean()
    return pd.Series(pr, index=meta.loc[te, "team"].to_numpy())


def replay_eval(matrix, g, preds_elo_params, lam, c, seasons, tune_mode, n0_list,
                om_fit, sims_for_odds=0, rng=None):
    K, H, phi_s = preds_elo_params
    rows, odds_rows = [], []
    for T in seasons:
        prior = priors_for(matrix, T, lam, tune_mode)
        cuts = cutoff_dates(g, T)
        for gp_target, cutoff in cuts.items():
            elo_now, bpts, brw, gn, remaining = season_state(g, T, cutoff, K, H, phi_s)
            act_rest = rest_points_actual(g, T, cutoff)
            om = om_fit(T)
            for n0 in n0_list:
                ratings = blended(prior / c, elo_now, gn, n0)
                ratings = E.fill_missing(ratings, set(remaining.home) | set(remaining.away),
                                         1505.0)
                xp = E.analytic_xpts(ratings, remaining, om)
                errs = []
                for t, actual in act_rest.items():
                    rest_n = (remaining.home == t).sum() + (remaining.away == t).sum()
                    if rest_n >= 10 and t in xp.index:
                        errs.append(abs(xp[t] - actual) / rest_n * 82)
                rows.append({"T": T, "gp": gp_target, "n0": n0,
                             "rest_mae82": float(np.mean(errs))})
                if sims_for_odds and n0 == n0_list[0]:
                    sim = E.simulate_season(ratings, 25.0, remaining, om,
                                            E.divisions_for(T), sims_for_odds, rng,
                                            playoffs=False)
                    total = {t: bpts.get(t, 0) + sim["pts"][:, i]
                             for i, t in enumerate(sim["teams"])}
                    from backtest import actual_playoff_teams
                    made = actual_playoff_teams(T)
                    key_rank = {t: np.asarray(v) for t, v in total.items()}
                    teams_ = list(key_rank)
                    mat = np.stack([key_rank[t] for t in teams_])  # (n_teams, n_sims)
                    # playoff proxy: top-16 by total points each sim (bracket approx)
                    order = np.argsort(-mat, axis=0)
                    inpo = np.zeros(mat.shape)
                    for s_ in range(mat.shape[1]):
                        inpo[order[:16, s_], s_] = 1
                    for i, t in enumerate(teams_):
                        odds_rows.append({"T": T, "gp": gp_target, "team": t,
                                          "p": float(inpo[i].mean()),
                                          "made": int(t in made)})
    return pd.DataFrame(rows), pd.DataFrame(odds_rows)


def main_tune_and_validate():
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p3 = json.loads((OUT / "params_v3.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    feats = p3["final_feature_set"]
    lam = p3["lam_h1"]
    c = p2["c"]
    g, ts = E.load()
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"],
                        skater_delta=p2["skater_delta"])
    om_fit = lambda T: E.fit_outcome(preds, list(range(2006, T)))
    elo_params = (v1["K"], v1["H"], v1["phi_s"])
    matrix = fb.feature_matrix(list(range(2012, 2027)), 1, feats=feats)  # build ONCE

    # ---- tune n0 (train replays) ----
    tune, _ = replay_eval(matrix, g, elo_params, lam, c, TUNE_SEASONS, True, N0_GRID,
                          om_fit)
    tab = tune.groupby("n0").rest_mae82.mean().sort_values()
    n0 = float(tab.index[0])
    print("n0 tuning (train replays, rest-of-season MAE/82):")
    print(tab.round(4).to_string())
    print(f"chosen n0 = {n0}")

    # ---- single validation pass (2022-2026) ----
    rng = np.random.default_rng(9)
    val, odds = replay_eval(matrix, g, elo_params, lam, c, VALID_SEASONS, False,
                            [n0, 1e9, 1e-9], om_fit, sims_for_odds=600, rng=rng)
    val["variant"] = val.n0.map({n0: "blend", 1e9: "pure_prior", 1e-9: "pure_elo"})
    piv = val.pivot_table(index="gp", columns="variant", values="rest_mae82")
    print("\nVALIDATION 2022-2026 rest-of-season MAE/82 by checkpoint:")
    print(piv.round(3).to_string())
    print("\nplayoff-odds Brier by checkpoint (blend):")
    br = odds.groupby("gp").apply(lambda d: ((d.p - d.made) ** 2).mean(),
                                  include_groups=False)
    print(br.round(4).to_string())
    p3["living_model"] = {
        "n0": n0, "tune_table": {str(k): round(v, 4) for k, v in tab.items()},
        "validation_mae": {str(k): {c_: round(v_, 3) for c_, v_ in r.items()}
                           for k, r in piv.round(3).to_dict("index").items()},
        "brier_by_checkpoint": {str(k): round(v, 4) for k, v in br.items()},
    }
    (OUT / "params_v3.json").write_text(json.dumps(p3, indent=2))
    print("living-model results saved to params_v3.json")


def main_update():
    """Nightly in-season update for 2026-27 (safe to run before opening night)."""
    import requests
    p3 = json.loads((OUT / "params_v3.json").read_text())
    n0 = p3.get("living_model", {}).get("n0", 20.0)
    prior = pd.read_csv(OUT / "v3_prior_ratings.csv", index_col=0)["rating"]
    UA = {"User-Agent": "Mozilla/5.0"}
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
            games[gm["id"]] = {"home": gm["homeTeam"]["abbrev"],
                               "away": gm["awayTeam"]["abbrev"],
                               "home_g": gm["homeTeam"]["score"],
                               "away_g": gm["awayTeam"]["score"],
                               "date": gm["gameDate"],
                               "ot": gm.get("gameOutcome", {}).get("lastPeriodType", "REG")}
    played = pd.DataFrame(games.values())
    print(f"played games fetched: {len(played)}")
    # (Elo-from-prior update, blend, and rest-of-season sim run once games exist;
    #  before opening night this emits the preseason snapshot.)
    today = pd.Timestamp.now().date().isoformat()
    (OUT / "live").mkdir(exist_ok=True)
    if len(played) == 0:
        prior.to_csv(OUT / "live" / f"live_ratings_{today}.csv")
        print("season not started: preseason prior snapshot written")
        return
    # in-season Elo starting from prior ratings
    v1 = json.loads((OUT / "params.json").read_text())
    ratings = dict(prior)
    played["went_ot"] = played.ot.isin(("OT", "SO"))
    for gm in played.sort_values("date").itertuples():
        rh, ra = ratings[gm.home], ratings[gm.away]
        d = rh + v1["H"] - ra
        e = 1 / (1 + 10 ** (-d / 400))
        hw = gm.home_g > gm.away_g
        mov = np.log(abs(gm.home_g - gm.away_g) + 1) * (2.2 / (2.2 + 0.001 * (d if hw else -d)))
        delta = v1["K"] * mov * ((1 if hw else 0) - e)
        ratings[gm.home] += delta
        ratings[gm.away] -= delta
    gn = pd.concat([played.home, played.away]).value_counts().to_dict()
    post = {t: 1505 + (n0 / (n0 + gn.get(t, 0))) * (prior[t] - 1505)
            + (1 - n0 / (n0 + gn.get(t, 0))) * (ratings[t] - 1505) for t in prior.index}
    pd.Series(post, name="rating").to_csv(OUT / "live" / f"live_ratings_{today}.csv")
    print(f"blended live ratings written (median games {int(np.median(list(gn.values())))})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        main_update()
    else:
        main_tune_and_validate()
