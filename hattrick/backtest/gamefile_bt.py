"""Backtest of the SHIPPED preseason game-file pipeline (hattrick/freeze.py).

The preseason-frozen numbers in games_bt.json validate
ratings.preseason_table (team history only, plug-in probabilities). The
2026-27 game file is built differently, and this script rebuilds it for past
seasons exactly the way freeze.py does, reusing its functions:

  1. targets  = blend of team views (points per 82 above the league mean):
                teams.market_rel82 (+ the teams top-down view td as shipped,
                or + the bottom-up roster view bu), convex weights from
                teams.fit_blend fitted on the OTHER clean market seasons
                (leave one season out);
  2. style    = teams.fit_style / predict_style (walk-forward);
  3. ratings  = calibrate.solve_ratings on the season's ACTUAL schedule with
                season.FittedModel(P_V), P_V = the game-model parameters
                fitted from seasons < V (ratings.fit_gamemodel_params), the
                league level set by freeze.league_level_mu (previous season's
                goals per game), rest/travel offsets from
                FittedModel.game_adjustments;
  4. rating SD from the blend's out-of-sample RMSE (leave-one-season-out on
     the other seasons) net of the game luck the scoring model implies
     (freeze.luck_sd_82), converted with calibrate.points_per_rating, split
     equally between o and d; league level by freeze.league_level_mu with
     the Jensen term for that dispersion and in-season drift; ratings
     re-solved at that level (the order of freeze.main);
  5. probabilities via freeze.game_file (averaged over rating draws).

Seasons: 2019, 2020, 2022, 2023, 2024 (market lines + clean first-10 roster
proxies). Scored against ratings.preseason_table frozen predictions and the
preseason Elo freeze on the same games, paired bootstrap CIs.

Differences from the 2027 freeze (stated, not hidden):
  * no 'news' adjustment (market line moves after the posting date are not
    available historically) and no Hellebuyck-type scenario;
  * no team-specific backup-on-back-to-back term (no historical goalie depth
    projections); the league-average backup effect is in the rest
    coefficients;
  * historical lines are 82-game late-preseason lines (no O/U prices), used
    as their own mean.
  * the bottom-up view's first-10-games roster proxy knows who dressed in a
    season's first 10 games (mild leakage for those games).

Run: python3 -m hattrick.backtest.gamefile_bt
Writes hattrick/output/backtest/gamefile_bt.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from hattrick import calibrate as K
from hattrick import config as C
from hattrick import freeze as F
from hattrick import ratings as R
from hattrick import season as SS
from hattrick import structural as S
from hattrick import teams as T
from hattrick.backtest.games_bt import paired

SEASONS = [2019, 2020, 2022, 2023, 2024]
VIEWS = {"mkt": ["mkt_rel82"], "mkt+td (shipped)": ["mkt_rel82", "td_rel82"],
         "mkt+bu": ["mkt_rel82", "bu_rel82"]}
OUT = C.OUT / "backtest"


def hist_frame() -> pd.DataFrame:
    """Per team-season views for the clean market seasons (as freeze.py)."""
    from hattrick import team_points_map as TPM
    h = T.blend_frame(SEASONS)
    bu_hist = pd.read_csv(C.OUT / "backtest" / "team_components_hist.csv")
    return h.merge(TPM.walk_forward_points(bu_hist), on=["team", "season_end"], how="left")


def schedule(V: int) -> pd.DataFrame:
    g = S.game_frame()
    g = g[g.season_end == V]
    return pd.DataFrame({"game_id": g.gid.to_numpy(), "date": g.date.to_numpy(),
                         "home": g.home.to_numpy(), "away": g.away.to_numpy()})


def game_file_season(V: int, hist: pd.DataFrame, views: list, P_V: dict) -> tuple[pd.DataFrame, dict]:
    """Mirror of freeze.main steps 1-5 for a past season (same order and the
    same leaf functions: solve_ratings, luck_sd_82, points_per_rating,
    league_level_mu, solve_ratings again, game_file). freeze.py's module
    constants V and GAMES are set to the past season for the calls that read
    them, and restored afterwards."""
    log: dict = {}
    sch = schedule(V)
    n_t = pd.concat([sch.home, sch.away]).value_counts()
    teams = sorted(n_t.index)
    P = json.loads(json.dumps(P_V))
    model = SS.FittedModel(P, V)
    adj = model.game_adjustments(sch)
    # targets: leave-one-season-out blend weights from the other seasons
    other = hist[hist.season_end != V]
    w = T.fit_blend(other, views)
    cur = hist[hist.season_end == V].set_index("team")
    rel82 = pd.Series(cur[views].to_numpy(float) @ w, index=cur.index).reindex(teams)
    if rel82.isna().any():
        raise ValueError(f"{V}: missing views for {list(rel82[rel82.isna()].index)}")
    targets = rel82 * n_t.reindex(teams) / 82.0          # this schedule's games
    lo = T.loso_blend(other, views)
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
        var = float(np.mean(net_sd ** 2) / 2 * 2 + 2 * drift ** 2 / 6)
        P["mu"] = F.league_level_mu(model, sch, adj, log, var)
        model = SS.FittedModel(P, V)
        r = K.solve_ratings(sch, targets, style, model, adj)
        r["o_sd"] = net_sd / np.sqrt(2)
        r["d_sd"] = net_sd / np.sqrt(2)
        games = F.game_file(sch, r, model, adj)
    finally:
        F.V, F.GAMES = old
    log.update({"weights": dict(zip(views, map(float, w))), "loso_rmse_other_82": rmse,
                "luck_sd_82": luck, "talent_sd_82": talent,
                "rating_sd_net_mean": float(np.mean(net_sd)), "mu": P["mu"],
                "max_target_miss": float((r.target - r.expected).abs().max())})
    return games.rename(columns={"game_id": "gid"}), log


def main():
    hp = R.load_hp()
    hist = hist_frame()
    preds = pd.read_csv(OUT / "games_bt_preds.csv.gz")
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocol": __doc__, "seasons": SEASONS, "variants": {}}
    params = {V: R.fit_gamemodel_params(V, hp, write=False) for V in SEASONS}
    for name, views in VIEWS.items():
        frames, logs = [], {}
        for V in SEASONS:
            gf, lg = game_file_season(V, hist, views, params[V])
            frames.append(gf.assign(season_end=V))
            logs[V] = lg
        gfa = pd.concat(frames, ignore_index=True)
        m = gfa[["gid", "season_end", "p_home_win"]].merge(
            preds[["gid", "home_win", "ht_frozen_p_home_win", "p_elo_frozen",
                   "ht_p_home_win"]], on="gid")
        assert len(m) == len(gfa)
        rows = []
        for V, x in list(m.groupby("season_end")) + [("all", m)]:
            y = x.home_win.to_numpy()
            rows.append({
                "season": V if V == "all" else int(V), "n": int(len(x)),
                "gamefile": R.logloss(x.p_home_win, y),
                "preseason_table_frozen": R.logloss(x.ht_frozen_p_home_win, y),
                "elo_frozen": R.logloss(x.p_elo_frozen, y),
                "hattrick_inseason_ref": R.logloss(x.ht_p_home_win, y),
                "d_gamefile_vs_preseason_table": paired(x.p_home_win, x.ht_frozen_p_home_win, y),
                "d_gamefile_vs_elo_frozen": paired(x.p_home_win, x.p_elo_frozen, y),
                "logit_sd_gamefile": float(np.log(x.p_home_win / (1 - x.p_home_win)).std()),
                "logit_sd_preseason_table": float(np.log(x.ht_frozen_p_home_win
                                                         / (1 - x.ht_frozen_p_home_win)).std())})
        res["variants"][name] = {"views": views, "by_season": rows,
                                 "season_logs": {int(k): v for k, v in logs.items()}}
        print(f"\n[{name}]  views={views}")
        print(f"{'season':>7} {'n':>5} {'gamefile':>9} {'pre_table':>9} {'elo_frz':>8} "
              f"{'(in-season)':>11}  {'GF - table [95% CI]':>26}  {'GF - Elo [95% CI]':>26}")
        for r_ in rows:
            d1, d2 = r_["d_gamefile_vs_preseason_table"], r_["d_gamefile_vs_elo_frozen"]
            print(f"{str(r_['season']):>7} {r_['n']:>5} {r_['gamefile']:.4f}    {r_['preseason_table_frozen']:.4f}"
                  f"    {r_['elo_frozen']:.4f}   {r_['hattrick_inseason_ref']:.4f}    "
                  f"{d1['diff']:+.4f} [{d1['ci95'][0]:+.4f},{d1['ci95'][1]:+.4f}]  "
                  f"{d2['diff']:+.4f} [{d2['ci95'][0]:+.4f},{d2['ci95'][1]:+.4f}]")
    (OUT / "gamefile_bt.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
