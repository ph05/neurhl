"""ORR game-level backtest against NeurHL on NeurHL's own game sets.

Protocol
  * Hyperparameters were tuned on 2012, 2014-2017 only (tune_games.py,
    ledger games_search_ledger.json) and are read frozen from
    orr/output/params/ratings_hp.json / elo_hp.json.
  * Test seasons 2018-2026: every prediction is walk-forward. Structural
    parameters (home ice, rest/travel, goalie coefficient, end-game layer,
    OT/SO), the preseason regression and the filter state are all refit or
    read from seasons / game days strictly before the prediction.
  * IN-SEASON: every game of a date is predicted from the state at the end
    of the previous date.
  * PRESEASON-FROZEN: every game of season V is predicted with information
    from before V only (the season's opening-night state), no updates. Rest
    and travel come from the schedule, which is known preseason; no
    starting goalies.
  * Two in-season variants: NO-GOALIE (morning / no-lineup information) and
    GOALIE-KNOWN (actual starting goalies, the like-for-like comparison with
    NeurHL's actual-lineup backtests). Starters are only available through
    2022-23 (and 2023-24 until 2023-11-07); later games fall back to the
    no-goalie prediction, which is flagged in every table.
  * NeurHL numbers are its published held-out predictions, built with the
    ACTUAL dressed skaters and ACTUAL starting goalies: labelled
    "NeurHL (actual lineups + starters)".

Run: python3 -m orr.backtest.games_bt
Writes orr/output/backtest/games_bt.json and games_bt_preds.csv.gz.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from orr import config as C
from orr import gamemodel as GM
from orr import ratings as R
from orr import structural as S
from orr.gametable import PREDS

OUT = C.OUT / "backtest"
TEST = list(range(2018, 2027))
FROZEN_SEASONS = list(range(2012, 2027))
NB = 2000
RNG = np.random.default_rng(C.SEED)


def _ll_vec(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(p_a, p_b, y, nb: int = NB) -> dict:
    """Mean per-game log-loss difference a - b with a paired bootstrap 95% CI."""
    d = _ll_vec(p_a, y) - _ll_vec(p_b, y)
    n = len(d)
    idx = RNG.integers(0, n, (nb, n))
    bs = d[idx].mean(1)
    return {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(n)),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def calib(p, y) -> dict:
    """Logistic recalibration (intercept, slope) and decile table."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    lg = np.log(p / (1 - p))

    def f(z):
        q = np.clip(1 / (1 + np.exp(-(z[0] + z[1] * lg))), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    a, b = minimize(f, [0.0, 1.0]).x
    dec = pd.DataFrame({"p": p, "y": y})
    dec["bin"] = pd.qcut(dec.p, 10, labels=False, duplicates="drop")
    t = dec.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return {"intercept": float(a), "slope": float(b),
            "deciles": t.round(4).reset_index(drop=True).to_dict("records")}


# ---------------------------------------------------------------------------
def build_predictions() -> pd.DataFrame:
    hp = R.load_hp()
    ehp_file = C.PARAMS / "elo_hp.json"
    ehp = R.EloHP(**json.loads(ehp_file.read_text())["hp"]) if ehp_file.exists() else R.EloHP()
    g = S.game_frame()
    frames = {}
    for tag, gk in (("ht", False), ("ht_gk", True)):
        pred = R.run_filter(hp, FROZEN_SEASONS, use_goalie=gk)
        pred = pred[pred.season_end >= 2010]
        ot = {}
        pin = R.predict_probs(pred, hp, ot_params=ot)
        frames[tag] = pin
        if not gk:
            frames["ht_frozen"] = R.predict_probs(pred, hp, frozen=True, ot_params=ot)
            frames["ot_params"] = ot
    e = R.elo_run(ehp)
    base = g[["gid", "game_id", "date", "season_end", "home", "away", "home_win",
              "extra", "reg_h", "reg_a", "home_g", "away_g", "gk_h", "gk_a"]]
    out = base.merge(e, on="gid", how="left")
    for tag in ("ht", "ht_gk", "ht_frozen"):
        f = frames[tag].drop(columns="season_end").rename(
            columns=lambda c: c if c == "gid" else f"{tag}_{c}")
        out = out.merge(f, on="gid", how="left")
    gt = S.goalie_game_talent(*hp.goalie)
    out = out.merge(gt[["gid", "gdiff_h", "gdiff_a"]], on="gid", how="left")
    out["gk_known"] = out.gdiff_h.notna() & out.gdiff_a.notna()
    out = out[out.season_end.isin(FROZEN_SEASONS)].reset_index(drop=True)
    out.attrs["ot_params"] = frames["ot_params"]
    out.attrs["hp"] = hp
    out.attrs["ehp"] = ehp
    return out


def neurhl_sets() -> dict:
    h = pd.read_csv(PREDS / "hier_restatement_games.csv")
    gg = pd.read_csv(PREDS / "g_gate_games.csv")
    gs = pd.read_csv(PREDS / "g_seal_games.csv")
    return {"hier_restatement": (h, "p_neurhl_h", "season"),
            "g_gate": (gg, "p_stack", "season_end"),
            "g_seal": (gs, "p_g", "season_end")}


def compare_sets(pr: pd.DataFrame) -> dict:
    out = {}
    for name, (nf, col, scol) in neurhl_sets().items():
        m = nf.merge(pr, on="game_id", how="inner", suffixes=("_n", ""))
        assert len(m) == len(nf), f"{name}: {len(m)} of {len(nf)} matched"
        assert (m.y == m.home_win).all()
        rows = []
        for V, x in list(m.groupby(scol)) + [("all", m)]:
            y = x.home_win.to_numpy()
            r = {"season": V if V == "all" else int(V), "n": int(len(x)),
                 "gk_known_share": float(x.gk_known.mean()),
                 "neurhl": R.logloss(x[col], y), "neurhl_elo": R.logloss(x.p_elo_n, y),
                 "hattrick": R.logloss(x.ht_p_home_win, y),
                 "hattrick_gk": R.logloss(x.ht_gk_p_home_win, y),
                 "hattrick_elo": R.logloss(x.p_elo, y),
                 "brier_neurhl": R.brier(x[col], y), "brier_hattrick": R.brier(x.ht_p_home_win, y),
                 "brier_hattrick_gk": R.brier(x.ht_gk_p_home_win, y),
                 "d_ht_vs_neurhl": paired(x.ht_p_home_win, x[col], y),
                 "d_htgk_vs_neurhl": paired(x.ht_gk_p_home_win, x[col], y),
                 "d_ht_vs_neurhl_elo": paired(x.ht_p_home_win, x.p_elo_n, y)}
            k = x[x.gk_known]
            if len(k) > 50:
                yk = k.home_win.to_numpy()
                r["gk_known_games"] = {
                    "n": int(len(k)), "neurhl": R.logloss(k[col], yk),
                    "hattrick": R.logloss(k.ht_p_home_win, yk),
                    "hattrick_gk": R.logloss(k.ht_gk_p_home_win, yk),
                    "d_htgk_vs_neurhl": paired(k.ht_gk_p_home_win, k[col], yk),
                    "d_htgk_vs_ht": paired(k.ht_gk_p_home_win, k.ht_p_home_win, yk)}
            rows.append(r)
        out[name] = rows
    return out


def frozen_table(pr: pd.DataFrame) -> list:
    rows = []
    for V, x in pr.groupby("season_end"):
        y = x.home_win.to_numpy()
        rows.append({"season": int(V), "n": int(len(x)),
                     "hattrick_frozen": R.logloss(x.ht_frozen_p_home_win, y),
                     "elo_frozen": R.logloss(x.p_elo_frozen, y),
                     "orr_inseason": R.logloss(x.ht_p_home_win, y),
                     "elo_inseason": R.logloss(x.p_elo, y),
                     "d_frozen_ht_vs_elo": paired(x.ht_frozen_p_home_win, x.p_elo_frozen, y),
                     "tuning_window": bool(V in (2012, 2014, 2015, 2016, 2017))})
    t = pr[pr.season_end.isin(TEST)]
    y = t.home_win.to_numpy()
    rows.append({"season": "2018-2026", "n": int(len(t)),
                 "hattrick_frozen": R.logloss(t.ht_frozen_p_home_win, y),
                 "elo_frozen": R.logloss(t.p_elo_frozen, y),
                 "orr_inseason": R.logloss(t.ht_p_home_win, y),
                 "elo_inseason": R.logloss(t.p_elo, y),
                 "d_frozen_ht_vs_elo": paired(t.ht_frozen_p_home_win, t.p_elo_frozen, y)})
    return rows


def outcome_calibration(pr: pd.DataFrame) -> dict:
    """Regulation ties, OT/SO shares, home wins, goal totals: predicted vs
    observed by season (in-season no-goalie predictions)."""
    rows = []
    for V, x in pr.groupby("season_end"):
        st = S.structural(V, pr.attrs["hp"].window, pr.attrs["hp"].h_halflife,
                          goalie_key=pr.attrs["hp"].goalie)
        P = {"layer": st["layer"], "kappa": st["kappa"], "ot": pr.attrs["ot_params"][V]}
        lh, la = x.ht_lam_h.to_numpy(), x.ht_lam_a.to_numpy()
        J = GM.reg_joint(lh, la, P, K=20)
        k = np.arange(20)
        tot = k[:, None] + k[None, :]
        et = (J * tot).sum((1, 2))
        vt = (J * tot ** 2).sum((1, 2)) - et ** 2
        obs_tot = (x.reg_h + x.reg_a).to_numpy()
        tie = (x.extra != "REG").to_numpy()
        rows.append({
            "season": int(V), "n": int(len(x)),
            "tie_pred": float(x.ht_p_tie.mean()), "tie_obs": float(tie.mean()),
            "so_pred": float(x.ht_p_so.mean()), "so_obs": float((x.extra == "SO").mean()),
            "so_share_of_ties_pred": float(x.ht_p_so.sum() / x.ht_p_tie.sum()),
            "so_share_of_ties_obs": float((x.extra == "SO").sum() / tie.sum()),
            "home_win_pred": float(x.ht_p_home_win.mean()), "home_win_obs": float(x.home_win.mean()),
            "reg_goals_pred": float(et.mean()), "reg_goals_obs": float(obs_tot.mean()),
            "reg_goals_var_pred": float(vt.mean() + et.var()),
            "reg_goals_var_obs": float(obs_tot.var()),
            "frozen_tie_pred": float(x.ht_frozen_p_tie.mean()),
            "frozen_home_win_pred": float(x.ht_frozen_p_home_win.mean()),
            "frozen_reg_goals_pred": float((x.ht_frozen_lam_h + x.ht_frozen_lam_a).mean())})
    return rows


def margin_validation(pr: pd.DataFrame) -> dict:
    """Out-of-sample check of the regulation-score model on 2018-2026 with the
    in-season expected goals: mean log-likelihood of the observed regulation
    margin (9 bins) under the end-game layer, independent Poisson, and a
    diagonal (Dixon-Coles-style) tie inflation whose theta is fitted on
    2012-2017 predictions."""
    hp = pr.attrs["hp"]
    tr = pr[pr.season_end.isin([2012, 2014, 2015, 2016, 2017])]
    from scipy.optimize import minimize_scalar
    th = minimize_scalar(lambda t: -GM.margin_loglik(GM.margin_pmf_diag(
        tr.ht_lam_h.to_numpy(), tr.ht_lam_a.to_numpy(), t), (tr.reg_h - tr.reg_a).to_numpy()),
        bounds=(0, 1), method="bounded").x
    rows = []
    obs_all, lay_all, poi_all = [], [], []
    for V, x in pr[pr.season_end.isin(TEST)].groupby("season_end"):
        st = S.structural(V, hp.window, hp.h_halflife, goalie_key=hp.goalie)
        lh, la = x.ht_lam_h.to_numpy(), x.ht_lam_a.to_numpy()
        mg = (x.reg_h - x.reg_a).to_numpy()
        Ml = GM.margin_pmf(lh, la, {"layer": st["layer"], "kappa": st["kappa"]})
        Mp = GM.margin_pmf_poisson(lh, la)
        Md = GM.margin_pmf_diag(lh, la, th)
        rows.append({"season": int(V), "layer": GM.margin_loglik(Ml, mg),
                     "poisson": GM.margin_loglik(Mp, mg), "diag_inflation": GM.margin_loglik(Md, mg)})
        obs_all.append(np.bincount(np.clip(mg, -4, 4) + 4, minlength=9) / len(mg))
        lay_all.append(GM._binned(Ml).mean(0))
        poi_all.append(GM._binned(Mp).mean(0))
    return {"diag_theta_fit_2012_2017": float(th), "by_season": rows,
            "margin_bins": [int(b) for b in GM.MARGIN_BINS],
            "share_obs": np.mean(obs_all, 0).round(4).tolist(),
            "share_layer": np.mean(lay_all, 0).round(4).tolist(),
            "share_poisson": np.mean(poi_all, 0).round(4).tolist()}


def market_comparison(pr: pd.DataFrame) -> dict | None:
    """Preseason points: ORR frozen expected points (actual schedule)
    vs the late-preseason sportsbook line, MAE on the 82-game scale.
    Descriptive only (no tuning on these seasons)."""
    m = R.market_history()
    if m is None:
        return None
    rows = []
    for V, x in pr.groupby("season_end"):
        if V not in set(m.season_end):
            continue
        h = x[["home", "ht_frozen_e_pts_h", "home_win", "extra"]].rename(
            columns={"home": "team", "ht_frozen_e_pts_h": "e"})
        a = x[["away", "ht_frozen_e_pts_a", "home_win", "extra"]].rename(
            columns={"away": "team", "ht_frozen_e_pts_a": "e"})
        h["pts"] = np.where(h.home_win == 1, 2, np.where(h.extra != "REG", 1, 0))
        a["pts"] = np.where(a.home_win == 0, 2, np.where(a.extra != "REG", 1, 0))
        t = pd.concat([h, a]).groupby("team").agg(e=("e", "sum"), pts=("pts", "sum"),
                                                  gp=("pts", "size"))
        t["e82"] = t.e * 82 / t.gp
        t["pts82"] = t.pts * 82 / t.gp
        t = t.join(m[m.season_end == V].set_index("team").line, how="inner")
        blend = 0.5 * t.e82 + 0.5 * t.line
        rows.append({"season": int(V), "n": int(len(t)),
                     "mae_hattrick_frozen": float((t.e82 - t.pts82).abs().mean()),
                     "mae_market": float((t.line - t.pts82).abs().mean()),
                     "mae_50_50_blend": float((blend - t.pts82).abs().mean()),
                     "corr_hattrick_market": float(np.corrcoef(t.e82, t.line)[0, 1]),
                     "sd_hattrick": float(t.e82.std()), "sd_market": float(t.line.std())})
    df = pd.DataFrame(rows)
    ok = df[df.season != 2021]
    return {"by_season": rows,
            "pooled_excl_2021": {c: float(np.average(ok[c], weights=ok.n))
                                 for c in ("mae_hattrick_frozen", "mae_market", "mae_50_50_blend")}}


# ---------------------------------------------------------------------------
def fmt(x):
    return f"{x:.4f}"


def print_tables(res: dict):
    print("\n=== IN-SEASON log loss on NeurHL's game sets "
          "(NeurHL = actual lineups + starters) ===")
    for name, rows in res["neurhl_sets"].items():
        print(f"\n[{name}]  HT = ORR no-goalie; HT-gk = ORR with actual starters "
              f"(starters known only through 2022-23 and 2023-24 until Nov 7)")
        print(f"{'season':>9} {'n':>5} {'gk%':>4} {'NeurHL':>7} {'HT':>7} {'HT-gk':>7} "
              f"{'EloN':>7} {'EloHT':>7}  {'HT-N [95% CI]':>26}  {'HTgk-N [95% CI]':>26}")
        for r in rows:
            d1, d2 = r["d_ht_vs_neurhl"], r["d_htgk_vs_neurhl"]
            print(f"{str(r['season']):>9} {r['n']:>5} {100 * r['gk_known_share']:>4.0f} "
                  f"{fmt(r['neurhl'])} {fmt(r['hattrick'])} {fmt(r['hattrick_gk'])} "
                  f"{fmt(r['neurhl_elo'])} {fmt(r['hattrick_elo'])}  "
                  f"{d1['diff']:+.4f} [{d1['ci95'][0]:+.4f},{d1['ci95'][1]:+.4f}]  "
                  f"{d2['diff']:+.4f} [{d2['ci95'][0]:+.4f},{d2['ci95'][1]:+.4f}]")
    print("\n=== PRESEASON-FROZEN log loss (opening-night information only) ===")
    print(f"{'season':>9} {'n':>5} {'HT frozen':>9} {'Elo frozen':>10} {'HT in-season':>12} "
          f"{'Elo in-season':>13}  {'HT-Elo frozen [95% CI]':>28}")
    for r in res["frozen"]:
        d = r["d_frozen_ht_vs_elo"]
        tag = " (tuning)" if r.get("tuning_window") else ""
        print(f"{str(r['season']):>9} {r['n']:>5} {fmt(r['hattrick_frozen']):>9} "
              f"{fmt(r['elo_frozen']):>10} {fmt(r['orr_inseason']):>12} "
              f"{fmt(r['elo_inseason']):>13}  {d['diff']:+.4f} "
              f"[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]{tag}")
    print("\n=== Outcome calibration (in-season, no-goalie) ===")
    print(f"{'season':>6} {'tie p/o':>13} {'SO|tie p/o':>13} {'home p/o':>13} "
          f"{'regG p/o':>13} {'var p/o':>13}")
    for r in res["outcome_calibration"]:
        print(f"{r['season']:>6} {r['tie_pred']:.3f}/{r['tie_obs']:.3f}  "
              f"{r['so_share_of_ties_pred']:.3f}/{r['so_share_of_ties_obs']:.3f}  "
              f"{r['home_win_pred']:.3f}/{r['home_win_obs']:.3f}  "
              f"{r['reg_goals_pred']:.2f}/{r['reg_goals_obs']:.2f}   "
              f"{r['reg_goals_var_pred']:.2f}/{r['reg_goals_var_obs']:.2f}")
    c = res["calibration_test"]
    print(f"\nlogistic recalibration 2018-2026 (in-season no-goalie): intercept "
          f"{c['hattrick']['intercept']:+.3f}, slope {c['hattrick']['slope']:.3f}; "
          f"frozen slope {c['hattrick_frozen']['slope']:.3f}; NeurHL-H slope "
          f"{c['neurhl_h']['slope']:.3f}")
    mv = res["margin_validation"]
    print("\nregulation-margin log-lik 2018-2026 (layer / Poisson / diag):",
          " ".join(f"{r['season']}:{r['layer']:.4f}/{r['poisson']:.4f}/{r['diag_inflation']:.4f}"
                   for r in mv["by_season"]))
    if res.get("market"):
        print("\n=== Preseason points vs sportsbook lines (MAE, 82-game scale; descriptive) ===")
        for r in res["market"]["by_season"]:
            print(f"{r['season']}: HT {r['mae_hattrick_frozen']:.2f}  market {r['mae_market']:.2f}  "
                  f"50/50 {r['mae_50_50_blend']:.2f}  corr {r['corr_hattrick_market']:.2f}  "
                  f"SD HT {r['sd_hattrick']:.1f} mkt {r['sd_market']:.1f}")
        print("pooled excl. 2021:", {k: round(v, 2) for k, v in res["market"]["pooled_excl_2021"].items()})


def main():
    pr = build_predictions()
    test = pr[pr.season_end.isin(TEST)]
    h = pd.read_csv(PREDS / "hier_restatement_games.csv")
    hm = h.merge(test, on="game_id")
    res = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": __doc__,
        "hp": {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in pr.attrs["hp"].__dict__.items()},
        "elo_hp": pr.attrs["ehp"].__dict__,
        "neurhl_sets": compare_sets(pr),
        "frozen": frozen_table(pr),
        "outcome_calibration": outcome_calibration(pr),
        "calibration_test": {
            "hattrick": calib(test.ht_p_home_win, test.home_win),
            "hattrick_gk": calib(test.ht_gk_p_home_win, test.home_win),
            "hattrick_frozen": calib(test.ht_frozen_p_home_win, test.home_win),
            "elo_hattrick": calib(test.p_elo, test.home_win),
            "neurhl_h": calib(hm.p_neurhl_h, hm.y)},
        "margin_validation": margin_validation(pr),
        "market": market_comparison(pr),
        "ot_params_by_season": {int(k): v for k, v in pr.attrs["ot_params"].items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "games_bt.json").write_text(json.dumps(res, indent=1, default=float))
    cols = ["gid", "game_id", "date", "season_end", "home", "away", "home_win", "extra",
            "reg_h", "reg_a", "gk_known", "p_elo", "p_elo_frozen",
            "ht_p_home_win", "ht_p_tie", "ht_lam_h", "ht_lam_a", "ht_gk_p_home_win",
            "ht_frozen_p_home_win", "ht_frozen_p_tie", "ht_frozen_lam_h", "ht_frozen_lam_a"]
    pr[cols].to_csv(OUT / "games_bt_preds.csv.gz", index=False, float_format="%.5f")
    print_tables(res)


if __name__ == "__main__":
    main()
