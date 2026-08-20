"""NeurHL Tier-0 — engineered-feature gradient-boosted baseline (gate G4 opponent).

A deliberately strong NON-neural baseline on walk-forward engineered features,
so the big-data hypothesis is tested, not assumed. Hyperparameters are FIXED a
priori (no tuning — a baseline that gets tuned on the tune window would spend
the same multiplicity budget the NN is charged for):
  xgboost: max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, min_child_weight=20, seed 711.

Features per regular-season game (all strictly pre-game, P1):
  v1 Elo expectation + rating diff; rest/b2b/travel/timezone (travel_games);
  days-into-season; era vector (prior-season league stats, P5); starting-goalie
  quality (pre-game EWMA save% vs expanding league mean, from event shards);
  roster form (pre-game EWMA points/60 and SOG/60 of dressed skaters, weighted
  by pre-game EWMA TOI share).

Walk-forward: predict season T with a model trained on seasons < T (tune
window: T in 2014-2017; features materialized for <= 2017 only — P7).
Writes tier0 results into neurhl/output/baselines_tune.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, PROJ, TENSORS  # noqa: E402

import backtest as B  # noqa: E402
import engine as E  # noqa: E402

TUNE_T = [2014, 2015, 2016, 2017]
MAX_SEASON = 2017                      # P7: nothing beyond the tune window
REMAP = {"PHX": "UTA", "ARI": "UTA", "ATL": "WPG"}
EWMA_ALPHA = 0.1
XGB_PARAMS = dict(max_depth=4, n_estimators=300, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  random_state=711, n_jobs=8, eval_metric="logloss")
ERA_COLS = ["prior_gpg", "prior_ot_share", "prior_so_share", "prior_margin_abs",
            "prior_parity", "flag_3v3", "flag_covid", "season_scaled"]


def load_tensors(kind: str, seasons) -> pd.DataFrame:
    dfs = [pd.read_parquet(TENSORS / f"{kind}_{se}.parquet") for se in seasons]
    return pd.concat(dfs, ignore_index=True)


def goalie_quality(events_seasons, dates: pd.Series) -> pd.DataFrame:
    """Per (game_id, goalie): pre-game EWMA sv% minus expanding league sv%."""
    ev = load_tensors("events", events_seasons)
    maps = json.loads((TENSORS / "maps.json").read_text())
    sog_i, goal_i = maps["event_type"]["shot-on-goal"], maps["event_type"]["goal"]
    sh = ev[(ev.event_type.isin([sog_i, goal_i])) & (ev.goalie > 0)
            & (ev.game_type == 2)]
    per = (sh.groupby(["game_id", "goalie"])
           .agg(sf=("event_type", "size"),
                ga=("event_type", lambda s: int((s == goal_i).sum())))
           .reset_index().rename(columns={"goalie": "player_id"}))
    per["date"] = per.game_id.map(dates)
    per = per.sort_values(["player_id", "date", "game_id"])
    per["sv"] = 1 - per.ga / per.sf
    grp = per.groupby("player_id")
    per["ewma_sv_pre"] = grp.sv.transform(
        lambda s: s.ewm(alpha=EWMA_ALPHA).mean().shift(1))
    per["n_pre"] = grp.cumcount()
    per = per.sort_values(["date", "game_id"])
    lg = (per.ga.cumsum().shift(1) / per.sf.cumsum().shift(1))
    per["lg_sv_pre"] = 1 - lg
    shrink = per.n_pre / (per.n_pre + 10)
    per["gq"] = (per.ewma_sv_pre - per.lg_sv_pre).fillna(0) * shrink
    return per[["game_id", "player_id", "gq"]]


def skater_form(pg: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Per (game_id, side): pre-game EWMA pts/60 & SOG/60, EWMA-TOI weighted."""
    sk = pg[(pg.pos_group < 2) & (pg.game_type == 2)].copy()
    sk["date"] = sk.game_id.map(dates)
    sk = sk.sort_values(["player_id", "date", "game_id"])
    sk["pts"] = sk.goals + sk.assists
    grp = sk.groupby("player_id")
    for src, dst in (("pts", "f_pts"), ("sog", "f_sog"), ("toi_sec", "f_toi")):
        sk[dst] = grp[src].transform(
            lambda s: s.ewm(alpha=EWMA_ALPHA).mean().shift(1))
    sk["n_pre"] = grp.cumcount()
    sk = sk.dropna(subset=["f_toi"])
    sk = sk[sk.f_toi > 60]
    shrink = sk.n_pre / (sk.n_pre + 10)
    sk["pts60"] = sk.f_pts / (sk.f_toi / 3600) * shrink
    sk["sog60"] = sk.f_sog / (sk.f_toi / 3600) * shrink
    w = sk.f_toi
    agg = (sk.assign(w=w, pw=sk.pts60 * w, sw=sk.sog60 * w)
           .groupby(["game_id", "is_home"])[["w", "pw", "sw"]].sum())
    agg["form_pts60"] = agg.pw / agg.w
    agg["form_sog60"] = agg.sw / agg.w
    return agg[["form_pts60", "form_sog60"]].reset_index()


def build_features() -> pd.DataFrame:
    cache = TENSORS / "t0_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    seasons = [s for s in range(2012, MAX_SEASON + 1)]
    gc = load_tensors("games_ctx", seasons)
    gc = gc[gc.game_type == 2].copy()
    gc["date"] = gc.date.astype(str)
    dates = gc.set_index("game_id").date

    # v1 Elo expectation, positionally aligned with engine.load() order
    v1 = json.loads((PROJ / "output" / "params.json").read_text())
    preds, _, _ = E.run_elo(B.g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    preds = preds.copy()
    preds["date"] = pd.to_datetime(B.g.date).dt.strftime("%Y-%m-%d").values
    preds["home_m"] = preds.home
    preds["away_m"] = preds.away
    key = pd.read_parquet(TENSORS / "_edacache" / "game_summary.parquet")
    key = key[key.game_type == 2].copy()
    key["home_m"] = key.home.replace(REMAP)
    key["away_m"] = key.away.replace(REMAP)
    key = key[["game_id", "date", "home_m", "away_m"]]
    elo = key.merge(preds[["date", "home_m", "away_m", "e_home", "rh", "ra"]],
                    on=["date", "home_m", "away_m"], how="inner")
    elo["elo_diff"] = elo.rh - elo.ra
    gc = gc.merge(elo[["game_id", "e_home", "elo_diff"]], on="game_id", how="inner")

    # goalie starter quality
    pg = load_tensors("player_games", seasons)
    gq = goalie_quality(seasons, dates)
    starters = pg[(pg.goalie_start == 1)][["game_id", "player_id", "is_home"]]
    sq = starters.merge(gq, on=["game_id", "player_id"], how="left").fillna(0)
    sq = sq.pivot_table(index="game_id", columns="is_home", values="gq",
                        aggfunc="first").rename(columns={True: "gq_h", False: "gq_a"})
    gc = gc.merge(sq, on="game_id", how="left")

    # roster form
    form = skater_form(pg, dates)
    fh = form[form.is_home].set_index("game_id")[["form_pts60", "form_sog60"]]
    fa = form[~form.is_home].set_index("game_id")[["form_pts60", "form_sog60"]]
    gc = gc.merge(fh.add_suffix("_h"), on="game_id", how="left")
    gc = gc.merge(fa.add_suffix("_a"), on="game_id", how="left")

    gc["y"] = (gc.outcome4.isin([0, 2])).astype(int)      # home win incl extra
    gc["season_end"] = gc.game_id.astype(str).str[:4].astype(int) + 1
    gc.to_parquet(cache, index=False)
    return gc


FEATS = (["e_home", "elo_diff", "days_in", "home_rest", "away_rest",
          "home_km3d", "away_km3d", "home_dtz", "away_dtz",
          "gq_h", "gq_a", "form_pts60_h", "form_sog60_h",
          "form_pts60_a", "form_sog60_a"] + ERA_COLS)


def _logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    """Tier-0 = the stronger of two fixed, self-regularizing procedures.

    Ledger note (t0-001/t0-002): the first fixed-parameter xgboost pooled
    0.69812 — worse than constant-home, i.e. mis-specified for 2-6k-game train
    sets. Before any NeurHL number exists, Tier-0 is redefined as
    max-strength of {standardized logistic regression, xgboost with early
    stopping on the last train season}. This only RAISES the G4 bar.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    df = build_features()
    results = {"logistic": {}, "xgb_es": {}}
    for T in TUNE_T:
        tr = df[df.season_end < T].fillna(0)
        te = df[df.season_end == T].fillna(0)
        # (a) standardized logistic regression
        sc = StandardScaler().fit(tr[FEATS])
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(sc.transform(tr[FEATS]), tr.y)
        results["logistic"][str(T)] = _logloss(
            te.y.to_numpy(), lr.predict_proba(sc.transform(te[FEATS]))[:, 1])
        # (b) xgboost, early stopping on the LAST train season (never on T)
        val_season = tr.season_end.max()
        fit, val = tr[tr.season_end < val_season], tr[tr.season_end == val_season]
        xgb = XGBClassifier(**XGB_PARAMS, early_stopping_rounds=30)
        xgb.fit(fit[FEATS], fit.y, eval_set=[(val[FEATS], val.y)], verbose=False)
        results["xgb_es"][str(T)] = _logloss(
            te.y.to_numpy(), xgb.predict_proba(te[FEATS])[:, 1])
        print(f"{T}: logistic={results['logistic'][str(T)]:.5f}  "
              f"xgb_es={results['xgb_es'][str(T)]:.5f}")
    n = {str(T): len(df[df.season_end == T]) for T in TUNE_T}
    pooled = {k: float(np.sum([v[str(T)] * n[str(T)] for T in TUNE_T])
                       / sum(n.values())) for k, v in results.items()}
    best = min(pooled, key=pooled.get)
    out_path = NOUT / "baselines_tune.json"
    blob = json.loads(out_path.read_text())
    blob["tier0"] = {"model": f"best of logistic/xgb_es (winner: {best})",
                     "ledger_note": "t0-001 fixed-xgb 0.69812 mis-specified; "
                                    "redefined pre-prereg to max-strength pair",
                     "features": FEATS,
                     "per_season": results,
                     "pooled_2014_2017": pooled,
                     "logloss_pooled_2014_2017": pooled[best]}
    out_path.write_text(json.dumps(blob, indent=1))
    print(f"tier0 (={best}) pooled 2014-2017 logloss: {pooled[best]:.5f}")


if __name__ == "__main__":
    main()
