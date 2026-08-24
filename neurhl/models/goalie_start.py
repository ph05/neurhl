"""NeurHL-3 — M2/S4b: goalie starter prediction + the GQ1 anchor test.

GS1: P(goalie g starts | rest, b2b, starts_7d, gq, team context) must beat
BOTH walk-forward heuristics — (a) most-recent-starter-repeats, (b) season
starts-share — on log loss, season-clustered (PLAN_NeurHL3 GS).

GQ1: does walk-forward GSAx predict NEXT-season save% better than `gq`
(EWMA-shrunk sv%)? Steiger's z on dependent correlations
(train/rapm_folds.steiger). If GQ1 fails, N5 uses gq only — declared.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scikit-learn --with scipy \
     python neurhl/models/goalie_start.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402
import windows as W  # noqa: E402
from train.rapm_folds import steiger  # noqa: E402

CFG = ROOT / "configs"
FEATS = ["rest_days", "b2b", "starts_7d", "gq", "gsax60", "gsax_ew",
         "share_todate", "is_last_starter"]


def load(seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        p = TENSORS / f"goalie_games_{s}.parquet"
        if p.exists():
            rows.append(pd.read_parquet(p).assign(season_end=s))
    d = pd.concat(rows, ignore_index=True)
    gc = []
    for s in seasons:
        q = TENSORS / f"games_ctx_{s}.parquet"
        if q.exists():
            gc.append(pd.read_parquet(q, columns=["game_id", "date",
                                                  "game_type"]))
    g = pd.concat(gc, ignore_index=True)
    g = g[g.game_type == 2]
    d = d.merge(g[["game_id", "date"]], on="game_id", how="inner")
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values(["team", "date", "game_id"], kind="stable")
    # walk-forward per (team, goalie): starts share to date; last starter flag
    d["b2b"] = (d.rest_days == 1).astype(float)
    d["started"] = d.goalie_start.astype(float)
    grp = d.groupby(["season_end", "team", "player_id"], sort=False)
    d["starts_todate"] = grp.started.cumsum() - d.started
    d["games_todate"] = grp.cumcount()
    d["share_todate"] = d.starts_todate / d.games_todate.clip(lower=1)
    # team's previous game starter
    tg = d[d.started == 1][["season_end", "team", "game_id", "date",
                            "player_id"]].sort_values(["team", "date"],
                                                      kind="stable")
    tg["prev_starter"] = tg.groupby(["season_end", "team"],
                                    sort=False).player_id.shift(1)
    d = d.merge(tg[["season_end", "team", "game_id", "prev_starter"]],
                on=["season_end", "team", "game_id"], how="left")
    d["prev_starter"] = d.groupby(["season_end", "team"],
                                  sort=False).prev_starter.ffill()
    d["is_last_starter"] = (d.player_id == d.prev_starter).astype(float)
    # recency-weighted GSAx (diagnosed repair: the career-cumulative gsax60
    # is stale next to gq's EWMA -- compute a shifted per-game EWMA here)
    d["_g60"] = np.where(d.toi_sec > 0, (d.xgf - d.ga) * 3600.0
                         / np.maximum(d.toi_sec, 1), np.nan)
    d = d.sort_values(["player_id", "date", "game_id"], kind="stable")
    d["gsax_ew"] = d.groupby("player_id", sort=False)["_g60"].transform(
        lambda s: s.ewm(alpha=0.05, adjust=False).mean().shift(1))
    d = d.sort_values(["team", "date", "game_id"], kind="stable")
    return d


def gs1(seasons_fit, seasons_score) -> dict:
    d = load(sorted(set(seasons_fit) | set(seasons_score)))
    d = d[d.games_todate >= 3]          # cold-open games have no basis
    res = []
    for v in seasons_score:
        fit = d[d.season_end.isin([s for s in seasons_fit if s < v])]
        te = d[d.season_end == v].copy()
        if not len(fit) or not len(te):
            continue
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=50, early_stopping=True,
            validation_fraction=0.1, random_state=20260824)
        m.fit(fit[FEATS], fit.started)
        te["p_model"] = m.predict_proba(te[FEATS])[:, 1]
        # renormalise within team-game (exactly one starter per side)
        tot = te.groupby(["game_id", "team"]).p_model.transform("sum")
        te["p_model"] = te.p_model / tot.clip(lower=1e-9)
        # repeat baseline CALIBRATED on the fit window (the naive 0.98
        # cliff scored ll ~2.2 -- that measured the cliff, not the heuristic)
        p_rep = float(fit[fit.is_last_starter == 1].started.mean())
        te["p_repeat"] = np.where(te.is_last_starter == 1, p_rep, 1 - p_rep)
        tot = te.groupby(["game_id", "team"]).p_repeat.transform("sum")
        te["p_repeat"] = (te.p_repeat / tot.clip(lower=1e-9)).clip(0.01, 0.99)
        tot = te.groupby(["game_id", "team"]).share_todate.transform("sum")
        te["p_share"] = (te.share_todate.clip(lower=0.02)
                         / tot.clip(lower=1e-9)).clip(0.01, 0.99)
        eps = 1e-12
        y = te.started.to_numpy()
        out = {"season": v, "n": int(len(te))}
        for k in ("model", "repeat", "share"):
            p = np.clip(te[f"p_{k}"].to_numpy(), eps, 1 - eps)
            out[f"ll_{k}"] = float(-(y * np.log(p)
                                     + (1 - y) * np.log(1 - p)).mean())
        res.append(out)
        print(f"  {v}: ll model {out['ll_model']:.4f}  repeat "
              f"{out['ll_repeat']:.4f}  share {out['ll_share']:.4f}")
    dm = [r["ll_model"] - r["ll_repeat"] for r in res]
    ds = [r["ll_model"] - r["ll_share"] for r in res]

    def t_p(x):
        x = np.asarray(x)
        if len(x) < 2:
            return 1.0
        t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        return float(2 * stats.t.sf(abs(t), df=len(x) - 1))
    return {"per_season": res,
            "vs_repeat": {"mean": float(np.mean(dm)), "p": t_p(dm)},
            "vs_share": {"mean": float(np.mean(ds)), "p": t_p(ds)},
            "pass": bool(np.mean(dm) < 0 and np.mean(ds) < 0
                         and t_p(dm) < 0.05 and t_p(ds) < 0.05)}


def gq1() -> dict:
    """gsax60 vs gq at predicting NEXT-season sv%, per goalie-season."""
    seasons = [s for s in range(2009, 2027)]
    d = load(seasons)
    d = d[d.sf > 0]
    agg = d.groupby(["season_end", "player_id"]).agg(
        sv=("ga", lambda x: np.nan), sf=("sf", "sum"), ga=("ga", "sum"),
        gq_last=("gq", "last"), gsax_last=("gsax_ew", "last")).reset_index()
    agg["sv"] = 1 - agg.ga / agg.sf.clip(lower=1)
    nxt = agg[["season_end", "player_id", "sv", "sf"]].copy()
    nxt["season_end"] -= 1
    j = agg.merge(nxt, on=["season_end", "player_id"],
                  suffixes=("", "_next")).dropna()
    j = j[(j.sf >= 300) & (j.sf_next >= 300)]
    r_gsax = float(np.corrcoef(j.gsax_last, j.sv_next)[0, 1])
    r_gq = float(np.corrcoef(j.gq_last, j.sv_next)[0, 1])
    r_xz = float(np.corrcoef(j.gsax_last, j.gq_last)[0, 1])
    t, p = steiger(r_gsax, r_gq, r_xz, len(j))
    return {"n": int(len(j)), "r_gsax_next_sv": round(r_gsax, 4),
            "r_gq_next_sv": round(r_gq, 4), "r_between": round(r_xz, 4),
            "p_steiger": round(float(p), 4),
            "pass": bool(r_gsax > r_gq and p < 0.05)}


def main():
    print("GS1 — starter prediction vs heuristics (dev+tune fit, eval score)")
    fit = W.PLAYER_DEV + W.PLAYER_TUNE + [2018, 2019, 2020, 2021]
    # NOTE: 2018-2021 used for FITTING only (training-side, never scored);
    # scoring happens on PLAYER_EVAL per the declared windows.
    g1 = gs1([s for s in range(2009, 2022)], W.PLAYER_EVAL)
    print(f"GS1 -> {'PASS' if g1['pass'] else 'FAIL'}  "
          f"vs repeat p={g1['vs_repeat']['p']:.4f}, "
          f"vs share p={g1['vs_share']['p']:.4f}")
    q1 = gq1()
    print(f"GQ1 -> {'PASS' if q1['pass'] else 'FAIL'}  gsax r "
          f"{q1['r_gsax_next_sv']} vs gq r {q1['r_gq_next_sv']}, "
          f"p={q1['p_steiger']} (n={q1['n']})")
    (CFG / "goalie_gates.json").write_text(json.dumps(
        {"GS1": g1, "GQ1": q1}, indent=1))
    print("-> configs/goalie_gates.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
