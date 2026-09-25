"""NeurHL-H walk-forward predictions and the paired-test battery.

Shared by the tune-window re-analysis (eval/hier_both_ways.py) and the one-shot
confirmation (eval/restate_hier.py), so both score exactly the same model:
Layer 3 is the thin head of eval/backtest_hier.py, refit for each predict-season
T on seasons < T, over Layer 1/2 projections read from `proj_dir`.

Nothing here writes a result; callers decide what is recorded.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
from eval.backtest_hier import COLS_DEFAULT, CS, nll  # noqa: E402
from train.train_game import elo_features, load_frames  # noqa: E402


def game_frame(proj_dir: Path, last_season: int) -> pd.DataFrame:
    """Every regular-season game with a pre-game Elo logit and projections."""
    gc, _ = load_frames(list(range(2008, last_season + 1)))
    elo = elo_features(gc)
    have = sorted(int(p.stem.split("_")[-1])
                  for p in Path(proj_dir).glob("proj_team_*.parquet"))
    proj = pd.concat([pd.read_parquet(Path(proj_dir) / f"proj_team_{s}.parquet")
                      for s in have])
    d = gc.set_index("game_id").join(elo).join(proj)
    d = d.dropna(subset=["elo_logit", "cfpct_h", "cfpct_a"])
    d["y"] = ((d.outcome4 == 0) | (d.outcome4 == 2)).astype(int)
    d["proj_diff"] = d.cfpct_h - d.cfpct_a
    d["proj_cl_diff"] = (d.p_clf_h / (d.p_clf_h + d.p_cla_h).clip(lower=1e-6)
                         - d.p_clf_a / (d.p_clf_a + d.p_cla_a).clip(lower=1e-6))
    d["rest_diff"] = (d.home_rest.clip(upper=7)
                      - d.away_rest.clip(upper=7)) / 7
    return d.fillna({"rest_diff": 0.0})


def predict(d: pd.DataFrame, seasons) -> pd.DataFrame:
    """Walk-forward NeurHL-H and Elo probabilities for each season in `seasons`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = []
    for T in seasons:
        tr, te = d[d.season_end < T], d[d.season_end == T]
        if len(tr) < 900 or not len(te):
            continue
        sc = StandardScaler().fit(tr[COLS_DEFAULT])
        best = (9.0, None, None)
        for C in CS:                      # chosen on TRAIN seasons, as frozen
            m = LogisticRegression(C=C, max_iter=4000).fit(
                sc.transform(tr[COLS_DEFAULT]), tr.y)
            l_tr = nll(m.predict_proba(sc.transform(tr[COLS_DEFAULT]))[:, 1],
                       tr.y.to_numpy()).mean()
            if l_tr < best[0]:
                best = (l_tr, m, C)
        p = best[1].predict_proba(sc.transform(te[COLS_DEFAULT]))[:, 1]
        out.append(pd.DataFrame({
            "game_id": te.index.to_numpy(), "season": T,
            "date": te.date.astype(str).to_numpy(), "y": te.y.to_numpy(),
            "p_neurhl_h": p,
            "p_elo": 1 / (1 + np.exp(-te.elo_logit.to_numpy())),
            "head_C": best[2]}))
    return pd.concat(out, ignore_index=True)


def murphy(p: np.ndarray, y: np.ndarray, bins: int = 20) -> dict:
    """Brier score = reliability - resolution + uncertainty (binned)."""
    base = y.mean()
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rel = res = 0.0
    for b in range(bins):
        m = k == b
        if m.any():
            w = m.mean()
            rel += w * (p[m].mean() - y[m].mean()) ** 2
            res += w * (y[m].mean() - base) ** 2
    return {"brier": float(np.mean((p - y) ** 2)), "reliability": float(rel),
            "resolution": float(res), "uncertainty": float(base * (1 - base))}


def battery(P: pd.DataFrame, seed: int = 711, block: int = 100,
            draws: int = 9999) -> dict:
    """The declared test battery (PLAN_NeurHL A5.2) on per-game predictions."""
    from itertools import product

    from scipy import stats

    P = P.sort_values(["date", "game_id"]).reset_index(drop=True)
    y = P.y.to_numpy()
    ll_h, ll_e = nll(P.p_neurhl_h.to_numpy(), y), nll(P.p_elo.to_numpy(), y)
    dx = ll_h - ll_e
    n = len(dx)
    se = dx.std(ddof=1) / np.sqrt(n)
    z = dx.mean() / se
    cl = pd.Series(dx).groupby(P.season.to_numpy()).mean()
    S = len(cl)
    cl_t = float(cl.mean() / (cl.std(ddof=1) / np.sqrt(S)))

    # wild cluster bootstrap on season with the null imposed: every one of the
    # 2^S Rademacher sign patterns applied to the season mean differences.
    # With S = 8 the smallest attainable p is 2/256 = 0.0078.
    obs = abs(cl.mean())
    pats = np.array(list(product((-1.0, 1.0), repeat=S)))
    wild = (pats * cl.to_numpy()).mean(axis=1)
    wild_p = float(np.mean(np.abs(wild) >= obs - 1e-15))

    # moving-block bootstrap over date-ordered games (percentile interval)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]
    boot = dx[idx].mean(axis=1)

    wins = int((cl < 0).sum())
    per = {}
    for s, g in P.assign(d_ll=dx, ll_h=ll_h, ll_e=ll_e).groupby("season"):
        sse = g.d_ll.std(ddof=1) / np.sqrt(len(g))
        per[str(int(s))] = {
            "n": int(len(g)), "neurhl_h": float(g.ll_h.mean()),
            "elo": float(g.ll_e.mean()), "diff": float(g.d_ll.mean()),
            "ci95": [float(g.d_ll.mean() - 1.96 * sse),
                     float(g.d_ll.mean() + 1.96 * sse)]}
    return {
        "n_games": int(n), "neurhl_h": float(ll_h.mean()),
        "elo": float(ll_e.mean()), "diff": float(dx.mean()), "se": float(se),
        "ci95": [float(dx.mean() - 1.96 * se), float(dx.mean() + 1.96 * se)],
        "z": float(z), "p_two_sided": float(2 * stats.norm.sf(abs(z))),
        "seasons": int(S), "clustered_t": cl_t,
        "clustered_p": float(2 * stats.t.sf(abs(cl_t), df=S - 1)),
        "wild_cluster_p_exact": wild_p,
        "block_bootstrap_ci95": [float(np.quantile(boot, 0.025)),
                                 float(np.quantile(boot, 0.975))],
        "block_bootstrap": {"block": block, "draws": draws, "seed": seed},
        "seasons_won": wins,
        "sign_test_p": float(stats.binomtest(wins, S, 0.5).pvalue),
        "murphy": {"neurhl_h": murphy(P.p_neurhl_h.to_numpy(), y),
                   "elo": murphy(P.p_elo.to_numpy(), y)},
        "per_season": per,
        "pass": bool(dx.mean() < 0 and abs(z) > 1.959964 and cl.mean() < 0)}


def default_proj_dir() -> Path:
    return TENSORS
