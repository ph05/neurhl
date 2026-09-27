"""NeurHL-G experiment runner (PLAN_NeurHL4 L): one config, one window, one ledger row.

  --config    configs/neurhl_g/<id>.json (overrides train_neurhl_g.DEFAULT;
              optional keys: parent, delta, stack_h)
  --protocol  quick (score 2016 and 2018, 1 seed) | full (score the window,
              cfg seeds)
  --window    iter (G_ITER) | gate (G_GATE; budget-capped)

For each predict-season T a snapshot trains on seasons < T. Snapshots also run
for earlier seasons back to 2011 so the walk-forward stack for season T is fit
on out-of-sample predictions of seasons < T. The stack is a logistic regression
on [Elo logit, NeurHL-G logit (, NeurHL-H logit)] and falls back to Elo alone if
leave-one-season-out cross-validation on its training seasons does not beat
Elo. Scores: game log loss (home win incl. OT/SO) for G raw, G stacked, Elo and
NeurHL-H on the same games; OT share; team and skater box-score errors.
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIGS, TENSORS  # noqa: E402
import windows as W  # noqa: E402
from train.train_neurhl_g import Data, predict, train_snapshot  # noqa: E402

LEDGER = CONFIGS / "search_ledger_g.csv"
RUNS = TENSORS / "g_runs"


def append_ledger(row: dict) -> None:
    """Append one run to the ledger, aligning columns with earlier rows."""
    new = pd.DataFrame([row])
    if LEDGER.exists():
        new = pd.concat([pd.read_csv(LEDGER), new], ignore_index=True)
    new.to_csv(LEDGER, index=False)


def model_sha() -> bytes:
    """Hash of the model and trainer source, so cached snapshot predictions
    are reused only for identical code."""
    root = Path(__file__).resolve().parents[1]
    return hashlib.sha256((root / "models" / "neurhl_g.py").read_bytes()
                          + (root / "train" / "train_neurhl_g.py").read_bytes()).digest()


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def h_preds(seasons) -> pd.Series:
    """NeurHL-H walk-forward home-win probabilities (hier_core), cached."""
    cache = TENSORS / "g_ref_neurhl_h.parquet"
    if cache.exists():
        c = pd.read_parquet(cache)
        if set(seasons) <= set(c.season.unique()):
            return c.set_index("game_id").p_neurhl_h
    from eval.hier_core import game_frame, predict as hpred
    d = game_frame(TENSORS, 2024)
    P = hpred(d, sorted(set(range(2011, 2025))))
    P[["game_id", "season", "p_neurhl_h"]].to_parquet(cache, index=False)
    return P.set_index("game_id").p_neurhl_h


def fit_stack(X, y, seasons):
    """Logistic stack; the column subset (always including Elo, column 0) is
    chosen by leave-one-season-out CV on the training seasons."""
    from sklearn.linear_model import LogisticRegression

    def loso(cols):
        tot, n = 0.0, 0
        for s in np.unique(seasons):
            tr, te = seasons != s, seasons == s
            if tr.sum() < 500:
                continue
            m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr][:, cols], y[tr])
            tot += nll(m.predict_proba(X[te][:, cols])[:, 1], y[te]).sum()
            n += te.sum()
        return tot / max(n, 1)
    from itertools import combinations
    others = list(range(1, X.shape[1]))
    subsets = [[0] + list(c) for r in range(len(others) + 1)
               for c in combinations(others, r)]
    if len(np.unique(seasons)) < 3:
        use = list(range(X.shape[1]))
    else:
        use = min(subsets, key=loso)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[:, use], y)
    return m, use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--protocol", choices=("quick", "full"), default="quick")
    ap.add_argument("--window", choices=("iter", "gate"), default="iter")
    a = ap.parse_args()
    cfg_path = CONFIGS / "neurhl_g" / f"{a.config}.json"
    cfg = json.loads(cfg_path.read_text())
    win = W.G_ITER if a.window == "iter" else W.G_GATE
    scored = [2016, 2018] if a.protocol == "quick" else list(win)
    W.assert_scorable_g(scored, a.window)
    seeds = [0] if a.protocol == "quick" else list(range(cfg.get("seeds", 5)))
    t0 = time.time()

    D = Data("train")
    meta = D.meta
    snaps = sorted(set(range(2011, max(scored) + 1)) - W.NO_SCORE | set(scored))
    if a.protocol == "quick":
        snaps = sorted(set(range(2013, max(scored) + 1)) | set(scored))
    rows = []
    for T in snaps:
        te = np.where(meta.season_end.to_numpy() == T)[0]
        if not len(te):
            continue
        acc = None
        train_cfg = {k: v for k, v in cfg.items()
                     if k not in ("stack_h", "parent", "delta", "seeds")}
        ck = hashlib.sha256(json.dumps(train_cfg, sort_keys=True).encode()
                            + model_sha()).hexdigest()[:16]
        for sd in seeds:
            cp = RUNS / "cache" / f"{ck}_{T}_{sd}.npz"
            if cp.exists():
                o = dict(np.load(cp))
            else:
                model, P = train_snapshot(D, T, {**train_cfg, "seed": sd})
                o = predict(model, P, te)
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cp, **o)
            acc = o if acc is None else {k: acc[k] + o[k] for k in acc}
        o = {k: v / len(seeds) for k, v in acc.items()}
        df = meta.iloc[te][["game_id", "season_end", "outcome4", "gh_reg", "ga_reg"]].copy()
        df["p_g"] = o["p_home_win"]
        df["p_tie"] = o["o4"][:, 2:].sum(1)
        df["elo_logit"] = D.A["CTX"][te, 0]
        for k in ("xgf", "sogf", "goals", "pp_opps", "pp_m"):
            df[f"{k}_h"], df[f"{k}_a"] = o[k][:, 0], o[k][:, 1]
        tmt = D.names["tm_tgt"]
        for k in ("xgf_all", "sogf", "gf_reg", "pp_opps", "pp_m"):
            j = tmt.index(k)
            df[f"y_{k}_h"], df[f"y_{k}_a"] = D.A["TMY"][te, 0, j], D.A["TMY"][te, 1, j]
        m = D.A["SKM"][te] > 0
        SKY = D.A["SKY"][te]
        sk = {}
        for j, (t, oo) in enumerate(zip(["toi_ev", "toi_pp", "isog", "ixg_all", "g", "a"],
                                        ["toi_ev", "toi_pp", "isog", "ixg", "g", "a"])):
            jj = D.names["sk_tgt"].index(t)
            y = SKY[..., jj][m]
            mu = o[oo][m]
            ok = np.isfinite(y)
            if t.startswith("toi"):
                sk[t] = float(np.abs(mu[ok] - y[ok]).mean())
            else:
                mu_, y_ = np.clip(mu[ok], 1e-6, None), y[ok]
                sk[t] = float(np.mean(mu_ - y_ * np.log(mu_)))
        df.attrs["sk"] = sk
        rows.append((T, df, sk))
        print(f"  snapshot {T}: {len(te)} games, {time.time() - t0:.0f}s", flush=True)

    allp = pd.concat([r[1] for r in rows], ignore_index=True)
    allp["y"] = allp.outcome4.isin([0, 2]).astype(float)
    use_h = bool(cfg.get("stack_h", False))
    if use_h:
        allp["p_h"] = allp.game_id.map(h_preds(snaps))
    # walk-forward stack
    allp["p_stack"] = np.nan
    for T in scored:
        tr = allp[(allp.season_end < T) & ~allp.season_end.isin(W.NO_SCORE)]
        cols = ["elo_logit", "lg"] + (["lh"] if use_h else [])
        f = lambda d: np.column_stack([d.elo_logit, logit(d.p_g)] +
                                      ([logit(d.p_h)] if use_h else []))
        if len(tr) < 500:
            continue
        mdl, usec = fit_stack(f(tr), tr.y.to_numpy(), tr.season_end.to_numpy())
        te = allp.season_end == T
        allp.loc[te, "p_stack"] = mdl.predict_proba(f(allp[te])[:, usec])[:, 1]
        allp.loc[te, "stack_cols"] = ",".join(cols[i] for i in usec)
    ev = allp[allp.season_end.isin(scored)].copy()
    ev["p_elo"] = 1 / (1 + np.exp(-ev.elo_logit))
    ev["p_h"] = ev.game_id.map(h_preds(snaps))
    res = {"n": int(len(ev))}
    for k in ("p_g", "p_stack", "p_elo", "p_h"):
        ok = ev[k].notna()
        res[f"ll_{k[2:]}"] = float(nll(ev.loc[ok, k], ev.loc[ok, "y"]).mean())
    d = nll(ev.p_stack, ev.y) - nll(ev.p_elo, ev.y)
    dh = nll(ev.p_stack, ev.y) - nll(ev.p_h, ev.y)
    res["d_vs_elo"] = float(d.mean())
    res["z_vs_elo"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    res["d_vs_h"] = float(dh.mean())
    res["z_vs_h"] = float(dh.mean() / (dh.std(ddof=1) / np.sqrt(len(dh))))
    per = ev.assign(d=d, dh=dh).groupby("season_end")[["d", "dh"]].mean()
    res["seasons_beat_elo"] = f"{int((per.d < 0).sum())}/{len(per)}"
    res["seasons_beat_h"] = f"{int((per.dh < 0).sum())}/{len(per)}"
    res["ot_pred"] = float(ev.p_tie.mean())
    res["ot_obs"] = float(ev.outcome4.isin([2, 3]).mean())
    for k in ("xgf", "sogf", "goals", "pp_opps"):
        yk = {"xgf": "xgf_all", "sogf": "sogf", "goals": "gf_reg", "pp_opps": "pp_opps"}[k]
        err = np.abs(np.r_[ev[f"{k}_h"] - ev[f"y_{yk}_h"], ev[f"{k}_a"] - ev[f"y_{yk}_a"]])
        res[f"mae_{k}"] = float(np.nanmean(err))
    sk = pd.DataFrame([r[2] for r in rows if r[0] in scored]).mean()
    for k, v in sk.items():
        res[f"sk_{k}"] = float(v)
    res["stack_cols"] = ";".join(sorted(ev.stack_cols.dropna().unique()))
    res["per_season_d_elo"] = {str(int(s)): round(float(v), 5) for s, v in per.d.items()}
    res["wall_s"] = round(time.time() - t0)

    RUNS.mkdir(exist_ok=True)
    ev.to_parquet(RUNS / f"{a.config}_{a.protocol}_{a.window}.parquet", index=False)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    row = {"run_id": f"{a.config}:{a.protocol}:{a.window}",
           "date": dt.datetime.now().isoformat(timespec="minutes"), "git": sha,
           "config_sha": hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:12],
           "parent": cfg.get("parent", ""), "delta": cfg.get("delta", ""),
           "seeds": len(seeds), **{k: v for k, v in res.items() if k != "per_season_d_elo"},
           "per_season_d_elo": json.dumps(res["per_season_d_elo"])}
    append_ledger(row)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
