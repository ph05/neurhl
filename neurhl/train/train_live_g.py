"""Train the NeurHL-G bundle used for live 2026-27 forecasts (PLAN_NeurHL4 C, LIVE).

  --config   configs/neurhl_g/<id>.json (the frozen configuration)
  --through  last training season: 2024 for v1 (seal unspent), 2026 for v2
             (after the seal; eval/seal_g.py unseals and calls build())
  --name     bundle name under neurhl/checkpoints/g/

The stack is fit on out-of-sample snapshot predictions of every scorable season
2011..through (each from a snapshot trained on earlier seasons; taken from the
runner's cache where present). The bundle's networks are trained on every
season <= through, one per seed, and saved with their training statistics.
Updates configs/live_models.json only when --activate is given.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CONFIGS  # noqa: E402
import windows as W  # noqa: E402
from eval.run_g import RUNS, fit_stack, logit, model_sha  # noqa: E402
from sim.g_forecast_core import save_bundle  # noqa: E402
from train.train_neurhl_g import Data, predict, train_snapshot  # noqa: E402


def oos_preds(D, cfg, seasons, seeds):
    import hashlib
    train_cfg = {k: v for k, v in cfg.items() if k not in ("stack_h", "parent", "delta", "seeds")}
    ck = hashlib.sha256(json.dumps(train_cfg, sort_keys=True).encode() + model_sha()).hexdigest()[:16]
    rows = []
    for T in seasons:
        te = np.where(D.meta.season_end.to_numpy() == T)[0]
        acc = None
        for sd in seeds:
            cp = RUNS / "cache" / f"{ck}_{T}_{sd}.npz"
            if cp.exists():
                o = dict(np.load(cp))
            else:
                m, P = train_snapshot(D, T, {**train_cfg, "seed": sd})
                o = predict(m, P, te)
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cp, **o)
            acc = o["p_home_win"] if acc is None else acc + o["p_home_win"]
        rows.append(pd.DataFrame({"season": T, "p_g": acc / len(seeds),
                                  "elo_logit": D.A["CTX"][te, 0],
                                  "y": D.meta.outcome4.iloc[te].isin([0, 2]).astype(float).values}))
    return pd.concat(rows, ignore_index=True)


def build(cfg_name, through, name, purpose="train"):
    cfg = json.loads((CONFIGS / "neurhl_g" / f"{cfg_name}.json").read_text())
    seeds = list(range(cfg.get("seeds", 5)))
    D = Data(purpose)
    assert D.meta.season_end.max() >= through, "training seasons not available (sealed?)"
    seasons = sorted(set(range(2011, through + 1)) - W.NO_SCORE)
    P = oos_preds(D, cfg, seasons, seeds)
    X = np.column_stack([P.elo_logit, logit(P.p_g)])
    mdl, use = fit_stack(X, P.y.to_numpy(), P.season.to_numpy())
    cols = ["elo_logit", "lg"]
    stack = {"cols": [cols[i] for i in use], "coef": mdl.coef_[0].tolist(),
             "intercept": float(mdl.intercept_[0]), "fit_seasons": seasons}
    train_cfg = {k: v for k, v in cfg.items() if k not in ("stack_h", "parent", "delta", "seeds")}
    models = []
    for sd in seeds:
        m, _ = train_snapshot(D, through + 1, {**train_cfg, "seed": sd})
        models.append(m)
    stats = models[0].stats
    h = save_bundle(name, models, stats, {**train_cfg, "through": through,
                                          "config": cfg_name}, D.names, stack)
    print(f"bundle {name}: {len(models)} seeds trained on seasons <= {through}; "
          f"stack {stack['cols']} coef {np.round(stack['coef'], 3).tolist()}; bundle.json sha {h[:16]}")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--through", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--activate", action="store_true")
    a = ap.parse_args()
    if a.through >= min(W.SEALED):
        raise SystemExit("training on sealed seasons happens only inside eval/seal_g.py")
    build(a.config, a.through, a.name)
    if a.activate:
        p = CONFIGS / "live_models.json"
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur["neurhl_g"] = a.name
        p.write_text(json.dumps(cur, indent=1))


if __name__ == "__main__":
    main()
