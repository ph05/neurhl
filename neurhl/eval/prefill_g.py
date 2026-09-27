"""Train a config's walk-forward snapshots in parallel into the runner's cache.

eval/run_g.py trains snapshots one (season, seed) at a time. This fills the
same prediction cache (same key: config minus stack/ledger keys, model and
trainer source, master-tensor fingerprint) with a process pool, so a run can be
resumed and finish from cache. Workers use fewer torch threads than the
config's value; that changes floating-point summation order only, within the
training nondeterminism the protocol already accepts (P9). Nothing is scored
here: scoring and the ledger row belong to run_g.py.

Usage: ... python neurhl/eval/prefill_g.py --config g1 --through 2024 [--workers 4]
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIGS  # noqa: E402
import windows as W  # noqa: E402

_D = None


def _key(cfg):
    from eval.run_g import model_sha
    tc = {k: v for k, v in cfg.items() if k not in ("stack_h", "stack_window", "parent", "delta", "seeds")}
    return tc, hashlib.sha256(json.dumps(tc, sort_keys=True).encode() + model_sha()).hexdigest()[:16]


def job(args):
    global _D
    cfg, T, sd, threads = args
    from eval.run_g import RUNS
    from train.train_neurhl_g import Data, predict, train_snapshot
    tc, ck = _key(cfg)
    cp = RUNS / "cache" / f"{ck}_{T}_{sd}.npz"
    if cp.exists():
        return T, sd, "cached"
    if _D is None:
        _D = Data("train")
    te = np.where(_D.meta.season_end.to_numpy() == T)[0]
    m, P = train_snapshot(_D, T, {**tc, "seed": sd, "threads": threads})
    o = predict(m, P, te)
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **o)
    tmp.rename(cp)
    return T, sd, "trained"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--through", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    assert a.through < min(W.SEALED), "sealed seasons are trained only inside eval/seal_g.py"
    cfg = json.loads((CONFIGS / "neurhl_g" / f"{a.config}.json").read_text())
    seeds = range(cfg.get("seeds", 5))
    snaps = sorted(set(range(2011, a.through + 1)) - W.NO_SCORE)
    jobs = [(cfg, T, sd, a.threads) for T in reversed(snaps) for sd in seeds]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for f in as_completed(futs):
            T, sd, st = f.result()
            print(f"  {T} seed {sd}: {st}", flush=True)


if __name__ == "__main__":
    main()
