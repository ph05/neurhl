# NeurHL

NeurHL is this repo's proprietary neural prediction model: a hierarchical network
trained on the event-scale corpora (play-by-play, shifts, shots, careers) to
simulate NHL **game, player, and season outcomes**. It lives entirely in this
directory plus the prereg `PLAN_NeurHL.md` at the repo root, and is report-only
with respect to the 2026-27 live holdout (production scoring stays v1/v4/HOWE).

Branding: the model name is **NeurHL** — exactly this capitalization — in all
prose, reports, and output model/column names. Filesystem artifacts are lowercase
(`neurhl/`, `params_neurhl.json`).

## Layout

| Path | Contents |
|---|---|
| `common.py` | shared paths/seeds + one-way import shim to repo `src/` |
| `configs/` | pinned env + locked hyperparams + `search_ledger.csv` (every tune-window config, committed) |
| `data/fetch/` | NeurHL-owned acquisition crawlers (HTM reports, NST, NHL player reports, lineup snapshots) |
| `data/` | vocab + tensorization (raw → `data/tensors/`, gitignored) |
| `eda/` | exploratory data analysis scripts + committed reports/figures |
| `models/` | Tier-0 baseline (xgboost/MLP), event-LM, career encoder, game model |
| `train/` | pretraining / training / calibration entry points |
| `eval/` | baselines harness, backtests, gates, one-shot 2018–2026 restatement |
| `sim/` | ratings bridge → `engine.simulate_season`; season sim emits the `howe.rebuild_sim` dict contract |
| `checkpoints/` | gitignored; SHA256s recorded in `output/params_neurhl.json` |
| `output/` | params contract, committed prediction artifacts (`preds/`), projections |
| `tests/` | `review_tests_neurhl.py` read-only verification battery |

## Run convention

No venv, no pyproject (house rule). Every script runs via uv, e.g.:

```
uv run --no-project --python 3.12 --with requests python neurhl/data/fetch/fetch_htm_reports.py
uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow --with matplotlib python neurhl/eda/eda_01_corpus.py
uv run --no-project --python 3.12 --with torch --with numpy --with "pandas<3" --with pyarrow python neurhl/train/pretrain_events.py
```

Resolved versions are pinned in `configs/env.json` at first successful run.

## Determinism policy

Torch-MPS training nondeterminism is accepted; checkpoints are ground truth
(SHA256 in `output/params_neurhl.json`), all gated/report numbers come from CPU
inference over saved checkpoints, and predictions are committed CSV artifacts
that `tests/review_tests_neurhl.py` re-derives and asserts against (max|Δp| ≤ 1e-6).
Sim/report seeds: 711 (h1) / 722 (h2).

## Methodology

See `PLAN_NeurHL.md` (repo root): protocol P1–P10 (vantage rule, pretrain
snapshots, MoneyPuck exclusion, era conditioning, COVID handling, strict
tune-window policy with a ≤40-config search budget, pre-committed restatement
decision rule, determinism, market firewall) and gates G1–G6. The prereg is
committed with locked numbers BEFORE any gate runs.
