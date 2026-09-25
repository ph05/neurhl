# neurhl/

The NeurHL package: data builders, models, training, evaluation, simulation
and tests. The project overview, findings and reproduction commands are in
the [top-level README](../README.md); the evidence behind each claim is in
[EVIDENCE.md](../EVIDENCE.md).

## Layout

| Path | Contents |
|---|---|
| `common.py` | Shared paths and seeds; makes the baselines in `src/` importable (one way only) |
| `windows.py` | Season roles (development, tune, confirmation, never-scored) and the guards that enforce them |
| `registry.py` | Feature registry: each feature's coverage and vantage rule, declared once |
| `manifest.py` | Artifact lineage; consumers refuse stale inputs |
| `data/fetch/` | Fetchers for NHL game reports, player reports, Natural Stat Trick, MoneyPuck and daily lineups |
| `data/` | Builders: events, shifts, stints, on-ice rates, absences, usage, goalie games, tensors |
| `models/` | xG, RAPM, event language model, event simulator, player-game chain, season player model, goalie starter, baselines |
| `train/` | Training entry points; `run_confirm_chain.sh` rebuilds the walk-forward artifacts for the NeurHL-H confirmation |
| `eval/` | Gates, backtests, one-shot confirmations (`confirm_player_2018_2020.py`, `restate_hier.py`), re-analyses and the live scorer |
| `sim/` | Season simulation and the 2026-27 projections (`project_2027.py`, `project_players.py`) |
| `site/` | Builds `docs/data.js` for the projection site |
| `configs/` | Locked configurations, gate records and the search ledgers |
| `output/` | Frozen 2026-27 predictions, committed prediction artifacts, live results |
| `eda/` | Exploratory analyses and their reports |
| `tests/` | Acceptance batteries and leakage audits |

`data/tensors/` and `checkpoints/` are large and are not committed; the
builders and training scripts regenerate them, and every checkpoint's
SHA-256 is recorded beside the numbers it produced.

## Conventions

- Every script runs through uv with explicit dependencies, for example
  `uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow python neurhl/sim/project_2027.py`.
  Resolved versions are pinned in `configs/env.json`.
- Seasons are named by the year they end (2027 is 2026-27).
- Any quantity used to predict season V is computed from seasons before V;
  in-season predictions may also use games of V played before the game being
  predicted.
- Neural training uses Apple MPS and is not bit-deterministic. Reported
  numbers come from CPU inference over saved checkpoints, and committed
  prediction files re-derive within 1e-6.
- The model name is written NeurHL; file and directory names are lowercase.
