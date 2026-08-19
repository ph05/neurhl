# HOWE — Hockey Outcomes via Weighted Ensemble

NHL team points/playoff/Cup projection system for 2026-27 and 2027-28. The flagship
model is **HOWE** (named for Gordie Howe, Mr. Hockey), a locked Elo-space equal blend
of two independently built engines:

```
r_HOWE = 1505 + 0.5*(r_v1 - 1505) + 0.5*(r_v4 - 1505)
```

- **v1** — Elo (K/H/phi tuned on 2006-2017) + xG-based season projection.
- **v4** — 19-feature walk-forward ridge (incl. prospect pipeline), amended roster
  overlay (rho*=0.421), availability 2.0, goalie game layer, b2b schedule effects.
- **HOWE** — the 50/50 blend, locked at preregistration (PLAN_V4 I3, before
  computation) under its working name "ENS"; renamed HOWE on 2026-08-19 with math,
  params, and seeds byte-identical. Report-only restatement: HOWE beats both parents
  at both horizons on 2018-2026 (h1 MAE 10.397 vs 10.457/10.550).

Every production run is deterministic (fixed seeds; reruns are hash-identical).
The 2026-27 season is a pre-registered live holdout scoring v1, v4, and HOWE on
rest-of-season MAE, CRPS, and playoff Brier.

## Layout

| Path | What |
|---|---|
| `src/howe.py` | Canonical HOWE module: blended ratings + deterministic sim rebuild |
| `src/report4.py` | Production report: ratings, sims, xlsx/CSV outputs |
| `src/live.py` | Nightly in-season updater (`update`) and parity `selftest` |
| `src/engine.py` | Elo, outcome model, season/playoff simulator |
| `src/price_milestones.py`, `src/optimize_board.py`, `src/optimize_full100.py` | Report-only market pricing & allocation off the HOWE sim |
| `output/` | Committed artifacts (`projections_*_howe.csv`, `v4_prior_ratings.csv`, …) |
| `PLAN*.md`, `NOTES.md` | Preregistration records and findings log (historical text keeps the prereg name "ENS") |
| `review_tests.py` | Independent read-only verification battery |

## Running

```
uv run --no-project --with numpy --with "pandas<3" --with openpyxl python review_tests.py
uv run --no-project --with numpy --with "pandas<3" --with openpyxl python src/report4.py
uv run --no-project --with numpy --with "pandas<3" --with openpyxl python src/live.py selftest
```

Market/allocation tooling additionally needs `--with scipy`. Nothing in the market
tooling feeds back into the model.
