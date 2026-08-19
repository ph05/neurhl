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

**v5 (2026-08-19, PLAN_V5):** the ridge gained gated features from an expanded data
layer — MoneyPuck shot-level data, a 17,647-game NHL play-by-play corpus
(2012-2026, cross-checked to r>0.99 against official aggregates), NHL stats-rest
reports, hockey-reference SRS/SOS, NHL EDGE tracking (report-only), and
sportsdataverse fastRhockey bulk data. Gate survivors: `corsi_dev`, `fo_dev`,
`pen_diff` (h1) and `hd_share` (h2). Post-lock restatement 2018-2026: **HOWE5**
(0.5·v1 + 0.5·v5) is the best h1 model on the board (MAE 10.359 vs HOWE 10.397).
Per PLAN_V5 H the live 2026-27 holdout still scores v1/v4/HOWE unchanged; v5/HOWE5
co-headline 2027-28. Raw corpora are gitignored (multi-GB); derived tables are
committed (`data/processed/team_seasons_v5.csv`, `pbp_team_seasons.csv`,
`hr_league.csv`, `edge_team.csv`).

**v6 (2026-08-19, PLAN_V6 @ b719257 — committed before fetch/gates):** eight more
data assets: NHL shift charts (pair TOI → line continuity, TOI concentration),
full-population prospect careers (records.nhl.com ids + player-landing),
hockey-reference coach records, a travel/timezone game table, playoff PBP,
player-level EDGE tracking archive, a cross-book odds logger
(`src/log_odds.py`, run daily-ish), and absence spells from rosterSpots. Gate
survivors: `toi_hhi_f` (both horizons) + `coach_tenure` (h1). Documented nulls:
line continuity, coach-change flag, production-weighted prospects, travel
effects (t≈0.3 after b2b control), playoff-specific params. The post-lock
restatement is candid: v6's gains do not add out-of-window (HOWE5 still best at
h1; HOWE/v4 best at h2) — and the S3 screen found spell-based injury propensity
beats the age-bucket availability baseline, queued for a future plan.

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
