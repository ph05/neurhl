# ORR changelog

ORR (Odds, Ratings & Rosters) is versioned by what it forecasts in-season. The scored 2026-27 preseason forecast (`orr/output/freeze_2027/`, ORR 1.0) is frozen; later versions change the daily in-season forecasts from the day they ship, and each forecast counts only for games after its publication. Release notes with evidence are in `orr/releases/`.

## 1.4 (2026-10-01): the standings race

- **In-season playoff odds backtested for the first time** (2021-23, run once). CRPS of final points is 3.84 against 4.99 for the preseason forecast; MAE is 5.47 against 7.07 for points pace. The rest-of-season drift multiplier is tuned (0.5; the surface is flat). 80% coverage is 0.86, slightly wide.
- **In-season goalie start shares** (Dirichlet update, α = 10 games). Rest-of-season starts error falls from 6.13 to 5.29 (CI −1.12 to −0.56).
- **Long-term absences in the rest-of-season simulation.** No measurable effect (+0.0008 CRPS, CI ±0.01), so the feature **ships off**.
- **Playoff odds over time** chart on the website.

## 1.3 (2026-10-01): the live loop matches the backtest

- **Box-score-first lineups and starters** for past games, ahead of NeurHL's pregame files, as in the X1 backtest (`box_first`).
- **In-season goalie talent**, using the backtests' own rule: preseason prior plus season saves above average. Each game uses only earlier evidence (`goalie_update`).
- **Per-game P(goal) and P(point) checked against 92,313 held-out skater-games.** They are already calibrated (log loss 0.3969 and 0.5973). The Platt correction made both slightly worse and is **not adopted**; it ships switched off.
- **Season-to-date columns on the website** (GP and points so far, rest-of-season points) once box scores exist.

## 1.2 (2026-10-01): player-level in-season

- **In-season skater rates.** Each skater's goals, assists and shots per game are updated with his own games so far, as a Gamma-Poisson posterior mean with a prior weight of 40 games for goals and assists and 20 for shots. On held-out 2021-22 and 2022-23, rest-of-season points error falls from 4.48 to 4.07 (−0.41, CI −0.49 to −0.34) and shots error from 10.71 to 9.23.
- **Box-score ingest.** `orr.ingest` stores every finished game's per-player box score (`orr/output/live/boxes_2027.csv`) from the NHL API. The daily player lines use them.
- **CI.** `.github/workflows/orr_tests.yml` runs every test suite on each push that touches `orr/`.
- **Changelog page** at `docs/orr/changelog.html`, linked from the projections page.

## 1.1 (2026-10-01): lineup-aware in-season forecasts

- **Lineup-aware forecasts (X1).** Both teams' ratings are adjusted for who dresses and who starts in goal, in the filter's prediction and its update. On the NeurHL-G gate games 2019-24, log loss improves by 0.00130 (CI −0.00255 to −0.00006). On two seasons no experiment had used (X3r), it improves by 0.00232 (CI −0.00438 to −0.00019).
- **Eight other pre-registered hypotheses were tested and not accepted.** See `orr/PLAN_1_1.md` and `orr/RESULTS.md`.
- **Daily GitHub Action** (`orr_daily.yml`) for results with shots from the NHL API.

## 1.0 (2026-10-01): first release

- Market-anchored standings, one scoring model for every game, player projections with conserved ice time, and the in-season filter. See `orr/README.md` and `orr/RESULTS.md`.
