# ORR changelog

ORR (Odds, Ratings & Rosters) is versioned by what it forecasts in-season. The scored 2026-27 preseason forecast (`orr/output/freeze_2027/`, ORR 1.0) is frozen; later versions change the daily in-season forecasts from the day they ship, and each forecast counts only for games after its publication. Release notes with evidence are in `orr/releases/`.

## 2.0.1 (2026-10-07): the daily job runs again

- **Daily job fixed.** From 2026-10-04, once every team had played, `evaluate_2027.py` wrote a numpy bool that JSON cannot encode. Every run then stopped after forecasting and before saving. The ORR daily forecasts for 10-04 to the morning of 10-07 were made but never published, so ORR has no eligible daily forecast for those games. The results are fetched again.
- **Opening-night box scores.** The 8 games of 09-29 and 09-30 came from NeurHL's results file without box scores or shots, and ingest never fetched them, so every player's season totals missed those games. Ingest now fetches any game that lacks either.
- **Re-runs made reliable.** Each forecast now records the SHA-256 of the box scores it read and the git tree of its code. `orr.reproduce` restores the exact box scores, and finds the code even after a rewrite of the repository's history (`orr/output/reproduce/rewritten_commits.json` covers the rewrite of 2026-10). A dry run of the whole daily job on a synthetic full-league season re-runs identically.
- **A change to the daily workflow runs it once on `main`.**

## 2.0 (2026-10-02): one system, accountable

- **Preregistered season-end evaluation** against NeurHL (`orr/EVALUATION_2027.md`, `orr/evaluate_2027.py`). It records the SHA-256 of every evaluated file and has four primary comparisons, each with a paired-bootstrap CI. A verdict counts only after the last regular-season game.
- **Reproducibility check.** `python3 -m orr.reproduce --date D` re-runs a published day from its recorded code commit and SHA-matched inputs. Both published days (10-01 with 1.7, 10-02 with 1.9) reproduce exactly. The daily Action checks the previous day.
- **Model card** at `docs/orr/model.html`: every layer and every switch, with its version, held-out evidence and state.
- **One accuracy section** on the website. It has one held-out results table for every layer, including failures, a season scorecard of every forecast, and the evaluation's status.
- Forecasts are unchanged from 1.9. Retrospective of 1.0 to 2.0 in `orr/releases/v2.0.md`.

## 1.9 (2026-10-02): is ORR beating NeurHL?

- **Paired live comparison** with NeurHL on the same eligible games: paired bootstrap CI and a plain verdict.
- **Live reliability of the player probabilities** (goal, point, 3+ shots) against box scores.
- **Clinch and elimination flags and playoff magic numbers** in the daily standings and on the website.
- **Freshness banner** that warns when the daily run is stale.

## 1.8 (2026-10-01): fix what 1.7 measured

- **Rest-of-season intervals reach nominal coverage.** A wider variance grid plus an injury-spell term (v = 6). On held-out 2021-23 the interval score is 25.07 vs 25.53 (CI -0.73 to -0.18), and coverage is 0.80.
- **Goalie rest-of-season file** with intervals for starts and season save %.
- **Team statistics follow the in-season view**, with shots from box scores plus rest-of-season lines.
- **Forecast diff:** each game's change since its previous forecast, and why.

## 1.7 (2026-10-01): what you see is current

- **In-season standings and odds** are the website default, with a View selector back to the frozen preseason file.
- **Rest-of-season player intervals** now simulate games played and inflate rate variance (v = 3). On held-out 2021-23 the interval score is 25.53 vs 28.29 (CI −3.07 to −2.43), and coverage rises from 0.62 to 0.73. Still under 80%.
- **Rest and travel** for both teams on the Today table.
- **Version stamps** on daily forecasts, and a version history on the changelog page.

## 1.6 (2026-10-01): the full picture

- **Upstream NeurHL in the scorer.** `orr.score --neurhl-live` scores NeurHL's committed pregame forecasts; the daily Action uses it.
- **In-season goalie table** on the website: so-far GP, SA and SV%, with preseason and updated talent and start share.
- **Remaining strength of schedule** in the daily standings file and the website.
- **Daily rest-of-season skater file** (`players_ros_<date>.csv`) with 80% intervals for rest-of-season points.
- **Fix:** teams with no starts yet keep their preseason goalie start shares.

## 1.5 (2026-10-01): calibration everywhere

- **Shot distributions.** Player lines publish P(≥2), P(≥3) and P(≥4) shots on goal from a negative binomial (r = 16.6) around the updated shot rates. It is slightly better than Poisson on held-out 2021-23 (mean log loss 0.48632 vs 0.48653) and is adopted.
- **Standings sharpness.** A multiplier on in-season rating uncertainty was tuned; it chose 1.0, the current setting, so there is no change. Tuning-season coverage is 0.816; 1.4's 0.86 test coverage looks like sampling variation.
- **Live accuracy panel** (running log loss by date on eligible games, plus a reliability table) and **biggest movers** on the website.

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
