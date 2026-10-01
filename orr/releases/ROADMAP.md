# ORR roadmap

A version ships every cycle. Each cycle opens with a brainstorm, and every feature considered there is planned into the next version. A feature that fails its backtest still ships, but behind a switch with the result documented; it is not silently dropped. Release notes are in this folder.

## 1.8 (planned): fix what 1.7 measured

Brainstorm, held after 1.7 shipped. Every item is planned into 1.8:

1. **Rest-of-season intervals, second pass.** 1.7's best v sat at the grid edge (3) with 73% coverage. Extend the tuning grid (v ∈ {3, 4, 6, 9}). Add an injury-spell term: with the player's historical probability of a multi-game absence, a share of the remaining games is removed in a block. Tune on ≤2017; test once on 2021-23.
2. **Goalie rest-of-season lines.** Mirror the skater file for goalies: season-to-date starts, saves and save %; rest-of-season starts from 1.4's shares × team games left; save % from 1.3's talent. 80% intervals by simulation. Published as `goalies_ros_<date>.csv` and linked.
3. **Team stat table in-season.** The website's team-statistics table follows the standings View selector. Goals for and against come from the daily run; shots come from rest-of-season player lines plus box-score totals to date.
4. **Forecast diff.** Each daily run records how each game's probability changed since the previous forecast of that game, and why: ratings, starters or lineups. Shown as a "Changed since last run" column on the Today table.

## 1.7 (planned): what you see is current

Brainstorm, held after 1.6 shipped. Every item is planned into 1.7:

1. **In-season standings on the website by default.** The projected standings, playoff odds and team-stat tables show the latest daily run, with its date. The frozen preseason file stays one click away (a toggle) and in the comparison page.
2. **Calibrated rest-of-season player intervals.** Backtest the 80% interval of rest-of-season points (1.6) at 25%, 50% and 75% of 2021-23 (test, once). Tune on ≤2017 a games-played variance term (each remaining team game is a Bernoulli dress with the player's rate) and a rate-variance multiplier, by coverage and interval score.
3. **Rest and travel on the Today table.** Rest days, back-to-back flags and kilometres travelled for both teams, from the schedule features the game model already uses.
4. **Version stamps.** Each daily file shows which ORR version produced it (from its run JSON), and the website gets a version history table linking every release note.

## 1.6 (planned): the full picture

Brainstorm, held after 1.5 shipped. Every item is planned into 1.6:

1. **Upstream NeurHL live files in the scorer.** Score NeurHL's pregame forecasts from upstream's committed files (`git archive origin/main neurhl/output/live`), not this branch's stale checkout, so the running table is complete. Applies to `score.py --neurhl-live` and the daily GitHub Action.
2. **In-season goalie table.** The website's goalie table gains season-to-date shots against, save %, the updated save talent (1.3) and the updated start share (1.4) beside the preseason projection.
3. **Remaining strength of schedule.** Each team's average opponent rating over its remaining games, from the current filter ratings, in the standings output and the website's standings table. Test: a unit test that it equals the mean of the opponents' net ratings.
4. **Rest-of-season skater projections file.** A daily `players_ros_<date>.csv`: season-to-date counts, plus rest-of-season goals, assists and points from the updated rates, times expected remaining games (dress probability times remaining team games), with 80% intervals. Linked from the website.

## 1.5 (planned): calibration everywhere

Brainstorm, held after 1.4 shipped. Every item is planned into 1.5:

1. **In-season standings sharpness.** Test coverage of the in-season 80% intervals is 0.86, so they are too wide. Tune a multiplier on the filter's rating uncertainty used in the rest-of-season simulation, on ≤2017 by CRPS. Test once on 2021-23, aiming for coverage closer to 0.80 and lower CRPS.
2. **Shots-on-goal distributions for player lines.** Fit a negative-binomial dispersion for skater shots per game, on ≤2017 box scores, around 1.2's updated shot rates. Publish P(≥2), P(≥3) and P(≥4) shots per player-game. Test calibration (log loss, reliability) once on 2021-23 against the Poisson.
3. **Live accuracy panel.** A website section that scores the daily forecasts as games are played:
   - running log loss by date for ORR and every NeurHL file, eligible games only;
   - a reliability table of ORR's daily game probabilities by probability bin.
4. **Biggest movers.** A website panel with the teams whose playoff odds moved most since the previous committed forecast, with the games that moved them.

## 1.4 (planned): the standings race

Brainstorm, held after 1.3 shipped. Every item is planned into 1.4:

1. **In-season standings backtest and drift calibration.** At 25%, 50% and 75% of a season, re-simulate the rest of the season from the filter's ratings, then score final points (CRPS, 80% coverage) against the preseason-only forecast and a points-pace baseline. Tune the rest-of-season drift multiplier on ≤2017 and test 2021-23 once. This shows whether the in-season odds are calibrated.
2. **In-season goalie start shares.** Each team's start shares are updated with the season's box-score starts (a Dirichlet update of the preseason shares). They feed the usual-starter reference in the starter offsets and the goalie mix of games with unknown starters. Test: on box-score seasons, rest-of-season start-share error, tune ≤2017, test 2021-23.
3. **Long-term absence in the rest-of-season simulation.** A player who has missed the team's last N games has his X1 lineup value removed from the team's future games until he returns. This applies the X1 machinery to the season simulation. It ships behind a switch with a backtest on 2021-23 standings CRPS; tuning of N is ≤2017.
4. **Playoff-odds history on the website.** A chart of each team's playoff odds by date, from the committed daily standings files.

## 1.3 (planned): close the live-versus-backtest gaps

Brainstorm, held after 1.2 shipped. Every item is planned into 1.3:

1. **Box-score-first lineups and starters.** Past games' actual dressed skaters and starting goalies come from `boxes_2027.csv`, ahead of NeurHL's pregame files, in the filter's update. This matches the X1 backtest, which used box scores. Live, today's games still use pregame files. Test: a unit test, plus a check that `--model 1.3` reproduces 1.2 when no box scores exist.
2. **In-season goalie talent.** Each goalie's save talent is updated with his season shots and goals against, by the same Gamma-Poisson logic as 1.2's skaters (prior weight in shots faced). It feeds the starter offsets. Test: a walk-forward backtest on the box scores; tune ≤2017, test 2021-23 once; metric is rest-of-season save% error (shot-weighted).
3. **Calibrated player-game probabilities.** P(goal) and P(point) for a game are scored against the box scores of 2021-23, using the dress-weighted updated rates. If they are miscalibrated, a dispersion (negative binomial) or Platt correction is fitted on ≤2017. Test: log loss and calibration slope on 2021-23, run once.
4. **Season-to-date on the website.** The skater table shows games, goals, assists and points so far, beside rest-of-season and full-season projections, when box scores exist.
