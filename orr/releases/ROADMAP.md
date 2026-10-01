# ORR roadmap

A version ships every cycle. Each cycle opens with a brainstorm, and every feature considered there is planned into the next version. A feature that fails its backtest still ships, but behind a switch with the result documented; it is not silently dropped. Release notes are in this folder.

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
