# ORR roadmap

A version ships every cycle. Each cycle opens with a brainstorm, and every feature considered there is planned into the next version. A feature that fails its backtest still ships, but behind a switch with the result documented; it is not silently dropped. Release notes are in this folder.

## 1.3 (planned): close the live-versus-backtest gaps

Brainstorm, held after 1.2 shipped. Every item is planned into 1.3:

1. **Box-score-first lineups and starters.** Past games' actual dressed skaters and starting goalies come from `boxes_2027.csv`, ahead of NeurHL's pregame files, in the filter's update. This matches the X1 backtest, which used box scores. Live, today's games still use pregame files. Test: a unit test, plus a check that `--model 1.3` reproduces 1.2 when no box scores exist.
2. **In-season goalie talent.** Each goalie's save talent is updated with his season shots and goals against, by the same Gamma-Poisson logic as 1.2's skaters (prior weight in shots faced). It feeds the starter offsets. Test: a walk-forward backtest on the box scores; tune ≤2017, test 2021-23 once; metric is rest-of-season save% error (shot-weighted).
3. **Calibrated player-game probabilities.** P(goal) and P(point) for a game are scored against the box scores of 2021-23, using the dress-weighted updated rates. If they are miscalibrated, a dispersion (negative binomial) or Platt correction is fitted on ≤2017. Test: log loss and calibration slope on 2021-23, run once.
4. **Season-to-date on the website.** The skater table shows games, goals, assists and points so far, beside rest-of-season and full-season projections, when box scores exist.
