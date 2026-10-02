# ORR 2.0 vs NeurHL 1.3

*Written 2026-10-02, after 8 of 1,344 regular-season games. It compares ORR 2.0, the in-season default, which uses the frozen ORR 1.0 preseason file, with NeurHL 1.3, NeurHL's current release (2026-09-30), and its game-day forecasts. Numbers that compare the two 2026-27 forecasts come from `python3 -m orr.compare_neurhl`, which writes `orr/output/compare_neurhl_1_3.json`. Backtest numbers come from `orr/RESULTS.md`. Live numbers come from `orr/output/scorecard_2027.json`. The author of this file also built ORR; read it with that in mind.*

## Bottom line

- **Live results so far show nothing.** Eight games have been played. No ORR daily forecast has yet been scored on a game it was eligible for. ORR's preseason file is eligible for one game and NeurHL 1.3 for three. Scores on that few games are noise.
- **Standings: ORR is better in backtests, mainly because it follows the betting market.** On NeurHL's own 2019-24 test window, ORR's standings beat NeurHL's engine layer: points MAE 9.23 vs 9.42. The market alone (9.32) also beats NeurHL. For 2026-27, ORR's team points are within 1.2 points of the preseason points lines on average, once the league average is removed. NeurHL's are 3.8 points away. Every large NeurHL departure is a bet against the market, led by Toronto at 14.5 points below it.
- **Game-by-game: a tie at best for ORR.** On NeurHL's 2019-24 gate games, ORR's in-season loop scores 0.6600 log loss with shots, against 0.6601 for NeurHL-G. On goals alone, NeurHL-G is better by 0.0021, which is not significant. ORR gets shots only from its daily GitHub Action, so its game forecasts depend on that Action running.
- **Skaters: level on NeurHL's headline test, behind on the other.** On points MAE for skaters with 40+ games, ORR scores 9.51 against NeurHL's 9.54. On NeurHL's per-82 test, NeurHL is better (9.08 vs 9.32). NeurHL also projects higher peaks and fuller seasons.
- **Goalies: neither has shown skill.** ORR's held-out goalie projections only match a league-average baseline. NeurHL publishes no comparable test. Their 2026-27 save-percentage projections agree only loosely (correlation 0.53).
- **One season probably cannot settle the game forecasts.** In backtests the gap between the two is under 0.003 log loss per game. With one season of eligible games, the 95% CI on the gap is about ±0.0055 (see the last section). The preregistered game comparison will most likely end "no difference shown", unless one system is much better live than in its backtests.

## The two systems

| | ORR 2.0 | NeurHL 1.3 |
|---|---|---|
| **Approach** | Top-down. A team rating anchored to the betting market comes first. Players are fitted inside the team totals. | Bottom-up. A player-game engine comes first. Team and season totals are sums of simulated player-games. |
| **Team strength** | A blend of preseason points lines, team history and roster change, with weights tuned on past seasons. It becomes offence and defence ratings on the actual schedule. | Emerges from the dressed players' projected ice time, shots, xG and goals. It is stacked with Elo and a neural player layer (NeurHL-H) for game probabilities. |
| **Game model** | One scoring model for every game: a Conway-Maxwell-Poisson base, a pulled-goalie layer, overtime and shootout models, rest and travel, and starting goalie. | A score-and-time hazard integration of the two lineups' scoring rates, with overtime and shootouts. |
| **Players** | Per-60 rates regressed to the mean and adjusted for age, a depth chart for who dresses, and team ice time conserved. In-season, Gamma-Poisson updates of goals, assists and shots per game. | Per-game ice time by strength, shots, xG, goals and assists from the engine. Rookies come from translated records in other leagues. Stat sheets include hits, blocks, giveaways, takeaways, faceoffs and PIM. |
| **Season simulation** | 40,000 seasons preseason and 20,000 per day in-season, with the exact tiebreakers and bracket. Rating drift is calibrated on held-out seasons. | 20,000 seasons. Lineups and goalie starts are sampled from rosters and an availability model. |
| **In-season** | A Kalman-style team filter on goals and shots, adjusted for dressed lineups and starters. Goalie talent, start shares and skater rates are updated. It publishes clinch flags, magic numbers and rest-of-season player and goalie files with intervals. | Each player's, goalie's and team's state is updated after every game. A morning forecast and a pregame forecast about an hour before puck drop. An exploratory nightly standings projection. |
| **Lineups** | Read from NeurHL's committed pregame lineup files, so ORR's daily forecasts depend on NeurHL's pipeline. | Its own pipeline. |
| **Validation** | Preregistered. Tuned on seasons up to 2016-17; each feature tested once on 2021-22 and 2022-23 under a rule declared beforehand. Three features failed and ship switched off. | Preregistered. Gates and sealed seasons, with final tests run once on seasons no decision had touched. |
| **Publication** | Preseason file published 2026-10-01 00:37 UTC, after 7 of the first 8 games had started, so it is scored on fewer games. | NeurHL 1.0 frozen before opening night; 1.3 issued before the 2026-09-30 games. |
| **Reproducibility** | Every daily run records its code commit and the SHA-256 of its inputs. `orr.reproduce` re-runs a published day. All three published daily forecasts re-run identically: 10-01 (1.7), and 10-02 with 1.9 and with 2.0. | Frozen files are identified by SHA-256 and forecasts are committed before the games. |

## Head-to-head backtests

Same games or players, same population. Lower is better.

| Test | ORR | NeurHL | Verdict |
|---|---|---|---|
| Standings, 2019-24 test window (158 team-seasons), points MAE / CRPS | **9.23 / 6.70** | 9.42 / 6.80 (engine layer) | ORR better. The market alone (9.32 / 6.71) also beats NeurHL. |
| Standings, NeurHL's backtest seasons 2012 and 2014-17, points-per-82 MAE | **9.02** | 9.99 | ORR better in 4 of 5 seasons |
| In-season games, NeurHL-G gate games 2019-24 (6,289), log loss, goals and shots | **0.6600** | 0.6601 | Tie (CI −0.0025 to +0.0025) |
| Same, goals only | 0.6622 | **0.6601** | NeurHL better by 0.0021, not significant (CI −0.0007 to +0.0047) |
| In-season games, NeurHL's sealed 2024-26 seasons (2,624), ORR 1.0 loop | **0.6712** | 0.6715 | Tie |
| Skaters, points MAE, ≥40 GP, 2022-26 ("sample A") | **9.51** | 9.54 | Level |
| Skaters, per-82 points MAE, 2022-26 ("sample B") | 9.32 | **9.08** | NeurHL better by 0.23 |
| Goalies, GSAx/60 MAE, ≥1,000 shots, 2022-26 | 0.237 | not published | ORR only ties league average (0.241) |

ORR 2.0's game forecasts follow the ORR 1.1 loop tested above, with two additions: in-season goalie talent and in-season start shares. Neither was tested on game log loss. NeurHL's numbers are for its gate engine. NeurHL 1.3 runs a refit engine (`g2027_v3`) that has not been compared head-to-head. NeurHL reports, descriptively, that the refit engine beat Elo by 0.0069 per game on 2025-26.

## How the 2026-27 forecasts differ

### Teams

They agree on the order of teams (rank correlation 0.90; points correlation 0.92), with a mean gap of 3.3 points per team. NeurHL spreads teams further apart: its points SD is 10.4, against 9.4 for the market lines and 8.7 for ORR. The two have almost the same 80% interval widths, 31.5 and 32.4 points on average.

| Team | ORR | NeurHL 1.3 | Market line | ORR playoff % | NeurHL playoff % |
|---|---|---|---|---|---|
| TOR | 93.5 | 81.0 | 95.5 | 44 | 12 |
| VAN | 75.7 | 68.5 | 74.5 | 11 | 5 |
| SJS | 90.3 | 83.9 | 93.5 | 48 | 30 |
| NSH | 84.4 | 90.1 | 85.5 | 28 | 44 |
| CBJ | 91.2 | 97.4 | 92.5 | 37 | 54 |
| OTT | 95.6 | 103.0 | 95.5 | 51 | 70 |

ORR is within 3.2 points of the market line for every one of these teams. NeurHL is 4.6 to 14.5 points away. So these six teams are where the season will show whether NeurHL's bottom-up view adds anything to the market. Toronto is the clearest test.

Both make Colorado and Carolina the Cup favourites, but NeurHL is more confident:

| | Colorado | Carolina |
|---|---|---|
| NeurHL | 16.5% | 14.9% |
| ORR | 12.3% | 11.7% |

### Skaters

Projected points agree closely (correlation 0.97, mean gap 4.2 points). Games played agree less (correlation 0.85, mean gap 6.3 games), and most large gaps come from games played, not from scoring rate. ORR's injury inputs take more games off. For example:

| Player | ORR | NeurHL |
|---|---|---|
| Barkov | 61 GP, 44 points | 76 GP, 78 points |
| Barzal | 57 GP, 59 points | 78 GP, 88 points |
| Fox | 60 GP, 49 points | 72 GP, 72 points |

NeurHL projects higher peaks:

| | Top projection | Players projected for 40+ goals |
|---|---|---|
| NeurHL | McDavid 137.9 points and 49.7 goals | 11 |
| ORR | McDavid 128.1 points and 41.5 goals | 6 |

Both figures are expected values. The real league leaders will beat both, because the leader is the luckiest of many players. ORR regresses on purpose: its projected points per game moves 0.86 for each point of last season's rate, against 0.96 for NeurHL. On NeurHL's per-82 test NeurHL scores better, but whether its lighter regression is the reason has not been tested.

### Games

The two preseason game files correlate at 0.91. Their home-win probabilities differ by 0.030 on average, at most by 0.15, and they favour different teams in 12.7% of games. NeurHL's probabilities are more spread out (SD 0.091 vs 0.079), which matches its wider team spread.

### Goalies

For the 68 goalies in both files, projected save percentages correlate at only 0.53. The two disagree more about goalies than about anything else. Neither has a held-out record that would justify trusting one view over the other.

## Live record (through 2026-09-30, 8 games)

| Forecast | Eligible games | Log loss |
|---|---|---|
| ORR preseason file | 1 | 0.481 |
| ORR daily (2.0) | 0 | — |
| NeurHL 1.3 | 3 | 0.673 |
| NeurHL 1.0 freeze | 8 | 0.703 |
| NeurHL-G pregame | 8 | 0.721 |
| NeurHL-H pregame | 8 | 0.708 |
| Elo pregame | 8 | 0.692 |

A forecast counts only for games that started after it was published. The season-end judgement follows the preregistered protocol in `orr/EVALUATION_2027.md`, fixed on 2026-10-02 with file hashes. It has four comparisons, decided only when a CI excludes zero:
- P1: ORR daily vs NeurHL-G, game log loss;
- P2: the two preseason game files;
- P3: team points MAE;
- P4: skater points MAE at 40+ GP.

It compares against NeurHL 1.1, the release current at ORR's data cutoff (2026-09-29 17:00 ET). NeurHL 1.3 is not in its primary comparisons.

## Where each is stronger

**NeurHL**
- Richer player output: full stat sheets, sampled lineups and a neural player layer.
- Better per-82 skater accuracy.
- Its own lineup pipeline.
- Preseason forecasts published on time.
- Better in-season game forecasts when ORR has no shots data.

**ORR**
- More accurate standings in backtests.
- Calibrated intervals, tested on held-out seasons: 80% rest-of-season player intervals cover 0.80.
- Day-by-day reproducibility.
- A public record of the features that failed.
- A simpler design with fewer moving parts.

**Neither has shown**
- Goalie skill beyond league average.
- Playoff-round probabilities tested on held-out playoffs. ORR's are untested, and NeurHL publishes no such test.

## What would change this comparison

- **Games will probably stay undecided.** The backtest gap between ORR and NeurHL-G has a per-game SD of about 0.10 log loss, from the gate test's CI of ±0.0025 on 6,289 games. With about 1,300 eligible games this season, the 95% CI on the live gap is about ±0.0055. Any real gap the backtests suggest (under 0.003) is too small to detect, so P1 will probably end "no difference shown".
- **Teams are the likeliest place for a clear result.** The forecasts differ by up to 12.5 points on individual teams, and the market sides with ORR. If NeurHL's departures from the market pay off (Toronto, San Jose, Ottawa, Columbus), that is evidence for its bottom-up approach. If they don't, ORR's standings edge holds. With 32 teams, only a large average difference will clear the CI.
- **Shots decide ORR's game forecasts.** If the daily Action supplies shots, ORR's game forecasts should be level with NeurHL-G. If it fails and ORR runs on goals alone, NeurHL-G should be slightly ahead.
