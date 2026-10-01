# HatTrick vs NeurHL: results

*Written 2026-10-01. Backtest outputs are in `hattrick/output/backtest/`. The 2026-27 forecast is in `hattrick/output/freeze_2027/`. Its `manifest_2027.json` records the SHA-256 of every input and of the code, and every repository input is read from commit `9580606` (2026-09-29 16:53 EDT), the last commit before the cutoff. Each number below names the file it comes from. Where HatTrick loses or ties, the tables say so.*

**Timing, stated plainly.**

- The review ([`review/REVIEW_NeurHL.md`](../review/REVIEW_NeurHL.md), commit `99cb632`) came before any HatTrick code.
- The 2026-27 forecast uses only information dated before 2026-09-29 21:00 UTC. It was built on 2026-09-30 and 2026-10-01, after the season had begun.
- No 2026-27 result enters any fitted quantity, but the builder could have seen opening-night scores. So the preseason file is compared with NeurHL on what it knew, not on when it was published.
- The scorer applies the same rule to every model: a file counts only for games that start after it was published. NeurHL 1.0 and 1.1 were published before the first puck drop; HatTrick's preseason file and NeurHL 1.2/1.3 were not.

## Head-to-head summary

| Layer | Test (same games or players, same population) | HatTrick | NeurHL | Verdict |
|---|---|---|---|---|
| **Standings** | NeurHL's 2019-24 judge window: 158 team-seasons, raw points over games played; MAE / CRPS | **9.23 / 6.70** (market + team history, convex weights, leave-one-season-out) | 9.42 / 6.80 (frozen engine layer); Elo 9.47 / 6.87 | HatTrick better. The market alone (9.32 / 6.71) already beats NeurHL. |
| | NeurHL's own backtest seasons 2012, 2014-17, points-per-82 MAE | **8.96** (team-history view) | 9.99 (shipped NeurHL-2 layer); 9.17 (its v1 Elo + xG) | HatTrick better in 4 of 5 seasons; 2012 lost, 7.20 vs 6.98 |
| **Games, preseason file** | Shipped game-file pipeline rebuilt for 2019-24 (6,289 games), log loss | **0.6674** | NeurHL has no historical preseason game file. A preseason-frozen Elo scores 0.6718. | −0.0044 vs Elo (95% CI −0.0067 to −0.0024). Also −0.0033 vs HatTrick's own team-history table (0.6707). |
| **Games, in-season** | NeurHL's 11,052 restatement games, 2017-18 to 2025-26 | 0.6641 with no lineups; **0.6635** with starters where known | 0.6645 (NeurHL-H, given actual dressed lineups and starters) | Tie, with far less information |
| | Same games vs NeurHL's own Elo baseline | 0.6641 | Elo 0.6691 | HatTrick −0.0050 (CI −0.0072 to −0.0029) |
| | NeurHL-G gate games, 2019-24 (n = 6,289) | 0.6610 / 0.6603 | **0.6601** (G stack) | NeurHL slightly better; not significant |
| | NeurHL seal games, 2024-25 and 2025-26 (n = 2,624) | **0.6712** | 0.6715 | Tie |
| **Skaters** | NeurHL's headline protocol: points MAE for players with ≥40 GP ("sample A"), held-out 2022-26 | **9.41** | 9.54 | HatTrick better (details and caveats below) |
| | Same, 2022-24 only (HatTrick's roster = first 10 games; NeurHL's = each player's actual season team) | **9.49** | 9.74 | HatTrick better by 0.25 |
| | Same, 2025-26 only (both use the actual season team; no box scores for a proxy) | 9.30 | **9.24** | NeurHL better by 0.05 |
| | NeurHL v2's per-82 restatement ("sample B"), 2022-26 | 9.22 | **9.08** | NeurHL better by 0.14 (tie on 2022-24, 9.30 vs 9.28) |
| | Regression toward the mean: slope of projected P/GP on last season's P/GP, 2027 file | 0.83 | 0.96 (1.3) | HatTrick regresses; NeurHL barely does |
| | 80% interval coverage of points, held out | 0.81 pooled (0.77 on 2022-24, 0.88 on 2025-26) | none published; intervals ~1.1× Poisson width | HatTrick near nominal |
| **Goalies** | GSAx/60 MAE, held-out 2022-26, goalies with ≥1,000 shots (n = 178) | 0.237 | n/a | Ties league average (0.241); beats last season (0.335) |
| **Live 2026-27** | 5 opening-night games, log loss | 0.750 (published after these games; not eligible) | 0.726 (1.0), 0.739 (1.1) | Noise (n = 5); eligible from 2026-10-01 |

**Summary.**

- **Standings:** HatTrick beats NeurHL on NeurHL's own judge window and on its backtest seasons. Most of the 2019-24 gain comes from using the sportsbook line, which is allowed new data; HatTrick's own modelling adds about 0.1 points of MAE on top.
- **Games:** HatTrick ties NeurHL's best in-season models with no lineup information and no neural network. Its shipped preseason pipeline beats a frozen Elo with a confidence interval that excludes zero.
- **Skaters:** HatTrick wins NeurHL's headline protocol, while NeurHL's backtest uses a future-roster leak that HatTrick's does not. HatTrick loses the per-82 restatement and the two most recent seasons.
- **Calibration:** HatTrick's projections are calibrated where NeurHL's are not: regression slope, interval coverage, and team spread versus the market.

## The 2026-27 forecast

`teams_2027.csv`, `games_2027.csv`, `skaters_2027.csv` and `goalies_2027.csv` come from 40,000 simulated seasons. The weights and diagnostics are in `state_2027.json`.

- **Standings target.**
  - **Weights.** 0.75 × the de-vigged 2026-08-17 points line, re-priced for later news, plus 0.25 × the regressed team-history view, in points per 82 above the league mean.
  - **How they were chosen.** Convex weights (non-negative, summing to 1) were picked by leave-one-season-out over six pre-declared view sets on the five clean seasons (2019, 2020, 2022-24) (`teams_bt.json`). The top three sets are within 0.03 MAE of each other, so the choice matters little.
  - **Calibration.** Team means have SD 8.7 (market 9.4; NeurHL 1.3 10.4). They correlate 0.988 with the line (NeurHL 1.3: 0.884), and the Cup odds correlate 0.944 with the de-vigged Cup market (NeurHL 1.3: 0.815).
- **Top of the table.**

  | Team | Points | Playoffs | Presidents' Trophy | Cup |
  |---|---|---|---|---|
  | COL | 109.4 | 92% | 17% | 12.3% |
  | CAR | 108.7 | 86% | 15% | 11.7% |
  | TBL | 104.9 | 78% | 9% | 7.8% |
  | FLA | 104.0 | 76% | 8% | 7.2% |

- **News priced between the 2026-08-17 line and the cutoff.** These are points per 82, from the player model: an event counterfactual with each move undone and each later-reported absence healed, at 0.33 standings points per goal of differential.

  | Team | News | Events priced |
  |---|---|---|
  | MTL | +3.0 | Kreider signed (his ANA contract was terminated) |
  | NJD | +1.6 | Evangelista acquired from NSH |
  | ANA | −2.6 | Kreider gone |
  | NSH | −2.3 | Evangelista traded, Barron to STL |
  | WPG | −2.0 | Hellebuyck's trade request and suspension. HatTrick gives a 10% chance he plays for Winnipeg and credits no other team with him. |
  | OTT | −1.9 | Foegele on no roster at the cutoff, Meriläinen to VAN |
  | CBJ | −1.7 | Marchenko and Merzlikins to TOR, Knies and Andrae in; Severson and Wood on no roster at the cutoff |
  | DET | −1.4 | Edvinsson and Bryson on no NHL roster at the cutoff |
  | FLA | −1.0 | Marchand on IR (09-17); two depth departures |

  A "departure" is a player with 50+ NHL games in 2025-26 who was removed from a club after 08-17 and is on no NHL roster or injured list at the cutoff. Injuries come from 46 researched timelines with sources dated on or before 2026-09-29. Five long-known injuries (Terry, Bedard, Jarvis, Demko's hip, Sandin) predate the line and are not counted as news.
- **Largest disagreements with NeurHL 1.3:**

  | Team | HatTrick | NeurHL 1.3 | Market line |
  |---|---|---|---|
  | TOR | 93.5 | 81.0 | 95.5 |
  | OTT | 95.6 | 103.0 | 95.5 |
  | VAN | 75.7 | 68.5 | 74.5 |
  | SJS | 90.3 | 83.9 | 93.5 |
  | CBJ | 91.2 | 97.4 | 92.5 |
  | NSH | 84.4 | 90.1 | 85.5 |
  | BUF | 94.7 | 99.6 | 95.5 |
  | BOS | 87.2 | 92.0 | 86.5 |

- **Game file.** Home-win probabilities average 0.539 (SD 0.079, range 0.33–0.76). Each game's probability is averaged over 400 joint draws of both teams' ratings, so preseason uncertainty widens it toward 50%. NeurHL 1.3's range is 0.27–0.79.
- **Skaters.**

  | Player | HatTrick (80% band) | NeurHL 1.3 |
  |---|---|---|
  | McDavid | 128.1 (88–168) | 137.9 |
  | Kucherov | 113.4 (82–147) | 104.3 |
  | MacKinnon | 110.5 (80–142) | 119.0 |
  | Draisaitl | 103.4 (70–136) | 93.5 |
  | Celebrini | 101.5 (55–140) | 98.8 |
  | Pastrnak | 94.1 (69–124) | 106.0 |
  | Robertson | 89.6 (62–119) | 103.2 |

  - HatTrick's intervals include talent, games-played, usage and scoring noise.
  - Team goals equal the sum of skater goals plus call-ups.
  - League scoring is anchored to 2025-26 (3.08 goals per team-game, no shootout goals).
  - Each player's simulation seed is derived from his id, so one player's line does not move when another player is added or removed.
- **Goalies.**
  - There is no start cap. Sorokin projects 62.0 starts, Vejmelka 61.3, Vasilevskiy 61.2 and Swayman 59.8; six goalies project above 55. NeurHL 1.3 caps every goalie at 52.7.
  - Goalie goals-against are reconciled to team goals-against, net of empty-net and shootout goals.
  - Save talent is heavily regressed: the projected SV% of every goalie with 15+ starts lies between .878 and .909.

## How each layer was tested

**Standings** (`teams_bt.json`):

- **Population.** Seasons with a preseason line (2019-26, 222 team-seasons, `data_market_history.csv`). Lines are hand-collected late-preseason values from Hockey-Reference, cross-checked.
- **Selection.** View sets are scored leave-one-season-out. Selection uses the five clean seasons, where a first-10-games roster proxy exists for the bottom-up and roster-delta views.
- **Two scales.**
  - *rel*: points per 82, centred on the league mean, HatTrick's own scale.
  - *judge raw*: uncentred points over games played, NeurHL's `season_layer_c2.py` scale. NeurHL's layer is quoted on this scale.

  The shipped mkt+td scores rel 9.42 MAE on the clean seasons and 9.83 on all seven.
- **NeurHL's backtest seasons (2012, 2014-17)** are scored walk-forward with the team-history view. No lines exist there, so the shipped blend cannot be evaluated on them.

**Games** (`games_bt.json`, `games_search_ledger.json`, `gamefile_bt.json`):

- **Tuning.** 295 configurations, scored on 2012 and 2014-17 only.
- **Walk-forward.** Every test prediction is walk-forward. The model's parameters (home ice, rest/travel, goalie coefficient, end-game layer, OT/SO) are re-estimated from earlier seasons only.
- **Scoring model.** A Conway-Maxwell-Poisson base with a mean-preserving pulled-goalie end-game layer. It wins regulation-margin log-likelihood in every test season: −2.107, against −2.159 for Poisson and −2.147 for diagonal tie inflation.
- **Test-window calibration.** Logistic slope 1.009; regulation ties 0.219 predicted vs 0.224 observed; shootout share of ties 0.337 vs 0.330; home win 0.545 vs 0.539.
- **`gamefile_bt.py`** rebuilds the shipped preseason pipeline for each past season with `freeze.py`'s own functions:
  - leave-one-season-out blend weights;
  - ratings solved on the season's actual schedule;
  - rating SD from out-of-sample error net of game luck;
  - league level;
  - rating-averaged probabilities.

  It has no historical news layer. Per season it is better than Elo in all five seasons (significantly in 2023). It ties the team-history table in 2022.

**Skaters** (`players_bt.json`, `players_search_ledger.json`):

- **Tuning** used only 2011-2017 (76 configurations, all logged). 2018-19 confirm; 2022-26 is the held-out test.
- **Rosters.** "Strict" uses opening rosters from each team's first 10 games and nothing else from season V. It is available through 2024. That proxy itself knows who dressed in those 10 games, a far smaller leak than NeurHL's full-season team. For 2025-26 no box scores exist, so both models use the season team, and that window is reported separately above.
- **Accuracy.** Pooled points MAE 9.41 sits beside goals MAE 4.31 and assists MAE 6.36; correlation is 0.845.
- **Final projection.** An average of HatTrick (0.6), Marcel (0.3) and HatTrick's rates times last season's GP (0.1). Rookies are calibrated by draft slot.

**Goalies** (`goalies_bt.json`):

- Tuned on 2011-17, tested on 2022-26.
- Starts MAE 11.3 over 2012-23.
- 80% GSAx/60 interval coverage 0.74.

## Where HatTrick does not win

- **Skaters on the per-82 protocol and on 2025-26.** NeurHL's A/B blend is 0.14 points better pooled and 0.33 better on 2025-26. HatTrick's skater advantage is on NeurHL's headline protocol, and it is largest where NeurHL's future-roster leak helps it most.
- **In-season games on NeurHL's gate window.** NeurHL-G is 0.0002 better than HatTrick with known starters, which is not significant.
- **Standings, 2012.** NeurHL-2 beats HatTrick's team-history view (6.98 vs 7.20).
- **Standings, own modelling.** On 2019-24 the sportsbook line does most of the work. HatTrick's own contribution over the market alone is about 0.1 points of MAE, smaller than the noise across seasons.
- **Goalie save talent.** It is barely predictable (correlation 0.23), so HatTrick's goalie projections tie "league average". Coverage is 0.74, below nominal.
- **Rookies.** Thin-history players remain the largest single error source after games played.
- **Timing.** NeurHL 1.0 and 1.1 were genuinely published before the season. HatTrick's preseason file was not, so it is not scored on opening night. Its 2026-27 comparison with NeurHL counts only games after its publication.

## Live scoring

- **Scorecard.** `python3 -m hattrick.score` scores every file on completed games:
  - HatTrick's preseason file;
  - NeurHL's 09-25 freeze and releases 1.0-1.3;
  - NeurHL-G, H and Elo pregame forecasts;
  - HatTrick's in-season forecasts.

  Each preseason file is also scored on the games after its own publication time (`after_publication`). A file or daily forecast counts for a game only if published before 15:00 UTC on the game's date, or before the 21:00 UTC first puck drop on opening night. The pushed commit time is the external evidence.
- **Interim team scoring** compares points earned with each model's expected points for the games actually played. At season end the scorer reports points MAE, RMSE and CRPS. The output is `hattrick/output/scorecard_2027.json`.
- **Daily loop.**
  1. `python3 -m hattrick.ingest` appends finished games, from the NHL API or by hand with `--add`, each checked against the schedule.
  2. `python3 -m hattrick.inseason --date YYYY-MM-DD` updates team ratings from results strictly before that date, then forecasts the day's games and players and re-simulates the season.

  The first in-season forecast (2026-10-01) was committed at 00:43 UTC on 10-01, before any 10-01 game. It uses results through 09-29: the 09-30 games were still in progress when it was made.
