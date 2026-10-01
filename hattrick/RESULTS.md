# HatTrick vs NeurHL: results

*Written 2026-10-01. Backtest files are in `hattrick/output/backtest/`. The 2026-27 freeze is in `hattrick/output/freeze_2027/`, and its `manifest_2027.json` records every input's SHA-256 and last-change time; the newest repository input predates the 2026-09-29 21:00 UTC cutoff. Each number below cites the file it comes from. Where HatTrick loses or ties, the table says so.*

## Head-to-head summary

| Layer | Test (same data, same population) | HatTrick | NeurHL | Verdict |
|---|---|---|---|---|
| **Skaters** | NeurHL's own protocol: points MAE, players with ≥40 GP, held-out 2022-26, raw points (its `player_backtest.json`) | **9.42** using preseason rosters only (first-10-games proxy) | 9.54 (path A; it knew each player's actual season-V team) | HatTrick better, with less information |
| | Same sample, NeurHL v2's per-82 restatement (`player_season_v2.json`) | 9.23 | **9.08** (its A/B blend) | NeurHL better by 0.15 |
| | Every rostered player with NHL history (no outcome filter) | 9.34, bias −0.31 | not published | — |
| | Regression toward the mean: slope of projected P/GP on last season's P/GP | 0.83 (2027) | 0.96 (1.3) | HatTrick regresses; NeurHL barely does |
| | 80% interval coverage, held out, preseason rosters (2022-24) | 0.79–0.81 | none published; NeurHL's intervals are ~1.1× Poisson width | HatTrick calibrated |
| **Games, in-season** | NeurHL's 11,052 restatement games 2017-18..2025-26, log loss | **0.6641** with no lineups or starters; 0.6635 with starters where known | 0.6645 (NeurHL-H, which was given actual dressed lineups and starters) | Tie (−0.0004, 95% CI −0.0020 to +0.0011), with far less information |
| | NeurHL-G gate games 2019-24 (n = 6,289) | 0.6610 / 0.6603 | **0.6601** (G stack) | NeurHL slightly better, not significant |
| | NeurHL seal games, 2024-25 and 2025-26 (n = 2,624) | **0.6712** | 0.6715 | Tie |
| | Same 11,052 games vs NeurHL's own Elo baseline | 0.6641 | Elo 0.6691 | HatTrick −0.0050 (CI −0.0072 to −0.0029) |
| **Games, preseason-frozen** | Every game of 2018-2026 predicted from opening-night information only | **0.6742** | no historical preseason game file exists; Elo freeze 0.6760 | HatTrick beats a frozen Elo, −0.0018 (CI −0.0033 to −0.0004) |
| **Standings** | NeurHL's matched seasons 2012, 2014-17 (points MAE per 82) | **8.96** (team-history view) | 9.99 shipped NeurHL-2; 9.17 its Elo | HatTrick better |
| | NeurHL's 2019-24 judge window (158 team-seasons), MAE / CRPS | 9.47 / 6.88 (market + roster, leave-one-season-out) | **9.42 / 6.80** (frozen engine layer); Elo 9.47 / 6.87 | NeurHL slightly better; all three effectively tied |
| | All seasons with preseason lines, 2019-26 (leave-one-season-out MAE) | 9.76 (market alone 9.90) | not tested | — |
| **Goalies** | GSAx/60, held-out 2022-26, goalies with ≥1000 shots (MAE) | 0.237 | n/a; NeurHL's starter model loses to a share heuristic | Ties league average (0.241), beats last season (0.336) |
| **Live 2026-27** | 5 opening-night games, log loss | 0.761 | 0.739 (1.1), 0.740 (1.3) | Noise (n = 5) |

On these backtests, HatTrick matches or beats NeurHL in game prediction with far less information (no lineups, no neural networks). Its skater projections beat NeurHL's published numbers on NeurHL's own headline protocol without NeurHL's future-roster leak, and lose narrowly on the per-82 protocol. On standings, HatTrick is clearly better on 2012-17 and level on 2019-24, where NeurHL's layer, Elo and the sportsbook lines are all within 0.1 points of each other. HatTrick's projections are calibrated where NeurHL's are not: regression slope, interval coverage, and team spread versus the market.

## The 2026-27 forecast

`teams_2027.csv`, `games_2027.csv`, `skaters_2027.csv` and `goalies_2027.csv` come from 40,000 simulated seasons.

- **Standings.** HatTrick is the August sportsbook line, re-priced for the news it had not seen, blended with a bottom-up roster view (weights 0.70 / 0.36, fitted on 2019-26). Its team means have SD 9.1 (market 9.4; NeurHL 1.3 10.4) and correlate 0.97 with the line (NeurHL 1.3: 0.88). Its Cup odds correlate 0.96 with the de-vigged Cup market (NeurHL 1.3: 0.82).
- **News priced between the 2026-08-17 line and the cutoff,** in points per 82, from the player model:

  | Team | News | What happened |
  |---|---|---|
  | WPG | −3.1 | Hellebuyck asked for a trade 08-27 and was suspended 09-17; HatTrick gives a 10% chance he plays for Winnipeg and spreads the trade-destination probability across CAR/BUF/UTA/SJS |
  | NSH | −2.3 | Evangelista traded 09-02, Barron to STL |
  | ANA | −2.2 | Kreider's contract terminated (he signed with MTL), Terry out ~30 games |
  | MTL | +3.0 | Kreider signed |
  | NJD | +1.6 | Evangelista acquired |
  | NYR | +1.7 | Tolvanen signed |

  The Knies/Marchenko/Andrae trade of 09-28 nets CBJ +0.7 and TOR +0.9. Injuries come from 46 researched timelines (sources dated ≤ 2026-09-29).
- **Largest disagreements with NeurHL 1.3:**

  | Team | HatTrick | NeurHL 1.3 | Market line |
  |---|---|---|---|
  | TOR | 96.7 | 81.0 | 95.5 |
  | OTT | 95.0 | 103.0 | 95.5 |
  | WSH | 104.4 | 96.9 | 102.5 |
  | SJS | 90.9 | 83.9 | 93.5 |
  | COL | 106.7 | 113.3 | 108.5 |
  | CBJ | 90.8 | 97.4 | 92.5 |
  | NYI | 85.3 | 91.7 | 86.5 |
  | VAN | 74.7 | 68.5 | 74.5 |
- **Game file.** Home-win probabilities average 0.541 (SD 0.079, range 0.32–0.76). Every game's probabilities are averaged over 400 draws of both teams' ratings, so they are not over-confident. NeurHL 1.3's range is 0.27–0.79.
- **Skaters.**

  | Player | HatTrick (80% band) | NeurHL 1.3 |
  |---|---|---|
  | McDavid | 129.3 (89–170) | 137.9 (119–158) |
  | Kucherov | 112.4 | 104.3 |
  | MacKinnon | 108.5 | 119.0 |
  | Draisaitl | 104.3 | 93.5 |
  | Celebrini | 101.4 | 98.8 |

  HatTrick's intervals include talent, games-played, usage and scoring noise; NeurHL's are near-Poisson. Team goals equal the sum of skater goals plus call-ups. League scoring is anchored to 2025-26 (3.08 goals per team-game, no shootout goals).
- **Goalies.** There is no start cap: Vejmelka 63.6 starts, Swayman 60.7, Vasilevskiy 58.9, Sorokin 58.3. NeurHL caps every goalie at 52.7. Goalie GA is reconciled to team GA net of empty-net and shootout goals.

## How each layer was tested

**Skaters** (`players_bt.json`, `players_search_ledger.json`):
- Tuning used only 2011-2017 (76 configurations, all logged). 2018-19 serve as confirmation and 2022-26 as the held-out test.
- "Strict" means opening rosters from each team's first 10 games and nothing else from season V. It is available for 2011-2024; 2025-26 fall back to the season team and are flagged.
- "Matched" uses NeurHL's own information set: the team a player played for most in season V.
- Points MAE by held-out season, strict protocol, sample A:

  | Season | HatTrick | NeurHL |
  |---|---|---|
  | 2022 | 9.72 | 10.47 |
  | 2023 | 9.48 | 9.60 |
  | 2024 | 9.29 | 9.17 |
  | 2025 | 9.07 | 9.09 |
  | 2026 | 9.53 | 9.39 |

- Points MAE 9.42 sits beside goals MAE 4.31 and assists MAE 6.36. Correlation is 0.844.
- The final projection averages HatTrick (0.6), Marcel (0.3) and HatTrick's rates times last season's GP (0.1). Rookies are calibrated by draft slot.

**Games** (`games_bt.json`, `games_search_ledger.json`, 295 configurations scored on 2012 and 2014-17 only):
- Every test prediction is walk-forward. The model's parameters (home ice, rest/travel, goalie coefficient, end-game layer, OT/SO) are re-estimated from earlier seasons only.
- The scoring model is a Conway-Maxwell-Poisson base with a mean-preserving pulled-goalie end-game layer. It wins regulation-margin log-likelihood in every test season: −2.107 against −2.159 for Poisson and −2.147 for diagonal tie inflation.
- Test-window calibration: logistic slope 1.009; regulation ties 0.219 predicted vs 0.224 observed; shootout share of ties 0.337 vs 0.330; home win 0.545 vs 0.539.

**Standings** (`teams_bt.json`):
- Seasons with a preseason line (2019-26) are scored leave-one-season-out over a pre-declared list of view sets.
- NeurHL's backtest seasons (2012, 2014-17) are scored walk-forward.
- The market lines are late-preseason values, hand-collected from Hockey-Reference and cross-checked (`data_market_history.csv`, 222 team-seasons).

## Where HatTrick does not win

- **Skaters on the per-82 protocol.** NeurHL's A/B blend is 0.15 points better (9.08 vs 9.23).
- **Standings on 2019-24.** NeurHL's frozen engine layer is 0.05 MAE and 0.08 CRPS better than HatTrick. The sportsbook line carries most of HatTrick's standings forecast, and on those seasons the line was not better than NeurHL's layer.
- **Goalie save talent.** It is barely predictable (correlation 0.23), so HatTrick's goalie projections tie "league average". HatTrick's goalie 80% intervals cover only 0.72.
- **Rookies.** Thin-history players remain the largest single error source after games played.
- **Opening night.** On the first 5 games HatTrick's frozen file (0.761) is worse than NeurHL 1.1 (0.739). That is too few games to mean anything; the season will decide.

## Live scoring

`python3 -m hattrick.score` scores every frozen file on completed games:

- HatTrick's freeze;
- NeurHL's 09-25 freeze, 1.0 and 1.1;
- 1.2 and 1.3, flagged as published after the first puck drop;
- NeurHL-G, H and Elo pregame forecasts;
- HatTrick's in-season forecasts.

Interim team scoring compares points earned with each model's expected points for the games actually played. The output is `hattrick/output/scorecard_2027.json`.

In-season, `python3 -m hattrick.inseason --date YYYY-MM-DD --results results.csv [--goalies starters.csv]` updates ratings from results, forecasts the day, and re-runs the season. A forecast counts only if it was pushed to GitHub before puck drop.
