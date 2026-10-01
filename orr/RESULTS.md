# ORR vs NeurHL: results

*Written 2026-10-01. Backtest outputs are in `orr/output/backtest/`. The 2026-27 forecast is in `orr/output/freeze_2027/`. Its `manifest_2027.json` records the SHA-256 of every snapshot input and of the code, and every repository input is read from commit `9580606` (2026-09-29 16:53 EDT), the last commit before the cutoff. `fitted_inputs_2027.json` adds the hashes of the fitted parameter files and hand-collected data the freeze read. Each number below names the file it comes from. Where ORR loses or ties, the tables say so.*

**Timing, stated plainly.**

- The review ([`review/REVIEW_NeurHL.md`](../review/REVIEW_NeurHL.md), commit `99cb632`) came before any ORR code.
- The 2026-27 forecast uses only information dated before 2026-09-29 21:00 UTC. It was built on 2026-09-30 and 2026-10-01, after the season had begun.
- No 2026-27 result enters any fitted quantity, but the builder could have seen opening-night scores. So the preseason file is compared with NeurHL on what it knew, not on when it was published.
- The scorer applies the same rule to every model: a file counts only for games that start after it was published. NeurHL 1.0 and 1.1 were published before the first puck drop; ORR's preseason file and NeurHL 1.2/1.3 were not.

## Head-to-head summary

| Layer | Test (same games or players, same population) | ORR | NeurHL | Verdict |
|---|---|---|---|---|
| **Standings** | NeurHL's 2019-24 judge window: 158 team-seasons, raw points over games played; MAE / CRPS | **9.23 / 6.70** (market + team history, convex weights, leave-one-season-out) | 9.42 / 6.80 (frozen engine layer); Elo 9.47 / 6.87 | ORR better. The market alone (9.32 / 6.71) already beats NeurHL. |
| | NeurHL's own backtest seasons 2012, 2014-17, raw points-per-82 MAE (NeurHL's scale) | **9.02** (team-history view; 8.96 centred) | 9.99 (shipped NeurHL-2 layer); 9.17 (its v1 Elo + xG) | ORR better in 4 of 5 seasons (2017 by 0.01); 2012 lost, 7.21 vs 6.98 |
| **Games, preseason file** | Shipped game-file pipeline rebuilt for 2019-24 (6,289 games), log loss | **0.6674** | NeurHL has no historical preseason game file. A preseason-frozen Elo scores 0.6718. | −0.0044 vs Elo (95% CI −0.0067 to −0.0024). Also −0.0033 vs ORR's own team-history table (0.6707). |
| **Games, in-season** | NeurHL's 11,052 restatement games, 2017-18 to 2025-26 | 0.6641 with no lineups; **0.6635** with starters where known | 0.6645 (NeurHL-H, given actual dressed lineups and starters) | Tie, with far less information |
| | Same games vs NeurHL's own Elo baseline | 0.6641 | Elo 0.6691 | ORR −0.0050 (CI −0.0072 to −0.0029) |
| | NeurHL-G gate games, 2019-24 (n = 6,289) | 0.6610 / 0.6603 | **0.6601** (G stack) | NeurHL slightly better; not significant |
| | NeurHL seal games, 2024-25 and 2025-26 (n = 2,624) | **0.6712** | 0.6715 | Tie |
| | The in-season loop started from the market-anchored preseason ratings (as live), NeurHL-G gate games 2019-24 | 0.6613 updating on goals and shots; 0.6606 with past starters too; **0.6634 on goals only** | 0.6601 (NeurHL-G); NeurHL Elo 0.6657 | With shots: tie with NeurHL-G (+0.0012, CI −0.0011 to +0.0036; +0.0006 with past starters). Goals only, which is what the live loop runs while results carry no shots: NeurHL-G better by 0.0033 (CI +0.0009 to +0.0056). Both beat Elo. |
| | **ORR 1.1** loop (adds dressed lineups and starters, item X1), same games and protocol (`inseason_bt_1_1.json`) | **0.6600** with goals and shots; 0.6622 on goals only | 0.6601 (NeurHL-G); NeurHL Elo 0.6657 | With shots: tie with NeurHL-G (−0.0000, CI −0.0025 to +0.0025). Goals only: NeurHL-G better by 0.0021, no longer significant (CI −0.0007 to +0.0047). |
| **Skaters** | NeurHL's headline protocol: points MAE for players with ≥40 GP ("sample A"), held-out 2022-26 | 9.51 | 9.54 | Level (ORR −0.04). By season: ORR wins 2022 (9.86 vs 10.47) and 2025 (9.07 vs 9.09); NeurHL wins 2023 (9.60 vs 9.62), 2024 (9.17 vs 9.45) and 2026 (9.39 vs 9.53). |
| | Same, 2022-24 only (ORR's roster = first 10 games; NeurHL's = each player's actual season team) | 9.64 | 9.74 | ORR better by 0.10, all of it 2022 |
| | Same, 2025-26 only (both use the actual season team; no box scores for a proxy) | 9.30 | **9.24** | NeurHL better by 0.05 |
| | NeurHL v2's per-82 restatement ("sample B"), 2022-26 | 9.32 | **9.08** | NeurHL better by 0.23 (by 0.18 on 2022-24, 0.33 on 2025-26) |
| | Regression toward the mean: slope of projected P/GP on 2025-26 P/GP, skaters with ≥60 GP in 2025-26 (n ≈ 460), 2027 file | 0.86 | 0.96 (1.3) | ORR regresses; NeurHL barely does |
| | 80% interval coverage of points, held out, every rostered skater (the unconditional sample) | 0.84 pooled (0.79 on 2022-24, 0.91 on 2025-26) | none published; intervals ~1.1× Poisson width | ORR near nominal pooled; too wide on 2025-26 |
| **Goalies** | GSAx/60 MAE, held-out 2022-26, goalies with ≥1,000 shots (n = 178) | 0.237 | n/a | Ties league average (0.241); beats last season (0.335) |
| **Live 2026-27** | 5 opening-night games, log loss | 0.750 (published after these games; not eligible) | 0.726 (1.0), 0.739 (1.1) | Noise (n = 5); eligible from 2026-10-01 |

**Summary.**

- **Standings:** ORR beats NeurHL on NeurHL's own judge window and on its backtest seasons. Most of the 2019-24 gain comes from using the sportsbook line, which is allowed new data; ORR's own modelling adds about 0.1 points of MAE on top.
- **Games:** ORR ties NeurHL's best in-season models with no lineup information and no neural network, when its update uses shots. The goals-only loop that runs while results carry no shots is significantly worse than NeurHL-G. Its shipped preseason pipeline beats a frozen Elo with a confidence interval that excludes zero.
- **Skaters:** level with NeurHL on its headline protocol (9.51 vs 9.54), with reserves projected without season-V information. ORR wins 2022, when NeurHL under-projected league scoring, and NeurHL wins three of the other four seasons. ORR loses the per-82 restatement by 0.23. On 2022-24 ORR uses first-10-game rosters where NeurHL uses each player's actual season team. On 2025-26 (40% of the pooled sample) both use the season team.
- **Calibration:** ORR's projections regress toward the mean (slope 0.86 vs 0.96) and its intervals cover near nominal pooled. Its team means track the market far more closely (correlation 0.988 vs 0.884). In spread, both models miss the market by similar amounts in opposite directions: ORR is 7% narrower, NeurHL 10% wider.

## ORR 1.1 (pre-registered hypotheses)

The plan, [`PLAN_1_1.md`](PLAN_1_1.md) (commit `6cb0904`), was committed before any of this work ran. It fixed nine hypotheses: three medium, three large and three extra large. For each it fixed the tasks, data windows, metric and acceptance rule.

- **Tuning.** Only seasons ending in 2017 or earlier were used, with 2013 excluded throughout. 2018-19 was used at most once, to confirm.
- **Test.** Each item was tested once on the windows used above, with paired bootstrap CIs.
- **Multiple testing.** An item counts only if it clears its own rule, because nine items were tested on windows that had been looked at before.

**Eight of the nine failed their rules. Only X1 (lineup-aware game forecasts) was accepted.** X3, the integration and full hindcast, also failed its rule: ORR 1.1's gain is significant with shots in the update, but not on goals alone. X1 therefore ships under the plan's fallback clause. That clause ships each individually accepted item that does not hurt in combination, and X1 lowers log loss in both variants.

Each item's code, logs and every configuration tried are in `orr/experiments/<ID>/` (`ledger.json`, `result.json`).

| Item | Size | Hypothesis | Test result (run once) | Rule | Accepted |
|---|---|---|---|---|---|
| M1 | Medium | Re-tuning the filter for goals-only updates (process noise, goal scale, prior scale) closes ≥30% of the 0.0033 gap to NeurHL-G. | Gate games 2019-24, goals-only loop: 0.6638 vs 0.6634, **worse** by 0.0005 (CI −0.0001 to +0.0010). | Gain ≥ 0.0010 with a CI that excludes 0 | **No** |
| M2 | Medium | A walk-forward stack of the in-season, preseason and Elo logits beats the in-season probability. | Goals only: +0.000003 (CI −0.00002 to +0.00003). Goals+shots: +0.000007 (CI −0.00001 to +0.00003). The stack converges on the in-season probability alone. | CI excludes 0, at least on goals only | **No** |
| M3 | Medium | A leave-one-season-out multiplier on the simulator's team-strength SD gives better-calibrated standings. | Leave-one-season-out CRPS 6.710 vs 6.682, **worse** by 0.028 (CI −0.008 to +0.066). 80% coverage 0.772 vs 0.816. | CRPS −0.05 and coverage closer to 0.80 | **No** |
| L1 | Large | A walk-forward model of who starts in goal beats the start-share mix when the starter is unknown. | 0.66123 vs 0.66131, −0.00008 (CI −0.00018 to +0.00003). Actual starters, the ceiling, gain only 0.0007. | CI excludes 0 | **No** |
| L2 | Large | Shrinkage set per component, position and situation closes the 0.23 per-82 (sample B) gap. | Sample B 9.314 vs 9.316, −0.002 (CI −0.017 to +0.014). Sample A +0.001. | Pooled B −0.08, and B better in ≥3 of 5 seasons; A no worse than +0.02 | **No** |
| L3 | Large | A player-specific games-played model improves GP and points error. | GP MAE 13.77 vs 13.93, −0.16 (CI −0.34 to +0.04). Points MAE (all) −0.066 (CI −0.110 to −0.022), better in every season. | GP −0.5 games **and** points better | **No**: the GP condition failed. The points gain is real but cleared no rule. |
| X1 | Extra large | Adjusting both teams for who dresses improves in-season log loss. The adjustment covers skaters' on-ice xG value and ice time against the expected lineup, plus the starting goalies. | Gate games, goals+shots: 0.66001 vs 0.66131, **−0.00130 (CI −0.00255 to −0.00006)**. Goals only: −0.00120 (CI −0.00250 to +0.00007). Restatement games 2018-24: 0.6603 vs NeurHL-H 0.6617 (CI includes 0). | CI excludes 0 | **Yes** |
| X2 | Extra large | Comparables projections in the style of PECOTA (nearest neighbours) improve skater samples A and B. | A 9.517 vs 9.506, **worse** by 0.011 (CI −0.009 to +0.032). B 9.321 vs 9.316, worse by 0.005. It had already failed its 2018-19 confirmation. | A and B both −0.05 | **No** |
| X3 | Extra large | The accepted items, combined in the daily loop, beat the shipped loop and NeurHL-G. | Goals+shots −0.00130 (CI −0.00255 to −0.00006). **Goals only −0.00120 (CI −0.00250 to +0.00007).** It ties NeurHL-G with shots and loses to it on goals only (not significantly). | CI excludes 0 in each variant | **No**: X1 ships under the fallback clause. |

**Combined hindcast** (`backtest/inseason_bt_1_1.py`, run once; `output/backtest/inseason_bt_1_1.json`).

- **Protocol.** The shipped in-season loop runs from the preseason pipeline's ratings and forecasts every game before it is played, exactly as `inseason_bt.py` does. It is scored on the NeurHL-G gate games 2019-24 (n = 6,289).
- **Combination.** With X1 the only accepted item, the combination is X1 itself: the "better than any accepted item alone" clause is an identity.
- **Integration check.** The run reproduces X1's own test to every decimal. Before the run, `orr/experiments/X3/checks.py` showed that on tuning season 2016 the core code equals X1's code exactly, and that the live `InSeasonFilter` reproduces the backtest filter exactly.

| Log loss (95% CI of the difference) | Goals and shots | Goals only (live until results carry shots) |
|---|---|---|
| ORR 1.0 shipped loop | 0.66131 | 0.66337 |
| **ORR 1.1** | **0.66001** | **0.66217** |
| ORR 1.1 − ORR 1.0 | **−0.00130 (−0.00255 to −0.00006)** | −0.00120 (−0.00250 to +0.00007) |
| NeurHL-G | 0.66006 | 0.66006 |
| ORR 1.1 − NeurHL-G | −0.00005 (−0.00250 to +0.00254) | +0.00211 (−0.00065 to +0.00468) |
| NeurHL Elo | 0.66566 | 0.66566 |
| ORR 1.1 − NeurHL Elo | −0.00565 (−0.00901 to −0.00206) | −0.00348 (−0.00602 to −0.00097) |

**ORR 1.1 − ORR 1.0, goals and shots, by season:**

| Season | Difference | Significant? |
|---|---|---|
| 2019 | −0.0023 | no |
| 2020 | +0.0010 | no |
| 2022 | −0.0042 | yes |
| 2023 | +0.0004 | no |
| 2024 | −0.0010 | yes (only 14% of its games have box scores) |

- **Subset with known lineups.** On the 79% of games where both lineups and starters are known, the difference is −0.0016.
- **Sensitivity.** With 20,000 bootstrap resamples instead of 2,000, the CIs are −0.00257 to −0.00004 (goals and shots) and −0.00245 to +0.00005 (goals only), so the verdicts do not change.

**What failed, stated plainly.**

- **No gain from the other in-season items.** Better hyperparameters (M1), stacking with Elo (M2) and a starter-choice model (L1) add nothing measurable. The goals-only gap to NeurHL-G is not a tuning problem. Lineup information (X1) is what narrows it, from 0.0033 (significant) to 0.0021 (not significant).
- **No skater item moves the 0.23 per-82 gap to NeurHL.** That covers shrinkage (L2) and comparables (X2). L3's games-played model did improve unconditional points MAE significantly, but its pre-registered GP threshold was not met, so it is not adopted.
- **The standings simulator's uncertainty is already calibrated (M3).** The z-score SD is 1.03.
- **X1's margin is thin.** Its CI only just excludes 0 (z = 2.01), and 1 of 5 alternative bootstrap seeds gives a CI that includes 0. Neither half is significant on its own:
  - the new skater lineup component, on top of starters: −0.00061, CI −0.00160 to +0.00045;
  - starters alone: −0.00069, CI −0.00143 to +0.00003.

  The backtest's "known starter" is the goalie with the most ice time, which is a post-game label (a pre-existing ORR convention). The reviewer bounded its effect on the 10% of games with an in-game goalie change. Treating those starters as unknown gives −0.00099 (CI −0.00224 to +0.00021), so the label does not inflate the result, but it shows how fragile it is.
- **The live gain is not established at 95%.** Live results currently carry no shots, so ORR 1.1's live loop runs goals only, the variant whose CI includes 0. Its live lineups also come from pregame lineup files that are sometimes wrong, where the backtest used box scores. Expect a smaller live gain than the backtest's.

**Adoption.**

- **Default.** `inseason.py` runs ORR 1.1 by default (`--model 1.0` restores the original loop).
- **Inputs.**
  - **Lineups and starters** come from NeurHL's committed pregame lineup files: per game, the latest pregame file, else the morning file, else the preview file. Starters marked FALLBACK count as unknown.
  - **Player values** come from `players.project(2027)`. True rookies outside the panel fall back to the frozen projection.
- **What enters the model.**
  - Past games' starters and lineups enter the update.
  - Today's lineups and starters enter the forecast.
  - A team's first known lineup carries no signal.
- **The 2026-10-01 forecast** was regenerated with ORR 1.1 at 04:09 UTC on 2026-10-01, before any 10-01 game. It differs from the 1.0 forecast by at most 0.0002 in any game's home-win probability. The whole difference comes from the five 09-29 starters entering the update:
  - each of those games was its teams' first known lineup, so its offset is 0;
  - no lineup file exists yet for 10-01.
- **The preseason file is unchanged.** `output/freeze_2027/` was not touched.
- **Changed code, unchanged defaults.** `ratings.py` changed: only options were added, and their defaults reproduce ORR 1.0 exactly (`experiments/X3/checks.json`). The `ratings.py` that the freeze manifest hashes is the one in git history up to `6cb0904`.

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
  | WPG | −2.0 | Hellebuyck's trade request and suspension. ORR gives a 10% chance he plays for Winnipeg and credits no other team with him. |
  | OTT | −1.9 | Foegele on no roster at the cutoff, Meriläinen to VAN |
  | CBJ | −1.7 | Marchenko and Merzlikins to TOR, Knies and Andrae in; Severson and Wood on no roster at the cutoff |
  | DET | −1.4 | Edvinsson and Bryson on no NHL roster at the cutoff |
  | FLA | −1.0 | Marchand on IR (09-17); two depth departures |

  A "departure" is a player with 50+ NHL games in 2025-26 who was removed from a club after 08-17 and is on no NHL roster or injured list at the cutoff. Injuries come from researched timelines; rows dated 2026-09-29 are excluded, because a date-only report may postdate the cutoff. One kept row, Frederik Andersen's (dated 09-28), takes its 31-game estimate from a 09-29 morning coach quote. That quote precedes the 21:00 UTC cutoff, but it is not covered by the date rule; without it his absence would default to 5 games, a few tenths of a point for EDM. Hellebuyck's 10% chance of playing for Winnipeg and his 12-game delay are judgments hard-coded in `goalies.py`, not fitted quantities. The trade request (08-27) and suspension (09-17) are pre-cutoff facts. The 12-game figure, however, matches a research row dated 09-29 that the date rule excludes elsewhere, so it is not strictly covered by that rule. Five long-known injuries (Terry, Bedard, Jarvis, Demko's hip, Sandin) predate the line and are not counted as news.
- **Largest disagreements with NeurHL 1.3:**

  | Team | ORR | NeurHL 1.3 | Market line |
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

  | Player | ORR (80% band) | NeurHL 1.3 |
  |---|---|---|
  | McDavid | 128.1 (88–168) | 137.9 |
  | Kucherov | 113.4 (82–147) | 104.3 |
  | MacKinnon | 110.5 (80–142) | 119.0 |
  | Draisaitl | 103.4 (70–136) | 93.5 |
  | Celebrini | 101.5 (55–140) | 98.8 |
  | Pastrnak | 94.1 (69–124) | 106.0 |
  | Robertson | 89.6 (62–119) | 103.2 |

  - ORR's intervals include talent, games-played, usage and scoring noise.
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
  - *rel*: points per 82, centred on the league mean, ORR's own scale.
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

**`inseason_bt.py`** runs the walk-forward filter from the preseason pipeline's ratings for 2019, 2020 and 2022-24. Those are the ratings `inseason.py` starts from live. OT/SO parameters are shared with the team-history run, so the starting prior is the only difference. Once games are played, the market-anchored start is no better than the team-history start: +0.0003 log loss, CI −0.0008 to +0.0015.

**Skaters** (`players_bt.json`, `players_search_ledger.json`):

- **Tuning** used only 2011-2017 (76 configurations, all logged). 2018-19 confirm; 2022-26 is the held-out test.
- **Rosters.** "Strict" uses opening rosters from each team's first 10 games. It is available through 2024. That proxy itself knows who dressed in those 10 games, a far smaller leak than NeurHL's full-season team. For 2025-26 no box scores exist, so both models use the season team, and that window is reported separately above.
- **Reserves.** The evaluation sample, like NeurHL's, is players who played in season V. Players on no opening roster who nonetheless played ("reserves") are projected at the preseason expectation for such players: 8.8 GP per 82, the average over earlier seasons including those who never played. An earlier version used the average of reserves who did play (19.5), which conditions on season-V participation; that version scored 9.41, about 0.09 better than the corrected 9.51.
- **Accuracy.** Pooled points MAE 9.51 sits beside goals MAE 4.34 and assists MAE 6.41; correlation is 0.842. A Marcel baseline scores 9.63 on the same players.
- **Final projection.** An average of ORR (0.6), Marcel (0.3) and ORR's rates times last season's GP (0.1). Rookies are calibrated by draft slot.

**Goalies** (`goalies_bt.json`):

- Tuned on 2011-17, tested on 2022-26.
- Starts MAE 11.3 over 2012-23.
- 80% GSAx/60 interval coverage 0.74.

## Where ORR does not win

- **Skaters.**
  - NeurHL's A/B blend is 0.23 points better pooled on the per-82 protocol, and 0.33 better on 2025-26.
  - On the headline protocol NeurHL wins 2023, 2024 and 2026. ORR's 0.04 pooled lead comes from 2022, when NeurHL under-projected league scoring (league ratio 0.88).
- **In-season games on NeurHL's gate window.**
  - **With shots in the update, ORR 1.0.** NeurHL-G is better than every ORR 1.0 variant, and no difference is significant:
    - by 0.0002 than the filter with known starters (team-history start);
    - by 0.0006 than the shipped loop with past starters;
    - by 0.0012 than the shipped loop without them.
  - **With shots in the update, ORR 1.1.** It ties NeurHL-G (0.6600 vs 0.6601).
  - **On goals only**, the variant the live loop runs until results carry shots:
    - NeurHL-G is 0.0033 better than ORR 1.0, which is significant;
    - it is 0.0021 better than ORR 1.1, which is not significant.
  - **Market-anchored start.** It wins preseason but adds nothing once games are played.
- **Standings, 2012.** NeurHL-2 beats ORR's team-history view (6.98 vs 7.20).
- **Standings, own modelling.** On 2019-24 the sportsbook line does most of the work. ORR's own contribution over the market alone is about 0.1 points of MAE, smaller than the noise across seasons.
- **Goalie save talent.** It is barely predictable (correlation 0.23), so ORR's goalie projections tie "league average". Coverage is 0.74, below nominal.
- **Rookies.** Thin-history players remain the largest single error source after games played.
- **Timing.** NeurHL 1.0 and 1.1 were genuinely published before the season. ORR's preseason file was not, so it is not scored on opening night. Its 2026-27 comparison with NeurHL counts only games after its publication.

## Live scoring

- **Scorecard.** `python3 -m orr.score` scores every file on completed games:
  - ORR's preseason file;
  - NeurHL's 09-25 freeze and releases 1.0-1.3;
  - NeurHL-G, H and Elo pregame forecasts;
  - ORR's in-season forecasts.

  Each preseason file is also scored on the games after its own publication time (`after_publication`). A file or daily forecast counts for a game only if published before that game's scheduled puck drop (from the NHL schedule), the rule NeurHL's pregame forecasts also follow. The pushed commit time is the external evidence.
- **Interim team scoring** compares points earned with each model's expected points for the games actually played. At season end the scorer reports points MAE, RMSE and CRPS. The output is `orr/output/scorecard_2027.json`.
- **Daily loop.**
  1. `python3 -m orr.ingest` appends finished games, from the NHL API or by hand with `--add`, each checked against the schedule.
  2. `python3 -m orr.inseason --date YYYY-MM-DD` updates team ratings from results strictly before that date, then forecasts the day's games and players and re-simulates the season.

  The first in-season forecast (2026-10-01) was committed at 00:43 UTC on 10-01, before any 10-01 game. It uses results through 09-29: the 09-30 games were still in progress when it was made.
