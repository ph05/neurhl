# ORR 1.1: nine pre-registered hypotheses

*Committed 2026-10-01, before any of the work below was run. Each item states its hypothesis, tasks, data windows, metric and acceptance rule in advance. Every item is reported whatever its outcome, including the ones that fail.*

## Ground rules (all items)

- **The scored 2026-27 preseason file is not replaced.** `orr/output/freeze_2027/` stays as published. Accepted changes enter ORR 1.1, which produces the daily in-season forecasts from the day it is adopted; those forecasts are scored only on games after their publication. A re-run of the preseason forecast, if any, is published as a separate file labelled 1.1 and is scored only from its publication time.
- **Windows.** Tuning uses only seasons that end in 2017 or earlier (2012-2017 for games, 2011-2017 for skaters and goalies; 2013 excluded throughout). 2018-2019 may be used once to confirm. The test windows are the ones already used in `RESULTS.md` (NeurHL-G gate games 2019-24 for games; held-out 2022-26 for skaters and goalies; the five clean market seasons 2019, 2020 and 2022-24 for standings). Each item runs its test once, after its configuration is fixed.
- **Multiple testing.** The test windows have been looked at before, and nine hypotheses are tested on them. An item is accepted only if it clears its rule below; a single test-set win that clears no rule is reported as "not accepted".
- **Comparisons.** Log loss on the same games with paired bootstrap 95% CIs (`orr.backtest.games_bt.paired`), MAE on the same players, CRPS on the same team-seasons.
- **Code.** Each item works in `orr/experiments/<ID>/` and does not edit existing files. Integration into the core (item X3) happens only for accepted items.

## Medium

### M1. Re-tune the in-season filter for goals-only updates
- **Hypothesis.** The filter's hyperparameters were tuned with shots in the update. When results carry no shots, as live now, they are mis-set. Re-tuning the process noise (`q_s`, `q_f`, `q_mu`), the goal observation scale (`phi_g`) and the prior scale for `use_shots=False` closes at least 30% of the 0.0033 log-loss gap to NeurHL-G.
- **Tasks.**
  1. Write a tuner over those `ratings.HP` fields with `use_shots=False` (coordinate or grid search, at most 60 configurations, every one logged), scored on 2012-2017 log loss with `ratings.run_filter` and `ratings.predict_probs`.
  2. Fix the best configuration.
  3. Score it once on the gate games 2019-24, against the current goals-only filter and NeurHL-G.
- **Accept if** the pooled gate log loss improves on the current goals-only filter with a 95% CI that excludes 0 and the gain is at least 0.0010.

### M2. Stack the in-season forecast with Elo
- **Hypothesis.** A walk-forward logistic stack of ORR's in-season logit, ORR's preseason logit and an Elo logit (ORR's Elo replica, `ratings.elo_run`, which needs only results and so runs live) beats ORR's in-season probability alone. NeurHL-G is itself a stack with Elo.
- **Tasks.**
  1. Build per-game features for 2012-2024 from the filter (both the goals-and-shots and the goals-only variants).
  2. Fit the stack for each season V on seasons before V (2012 onward).
  3. Choose the regularisation on 2012-2017 only.
  4. Score once on the gate games 2019-24, for both variants.
- **Accept if** the stack improves on the same variant without it, with a 95% CI that excludes 0, for at least the goals-only variant.

### M3. Calibrate the standings uncertainty
- **Hypothesis.** The season simulator's team-strength SD is derived from a formula: the blend's out-of-sample RMSE net of game luck, plus drift. Choosing a multiplier on that SD by leave-one-season-out CRPS gives better-calibrated season point distributions.
- **Tasks.**
  1. For each clean market season (2019, 2020, 2022-24), rebuild the preseason ratings as `backtest/gamefile_bt.py` does (shipped views `mkt_rel82` + `td_rel82`).
  2. Simulate each season with `season.simulate` at SD multipliers {0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4} (at least 5,000 seasons each).
  3. Score team points CRPS and 80% coverage.
  4. Choose the multiplier leave-one-season-out.
- **Accept if** the leave-one-season-out CRPS improves by at least 0.05 points and 80% coverage moves closer to 0.80.

## Large

### L1. Predict unknown starting goalies
- **Hypothesis.** When the starter is unknown, the forecast uses the team's start-share mix. A walk-forward model of who starts improves in-season log loss over the mix. It would use rest days and back-to-backs, recent start share, consecutive starts, home or away, and the goalies' talent gap.
- **Tasks.**
  1. Build the starter-choice dataset from the box scores (`data.goalie_games`, 2011-2024).
  2. Fit a per-team-game choice model on 2011-2017, confirming on 2018.
  3. Turn its probabilities into expected goalie offsets (`gamemodel.goalie_offset`).
  4. Run the in-season filter with predicted offsets in place of the mix and score once on the gate games 2019-24, against the start-share mix (lower bound) and actual starters (upper bound).
- **Accept if** the improvement over the mix has a 95% CI that excludes 0.

### L2. Component-specific reliability for skater rates
- **Hypothesis.** ORR loses NeurHL's per-82 protocol (sample B) by 0.23. Shrinkage strength tuned separately is closer to each component's true reliability, which improves the per-game rates that sample B measures. The components are goals, primary assists, secondary assists and shots; the strength is set per position and per situation (even strength and power play).
- **Tasks.**
  1. Expose per-component shrinkage constants in a copy of the rate step (`players.py`), keeping everything else fixed.
  2. Tune them on 2011-2017 (at most 80 configurations, logged) on per-82 MAE, then confirm on 2018-19.
  3. Run the strict protocol of `backtest/players_bt.py` once on 2022-26.
- **Accept if** pooled sample-B MAE improves by at least 0.08 and in at least 3 of the 5 test seasons, with sample-A MAE no worse by more than 0.02.

### L3. Player-specific games-played projections
- **Hypothesis.** Games played is the largest single error source after rates. A GP model that uses each player's own availability history improves both GP error and the unconditional points error. The history covers games missed in each of the last three seasons, age, position and usage tier.
- **Tasks.**
  1. Build a walk-forward GP model (for example a beta-binomial or GBM on games share) trained on 2011-2017 and confirmed on 2018-19.
  2. Substitute it for the current games share in a copy of the projection step.
  3. Run the strict protocol once on 2022-24 (first-10 rosters) and report 2025-26 separately.
- **Accept if** GP MAE (all rostered skaters) improves by at least 0.5 games and sample-"all" points MAE improves, both on 2022-24.

## Extra large

### X1. Lineup-aware game forecasts
- **Hypothesis.** NeurHL-H uses each game's dressed lineup; ORR uses none. Adjusting both teams' ratings for who actually dresses improves ORR's in-season log loss. The adjustment is the summed on-ice value of present minus expected players, from ORR's player projections (`rel_xgf60`, `rel_xga60`, ice time) and its goalie layer. Live, the lineups are NeurHL's committed pregame lineup files (DailyFaceoff, posted before puck drop).
- **Tasks.**
  1. From the box scores (2011-2024), build each game's dressed skaters.
  2. Build each player's walk-forward preseason value (projections for season V use seasons before V only).
  3. Build the team's expected lineup value (the dress-probability-weighted sum).
  4. Add the lineup delta as an offset to both the filter's update and its prediction. Fit its coefficient on 2012-2017.
  5. Score once on the gate games 2019-24 against ORR without lineups, and on NeurHL's restatement games 2018-24 against NeurHL-H.
- **Accept if** the improvement over ORR without lineups has a 95% CI that excludes 0.

### X2. Comparables (PECOTA-style) skater projections
- **Hypothesis.** A nearest-neighbour comparables projection improves both samples A and B, especially in 2023, 2024 and 2026 where ORR lost to NeurHL. Each player is matched to historical players on age, position, the last three seasons' per-60 rates and usage, and projected from his comparables' next-season changes. It is blended with ORR's rates by a tuned weight.
- **Tasks.**
  1. Build the comparables engine (standardised features, k nearest neighbours with distance weights, trajectories from seasons before V only).
  2. Tune k, the feature weights and the blend weight on 2011-2017 and confirm on 2018-19.
  3. Run the strict protocol once on 2022-26.
- **Accept if** pooled sample-A and sample-B MAE both improve by at least 0.05.

### X3. ORR 1.1 integration and full in-season hindcast
- **Hypothesis.** The accepted items, combined in the daily loop, improve ORR's in-season forecasts as run live. The loop starts from the preseason pipeline, updates daily and forecasts each game before it is played. On the gate games 2019-24 the combination does better than any accepted item alone, and better than NeurHL-G.
- **Tasks.**
  1. Integrate every accepted item into the core behind explicit settings, leaving the frozen preseason outputs untouched.
  2. Re-run `backtest/inseason_bt.py` with the combination, in both the goals-only and the goals-and-shots variants.
  3. Report the result against NeurHL-G and NeurHL's Elo.
  4. Run all test suites.
  5. Update `inseason.py` so the 2026-27 daily forecasts use ORR 1.1 from its adoption date, and update `RESULTS.md` with every item's outcome.
- **Accept if** the combined loop improves on the current shipped loop (same variant) with a 95% CI that excludes 0. Otherwise ship only the individually accepted items that do not hurt in combination.

## X3r. Revision of X3 (added 2026-10-01 after X3's result, before any of this ran)

X3 failed for one reason. ORR 1.1 beat the shipped loop with a CI below 0 when the update used shots, but not on goals alone (−0.00120, CI −0.00250 to +0.00007). The live loop runs on goals alone only because this build container cannot reach the NHL API for shot counts. The revision does not loosen X3's rule and does not re-test on the gate games. It changes two things.

**A. Remove the cause.** A daily GitHub Actions job (`.github/workflows/orr_daily.yml`) runs on GitHub's runners, which can reach the NHL API. It:
1. ingests results with shots on goal (`orr.ingest`);
2. runs the ORR 1.1 daily forecast with NeurHL's committed pregame lineups (`orr.inseason`);
3. rebuilds `docs/orr/`;
4. commits.

The live loop then runs the goals-and-shots variant, the one that passed. Scheduled jobs run from the default branch, so this takes effect once the branch is merged.

**B. Confirm on data no experiment has touched.**
- **Data.** Box-score lineups also exist for 2010-11 (1,230 games) and 2020-21 (868 games). No experiment, including X1's tuning and confirmation, has used them. Neither season has market lines, so both arms start from the team-history prior. X1 also tested that start on the gate games: −0.00139, CI −0.00266 to −0.00013.
- **Comparison.** ORR 1.1 (the fixed X1 configuration: lineup offsets plus known starters) against ORR 1.0 (neither), in the same walk-forward filter. Each arm takes its OT/SO parameters from its own run, as in X3. Paired bootstrap 95% CIs use `games_bt.paired`.
- **Primary metric.** Pooled 2010-11 and 2020-21 log loss, goals-and-shots variant. Secondary: goals only, per season, and an inverse-variance combination of this result with X3's gate result. The combination is reported but not used to decide.
- **Accept X3r if** the primary difference has a 95% CI that excludes 0, below zero.
- **Power, stated in advance.** About 2,100 games give a CI half-width near 0.0022, while the gate effect was about 0.0013, so an inconclusive result is the most likely outcome. An inconclusive result is reported as "not confirmed". It does not count as a pass, and X1 keeps shipping only under the original fallback clause.
- **Run.** Once, with `python3 -m orr.backtest.inseason_bt_x3r`. The script refuses to overwrite its output.
