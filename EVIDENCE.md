# NeurHL evidence ledger

Every claim NeurHL makes, with the strength of the evidence behind it. A
result is **confirmed** only if it came from a window that no decision had
used, scored once after the model was frozen. **Exploratory** results came
from windows that were consulted while the model was being built; they are
estimates, not tests. **Development** results have no out-of-sample test at
all. Each row names the committed file that holds the numbers.

Seasons are named by the year they end: 2018 is the 2017-18 season.

## Confirmed

| Claim | Result | Window | Record |
|---|---|---|---|
| The player-game layer beats each skater's own recent average on all four targets | Ice-time share MAE -4.1%, shots deviance -10.2%, P(goal) log loss -2.8%, P(assist) log loss -2.6%. Holm p < 1e-5; two-way clustered z from -8.1 to -23.9; better in every season | 2018-2020, 130,092 skater-games, one config, spent once | `neurhl/configs/player_game_confirm_2018_2020.json`, `player_game_twoway_2018_2020.json` |
| NeurHL-H beats Elo game by game, given the dressed lineup | Log loss 0.66495 vs 0.66973, diff -0.00478 (95% CI -0.00650 to -0.00306), p < 1e-6; season-clustered p = 0.0007; better in 8 of 8 seasons. The gain is resolution (0.0122 vs 0.0098), not recalibration. Against a constant home-win rate (0.6899) it improves Elo's skill by 24% | 2018-2026 excluding 2021, n = 10,184; frozen in PLAN_NeurHL A5, run once (including 2021: -0.00459, 9 of 9 seasons) | `neurhl/output/hier_restatement.json`, `neurhl/output/preds/hier_restatement_games.csv` |

NeurHL-H uses which skaters dressed for each game, known about an hour before puck drop; Elo does not. The confirmed gain is conditional on that lineup.

## Exploratory

| Claim | Result | Why it is not a test | Record |
|---|---|---|---|
| NeurHL-H beats Elo game by game (tune window) | Log loss 0.67314 vs 0.67597, diff -0.00283, p = 0.0075; won 5 of 5 seasons. Superseded by the confirmation above | The scored seasons, Layer 1 and the scoring rule changed on this window while results were visible (PLAN_NeurHL A5.1). The estimate does not depend on the season choice: p runs 0.0075-0.019 across the four scored sets | `neurhl/output/hier_result.json`, `neurhl/configs/hier_both_ways_original.json` |
| Rolling shot-share form adds to Elo | All six variants improve log loss (-0.0009 to -0.0022); best single p = 0.041 | Exploration on 2015-2017; not significant after correcting for six tests | PLAN_NeurHL A3.1 |

## Development only

| Claim | Result | Caveat | Record |
|---|---|---|---|
| The season simulator produces calibrated standings | 80% intervals cover 0.794 of team-seasons; standings MAE 9.61 | The interval width (sigma = 0.07) was tuned on the same seasons whose coverage is reported | `neurhl/configs/season_backtest.json` |
| The season simulator is as accurate as the Elo baseline | On the five seasons both records share, v1 is better: MAE 9.17 vs 9.99, CRPS 6.43 vs 7.17, v1 ahead in 4 of 5 seasons (p = 0.19) | Five seasons cannot separate the two | `neurhl/configs/season_matched_comparison.json` |
| The player-season blend beats either path alone | Points MAE per 82 games: blend 9.07, Path B 9.41, Path A 9.77, pooled over 11 seasons; Path B alone wins only 4 of 11 | Several of the 11 seasons informed the model's design; no uncertainty is attached | `neurhl/configs/player_season_v2.json` |

## Documented nulls

| Hypothesis | Result | Record |
|---|---|---|
| A neural game network over event-language-model player embeddings beats Elo | Five configurations; best 0.6966 vs Elo 0.67585 and a constant home-win rate of 0.6898 | PLAN_NeurHL R.1, `neurhl/configs/search_ledger.csv` |
| Pooled player embeddings carry team strength | Logistic probe 0.6888, level with the constant; adding embeddings to team form makes it worse | `neurhl/eda/eda_09_signal_location.md` |
| A career encoder can place rookies | Validation MSE 0.99-1.00 of predicting the mean, at every vantage | ledger row ce-null-1 |
| Neural player-rate heads beat a player's own average | Worse on ice time (0.00934 vs 0.00619 MAE) and shots; level with a shrunk average on goals | PLAN_NeurHL R.6 |
| The event simulator's rates beat Elo per game | Blend 0.67763 vs Elo 0.67668; the confirmation stopped at its pre-gate | PLAN_NeurHL2, final status |
| Absences, goalie quality, schedule density or schedule strength improve the game model | Each block adds log loss over the nested model (+0.00024 to +0.00541) | `neurhl/configs/game_v3_pregate.json` |
| A starter model beats simple heuristics | Season start-share wins (log loss 0.66-0.68 vs 0.71-0.77) | `neurhl/configs/goalie_gates.json` |
| Goals saved above expected predicts next season better than shrunk save percentage | r 0.159 vs 0.218, Steiger p = 0.007 in favour of save percentage | `neurhl/configs/goalie_gates.json` |
| Ridge RAPM transfers to a new team better than team-demeaned raw rates | Indistinguishable (movers r 0.387 vs 0.371, p = 0.60, pooled over three folds) | `neurhl/configs/rapm_folds.json` |
| Shot-quality model calibrated to 0.005 per decile | Best 0.00653 after three declared correction ladders; the residual is a 2023 change in how shot locations are recorded | `neurhl/configs/xg_gates.json` |

## Corrected or withdrawn

| Earlier statement | Correction |
|---|---|
| NeurHL-G's calibration gate passed (PLAN_NeurHL4 FREEZE) | The gate script judged it on slope and OT share only. The declared randomised-PIT coverage, computed afterwards from the same predictions, fails for team SOG (0.860 against 0.80 ± 0.03). Recorded in PLAN_NeurHL4 A2; no decision changes. |
| "The season layer beats both house benchmarks" | The HOWE figure came from 2018-2026 and the NeurHL figure from 2011-2017. On matched seasons v1 is better (above). Withdrawn. |
| NO_SCORE {2013, 2021} "carried verbatim" from amendment A4 | A4 required reporting both ways; the rule was replaced after results were visible. Recorded in PLAN_NeurHL A5.1; both-ways numbers above. |
| The PS1 player blend "ships" | The shipped file was still Path A alone. The blend now ships for 2026-27. |
| Playoff odds | Ties were broken alphabetically, worth up to 1.6 playoff points for some clubs. Fixed; only playoff probabilities moved. |

## Diagnostics that passed

- **Leakage.** Two leaks were found in the event simulator (it could see the
  next event's team and the next on-ice set). Both gates had passed with the
  leaks present. A causality audit now corrupts everything after a cut point
  and requires every earlier input to be bit-identical; it runs clean for the
  simulator and for every tabular builder (`neurhl/tests/audit_leakage*.py`).
- **Event realism.** Simulated goals, shots, penalties and faceoffs per game
  and per period sit within 10% of observed; score effects reproduce at a
  gradient ratio of 0.979 (`neurhl/configs/event_sim_gates.json`).
- **Shot quality.** Beats a distance-and-angle baseline in 18 of 18 seasons;
  team xG differential predicts future goal share better than Corsi or
  Fenwick (`neurhl/configs/xg_gates.json`, `x2_diagnostics.json`).
- **Reproducibility.** The four acceptance batteries pass (25/25, 40/40,
  25/25 and, for NeurHL-G, 10/10); committed predictions re-derive on CPU
  within 1e-6.

## NeurHL-G, the single-game engine (PLAN_NeurHL4)

NeurHL-G simulates each game from the two dressed rosters: every skater's ice
time by strength, shots, individual xG, goals and assists, both goalies,
power-play opportunities and the score, with a win probability that stacks
the engine with Elo and NeurHL-H. Its gate window (2019, 2020, 2022-2024) was
scored once, for one candidate declared in advance; those seasons had already
been used at game level by NeurHL 1.0, so this is gate evidence, not a clean
confirmation. The one-shot seal (2025-2026) was not spent: the pre-gate failed
exactly as the candidate commit predicted. The 2026-27 live season is the
prospective test.

| Claim | Result (G_GATE, n = 6,289 games) | Record |
|---|---|---|
| Beats Elo game by game | Log loss 0.66006 vs 0.66566, diff -0.0056 (SE 0.0013); better in 5 of 5 seasons | `neurhl/output/g_gates.json` |
| Adds win-probability information beyond NeurHL-H | Not shown: diff -0.0005 (SE 0.0006), better in 3 of 5 seasons; the pre-gate required -0.0010 | same |
| Player heads beat the confirmed player-game layer | Better on all four heads: ice-time share -0.59%, SOG -0.45% (Poisson log loss), P(goal) -0.45%, P(assist) -0.36%; every upper 95% bound below zero (225,752 skater-games, two-way clustered) | same |
| Team box score beats team history and the summed player-game layer | SOG, xGF, goals and PP opportunities beat team history; SOG and goals also beat the player-game layer summed over the dressed skaters, on deviance and CRPS | same, and `g_gates_record.json` |
| Calibrated | Mixed. Slope 0.968 and OT share 22.4% predicted vs 22.0% observed pass; randomised-PIT 80% coverage passes for team goals (0.803) and skater SOG (0.783) but fails for team SOG (0.860: the intervals are too wide), so the declared calibration gate fails. First recorded as a pass; corrected in PLAN_NeurHL4 A2 | `neurhl/output/g_gates.json`, `g_gates_record.json` |

Components that did not earn their place on the iteration window (G_ITER,
7,421 games): an Elo anchor on scoring rates, lineup-versus-usual multipliers,
NeurHL-H's projections as inputs, and attention across rosters. Gradient-boosted
trees and a logistic model on game-level features, stacked with Elo, did no
better than Elo. Every run is in `neurhl/configs/search_ledger_g.csv`.

## NeurHL 1.0, the unified model (PLAN_NeurHL_1_0)

NeurHL 1.0 joins the confirmed and gated components into one model for
player-games, games and season totals. Its claims are of two kinds:

| Claim | Status | Record |
|---|---|---|
| Its levels agree: skater totals sum to team totals, games to seasons, player-games to season lines, with 18 skater-games and one goalie start per team-game, and league points equal to 2 x games + overtime games in every simulated season | Verified on the frozen outputs by an independent check | `neurhl/tests/check_neurhl_1_0.py`, `neurhl/output/neurhl_1_0/checks_2027.json` |
| Its 2026-27 forecasts are accurate | No evidence yet. Frozen before the first game and scored once, after the regular season. Its game level is the gated engine; its season layer carries a season-level uncertainty calibrated for an earlier model | `PLAN_NeurHL_1_0.md` |

## NeurHL 1.1 layers (PLAN_NeurHL_1_1)

These layers came from a review of every work the project cites. Each is
preregistered before the games it is judged on, and ships as a dated file
set scored beside the NeurHL 1.0 freeze. The frozen files are never edited.

| Claim | Status | Record |
|---|---|---|
| The stat sheet's team-shot intervals are too wide, and its team goal means too extreme | Development evidence on 2019-2024 (G_GATE): with the frozen r = 40, 80% intervals cover 0.861; with the fitted r = 98.68 they cover 0.797 (+0.021 nats per team-game). A goal slope of 0.70 brings the top predicted decile from 4.17 to 3.80 against 3.73 realised. The 2026-27 test was fixed before the first game, with one inference at season end | `PLAN_NeurHL_1_1.md` C1, `neurhl/configs/calibration_1_1.json`, `neurhl/eval/score_calibration_1_1.py` |
| Goal totals and skater points can be sharpened without changing any win probability | Dated sets `neurhl/output/neurhl_1_0/cal_20260929/` (goal slope; team goals-for spread 0.303 to 0.215 per game at an unchanged league level) and `skaters_blend_20260929/` (50/50 engine and season-model paths). No evidence yet; scored at season end | `PLAN_NeurHL_1_1.md` C1, A3; `neurhl/eval/score_dated_1_1.py` |
| Team-xG and skater-shot intervals are also mis-sized | Development evidence on 2019-2024: team xG with gamma shape 9 covers 0.855 at a nominal 0.80, and shape 11.86 covers 0.798 (+0.018 nats per team-game). Skater shots, fitted NB r = 18.75: coverage 0.798 against Poisson's 0.782. Both are tested on 2026-27, fixed before the first game | `PLAN_NeurHL_1_1.md` A6, A7; `neurhl/configs/calibration_1_1b.json`, `calibration_1_1c.json` |
| Fine-tuning the engine on the current season's games improves it | Null (exploratory, 2012 and 2014-2016): against a single frozen seed the gains looked large, but against the five-seed ensemble the stacked difference is -0.00006 (SE 0.00034) | `PLAN_NeurHL_1_1.md` A5, `neurhl/output/neurhl_1_1/finetune_c4_stack_iter.json` |
| More seeds in the engine ensemble would help | Null (exploratory): the stacked log loss goes 0.67242, 0.67230, 0.67226, 0.67224, 0.67223 for 1 to 5 seeds; 10 seeds would add about 0.00002 | `PLAN_NeurHL_1_1.md` |
| The NeurHL 1.0 season layer is calibrated for its engine | Supported by a walk-forward preseason backtest (engine snapshots trained on earlier seasons, opening-night states). The fit seasons (2012, 2014-2018) select the frozen shock size, 0.07. On held-out 2019-2024: points CRPS 6.80, MAE 9.42, 80% coverage 0.823, better than a preseason Elo season model (6.87, 9.47). Evolving-strength and Elo-blend variants did not beat it | `PLAN_NeurHL_1_1.md` A9, `neurhl/output/neurhl_1_1/season_layer_c2.json` |
| Updating team strength in season sharpens the standings projection | Exploratory: in the backtest the update beats "no update" on CRPS (-0.39, 95% CI -0.66 to -0.12; MAE 5.74 vs 6.24), but its intervals were too wide on the held-out seasons (coverage 0.876), so it was not adopted. It is published nightly as an exploratory projection and scored at season end | `PLAN_NeurHL_1_1.md` A9, `neurhl/output/live/standings_1_1_2027.csv` |
| A filtered team rating with weighted overtime results beats Elo | Null (exploratory): slightly worse out of sample (+0.00038 and +0.00055 log loss) and adds nothing to the stack (+0.00007, SE 0.00007) | `PLAN_NeurHL_1_1.md` A1 |
| Goals excite goals (Hawkes self-excitation) | Rejected: in periods 1-2 of 2012 and 2014-2018, the goal rate in the 30 s after a goal is 2.9 per 60 minutes, against about 5.5 later, at every lead | `PLAN_NeurHL_1_1.md` |
| Stack weights should vary with the season phase | Rejected: walk-forward on 2014-2018, +0.00074 log loss (SE 0.00029) | `PLAN_NeurHL_1_1.md` |

## The 2026-27 season

Predictions for every game, team and skater were frozen before opening night
and are scored as the season is played (`PLAN_NeurHL_LIVE.md`). Inference is
made once, at the end of the regular season.
