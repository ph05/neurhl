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
| NeurHL-H beats Elo game by game | **Pending.** Frozen in PLAN_NeurHL A5 before scoring; runs once | 2018-2026, n = 10,184 | `neurhl/output/hier_restatement.json` (written by the run) |

## Exploratory

| Claim | Result | Why it is not a test | Record |
|---|---|---|---|
| NeurHL-H beats Elo game by game (tune window) | Log loss 0.67314 vs 0.67597, diff -0.00283, p = 0.0075; won 5 of 5 seasons | The scored seasons, Layer 1 and the scoring rule changed on this window while results were visible (PLAN_NeurHL A5.1). The estimate does not depend on the season choice: p runs 0.0075-0.019 across the four scored sets | `neurhl/output/hier_result.json`, `neurhl/configs/hier_both_ways_original.json` |
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
- **Reproducibility.** Acceptance batteries pass 25/25 and 40/40; committed
  predictions re-derive on CPU within 1e-6.

## The 2026-27 season

Predictions for every game, team and skater were frozen before opening night
and are scored as the season is played (`PLAN_NeurHL_LIVE.md`). Inference is
made once, at the end of the regular season.
