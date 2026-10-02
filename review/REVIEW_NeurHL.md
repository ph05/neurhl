# NeurHL: A Review

*Reviewed 2026-09-30, one day into the 2026-27 season. The code reviewed is the repository at HEAD `420cd71` (NeurHL 1.3); the history available goes back to graft `f50c942`. Every finding cites a file and line, a commit, or a number. Every number marked **(R)** is recomputed by `review/evidence/recompute.py`, whose output is `review/evidence/recomputed.json`. That script reads only NeurHL's own files, `data/`, the git history, and 222 hand-collected preseason sportsbook points lines (`review/data/nhl_point_totals_history.csv`).*

---

## Verdict

NeurHL is the best-documented NHL model I have reviewed that cannot show it forecasts anything better than a well-tuned Elo, a free sportsbook line, or its own simplest component.

It spans about 44,000 lines of Python. Its parts:

- a transformer pretrained on 7.6 million play-by-play events;
- a career encoder and a neural player layer;
- a hazard-integrated game engine with conserved ice time;
- a gradient-boosted xG model and RAPM priors;
- an availability model;
- a preregistration apparatus with about 41 amendments.

Here is what all of it produces, set against simple references:

- **The game engine, on its own, scores exactly what Elo scores:** 0.66568 against 0.66566 on 6,289 held-out games **(R)**. Every claimed gain over Elo comes from NeurHL-H, a far simpler stacked layer. That layer's edge sits almost entirely in games where one team's lineup is depleted:
  - −0.0087 nats when more than 40 minutes of regulars' ice time are missing;
  - −0.0003 (SE 0.0045) when nobody is missing **(R)**.

  Its backtests were fed the *actual* dressed lineup and the *actual* starting goalie. The preseason forecast has neither.
- **The season layer has never beaten Elo.**
  - The shipped layer scored MAE 9.99 against Elo's 9.17 (better in 1 of 5 seasons).
  - The current one scored 9.42 against 9.47 on 2019-24 **(R)**, a tie.
  - Late-preseason sportsbook lines score 9.51 on those same 158 team-seasons **(R)**.

  The neural stack ties a spreadsheet Elo and a free public number, and never tries combining with either.
- **The 2026-27 forecast is overconfident, and the project knows it.**
  - Team goal predictions are 43% too spread by NeurHL's own fitted calibration (b = 0.70) **(R)**, and the fit is never applied.
  - Projected team goal differentials have SD 44 **(R)**, wider than realised seasons.
  - Skater projections barely regress: the slope of projected on last-season points per game is 0.96 **(R)**.
  - Every elite goalie is capped at 52.7 starts **(R)**.
- **The integrity machinery was bent every time it bound.**
  - A failed stop rule was waived by "owner directive".
  - The sealed test seasons were spent "descriptively" and folded into training 17 minutes later **(R)**.
  - Twenty-one amendments were written on opening day.
  - Four season forecasts shipped in 43 hours, two after the puck dropped **(R)**.
  - The documents recording all of this were removed from the public tree 20 minutes after the first puck drop **(R)**.

It is not worthless. The conservation-law bookkeeping, the walk-forward state bank, the forecast-timing scorer and the candour about failed experiments are genuinely good, and a competitor should take all of them (section 5). But as a forecasting product it is an elaborate machine whose outputs tie simple references, and whose paperwork is louder than its evidence.

### Scorecard

| Area | Grade | Reason |
|---|---|---|
| In-season game probabilities | C+ | A real gain over Elo, but measured with puck-drop lineups; the engine adds nothing; no market benchmark |
| Preseason game file (1,344 frozen probabilities) | D+ | Opening-night inputs patched by six post-hoc multipliers; it lacks exactly the lineup information where the edge lives |
| Standings, playoff and Cup odds | D | Ties Elo and a free line; never blends; spread too wide; uncertainty borrowed from an older layer |
| Skater projections | C− | Good conservation laws, but barely regressed, near-Poisson intervals, and PP stars anti-regressed |
| Goalies | D | Hard start cap, no goalie injuries, a stale career-cumulative talent input, a starter model that loses to a heuristic |
| Uncertainty and calibration | D | Goal spread 43% too wide with the fix unused; team shock 2× its documented size; player intervals too narrow |
| Research process and preregistration | D+ | Genuine guard rails, then overridden, amended ~41 times, and hidden |
| Live operations | B− | Forecasts committed about an hour before puck drop with provenance; but "on time" is judged by the author's own clock |
| Engineering | C− | Guards and hashes; but 5 copies of the standings simulator, a 425-line `main()`, a 189 MB repo, pickles |
| Honesty about failures | A− | Nulls, failed gates and bad seasons are written down |

---

## 1. The README's claims against the evidence

| README claim | What the repository shows |
|---|---|
| "Neural player layer … beats Elo on 10,184 held-out games (log loss 0.6650 vs 0.6697)" | True as measured (0.66455 vs 0.66914 on the 11,052-game restatement **(R)**). But every prediction was given the players who **actually dressed** and the goalie who **actually started** (`neurhl/data/build_g_tensors.py:137-168`; the docstrings of `build_absences.py` and `models/player_game.py` call it "INGAME-at-puck-drop information"). The gain is −0.0003 when no regular is missing (n=367) and −0.0087 when more than 40 TOI-minutes are vacated (n=2,142) **(R)**. It is a lineup-news detector, and the preseason freeze has no lineup news. |
| "Game engine (NeurHL-G), stacked with Elo and NeurHL-H … beats Elo … −0.0056, 5 of 5 seasons" | The raw engine scores 0.66568 against Elo's 0.66566 on the same 6,289 games **(R)**. The stack minus H alone is −0.00049 (SE 0.0006) **(R)**. On the iteration window, H alone (0.67244) beats every engine configuration in the ladder (best 0.67258, `neurhl/configs/search_ledger_g.csv`). The "5 of 5" belongs to H; the stop rule comparing G with H *failed* (`neurhl/output/g_sstop.json`: `pass: false`, 3 of 5 seasons) **(R)**. |
| "The same engine produces all three, and they are built to agree" | Win probabilities come from an Elo+H+engine stack; goals come from engine × m0. `neurhl/live/goal_calibration.py` says so outright: "m scales stat-sheet goal means only; the win probability never changes". Across the 1,344 games, logit(p) against log goal ratio has slope 1.39, versus the engine's own 1.93. Across teams, points on goal differential has slope 0.227, versus 0.329 historically. The hazard integration allows at most one goal per team per two-minute step and cancels simultaneous goals (`neurhl/models/neurhl_g.py:287-307`), so expected goals come out 5% below the rates it integrates. |
| "Built under preregistration … final tests ran once on seasons no decision had touched" | S-STOP failed. The plan's pre-committed consequence ("the seal stays unspent, NeurHL-H stays primary", PLAN_NeurHL4 §C) was overridden by amendment A5, "Owner directive". The seal was spent "descriptively" at 18:36Z on 09-29 (`43aa58d`) and absorbed into training at 18:53Z (`5816561`) **(R)**. The 2018-26 confirmation seasons were reused as NeurHL-G iteration and gate windows (`neurhl/windows.py:116-118`). No untouched holdout remains. |
| "NeurHL 1.1 was issued before the first game" | The 1.1 commit `9580606` is stamped 20:53:49Z by the author's clock; FLA@CAR dropped at 21:00:00Z. That is a 6-minute margin **(R)** on a clock the author controls. The release touches no `neurhl/output/live/**` path, so the server-side stamp workflow (`.github/workflows/stamp.yml`) never ran for it. |
| "1.0 … and the 1.1 and 1.2 files stay unchanged and are scored beside 1.3" | No scorer reads `neurhl_1_1/`, `neurhl_1_2/` or `neurhl_1_3/`: `neurhl/eval/score_dated_1_1.py:81` only globs `neurhl_1_0/season_*` and `v*_20*`. |
| "Season simulation … 80% intervals covered 0.794" | That figure belongs to an **earlier** season layer; README row 5 says it "is carried over". The team shock it rests on is documented as 0.07 on the log goal rate but applied as GF×e^s and GA×e^−s. Its effect on the log goal ratio is therefore 0.14 (`neurhl/sim/game_model.py:283-298`). |
| "No betting-market data enters any model" (`docs/index.html:93`) | True, and it is a choice, not a virtue: the market is an independent forecast of equal accuracy that costs nothing to combine with. Meanwhile `src/optimize_board.py` and `src/optimize_full100.py` size joint-Kelly bets against it (`output/final_100_allocation.csv`: 20 tickets, $100). Neither the README nor the site mentions it. |

---

## 2. Critical flaws

### C1. Every headline backtest uses information the forecasts do not have

- **Game backtests.** Each game is filled with the skaters who actually dressed and the goalie with `gk.start == 1` (`neurhl/data/build_g_tensors.py:137-168`).
  - The live goal calibration then had to use fallback lineups for 178 of 178 team-games (`neurhl/configs/live_goal_calibration.json`), and the season simulation samples lineups.
  - No backtest anywhere uses *projected* lineups.
  - So 0.6650 and 0.6601 are upper bounds. Neither the morning forecasts nor the 1,344-game preseason file can reach them.
- **Player backtests leak the future roster.** `neurhl/sim/project_players.py:64-76` (`team_map_actual`) assigns each player to the team he played for most **in the season being predicted**, using that season's player-games.
  - `to_totals` (`neurhl/models/player_proj.py:309-356`) then splits each team's ice-time budget over exactly the players who actually played for it that season.
  - That is not a preseason forecast. It already knows who made the team, who was traded, who was called up and who got hurt.
  - The published points MAE of 9.1–10.5 (`neurhl/configs/player_backtest.json`) and 8.5–10.0 (`player_season_v2.json`) were earned with that knowledge.
- **The evaluation population is chosen on the outcome.** Points MAE is computed only for players with at least 40 actual GP (`project_players.py:96`). The code's own comment concedes this selection shifts the bias by −1.9 points. It is still the headline metric, and the live scorecard reuses it.

### C2. The expensive part of the model adds no win-probability skill

- The engine is a hazard-integrated network with conserved ice time, softmax deployment and zero-initialised residual heads. It is fed by a tensor pipeline, a state bank, an xG model and RAPM priors.
- It scores 0.66568 against Elo's 0.66566 on its own gate **(R)**.
- It is also *overconfident* **(R)**:
  - its logit SD is 0.637 against the stack's 0.539;
  - its bottom decile predicts 0.287 against 0.329 observed;
  - its top decile predicts 0.791 against 0.742 observed.
- The shipped configuration (g1rk / `g2027_v3`) never faced the seal. Its only evidence is a second run on the already-spent G_GATE window. Its rookie-input gain (−0.00017, SE 0.00018) passed because the rule only required "lower" (PLAN_NeurHL_1_1 A18).
- The README's own Findings section concedes the engine is "level with the much simpler neural player layer". That is the finding: the complexity buys nothing.

### C3. The standings layer ties Elo and a free line, and never combines them

- **Shipped NeurHL-2 layer** (`neurhl/configs/season_matched_comparison.json` **(R)**): MAE 9.99 against v1 Elo's 9.17, better in 1 of 5 seasons. The variant *without* the roster term does better (9.46), so the roster model made standings worse.
- **Current layer, 2019-24 judge seasons** (`neurhl/output/neurhl_1_1/season_layer_c2.json` **(R)**):

  | Forecast | MAE |
  |---|---|
  | NeurHL | 9.42 |
  | NeurHL's Elo | 9.47 |
  | Late-preseason sportsbook points lines, same 158 team-seasons **(R)** | 9.51 |

  NeurHL built a neural season engine to tie two forecasts that cost nothing to produce.
- **Never combined.** Two forecasts of similar accuracy with imperfectly correlated errors should be averaged; it is the cheapest accuracy there is. On the seven seasons of recovered lines (222 team-seasons), a 75/25 blend of the line with even a crude two-variable team-history regression beats both inputs **(R)**:

  | Forecast | MAE |
  |---|---|
  | Line alone | 9.86 |
  | Regression alone | 10.26 |
  | 75/25 blend | 9.80 |

  NeurHL's only use of the market is to bet against it.
- **Its 2026-27 disagreements are bets its history does not license.** 1.3 is 0.88-correlated with the August market lines but more spread out (SD 10.39 against 9.42) **(R)**. It sits 14.5 points below the line on TOR, 9.6 below on SJS and 7.5 above on OTT, and implies P(side) ≥ 0.65 on 10 of the 32 lines **(R)**. To be fair, those lines were recorded on 2026-08-17 and do not know about camp injuries or the Knies–Marchenko trade. But a layer that has never beaten Elo has not earned confidence at the ten-mispriced-futures level.

### C4. Overconfidence is built in, and the project's own fix sits on the shelf

- **Team goals.** `neurhl/configs/calibration_1_1.json` fits `goal_slope.b_hat = 0.7015` on 12,536 team-games **(R)**: the engine's goal predictions are about 43% too spread.
  - Top decile: predicts 4.17 against 3.73 observed. Bottom decile: 2.16 against 2.37.
  - The fix was applied only to one dated side-set (`neurhl_1_0/cal_20260929`).
  - The 1.1, 1.2 and 1.3 season sets use raw engine goals: projected GD SD is 44 **(R)**. That is larger than the *realized* GD spread of 2025-26 (41.5 over 84 games), and an expectation cannot be more dispersed than the outcomes it forecasts.
- **Team points.**
  - The SD of projected means is 10.2; the within-team SD is 12.1. Together they imply a realized league SD of 15.9 over 84 games, against 13.3 and 15.0 in the last two seasons.
  - The projection's slope on last season's points is 0.64, against a historical year-over-year slope near 0.45.
- **Skaters.**
  - The slope of projected points per game on last season's points per game is 0.96 across 469 regulars **(R)**. A preseason projection should be well below 1, because players do not repeat last season.
  - The 80% intervals are 1.07× (goals) and 1.17× (points) as wide as pure Poisson (`neurhl/sim/unified_2027.py:763-774`). There is no talent or games-played uncertainty, and goals and assists are drawn independently.
  - McDavid is projected at 137.9 points with an 80% interval of 119–158 **(R)**.
- **Season shock.** The shock is 2× its documented size (section 1). It was calibrated on 2011-17 for a different layer (`game_model.py:83-92`).

### C5. Four forecasts in 43 hours, two after the season began, none scored

**(R)** throughout:

| Release | Commit | Time | Timing |
|---|---|---|---|
| 1.0 | — | 02:12Z 09-29 | superseded by "1.0 v2" at 18:53Z |
| 1.1 | `9580606` | — | 6 minutes before the first puck drop |
| 1.2 | `a921158` | 01:41:45Z 09-30 | 4 h 42 min after it: games 1–2 final, game 3 under way |
| 1.3 | `420cd71` | 21:06Z 09-30 | 12 hours after every opening-night result had been committed (`0b90332`) |

- Counting legacy sets, the repository holds **16** team-points projections for 2026-27.
- The 1.2 and 1.3 game files still carry probabilities for games already played, with no valid-from field.
- The site headlines 1.3 and labels it "frozen: 2026-09-30".
- The README still names the 09-25 freeze as "the live test". That is the worst performer so far (log loss 0.856 against Elo's 0.739 on five games).
- With no scorer for 1.1–1.3 and no rule for which release counts, the design allows the best release to be chosen once the season is over.

### C6. Preregistration theatre

- **Amendments.** About 41 across seven plans. Twenty-one of them (PLAN_NeurHL_1_1 A1–A21) were written on 2026-09-29, opening day.
- **Run budget.** Seventeen G runs were made against a cap of 14, allowed by a carve-out (`neurhl/tests/review_tests_neurhl4.py:90-97`). Two acceptance checks were "adapted" so the battery reads 12/12 (A21).
- **The "last" attempt.** The first preregistered architecture scored 0.6966, worse than always picking the home team (0.6898). A2 declared itself "the LAST modelling attempt"; A3 onward followed. A5.1 concedes the tune-window pass was reached "with results in view".
- **Hidden documents.** `.gitignore:24-25` hides every `*.md`. 208 tracked files reference PLAN/NOTES/EVIDENCE documents the public cannot read **(R)**. The documents were deleted from the tree at `d523c85`, 21:19:55Z on opening day, 20 minutes after the first puck drop **(R)**.
- **Rewritten history.** `neurhl/configs/commit_map.csv` remaps 80 commit IDs, and the seal's `freeze_commit` `f467931` resolves nowhere.
- **A false claim on the site.** `docs/index.html:256` still says "All inputs, code, plans … are in the repository".

Preregistration is valuable because it makes a claim checkable. This one is neither binding nor checkable.

### C7. It ignores the market in its forecasts and bets against it anyway

- `src/optimize_board.py` and `src/optimize_full100.py` implement joint-Kelly sizing on a 50/50 model/market blend.
- The allocation stakes $15 on TOR U95.5, $15 on SJS U93.5 and $14.5 on FLA U108.5.
- The FLA bet uses `p_model = 0.912` from the HOWE baseline, which put Florida at 90.8 points against a 108.5 line. That is an 18-point disagreement with the consensus.
- Repriced with NeurHL 1.3's own quantiles, $21.5 of the $98.5 priced stake is negative-EV.
- There is no record of settlement and no closing-line-value tracking.

A forecasting project that stakes money on 60–90% "edges" on main season-points lines is demonstrating that its models are miscalibrated, not that the market is wrong.

---

## 3. Major flaws

1. **Level drift, patched with six multipliers instead of fixed at the source.**
   - The season file evaluates every game with opening-night inputs (`days_in = 0`, convention P). The project's own test (`neurhl/output/neurhl_1_2/season_convention.json`) found the alternative, convention C, nearly unbiased:

     | Measure | Convention C | Convention P |
     |---|---|---|
     | Power-play opportunity ratio | 1.035 | needs ×0.776 |
     | Team MAE, power-play opportunities | 0.26 | 0.99 |

   - P was kept, and now carries six multipliers, while the fitted b = 0.70 goal-spread correction is still missing:

     | Quantity | Multiplier |
     |---|---|
     | Power-play opportunities | ×0.776 |
     | Power-play minutes | ×0.826 |
     | xG | ×0.961 |
     | Goals | ×1.0055 |
     | Shots | ×0.928 |
     | Attempts | ×1.033 |

   - The goal multiplier wandered by engine version: 0.923, 0.972, 1.011, 1.005. On the seal, goals were predicted at 3.204 against 2.980 observed (+7.5%).
   - The root causes are still there:
     - `season_scaled = (season − 2006)/20`, a linear time trend extrapolated to 1.05 for 2027 (`neurhl/sim/g_live.py:74`);
     - no offseason regression in team state;
     - skater baselines shrunk toward 2008–2011 medians despite a documented ~13% shift in the scoring era (`build_g_tensors.py:69-75`).
2. **The 1.2 power-play fix never reached the players.**
   - `unified_2027.py:548-563` rescales PP and SH minutes and xG, but not skater goals or assists.
   - Every PP-usage quintile's goals move by the same ×1.02 while PP time falls ×0.826.
   - The top-PP quintile ends up projected at 1.053× its three-year points per game, against 0.85–0.90× for the middle quintiles. PP stars are *anti*-regressed: Bouchard 1.17 vs 1.00, Werenski 1.16 vs 0.97, Dahlin 1.05 vs 0.87.
   - The per-game probabilities feeding the standings still include the ~33% inflated PP opportunities.
3. **Goaltending, the highest-leverage position, is the weakest module.**
   - `G_START_MAX = 0.72` (`neurhl/sim/availability_2027.py:130`) caps every goalie at 52.7 starts. Vejmelka, Shesterkin, Vasilevskiy and Ullmark all sit at 51.9–52.7 **(R)**, while 10–14 goalies a season historically reach 55+ GP.
   - There are no random goalie injuries (`:947-958`), and 28 of 32 teams use exactly two goalies.
   - The engine's goalie input is career-cumulative GSAx/(xGA+150) with no decay (`build_g_state.py:228`); the project's own ledger calls it "stale".
   - The start-probability GBM loses to a share heuristic (log loss 0.71–0.77 vs 0.66–0.68, `neurhl/configs/goalie_gates.json` GS1), even though the share is one of its features. That points to a bug.
   - NeurHL-H, which carries 0.54–0.58 of the stack weight, has no goalie input at all.
4. **xG.**
   - The sequential isotonic calibrator makes held-out log loss *worse* in 17 of 18 vantages (`neurhl/configs/xg_gates.json`).
   - Check X3 fails.
   - Season-level predicted-to-observed goal rates range from 0.941 to 1.081. The season that feeds the 2027 inputs (2025-26) runs 8% hot.
   - The X1 baseline it "beats in 18 of 18 seasons" uses only distance and |angle|.
   - The xG specification was chosen on pooled vantages that include the gate and seal seasons (PLAN_NeurHL A5.3).
5. **Baselines and ablations are broken or mislabelled.**
   - A second Elo (`neurhl/sim/game_model.py:156`: K=8, H=30, no margin of victory) claims to "match the house Elo" and doesn't. It feeds the opponent-Elo feature of the player-game layer.
   - The R3 "trees vs network" rung reports a logistic-plus-Elo stack at 0.6970, worse than predicting home ice; the first run logged `inf`. That is a bug, and the conclusion "no better than Elo" rests on it.
6. **Leakage defences are partial.**
   - Features are selected by *blacklist* (`build_g_tensors.py:122-132`), so any new numeric column silently becomes a same-game feature.
   - `audit_leakage_g.py:53` audits only `add_state` outputs. It does not audit days_since, team_change, gp_with_team, the RAPM merge, CTX, Elo, the H projections or the rookie features.
   - `tensorize_games.py:98` flags all of 2019-20 as COVID, which is hindsight.
7. **Rookies: survivorship bias, then more weight on it.** Training rookies must have played 20+ NHL games in the rookie season, and the translation factors use only NHL seasons of 20+ games (`neurhl/eval/rookie_priors.py:68,118`). Both select on the outcome. 1.2 then *raised* the translated weight on rookie goals to 75%.
8. **NeurHL-H training hygiene.**
   - There is a known train/serve skew: training rows get embeddings fitted on their own season (PLAN_NeurHL4 §M).
   - Early stopping uses a random 10% split, not a temporal one (`neurhl/train/train_player.py:168`).
   - The head's regularisation C is chosen by *training* loss (`hier_core.py:53`), so it always picks the least regularisation.
   - The event-LM and career embeddings, which the README calls a documented null, were never ablated out of H.
9. **Uncertainty discarded in the season simulation.**
   - The engine runs 64 lineup draws per game, and probabilities are averaged over them (`unified_2027.py:490-519`).
   - The 20,000 seasons then Bernoulli-sample the averaged p plus one fixed team shock.
   - That discards within-draw injury correlation, parameter uncertainty and goalie-injury scenarios.
10. **Live-forecast integrity rests on the forecaster's own clock.**
    - "On time" means committer date < start (`neurhl/eval/score_live_g_2027.py:289`).
    - The GitHub stamp is recorded but "never required" (`:18-20`; `neurhl/live/deadline.py:69,81`). A test asserts that a late stamp leaves a forecast valid (`test_score_live_g.py:311`).
    - Commits are unsigned by force (`publish.sh:92`).
    - A missed or late pregame forecast removes that game for *every* model (`score_live_g_2027.py:21-24`), so an awkward game can be dropped by not publishing.
    - The pipeline runs on one laptop under launchd and caffeinate.
11. **Scorecards mislabel and mix.**
    - The column labelled "1.0" in `scorecard_g_2027.json` is actually the 09-25 freeze (`neurhl/live/forecast.py:124,140`).
    - The "NeurHL-G" live series mixes three engines on opening day: g2027_v1 in the morning, v2 for game 1, v3 for games 2–5.
    - Two different Elo log losses appear across scorecards (0.73909 vs 0.73852).
    - There is no market benchmark, no RMSE and no reliability curve.
12. **The tests test bookkeeping.**
    - Seventeen check scripts hold about 310 `check()` calls. None tests spread, regression toward the mean or interval coverage; they are identities and wide bands. "Goals per game in [2.7, 3.3]" would pass a coin.
    - Only `test_score_live_g.py` has conventional unit tests.
    - 1.1 passed 41 of 41 checks while carrying PP opportunities 33% high and shots at 30.0 against 27.8. The count then grew 41 → 45 → 48 as each error was found in production.
    - The batteries need gitignored tensors and checkpoints, so no outsider can run them, and running them writes to `configs/`.

---

## 4. Nitpicks (every one of them real)

- **Shootout share.** `P_SO_GIVEN_TIE = 0.38` is labelled "2016-2026 share" (`unified_2027.py:66`, `boxscore_mc.py:30`). The actual 2016-26 share is 0.337, and 0.284–0.315 in three of the last four seasons **(R)**. It feeds shootout goals and the ROW tiebreaker.
- **Assists per goal** are exactly 1.676 for every team (SD 0.000), while `boxscore_mc.ASSIST_MIX` implies 1.58.
- **Shutouts** are computed as exp(−regulation GA): no overtime, no overdispersion.
- **Age** is season_end minus birth year, so January and December birthdays get the same age.
- **Team state** decays per game, not per day, so a five-day break counts the same as a back-to-back. Team shrinkage uses five pseudo-games.
- **Hellebuyck** is excluded for all 84 games, with no mixture over a return or a trade.
- **IR handling.**
  - Twenty DailyFaceoff "ir:out" players have `injured=False` in the 1.0 availability file.
  - Every IR player gets a flat 15 games out (`availability_2027.py:127`).
  - As a result, McAvoy (74.0 GP), Rust (74.4) and Fiala (72.4) project near-full seasons.
- **Old playoff format.** The 09-25 freeze uses a top-8-per-conference playoff format rather than the division/wild-card format in force (`project_2027.py:269-289`), and a hardcoded home OT win share of 0.53.
- **Team SOG spread.** The negative binomial with r = 40 is too wide: 80% coverage is 0.861 **(R)**. The fitted r is 98.7 **(R)** and sits unused in `calibration_1_1.json`.
- **1.3's shot fix** is a uniform ×0.928 on shots and ×1.033 on attempts for every team and player (`shot_level_1_3.py:47-72`).
  - As a side effect, xG per shot jumps from 0.102 to 0.110.
  - The engine's shots-per-attempt ratio (0.532) is outside every recent season (0.478–0.498). That is a miscalibration, not an input lag.
- **Code hash.** `neurhl_1_2/season/run_2027.json` records code `b001285`, whose `unified_2027.py` has none of the 1.2 flags. The recorded hash does not identify the code that made the files.
- **Seal script.**
  - Its secondary Holm family counts team_xga, which is identical to team_xgf (`seal_g.py:176`).
  - A missing H logit silently becomes 0 (`:129`).
  - Cached predictions are reused if present (`:107`).
- **Duplicated simulator.** The standings and playoff simulator exists in five copies: `unified_2027.season_mc`, `season_1_1.season_mc_dyn`, `project_2027`, `eval/backtest_season.simulate`, `ratings_bridge.simulate`.
- **Oversized code.** `unified_2027.main()` is about 425 lines, and `availability_2027.py` is 1,622 lines.
- **Magic numbers:** 57.0, 5.0/4.0, /18, fake game IDs 9_800_000/9_900_000, and a sort on an undocumented `SKB[...][0]`.
- **Hardcoded network constants.** SOG weights 1.3/0.4, ixG offsets +0.3/+0.05 and `shrink_k=5` are hardcoded inside the network (`neurhl_g.py:235-238`). `T_ev.clamp(min=30)` (line 185) quietly breaks the "exact" conservation claim.
- **Divisions** are read from the HOWE baseline's output CSV (`unified_2027.py:88-92`).
- **Dead code.**
  - `season_sim.py` reads a missing `preds/games_2027_neurhl.csv`.
  - `project_2026_27.py` and `ratings_v2.py` are unreferenced.
  - `build_g_state.py:273` is dead.
  - The officials builder's output is never used.
- **Version sprawl.**
  - `backtest{,2..6}.py`, `report{,2..6}.py` and `report_only_{v5,v6,diagnostics}.py`
  - four search ledgers and five calibration files
  - two `game_model.py` files
  - 64 versioned params and projection files
  - seven xlsx workbooks
  - six ~5.5 MB copies of one season's player-games
  - 189 MB on disk
- **2027-28 projections** (nine files), two seasons out. Their team SD (13.19) barely differs from 2026-27's (13.06), yet they are called a "co-headline".
- **Reproducibility.**
  - Only scikit-learn is pinned (`==1.9.1`, because the xG GBM is a pickle).
  - The pickle's SHA check runs only if a sidecar file happens to exist (`ingest_2027.py:686-690`).
  - `torch.load(weights_only=False)` appears (`validate_event_sim.py:56`).
  - Networks are trained on Apple MPS, which is not bit-deterministic.
  - CPU thread counts vary (3/5/8) across ledger runs. Two runs with the same config hash differ by 0.0006, twice the 0.0003 adoption threshold.
- **Scraping and attribution.** `common.py` scrapes with a spoofed Safari user agent, and `publish.sh:42,95` rejects commit messages that credit an AI co-author.
- **`manifest.py`** claims "changed content never reads as fresh". That is false for copies that preserve size and mtime.
- **The site.**
  - It renders entirely in JavaScript, with no `<noscript>` fallback.
  - Sortable headers are `th.onclick`, with no keyboard access and no `aria-sort`.
  - Text is 11–13 px, and `.grid4 minmax(540px)` overflows phones.
  - There is no dark mode.
  - It shows no live accuracy. NeurHL-G's pregame forecasts are 0.0104 worse than Elo on five games; that is noise, but noise the site chooses not to show.

---

## 5. What NeurHL gets right, and what I am taking

Credit where due. Each of these is worth adopting in any competitor, and I intend to take all of them:

| NeurHL strength | Take it as |
|---|---|
| **Conservation laws.** Team ice time is conserved, one side's SH minutes equal the other's PP minutes, and player stats are shares of team totals, so players sum to teams and teams to games | Normalise skater TOI within team and situation; reconcile player goals to team goals; balance faceoffs |
| **Walk-forward state**, read strictly before each game, with an effective sample size (`build_g_state.py:39-50`) | Ratings as a filter whose state is only ever read before a game; every feature builder takes the target season and reads only earlier seasons |
| **Causality audit.** Corrupt everything after a cut date and require bit-identical features before it | Go further: read every freeze input from the git tree of the last commit before the information cutoff, and record each file's SHA-256 and last-change time in the freeze manifest |
| **Timing discipline for live forecasts.** Committed about an hour before puck drop; the first-added blob is what gets scored; misses are counted by reason | Keep it, but *require* a server-side timestamp for a forecast to count, and score a missed forecast as the preseason number rather than dropping the game for everyone |
| **Availability modelling.** Depth-aware dress probabilities, absence spells, dated returns, call-up tiers | Keep it, and extend it to goalies: injury risk and no hard start cap |
| **Exact best-of-seven** with 2-2-1-1-1 home ice, a vectorised tiebreak key, and the correct division and wild-card bracket | Keep it; add goal differential and goals for to the key |
| **Full quantiles published**, so CRPS can be scored | Keep it |
| **Paired, clustered tests** with a minimum detectable effect computed in advance | Keep them: paired bootstrap intervals on per-game log-loss differences |
| **Honest reporting of nulls and failed gates** | Keep it, including where the competitor loses |

---

## 6. What a competitor has to do differently

This is a design brief derived from the flaws above, not a claim that it has been done.

1. **Anchor to the market, then add what the market lacks.** Preseason standings should be a walk-forward blend of the de-vigged points line, a regressed team-history view and a bottom-up roster view. The model earns its keep by moving the line for news it has not priced: roster moves after the line was posted, camp injuries, the Hellebuyck suspension.
2. **Regress everything, and measure the regression.** Shrink player rates by exposure-based reliability toward usage-aware priors, fit the aging curves, and regress finishing heavily. Report the slope of projection on last season rather than assuming it.
3. **Propagate uncertainty honestly.** Draw each simulated season's team strengths from a posterior whose SD comes from walk-forward residuals, and let strength drift within the season. Let goalie health and start shares vary by simulation.
4. **Use one scoring model for wins and goals**, with the late-game (pulled-goalie) margin structure built into the model rather than patched afterwards.
5. **Backtest only on preseason information.** Build rosters from an opening-roster proxy, never from the team a player ended the season with. Report every protocol difference from NeurHL next to the numbers.
6. **Ship one release, frozen before puck drop**, with a provenance manifest and pre-declared scoring for every file.

---

## Appendix: recomputed numbers **(R)**

All values are produced by `review/evidence/recompute.py` and stored in `review/evidence/recomputed.json`.

| Quantity | Value |
|---|---|
| Log loss, gate games (n = 6,289): stack / H / Elo / raw engine | 0.66006 / 0.66056 / 0.66566 / 0.66568 |
| Raw engine logit SD vs stack; bottom and top decile predicted vs observed | 0.637 vs 0.539; 0.287 vs 0.329; 0.791 vs 0.742 |
| Stack minus H, per game (SE) | −0.00049 (0.00060) |
| Log loss, restatement (n = 11,052): H / Elo / constant | 0.66455 / 0.66914 / 0.69003 |
| Log loss, seal (n = 2,624): stack / H / Elo | 0.67151 / 0.67365 / 0.67842 |
| H − Elo by vacated-TOI bucket: none / 0–20 / 20–40 / 40+ min | −0.0003 / −0.0029 / −0.0055 / −0.0087 |
| S-STOP | pass = false; G − H −0.0005; beat H in 3 of 5 seasons |
| Season MAE: v1 Elo / shipped NeurHL-2 / NeurHL-2 without roster term | 9.17 / 9.99 / 9.46 |
| Season MAE, 2019-24 judge (n = 158): NeurHL layer / Elo / sportsbook lines | 9.42 / 9.47 / 9.51 |
| Line vs crude team-history regression vs 75/25 blend (222 team-seasons) | 9.86 / 10.26 / 9.80 |
| Goal-spread slope fitted but not applied | b = 0.7015 (n = 12,536) |
| Team SOG 80% coverage at r = 40; fitted r | 0.861; 98.7 |
| Shootout share of extra-time games, 2016-26 | 0.337 (0.284–0.389 by season) |
| 1.3 team points SD / market SD / correlation / mean absolute gap | 10.39 / 9.42 / 0.884 / 3.80 |
| 1.3 goal differential SD | 44.1 |
| 1.3 largest gaps to market | TOR −14.5, SJS −9.6, OTT +7.5, VAN −6.0, FLA −5.7, WSH −5.6, BOS +5.5, NYI +5.2 |
| 1.3 lines with P(side) ≥ 0.65 | 10 of 32 |
| 1.3 skaters: slope on last-season P/GP (n = 469); 100-point / 40-goal players | 0.96; 5 / 11 |
| 1.3 top goalie starts | 52.7, 52.5, 52.4, 51.9 |
| Minutes from the first puck drop: 1.1 / docs removed / 1.2 / 1.3 | −6.2 / +19.9 / +281.8 / +1,446.1 |
| Seal spent → seal in training | 18:36Z → 18:53Z |
| Tracked files referencing hidden plan documents | 208 (tracked `.md` files: 1) |
