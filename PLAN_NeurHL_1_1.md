# PLAN NeurHL 1.1: calibrated count distributions, then further layers

STATUS: COMMITTED 2026-09-29, BEFORE THE FIRST 2026-27 REGULAR-SEASON GAME
(2026-09-29 21:00 UTC). Append-only. C1 is fully specified here. C2 and C3
are declared as next steps; each gets its own dated amendment before any
number it will be judged on is computed.

## Why

A review of the 53 works cited by the project's write-up asked whether any of
them justifies new architecture. The review read each work, in full where a
copy was reachable.

**The finding matches the project's own record.** No cited method offers a
change to the win-probability engine with an expected gain the sealed test
could detect:
- the sealed test (2,624 games) has a minimum detectable effect of 0.0026
  against NeurHL-H;
- the candidates found are worth 0.0003 to 0.002.

The sealed 2025 and 2026 seasons therefore stay unspent, kept for new
information.

**Three places do justify new components:**

| # | Component | Layer | Basis |
|---|---|---|---|
| C1 | Fitted count distributions: team-shots dispersion and goal-mean slope | Stat sheet and goal totals; win probabilities unchanged | Gneiting and Raftery (2007), optimum score estimation; Czado, Gneiting and Held (2009), count dispersion; Gneiting, Balabdaoui and Raftery (2007), a hump-shaped PIT means too wide; Dawid (1984), prequential tuning; Efron and Morris (1975), shrinking extremes. Also PLAN_NeurHL4 A2: gate C fails on team-shots coverage, 0.860 against a nominal 0.80. |
| C2 | Team strength that evolves within the season, and a season shock estimated for this engine | Season simulator | Lopez, Matthews and Baumer (2018), state-space team strength (structure only: their estimates come from betting markets, which never enter this project); Efron and Morris; Gneiting and Raftery (fitting by CRPS) |
| C3 | Rating with a gain that shrinks as a season's games accumulate, and weighted overtime and shootout results | A candidate stack column | Elo (1978), §3.73 "development coefficient"; Whelan and Klein (2021), four-outcome Bradley-Terry |

Rejected, with evidence:
- **Excitation after goals (Hawkes 1971).** On 2012 and 2014-2018, periods 1-2,
  the goal rate in the 30 s after a goal is 2.9 per 60 minutes, against about
  5.5 later, at every lead. That is inhibition, not excitation.
- **Stack weights gated by season phase.** Walk-forward on 2014-2018, it adds
  +0.00074 log loss (SE 0.00029).
- **A tree-model stack column.** Rung R3 already lost to Elo.
- **A manpower state in the outcome chain.** About zero effect pre-game.
- **Neural point-process game predictors.** A documented null.
- **Training the network on CRPS.** It already trains on proper losses.
- **Anything market-based.** Firewall.

## C1: fitted count distributions

### Phase A (declared before fitting)

- **Targets:**
  - team shots on goal per team-game;
  - team regulation goals per team-game (the overtime or shootout winner's
    extra goal removed, as `eval/score_live_g_2027.py` counts them).
- **Forms.** Each calibrated forecast is a fixed function of numbers frozen in
  each pregame forecast file, plus the constants below.
  - **Shots:** NB2(mu, r_hat), Var = mu + mu^2 / r_hat, where mu is the
    file's `sog_home` / `sog_away`. The frozen comparator is NB2(mu, 40), the
    stat sheet's dispersion.
  - **Goals:** Poisson(m_t * M * (raw / M) ^ b_hat). Here raw is
    `goals_home_raw` / `goals_away_raw`, m_t is the file's `goal_mult`, and
    M = 3.2534 is the mean raw projection in
    `configs/live_goal_calibration.json`. The frozen comparator is
    Poisson(m_t * raw), the A1 mean. At b_hat = 1 the two are identical.
- **Fit.** `neurhl/eval/fit_calibration_1_1.py` maximises the NB log score
  (for r) and the Poisson log score (for b, with a free level per season)
  over NeurHL-G's out-of-sample predictions on G_GATE:
  - source: `neurhl/output/preds/g_gate_games.csv`;
  - configuration g1, the live bundle's configuration;
  - seasons 2019, 2020 and 2022-2024;
  - 12,578 team-games for shots and 12,536 for goals.

  Rows from 2025 or 2026 stop the fit.
- **Seen before fitting (postdictions, disclosed).** During the review:
  - on the iteration window (2012, 2014-2018), r = 83 and a goal slope of
    0.60 (linear);
  - walk-forward on G_GATE seasons, r = 90-100 and b = 0.68-0.73.

  None of these numbers is evidence for C1; only the 2026-27 games are.

### Phase B (fitted values, frozen here)

| Quantity | Value |
|---|---|
| r_hat | 98.68 |
| G_GATE 80% randomised-PIT coverage, r = 40 / r_hat | 0.861 / 0.797 |
| G_GATE NB log-score gain per team-game | +0.0206 |
| b_hat | 0.7015 |
| G_GATE Poisson log-score gain per team-game | +0.0046 |
| G_GATE top decile, predicted / calibrated / realised | 4.167 / 3.800 / 3.732 |
| G_GATE bottom decile, predicted / calibrated / realised | 2.163 / 2.393 / 2.373 |
| `neurhl/configs/calibration_1_1.json` | sha256 `230d88eab2c7614872f1723cea1a50dc74db3056d26f4c1f8993448843f68b6f` |

Running the fit again reproduces the file byte for byte
(`neurhl/tests/test_calibration_1_1.py`).

### Test (one inference, after the last regular-season game, 2027-04-10)

- **Games:** every 2026-27 regular-season game counted by
  `eval/score_live_g_2027.py` for the pregame forecast (the same selection
  and validity rules, reused in code).
- **Primary, one Holm family of two:** the mean paired log-score difference
  per team-game (calibrated minus frozen) for shots and for goals. Each test
  is two-sided at 0.05 by week-block bootstrap (9,999 draws, seed 711).
- **Also reported:**
  - 80% randomised-PIT coverage for shots (nominal 0.80; the calibrated
    forecast passes if within ±0.03);
  - goal quintiles, frozen against calibrated against realised;
  - a team-clustered standard error.
- **Interim:** scorecards (`neurhl/output/live/scorecard_1_1_2027.json`,
  nightly) are descriptive only.
- **Stopping rule:** the full regular season. No earlier inference.

### Consequences, fixed now

| Outcome | Consequence |
|---|---|
| Both pass | From 2027-28, the stat sheet uses the fitted dispersion and slope. The 2026-27 frozen files are never edited. |
| One passes | That one is adopted for 2027-28 and the other is a documented null. |
| Neither passes | Both are documented nulls. The frozen stat sheet stands. |

### The dated file set beside the NeurHL 1.0 freeze

`neurhl/sim/calibrate_1_0_goals.py` applies the goal slope to the frozen
NeurHL 1.0 files and writes `neurhl/output/neurhl_1_0/cal_20260929/`. The
frozen files are read, never written.
- Each team-game is scaled as a unit by f = k * (g / m0 / M)^(b_hat - 1),
  with one league constant k = 1.0050. So the league's goals are unchanged
  (3.034 per team-game) and every sum rule still holds.
- Affected: team goals for and against, power-play goals and percentages,
  skater goals, assists and points, and goalie goals against.
- Unchanged: win probabilities, standings, shots and ice time.
- The spread of team goals-for per game falls from 0.303 to 0.215.
- b_hat was fitted on game-day predictions, while these are
  preseason-convention predictions, so the set is an approximation.
- It is scored beside the frozen files, under the frozen files' own
  measures (PLAN_NeurHL_1_0, Scoring: team goals and skater points MAE).
  The frozen files remain NeurHL 1.0's forecast.

| File | SHA-256 |
|---|---|
| `neurhl/output/neurhl_1_0/cal_20260929/games_2027.csv` | `c642e5b2c89aac5987bf7b6def2bba05a2e9fb3d0f1be1c9a6bec8d9d196aa6b` |
| `neurhl/output/neurhl_1_0/cal_20260929/teams_2027.csv` | `d2e3a2b2a0301798ada9bcb147913cfbc652e776d91edd6c43eae41d7d8e9fc4` |
| `neurhl/output/neurhl_1_0/cal_20260929/skaters_2027.csv` | `7b4254405e685c6eb5898f490fbc7dab9fe4554a85961aa1805a83918ac1e496` |
| `neurhl/output/neurhl_1_0/cal_20260929/goalies_2027.csv` | `bf1291ff42f3ae7748fb3d820d8a5be80709e48217c532c584262bc1fe3d172b` |
| `neurhl/output/neurhl_1_0/cal_20260929/player_games_2027.csv.gz` | `c185d7b4974a9542b795ac15c5b8fb17f181abc365b5d9ba52d7353c0c236964` |
| `neurhl/output/neurhl_1_0/cal_20260929/checks.json` | `8e44928b82c31b89425a50fca750934f89dd77000efdf04653f9e73bfab48599` |

### Known limitation

Player shots in the stat sheet are a multinomial split of team shots. A
sharper team total narrows player intervals, which the gate already found a
little too narrow (coverage 0.783). C1 does not change player distributions.

## C2 and C3: declared next steps

- **C2.** A historical backtest of the NeurHL 1.0 season model, in the
  preseason convention:
  - G snapshots trained walk-forward with configuration g1, and the
    availability model fitted on earlier seasons;
  - seasons 2012, 2014-2020 and 2022-2024; never 2025 or 2026.

  It compares the constant season shock, a weekly AR(1) team-strength path,
  and AR(1) with shrinkage of later games toward the mean, on points CRPS,
  MAE, 80% coverage and playoff Brier. Parameters are fitted on 2012-2018
  and judged on 2019-2024, under a rule committed in an amendment before the
  judging run.
- **C3.** A rating with a Kalman-style gain (high at season start and after
  roster turnover, falling as games accumulate) and outcome targets {1, p_OT,
  1 - p_OT, 0}, with p_OT learned. In the review, overtime and shootout
  winners' slope on the Elo logit was 0.34, against 1.28 for regulation
  winners (2012 and 2014-2018). It is fitted on 2009-2017 and judged as a
  stack-column candidate on the iteration and gate windows. If it helps, it
  becomes a live exploratory column, declared before the games it is scored
  on. The frozen Elo comparator is never changed.
- **Neither spends the sealed seasons.** An engine candidate may: in-season
  fine-tuning with decoupled L2-SP (Li, Grandvalet and Davoine 2018;
  Loshchilov and Hutter 2019). But only under its own plan, and only if its
  pre-gate gain reaches the sealed test's detectable size.

## A1 (2026-09-29 04:30 UTC, before any C2 or C3 number is judged)

### C3: dropped as an exploratory null

The C3 rating was built (`neurhl/sim/ratings_v2.py`) and tuned by walk-forward
log loss on 2010-2012 and 2014-2017 before a rule was written here. So these
results are reported as exploratory. The tuned values:
- p_ot = 0.69;
- season carry-over 0.81;
- the margin exponent went to 0.

Out of sample, without retuning, against the frozen Elo:

| Window | Difference (SE) |
|---|---|
| 2012, 2014-2018 | +0.00038 (0.00048) |
| 2019-2024 | +0.00055 (0.00046) |

As a stack column (walk-forward on 2019-2024), it changes log loss by
+0.00007 (SE 0.00007). C3 is dropped: no live column, and nothing is
adopted.

### C2: the decision rule, fixed before the judging run

**Inputs.** `neurhl/eval/backtest_unified_season.py` writes preseason-convention
per-game outputs for 2012, 2014-2020 and 2022-2024:
- the g1 engine trained on seasons < V, 5 seeds;
- opening-night rows for every game;
- the frozen Elo entering V;
- the live bundle's Elo + NeurHL-G fallback stack.

Seasons 2025 and 2026 are refused.

**Variants.** `neurhl/eval/season_layer_c2.py` simulates 4,000 seasons per
variant. Team strength moves each game's logit by
k * (s_home - s_away), with k the snapshot's own sensitivity, exactly as
NeurHL 1.0:
- s_team(week) = s0 + a random walk with weekly innovation sd sw;
- s0 has sd sigma0;
- each game's logit is shrunk toward the season's mean logit by
  rho^(weeks from opening night).

The grid:

| Parameter | Values |
|---|---|
| sigma0 | 0, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13 |
| sw | 0, 0.005, 0.01, 0.015, 0.02 |
| rho | 1, 0.995, 0.99, 0.98 |

The frozen NeurHL 1.0 layer is (0.07, 0, 1).

**Selection.** The variant with the lowest mean team-season points CRPS on the
fit seasons 2012 and 2014-2018.

**Judging.** On 2019, 2020 and 2022-2024 (never used for a season-level
decision about this engine), the selected variant is compared with (0.07, 0, 1)
on:
- mean team-season points CRPS, paired by team-season, with a season-block
  bootstrap interval;
- points MAE;
- 10th-90th percentile coverage.

**Adoption requires both:**
- the judged CRPS difference below 0;
- the selected variant's coverage within 0.80 ± 0.05.

If it is adopted, the season layer is rerun on the frozen NeurHL 1.0 games
file (`games_2027.csv`: its probabilities, outcome4 and rate sensitivities are
used unchanged). The result is issued as a dated file set beside the 1.0
freeze and scored at season end with the frozen set's own measures (points
MAE, CRPS, coverage). Otherwise C2 is a documented null and 1.0's season
layer stands.

**Also reported, not deciding:** the same grid on a preseason Elo-only
season model (the logit being the frozen Elo entering V).

## A2 (2026-09-29, before any C2b number is judged): in-season standings (C2b)

**What it projects.** A nightly projection of each team's final points,
built from three pieces:
- the points already earned;
- the frozen NeurHL 1.0 probabilities and outcome4 of the remaining games;
- a team-strength shock updated from the games played
  (`neurhl/sim/live_standings_1_1.py`).

The update is a Laplace posterior of the logistic model
sigmoid(z + k (s_home - s_away)) with prior sd sigma0, plus weekly drift sw
after the last game played. A game's frozen probability is never changed.

**Backtest.** `neurhl/eval/live_standings_c2b.py` runs on the C2 preseason
outputs.
- Checkpoints: 4, 8, 12, 16 and 20 weeks after opening night.
- Comparator: "no update", the prior left in place and drift from opening
  night, which is what adding actual points to the frozen preseason
  simulation does.
- Grid, for each of the two:

  | Parameter | Values |
  |---|---|
  | sigma0 | 0.05, 0.07, 0.09, 0.11, 0.13 |
  | sw | 0, 0.005, 0.01, 0.02 |

- Selection: the lowest mean points CRPS over the fit seasons 2012 and
  2014-2018.
- Judging: on 2019, 2020 and 2022-2024, the update's mean CRPS minus the
  comparator's, paired by team, season and checkpoint, with a season-block
  bootstrap interval.

**Adoption requires both:**
- a difference below 0;
- the update's 10th-90th percentile coverage within 0.80 ± 0.05.

**If adopted:** from the first night with completed games, the projection is
published as an exploratory NeurHL 1.1 product (not a replacement for the
frozen files). It is scored at the end of the season at the same checkpoints
against the final table, beside the frozen preseason files plus actual
points.

## A3 (2026-09-29, before the first 2026-27 game): blended skater season points (C4)

**The blend.** The earlier player-season work declared a 50/50 blend of two
paths and found it had the lowest points error over 11 vantages: 9.07 per 82
games, against 9.41 and 9.77 for the paths alone. The paths are:
- path A, the gradient-boosted season model;
- path B, the summed player-game layer.

NeurHL 1.0's skater totals come from the engine alone. Its player heads beat
path B on every shared target (PLAN_NeurHL4 gate PG).
`neurhl/sim/skaters_blend_1_1.py` blends the two surviving paths 50/50 per
game played:
- points per game = 0.5 x the engine's + 0.5 x c x path A's;
- path A is `proj_p_path_a / exp_gp` in the dated U1 file
  `neurhl/output/player_proj_2027_20260928.csv`;
- c = 0.9625 puts path A on the engine's league scoring level (path A runs
  about 4% higher), so the league's skater points are unchanged (21,830);
- games played stay the engine's (the availability model);
- goals and assists scale with points;
- 725 of 925 skaters have a path-A row; the others keep their 1.0 values.

**Scope and scoring.** Skater totals only. Team files are unchanged, and a
team's skater sums may differ from its team goals by the redistribution. It
is scored at season end beside the frozen skaters file, on the frozen file's
own measure: skater points MAE for skaters with at least 40 games
(PLAN_NeurHL_1_0, Scoring). It is exploratory: it replaces nothing.

| File | SHA-256 |
|---|---|
| `neurhl/output/neurhl_1_0/skaters_blend_20260929/skaters_2027.csv` | `8ce690f3de7a473ceb4a69981b05cdbe1df1e1095acfa3a1c5c0654913d6f5fd` |
| `neurhl/output/neurhl_1_0/skaters_blend_20260929/run.json` | `1bd76d1e487967bff21d279f049abf8a8faed708d7c80efba334f4c375f861a2` |

## A4 (2026-09-29, before any C2 judging season has been computed): season-level blend (C2c)

**Motivation.** On matched seasons, the preseason Elo projection beat the
earlier season layer. NeurHL 1.0's game probabilities already stack Elo with
the engine, but with weights fitted on game-day predictions; opening-night
predictions of games months away may want a different mix.

**Variants.** C2c adds a blended game logit:

z = a x z_stack + (1 - a) x z_elo_pre, with a in {0, 0.25, 0.5, 0.75, 1}

- z_stack is the preseason stacked logit and z_elo_pre the frozen Elo logit
  entering the season.
- It is crossed with the A1 grid (sigma0, sw, rho). k and outcome4 are the
  game's own.

**Selection and judging.**
- Selection: the (a, sigma0, sw, rho) with the lowest mean fit-season CRPS
  (2012, 2014-2018).
- Judging: on 2019, 2020 and 2022-2024, against the frozen layer
  (a = 1, 0.07, 0, 1) and against A1's selected variant.

**Adoption.** C2c replaces A1's selection only if its judged CRPS is lower
than both, with coverage within 0.80 ± 0.05. When C2c is adopted, its
variant is the one rerun for 2026-27, with z_elo_pre from the frozen games
file's `p_home_win_elo`.

## A5 (2026-09-29): C4, in-season fine-tuning of the engine, is an exploratory null

**Method.** Iteration window only (`neurhl/eval/finetune_g_c4.py`,
`finetune_g_c4_stack.py`). In each season, from game 300 on, every block of
150 games is predicted two ways:
- by the season's frozen g1 snapshots;
- by copies fine-tuned on the season's games already played: 3 epochs, AdamW
  with no decay toward zero, and decoupled L2-SP toward the frozen weights
  with lam = 10.

**A single seed misleads.** Against a single frozen seed, the tuned copy
looked strong: 8 of 9 season-seed runs improved, by up to 0.006 nats per
game.

**The live recipe shows no gain.** Measured the way the model is used (a
five-seed ensemble, stacked with Elo and NeurHL-H by the live bundle's
coefficients) on 2012 and 2014-2016 (3,720 games):

| Measure | Tuned minus frozen (SE) |
|---|---|
| Raw ensemble | -0.00116 (0.00117) |
| Final stacked probability | -0.00006 (0.00034) |

Mostly, the single-seed gains came from tuning partly undoing one seed's
noise, which the ensemble already averages away. C4 is not taken to G_GATE,
and the sealed seasons stay unspent.

## A6 (2026-09-29, before the first 2026-27 game): C1b, team-xG dispersion

**The problem.** The frozen stat sheet draws team xG from a gamma
distribution around the model mean with shape 9. On G_GATE (the same
12,574 out-of-sample team-games), its 80% intervals cover 0.855.

**The fit.** `neurhl/eval/fit_calibration_1_1b.py` fits the shape by maximum
likelihood:
- k_hat = 11.86;
- coverage 0.798;
- +0.0179 nats per team-game.

This was seen once, in exploration, before fitting. It is disclosed as a
postdiction.

**The test.**
- The calibrated forecast is gamma(k_hat) around each pregame forecast's
  frozen `xgf_home` / `xgf_away`. The comparator is gamma(9).
- The target is the ingest's team xG (`xgf_all`); degraded games are left
  out.
- It uses the same games, the same bootstrap and the same single inference
  after the regular season as C1, but a separate Holm family of one.
- C1's two declared tests are unchanged.

**Seen but not tested.** Power-play opportunities are under-dispersed
(variance/mean 0.665; Poisson 80% coverage 0.883). They cannot be tested
this season, because the forecast files do not store their mean. This is
recorded for the 2027-28 stat sheet.

| File | SHA-256 |
|---|---|
| `neurhl/configs/calibration_1_1b.json` | `53cae3662e8ea40f5f50db952987500cb113c55fbee0dbe37e637152e54e6956` |

## A7 (2026-09-29, before the first 2026-27 game): C1c, skater-shot intervals

**The fit.** Skater shots on goal per game ~ NB2(mu, r_s) around the
engine's skater mean. `neurhl/eval/fit_calibration_1_1c.py` fits r_s by
maximum likelihood on the cached G_GATE predictions (226,354 dressed
skater-games):
- r_s = 18.75;
- randomised-PIT 80% coverage 0.798 against 0.782 for Poisson;
- +0.0023 nats per skater-game.

**The test.**
- Games: the same as C1 (pregame, as first committed, before the start),
  with each file's skater lines as first committed. Degraded games are left
  out.
- Comparison: the calibrated 80% interval, the NB(`sog_mean`, r_s) 10th and
  90th percentiles, against the frozen stat sheet's own `sog_p10`-`sog_p90`.
- Score: the interval score at alpha = 0.2 (Gneiting and Raftery 2007),
  width plus 10 times any miss beyond either end; lower is better.
- One inference after the regular season, with the week-block bootstrap, in
  a separate Holm family of one. Coverage is also reported.

**Expectation, stated now.** The frozen intervals come from simulated draws
that already include team-shot spread, so the gain may be small or negative.

| File | SHA-256 |
|---|---|
| `neurhl/configs/calibration_1_1c.json` | `f19291cf492326a5510364ec28cb08ff168f9e3934aa8ca1d14f7782775f94c8` |

## A8 (2026-09-29): ensemble size, an exploratory null

The question is whether more seeds would help. On 2012 and 2014-2017 (the g1
snapshots saved by the C2 backtest; game-day states; the live bundle's
stack), log loss averaged over every seed subset of each size:

| Seeds | Raw engine | Stacked |
|---|---|---|
| 1 | 0.67923 | 0.67242 |
| 2 | 0.67780 | 0.67230 |
| 3 | 0.67733 | 0.67226 |
| 4 | 0.67710 | 0.67224 |
| 5 | 0.67696 | 0.67223 |

The gains follow a + b/k. Doubling to 10 seeds would improve the stacked
probability by about 0.00002, so the live bundle keeps its five seeds.

## A9 (2026-09-29, before the first 2026-27 game): C2, C2b and C2c results under the rules of A1, A2 and A4

The inputs are `neurhl/data/tensors/_season_bt/pre_<V>.parquet` (g1 snapshots,
five seeds, for every season 2012, 2014-2020 and 2022-2024). Results are in
`neurhl/output/neurhl_1_1/season_layer_c2.json` and `live_standings_c2b.json`.

### C2 (A1): not adopted

The fit seasons selected (sigma0, sw, rho) = (0.07, 0.005, 0.98). On the
judge seasons (158 team-seasons):

| Layer | Points CRPS | MAE | 80% coverage |
|---|---|---|---|
| Selected (0.07, 0.005, 0.98) | 6.836 | 9.47 | 0.804 |
| Frozen NeurHL 1.0 (0.07, 0, 1) | 6.799 | 9.42 | 0.823 |

The difference is +0.037, 95% CI -0.19 to +0.28. The rule requires a
negative difference, so C2 is not adopted.

**Reading.** This backtest supports the frozen season layer. Its season-shock
size, carried over from NeurHL-2, is also the size the fit seasons select
for this engine (sigma0 = 0.07). On held-out seasons its 80% intervals cover
0.823. The same engine in the preseason convention beats a preseason
Elo-only season model:

| Season model | CRPS | MAE |
|---|---|---|
| Engine | 6.80 | 9.42 |
| Elo only | 6.87 | 9.47 |

This replaces the earlier known limitation ("the season shock is carried
over, not recalibrated for this engine") with evidence.

### C2c (A4): not adopted

The selected variant was a = 0.5, (0.07, 0.005, 0.99). Its judged CRPS,
6.807, is above the frozen layer's 6.799.

### C2b (A2): not adopted

- **The update is better.** Updating team strength from games played beats
  leaving the prior, on CRPS: difference -0.39, season-block 95% CI -0.66 to
  -0.12. It is better at every checkpoint (4 to 20 weeks), with MAE 5.74
  against 6.24.
- **The coverage rule fails.** On the judge seasons, the selected variant's
  (0.07, 0.02) 10th-90th percentile coverage is 0.876, outside 0.80 ± 0.05.
  (It was 0.819 on the fit seasons.)

### A separate, prospective exploratory test

The judge seasons are now spent for C2b, and 2026-27 is unseen. So the same
variant, unchanged, is published nightly from the first night with completed
games, as an exploratory projection.
- **Code:** `neurhl/sim/live_standings_2027.py`, run by the nightly job,
  non-fatal.
- **Outputs:** `neurhl/output/live/standings_1_1_2027.csv`, plus one dated
  copy per night in `neurhl/output/live/standings_1_1/`.
- **Labels:** each row says "exploratory", and the config states the backtest
  status.
- **Scoring, at season end:** for the copies nearest to 4, 8, 12, 16 and 20
  weeks after opening night, final-points CRPS, MAE and coverage against the
  final table. The comparator is the frozen NeurHL 1.0 preseason simulation
  plus actual points.
- **Status:** nothing is adopted from this test; it replaces nothing.

| File | SHA-256 |
|---|---|
| `neurhl/configs/live_standings_1_1.json` | `6611ca2f0bfb105e74be652a8b53a8ccc35794f039fa1e4d4a59e0eb7c2a2497` |

## A10 (2026-09-29, before the first 2026-27 game): C1d, team-xG mean slope; and further exploratory nulls

### C1d

**The fit.** Team xG ~ gamma(k_x) with mean M x (mu / M) ^ b_x, where:
- mu is each pregame forecast's frozen `xgf_home` / `xgf_away`;
- M is the mean of those frozen forecasts over the games scored. M depends
  only on published forecasts, never on outcomes.

`neurhl/eval/fit_calibration_1_1d.py` fits both parameters jointly on
G_GATE:
- b_x = 0.7849 and k_x = 11.98;
- 80% coverage 0.797;
- +0.0233 nats per team-game against the frozen gamma(9).

Exploration (disclosed): walk-forward within G_GATE, it beat C1b's shape-only
layer by +0.0066, +0.0033 and +0.0045 in 2022-2024.

**The test.** Against the frozen gamma(9), with the same games, bootstrap and
single season-end inference as C1, in a separate Holm family of one.

| File | SHA-256 |
|---|---|
| `neurhl/configs/calibration_1_1d.json` | `c54bf85707ee0b1aab823740bab88790526084f97e4f849aa1507540b6f51683` |

### Exploratory nulls (none adopted)

- **Venue-specific home advantage.** A shrunk per-venue residual from the
  previous three seasons, added to the live stack's inputs, walk-forward on
  2016-2024 (10,020 games): -0.00002 (SE 0.00009). Its coefficient is
  negative: venue residuals tend to reverse.
- **Skater point probabilities.** On G_GATE they are already calibrated:
  every decile is within 0.014, and the mean is 0.340 predicted against
  0.348 realised. No layer.
- **Shot-feature drift (population stability index, 2021-22 against
  2022-23 and 2023-24).** Every feature stays below 0.05: angle 0.047,
  distance 0.033, time since the last event 0.026. Conversion inside 10 ft
  fell from 16.6% to 14.5%. The xG model's close-shot-conversion era
  feature already targets this. Diagnostic only.

## A11 (2026-09-29, before the first 2026-27 game): corrections from a code review

These corrections change no declared test. All were made before any game.

- **The live standings (A9) could not have run.** The nightly environment has
  no torch, and the runner imported `sim/unified_2027.py`, which imports it.
  It now carries its own logit and division helpers. Checked in the nightly
  environment.
- **The season-end comparator for the live standings is now the frozen 1.0
  layer (0.07, 0, 1),** as A9 declares. The code had used the update's own
  drift (0.02), which widens the comparator's intervals by 16-20%.
- **The team-clustered standard error in the C1 scorer is now centred.** It
  had omitted subtracting the mean, which inflated it about threefold. It is
  reported only, never decisive.
- **C1d's centre M is the mean frozen xG forecast over every scored game.**
  It no longer depends on which games have ingested xG actuals: forecasts
  only, as A10 says.
- **C1c skips skater rows with missing values.** Before, one NaN turned the
  whole comparison into NaN.
- **Guards.** `live/freeze_opening.sh` now stops before any step once the
  FREEZE section exists. `goal_calibration.py --freeze` refuses too, because
  its M is a C1 input.
- **Unchanged, noted.** Skater goals MAE for the calibrated set is reported
  only if the season-end skater file carries goals. The final branches need
  all 1,344 results; a game postponed beyond 2027-04-10 would delay them.

## A12 (2026-09-29, before the first 2026-27 game): C1e, power-play opportunities

**New forecast columns.** From this commit, every game-day forecast file adds
two last columns: `pp_opps_home` and `pp_opps_away`, the engine's
power-play opportunity means. They already enter the stat-sheet simulation.
No existing column, number or selection rule changes, and the PLAN_NeurHL4
LIVE scorer reads columns by name. This was checked with a preview run into
a temporary folder.

**The fit.** `neurhl/eval/fit_calibration_1_1e.py` fits the calibrated
distribution, binomial(n, mu / n), by maximum likelihood over integers 6-40
on G_GATE (12,578 team-games):
- n_hat = 10;
- 80% coverage 0.816, against 0.883 for the frozen Poisson;
- +0.0335 nats per team-game.

This was seen in exploration first (variance/mean 0.665).

**The test.**
- Games: every pregame forecast that carries the new columns. Games before
  the change have none and are skipped.
- Target: the ingest's official power-play opportunities (`pp_opps` in the
  team-game table). Degraded games are left out.
- Comparison: the paired log-score difference, binomial minus Poisson, with
  the week-block bootstrap.
- One inference after the regular season, in a separate Holm family of one.

| File | SHA-256 |
|---|---|
| `neurhl/configs/calibration_1_1e.json` | `5335aefc02f879cfbfe0f1ca30b9e26ebe6ff37f62a4266b169a1542fea5f4f1` |

## A13 (2026-09-29, before the first 2026-27 game): C1p, prequential dispersion; goals need no dispersion layer

### C1p

**What it tests.** Whether the fitted dispersions should keep learning in
season (Dawid 1984).

**The update.** For team shots (1/r) and team xG (1/k), each team-game uses a
moment estimate from the 2026-27 team-games on strictly earlier dates, shrunk
to the G_GATE value (1/98.68 and 1/11.86) with a prior weight of 2,000
team-games (`eval/score_calibration_1_1.py`, `prequential_phi`):
- shots: ((y - mu)^2 - mu) / mu^2;
- xG: (y / mu - 1)^2.

**The test.** The paired log score against the fixed C1 / C1b parameters, on
the same games, with the same bootstrap and one season-end inference, in a
separate Holm family of two. The fixed layers stay the declared primaries.

### Exploratory: goals are close to Poisson

With C1's calibrated means, regulation goals on G_GATE have variance/mean
0.943. Poisson 80% coverage is 0.808, and a fitted negative binomial gains
nothing. No layer.

## A14 (2026-09-29, before the first 2026-27 game): the engine replacement (PLAN_NeurHL4 A5)

From the switch to `g2027_v2`, every scored game-day forecast comes from the
refit engine (configuration g1, seasons <= 2026).

- **C1's goal form.** It uses M from `configs/live_goal_calibration.json` as
  re-derived for `g2027_v2` by A1's procedure (the value is recorded when
  it is computed), and each forecast's own `goal_mult`. The dispersion and
  slope parameters (r_hat, b_hat, k_hat, b_x, k_x, n_hat, r_s) are unchanged.
  They were fitted on the out-of-sample predictions of configuration g1,
  which is the refit engine's configuration.
- **The C1 dated goal set and the skater blend** were built from the
  `g2027_v1` season files and stay as declared. Equivalents built from the
  `g2027_v2` season set are issued beside it if time allows before the first
  game, as their own dated sets.

## A15 (2026-09-29): rookies and injuries

**Pre-NHL records.** Rookies' records in other leagues now come from the
NHL's player landing records: AHL, CHL, NCAA, USHL, KHL, SHL, Liiga, NL, DEL,
Czech, the European second tiers and junior leagues
(`neurhl/data/build_prenhl.py`). From them, `neurhl/eval/rookie_priors.py`
learns league translation factors and shrinkage, walk-forward.

Results on 938 rookies from 2014-2024 (goals and assists per game against
each player's position average; season points MAE):

| Measure | Translated | Position average |
|---|---|---|
| Goals per game, error | | -12% |
| Assists per game, error | | -14% |
| Season points MAE | 5.28 | 6.29 |

**Inside the engine at inference.** Priors injected as engine baselines
without retraining did not help (`neurhl/eval/rookie_engine_test.py`,
three variants). They are not used live; the hook `sim/rookie_hook.py`
stays disabled.

**Season totals: adopted for the refit engine's season set.**
`neurhl/eval/rookie_season_test.py` compares preseason engine projections for
opening-lineup rookies with the translated ones. The weight was chosen on
2012 and 2014-2018 and judged on 2019-2024 (161 rookies):

| Measure | Engine | Translated |
|---|---|---|
| Goals per game, error | 0.063 | 0.049 |
| Assists per game, error | 0.100 | 0.089 |
| Season points, error | 8.58 | 6.83 (6.75 at a 0.75 blend) |

The selected weight was 1. `sim/unified_2027.py --rookie-priors` sets each
rookie's goals and assists per game from `configs/rookie_priors_2027.csv`,
scaled by his chance of dressing. Teammates are rescaled within each
team-game, so the team totals stay the engine's.

**Injuries.**
- MoneyPuck's injury list (status, return date, games still to miss) is now
  snapshotted daily and every half hour on game days. It enters the
  availability model and non-confirmed game-day lineups (injured reserve or
  out, with a dated return).
- MoneyPuck's placeholder dates (a return a year away) were researched one
  by one from current reporting. The results, with their sources, are in:
  - `data/manual/player_status_2027.csv`: Pietrangelo and Ellis, LTIR for
    the season;
  - `neurhl/configs/injury_overrides_2027.csv`: 13 games-out values.
- The refit engine's season set uses the 2026-09-29 rosters. The 2026-09-28
  snapshot predates the Marchenko-Knies trade.

## A16 (2026-09-29, before the candidate is run): candidate g1rk, rookie inputs in the engine

**The candidate.** `configs/neurhl_g/g1rk.json` is g1 with three extra
skater inputs:
- the rookie's translated goals per game;
- his translated assists per game;
- a rookie flag.

The values are walk-forward: each season's come from moves and rookies of
earlier seasons (`data/build_rookie_features.py`, `data/build_g_tensors_rk.py`;
tensors `g_master_rk`, loaded with NEURHL_MASTER_TAG=rk).

**Iteration run.** `eval/run_g.py --config g1rk --protocol full --window iter`
(five seeds), compared on the same games with g1's cached iteration
predictions.

**Adoption rule, fixed now.** Both must hold on the iteration window:
- the rookie skater-games' Poisson loss for goals and for assists is lower
  than g1's;
- the final-probability log loss is no worse than g1's + 0.0003.

**If adopted.**
- One G_GATE run (the second of the two the gate budget allows) is reported
  with the same rule.
- If that also holds, a live bundle `g2027_v3` (configuration g1rk, seasons
  <= 2026, 2026-27 rookies' inputs from `configs/rookie_priors_2027.csv`)
  replaces `g2027_v2` from a declared date.

The sealed seasons are spent, so there is no confirmatory test. The live
2026-27 season scores it.

## A17 (2026-09-29): g1rk on the iteration window meets the A16 rule

Iteration window 2012 and 2014-2018, five seeds each, the same games for g1
and g1rk:

| Measure | g1 | g1rk | Difference (SE) | Seasons better |
|---|---|---|---|---|
| Rookie goals, Poisson loss (32,328 rookie skater-games) | 0.34274 | 0.34234 | -0.00040 (0.00017) | 4 of 6 |
| Rookie assists, Poisson loss | 0.47973 | 0.47944 | -0.00029 (0.00017) | 5 of 6 |
| Final-probability log loss | 0.67260 | 0.67272 | +0.00011 | |

The last row is within the +0.0003 allowance. So, as A16 declares, one
G_GATE run follows under the same rule.

## A18 (2026-09-29): g1rk on G_GATE meets the A16 rule; live bundle g2027_v3

G_GATE 2019, 2020 and 2022-2024, five seeds, the same games as g1:

| Measure | g1 | g1rk | Difference (SE) | Seasons better |
|---|---|---|---|---|
| Rookie goals (22,539 rookie skater-games) | 0.34761 | 0.34743 | -0.00017 (0.00018) | 3 of 5 |
| Rookie assists | 0.47827 | 0.47778 | -0.00049 (0.00022) | 5 of 5 |
| Final-probability log loss | 0.66006 | 0.65994 | -0.00012 | |

For g1rk, the final probability against Elo is -0.00572 (5 of 5 seasons) and
against NeurHL-H -0.00061 (4 of 5).

**The live bundle.** `g2027_v3` (configuration g1rk, seasons <= 2026, five
seeds; the stack refit on out-of-sample predictions of 2011-2026) is trained
with `train/train_live_g.build` on the loader's all-seasons path. The sealed
seasons are spent (PLAN_NeurHL4 A6).

**Live rookie inputs.** 2026-27 rookies' inputs come from
`configs/rookie_priors_2027.csv` (`sim/g_live.py` fills them when the bundle
lists them).

**The switch.** It happens only after a preview forecast with `g2027_v3`
runs cleanly in a temporary folder, and from the first forecast after it is
declared in an appendix to this amendment. The season set is rerun with
`g2027_v3` as `output/neurhl_1_0/v3_20260929/`, which becomes the primary
once it passes the consistency check.
