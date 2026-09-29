# PLAN NeurHL-4 — NeurHL-G, the single-game simulation engine (prereg)

STATUS: COMMITTED AND PUSHED 2026-09-27, BEFORE ANY NeurHL-G NUMBER EXISTS on
any window. GitHub's push record dates this file; the house rule is that the
numbers that test a plan come after the plan.

NeurHL-G predicts one game at a time from the player-games of both dressed
rosters. It simulates a full stat sheet (ice time by strength, shots on goal,
attempts, individual xG, goals, assists, on-ice and team xGF/xGA, goalie
results, power-play opportunities, score distribution) and the outcome,
including overtime and shootout. Every prediction uses every game played before
it: game 50 of a season sees games 1-49 plus all earlier seasons.

## S. Standing constraints (carried)

- Market firewall (P10): no odds in any feature, target, calibration or
  selection step. No historical game odds exist in the repository; the market
  can be compared live only, as evaluation.
- Vantage rule (P1): anything used to predict a game comes from earlier games.
  The dressed lineup and the announced starting goalie are legitimate inputs,
  as in NeurHL-H.
- Reported numbers come from CPU inference over saved, hashed checkpoints.
- 2013 and 2021 train but never score (A4/NO_SCORE).
- The 1.0 frozen predictions (PLAN_NeurHL_LIVE.md) are never edited.

## W. Windows (neurhl/windows.py)

| Window | Seasons | Games | Use |
|---|---|---|---|
| Burn-in | 2008-2011 | — | player state and training only |
| G_ITER | 2012, 2014-2018 | 7,421 | iteration and ablations |
| G_GATE | 2019, 2020, 2022-2024 | 6,289 | gates and the pre-gate |
| SEALED | 2025, 2026 | 2,624 | scored once, by eval/seal_g.py |
| LIVE | 2027 | 1,344 | morning and pregame forecasts |

Disclosure: every season through 2026 has been used at game level by NeurHL
1.0 (NeurHL-H's confirmation scored 2018-2026). The seal therefore protects
against adaptive NeurHL-G choices; it is procedurally sealed, not unseen. The
2026-27 live season is the only clean test.

**Seal mechanics.** The plan approved physical quarantine of 2025-2026
tensors. That would break the 1.0 batteries, the 1.0 live scorer and the live
pipeline, which all read those files, so the seal is enforced logically
instead, and this deviation is declared here:
- `neurhl/data/g_loader.py` is the only reader NeurHL-G model, training and
  evaluation code may use. It refuses SEALED seasons unless
  `windows.unseal()` was called by `eval/seal_g.py`, or the caller declares
  `purpose="live_inputs"` (feature histories for live inference; every such
  access is logged).
- `tests/review_tests_neurhl4.py` fails any NeurHL-G module that reads the
  tensor directory directly.
- `neurhl/configs/sealed_inputs.sha256` fixes the 37 sealed-season input files
  as of this commit; the seal script refuses if any hash has changed.

## H. Hypotheses

- H1 (primary): NeurHL-G has lower per-game log loss than v1 Elo.
- H2: NeurHL-G has lower per-game log loss than NeurHL-H.
- H3: NeurHL-G's player heads are no worse than the confirmed player-game layer.
- H4: team SOG, xGF, xGA, goals and PP opportunities beat team baselines.
- H5: the predicted distributions are calibrated.

## M. Model (summary; configurations logged in configs/neurhl_g/)

- **Player state**, updated after every game: shifted ratio-of-sums EWMA bank
  (alphas 0.02, 0.05, 0.15, 0.4, with effective sample sizes), season- and
  career-to-date totals, a Bayesian cold-start prior, and as-of priors (a row
  in season s uses `embeddings_v{s}` and `rapm_prior_{s}`). Goalie state:
  shrunk save percentage, shrunk GSAx per xG, workload, replacement prior.
- **Game network**: shared player encoder; heads residual on the confirmed
  player-game layer (zero-initialised, so the untrained network reproduces
  it); a team layer that conserves ice time and makes player outputs sum to
  team totals by construction; team scoring rates log-linear in a log5 team
  baseline plus the dressed players' on-ice effects.
- **Outcome**: differentiable integration over a joint score grid with a
  score-by-time hazard table fitted on seasons before T, a shared pace shock
  for tie mass (no ad-hoc tie scalar), and an overtime model.
- **Final probability**: per season T, a logistic stack over [Elo logit,
  NeurHL-G logit, NeurHL-H logit, rest difference], fitted on out-of-sample
  predictions from seasons before T, falling back to Elo alone if it does not
  beat Elo in blocked cross-validation on the training seasons.
- **Stat sheet**: Monte Carlo box-score simulation, 10,000 draws per game.
- **Training**: snapshot T is trained on seasons before T; 5-seed ensemble.
  In-season weight fine-tuning is an ablation, not the default.

Known issue carried from 1.0: NeurHL-H's Layer 1 gives training rows
embeddings fitted on their own season, while test rows see only earlier
seasons. Its confirmed result stands (test inputs are clean), but NeurHL-G
uses as-of priors throughout.

## L. Ladder and budget

Rungs, each against its parent on the same games: R0 constant, R1 Elo, R2
NeurHL-H, R3 gradient-boosted and logistic models on game-level features,
R4 non-neural engine (player-game-layer sums → integration → stack),
R5 network box-score pretraining, R6 plus outcome fine-tuning, R7 without xG,
R8 league-average goalie, R9 attention, R10 sequence encoder, R11 stacked with
NeurHL-H, R12 in-season fine-tuning.

- G_ITER: at most 14 full runs (quick runs logged too).
- G_GATE: at most 2 runs; the second only as one pre-declared change after a
  gate failure.
- Every run gets a row in `neurhl/configs/search_ledger_g.csv`.
- Adoption: a component is kept if it cuts G_ITER log loss by at least 0.0003
  without losing in 4 or more of the 6 seasons, or else meets the declared
  box-score margins (team xG deviance -1%, player SOG deviance -0.5%);
  otherwise the simpler model wins.
- Stop rule: if no variant beats NeurHL-H on G_ITER by run 8, the stack of
  NeurHL-H with the best NeurHL-G ships if it beats NeurHL-H; otherwise
  NeurHL-H keeps the win probability and NeurHL-G ships for the stat sheet.

## G. Gates on G_GATE (baselines on the same games)

- **PG:** TOI share, SOG, P(goal), P(assist), ixG no worse than the confirmed
  player-game layer: upper confidence bound below +1% relative, two-way
  clustered (player and game).
- **T:** team SOG, xGF, xGA, goals and PP opportunities beat (a) a log5 team
  EWMA baseline and (b) the player-game-layer sum over dressed skaters, on
  deviance and CRPS.
- **C:** outcome calibration slope in [0.9, 1.1]; randomised PIT and 80%
  coverage within 3 points of nominal for team goals, team SOG and player SOG;
  predicted overtime share within 1 point of observed.
- **S-STOP (pre-gate):** mean paired NeurHL-G minus Elo <= -0.0045 AND NeurHL-G
  minus NeurHL-H <= -0.0010, direction agreeing in at least 4 of 5 seasons.
  If S-STOP fails, the seal is not spent.

## SEAL. The one-shot test (eval/seal_g.py)

Refuses unless: the FREEZE commit (config hash, checkpoint hashes, G_GATE
standard errors) is on origin/main; S-STOP passed; sealed input hashes match;
and it has never run. It trains snapshot 2025 on seasons <= 2024 and snapshot
2026 on seasons <= 2025 with the frozen configuration, then scores once.

- **S1 (primary):** mean paired per-game log loss, NeurHL-G minus Elo, over
  2,624 games; two-sided p < 0.05 and the same direction in both seasons.
- **S2:** only if S1 passes, NeurHL-G minus NeurHL-H, two-sided p < 0.05
  (fixed-sequence; full alpha).
- Power: the paired SE is about 0.0016, so the minimum detectable effect at
  80% power is about 0.0046 against Elo, which is NeurHL-H's confirmed size.
  S2 is likely underpowered; its SE and minimum detectable effect are
  recorded from G_GATE in the FREEZE commit.
- Secondary (Holm, 9 tests): team SOG, xGF, xGA, goals; player TOI, SOG, ixG,
  goals, assists.
- Sensitivity: moving-block bootstrap (100 games, 9,999 draws, seed 711), sign
  test by month, Murphy decomposition.

## C. Consequences, fixed now

| Outcome | Consequence |
|---|---|
| S-STOP fails | Seal unspent. NeurHL-G v1 (trained <= 2024) goes live as exploratory; NeurHL-H is the live primary. |
| S1 and S2 pass | Retrain on <= 2026 with the frozen configuration; NeurHL-G becomes the live primary. |
| S1 passes, S2 fails | Partial result, reported as partial; both live, NeurHL-H stays the headline. |
| S1 fails | Documented null. NeurHL-G stays live as exploratory. The seal is never re-spent. |

## D. Data refresh and inputs

- Rosters are pulled into dated snapshots (`data/raw/rosters/<date>/`) with a
  moves file against the previous snapshot. The August roster files are frozen
  1.0 inputs and are never overwritten. The final pre-season pull follows the
  NHL roster deadline (2026-09-28, 17:00 ET).
- `data/manual/player_status_2027.csv` records availability that rosters do
  not show. Entries change inputs (who can dress) only, never outputs, and
  each is committed before the first prediction that uses it. First entry:
  Connor Hellebuyck, suspended for not reporting, effective 2026-09-17.
- Refreshed 1.0 preseason files are issued as new dated files with their own
  hashes; the 1.0 model is unchanged (rosters only).

## LIVE. The 2026-27 season

- Each game day, a **morning forecast** (about 11:00 ET, projected lineups)
  and a **pregame forecast** (about 60 minutes before the scheduled start,
  confirmed lines and starting goalies where available). Both are scored; the
  pregame forecast is primary.
- A forecast counts only if its commit is on origin/main before the game's
  scheduled start (startTimeUTC from the NHL API), checked with
  `git ls-remote`, and dated by a GitHub Actions run. A game without a valid
  forecast is MISSED for every model, so missingness cannot favour one model.
  No backfill.
- Each forecast records its lineup source: NHL API, DailyFaceoff confirmed,
  DailyFaceoff projected, or fallback (last dressed roster minus unavailable
  players, starter by share of starts to date).
- Comparators on the same games: in-season house Elo (src/live.py) and
  NeurHL-H with the same lineup inputs; the 1.0 frozen probabilities are
  reported descriptively.
- Inference once, after the last regular-season game (2027-04-10), with a
  week-block bootstrap. Expected SE about 0.0024 at about 1,300 games, so only
  gaps of about 0.007 or more will resolve within one season.

## Q. Acceptance

`neurhl/tests/review_tests_neurhl4.py`: window guards and sealed hashes; the
loader scan; ledger caps; every live forecast commit precedes its game's start;
leakage audits pass for every new builder; reported numbers re-derive on CPU;
the seal ran at most once.

## CANDIDATE (2026-09-27, declared before any G_GATE number)

The one configuration taken to G_GATE is config `g1`
(neurhl/configs/neurhl_g/g1.json, sha256 `587746c6367d7ef78276162fc194d902e09a7c5a617f6beb19cb4239706ea750`), on the master tensor
with sha256 `325390c3977c3be938b24e6a2327657fc0b039cd565c6404961863620821244c`.

How it was chosen, on G_ITER (2012, 2014-2018; 7,421 games; Elo 0.67528,
NeurHL-H 0.67244; every row in configs/search_ledger_g.csv):

| Rung | Stack log loss | Decision |
|---|---|---|
| R4 non-neural engine (quick) | 0.67581 | baseline |
| R3 trees / logistic on game features, Elo-stacked | 0.67611 / 0.69699 | no better than Elo |
| R6 network (official PP target) | 0.67301 | parent |
| R6e Elo anchor on scoring rates | 0.67330 | rejected |
| R6L lineup-vs-usual multipliers | 0.67299 | rejected (< 0.0003) |
| R6c NeurHL-H projections as team inputs | 0.67344 | rejected |
| R13 R6c + direct H terms | 0.67344 | rejected |
| R9 attention across rosters | 0.67366 | rejected |
| R11 R6 with NeurHL-H in the stack (the form section M declares) | 0.67258 | carried forward |

g1 is R11 with the 5-seed ensemble section M declares (the G_ITER runs used 3).
On G_ITER the network carries no win-probability information beyond Elo and
NeurHL-H: R11 trails NeurHL-H alone by 0.00014. The S-STOP threshold is left
exactly as declared; the expectation, written before the gate run, is that
S-STOP fails and the seal stays unspent.

## FREEZE (2026-09-27): G_GATE results and the decision

Frozen candidate: config `g1` (sha256 `587746c6367d7ef78276162fc194d902e09a7c5a617f6beb19cb4239706ea750`), unchanged since the
CANDIDATE section. One G_GATE run (1 of 2 allowed; ledger row
`g1:full:gate`); every number below is in neurhl/output/g_gates.json and
neurhl/output/preds via the runner's saved predictions.

G_GATE: 2019, 2020, 2022-2024, n = 6,289 games.

| Final probability | Log loss | NeurHL-G minus it |
|---|---|---|
| NeurHL-G (stack over Elo, NeurHL-G, NeurHL-H) | 0.66006 | — |
| Elo | 0.66566 | -0.00559 (SE 0.00131), better in 5 of 5 seasons |
| NeurHL-H | 0.66056 | -0.00049 (SE 0.00060), better in 3 of 5 seasons |

| Gate | Result |
|---|---|
| PG, player heads vs the confirmed player-game layer (225,752 skater-games, two-way clustered) | PASS: better on all four heads (TOI share -0.59%, SOG -0.45%, P(goal) -0.45%, P(assist) -0.36%; every upper 95% bound below zero) |
| T, team box score vs log5 team history | PASS: SOG, xGF, goals, PP opportunities all better |
| C, calibration | PASS: slope 0.968; OT share 0.224 predicted vs 0.220 observed |
| S-STOP | **FAIL**: the Elo condition holds (-0.00559 <= -0.0045) but the NeurHL-H condition does not (-0.00049 > -0.0010; 3 of 5 seasons < 4) |

Recorded for the seal, had it been spent: at n = 2,624 the minimum detectable
effect at 80% power would be 0.0057 against Elo and 0.0026 against NeurHL-H.

**Decision, applied as section C declares:** the seal is not spent; 2025 and
2026 remain sealed for a later version. NeurHL-G v1, trained on seasons
<= 2024 with configuration g1, goes live for 2026-27 as an exploratory model:
its stat sheets (ice time, shots, xG, goals, assists, goalie results) are
published for every game, its win probability is reported and scored, and
NeurHL-H is the live primary for win probability. Read plainly: NeurHL-G
beats Elo by a margin comparable to NeurHL-H's and is calibrated, but it has
not been shown to add win-probability information beyond NeurHL-H.

## A1 (2026-09-27, before any 2026-27 game): stat-sheet goal level

**Finding (from inputs, before any live game).** A dry run of NeurHL-G v1 on
opening night projects 3.36 regulation goals per team-game, against a league
level that has held at about 3.0 for years. The cause is measurement drift in
the house xG, visible in the training-side tables: goals per team-game were
flat (3.08, 3.01, 3.08 in 2024-2026) while measured xG rose (2.89, 3.03, 3.32),
so the goals-to-xG ratio fell from 1.067 to 0.927. This is the recording drift
that failed gate X3 in NeurHL 1.0. v1, trained through 2024, learned the older
ratio and reads 2025-26-era xG inputs.

**What it affects.** Only the stat sheet: goals, assists, points and score
distributions. The scored win probability is unaffected: the stack is fitted
on the engine's unscaled outputs, and the Monte Carlo conditions every score
on the stacked outcome category.

**Rule.** Each forecast multiplies the engine's regulation goal means (and so
assists) by m:
- m0 = L / M, fixed at the pre-season freeze: L is last season's league
  regulation goals per team-game computed from the era inputs the model
  already receives (prior_gpg / 2 minus half the prior OT and shootout shares);
  M is v1's own mean projected regulation goals over every game scheduled in
  the first 14 days (fallback lineups, inputs only).
- In season, m = (A + k L) / (P + k M) with k = 300 team-games of prior weight
  (m = m0 before any game), where A and P are the actual and
  unscaled-predicted regulation goals over completed 2026-27 games that have a
  primary forecast. Walk-forward: a forecast uses only games already finished.
NeurHL-G's weights, stack and win probabilities are untouched, and no 2025 or
2026 game is scored or used to fit anything; the seal stays intact.

## A2 (2026-09-28): gate record correction

While the gates were being written up for publication, three declared
components of section G were found missing from `eval/gate_g.py`:

1. **C, coverage.** The rule declares randomised PIT and 80% coverage within
   3 points of nominal. The script computed coverage of discrete 80%
   intervals, which over-cover by construction, reported it, and judged C on
   slope and overtime share alone.
2. **T(b).** The comparison against the player-game-layer sum over dressed
   skaters was not computed.
3. **T, CRPS.** Only deviance was computed.

The PG rule also names ixG, which has no counterpart in the player-game layer
and so has no baseline to beat.

`eval/gate_g_record.py` computes items 1-3 from the same saved G_GATE
predictions, with the predictive distributions the gate script already used
for its intervals (Poisson; negative binomial with r = 40 for team SOG).
Nothing is refitted. Record: `neurhl/output/g_gates_record.json`.

| Component | Result |
|---|---|
| C coverage, team regulation goals (Poisson) | 0.803, within 3 points |
| C coverage, team SOG (negative binomial, r = 40) | **0.860, outside 3 points: intervals too wide** |
| C coverage, skater SOG (Poisson) | 0.783, within 3 points |
| T(b) team SOG vs player-game-layer sum, Poisson deviance | better: -0.0248 (SE 0.0038) |
| T(b) team goals vs player-game-layer sum, Poisson deviance | better: -0.0055 (SE 0.0026) |
| T, CRPS, team SOG, vs history and vs layer sum | better: -0.029 (SE 0.006) and -0.059 (SE 0.010) |
| T, CRPS, team goals, vs history and vs layer sum | better: -0.018 (SE 0.003) and -0.007 (SE 0.003) |

The team goals comparison uses the Poisson mean implied by each skater's
P(goal >= 1).

**Correction.** Gate C, as declared, **fails** on team-SOG coverage, which
supersedes "C: PASS" in the FREEZE table. The slope (0.968) and overtime share
(0.224 predicted vs 0.220 observed) still pass. Gate T passes on every
declared comparison. No decision changes: S-STOP failed on its NeurHL-H
condition, the seal stays unspent, and NeurHL-G v1 stays live as exploratory.

The stat-sheet simulator draws team SOG from the same negative binomial
(r = 40), so its team-SOG spread is too wide. Team means are unaffected. The
dispersion stays frozen for 2026-27 and is recorded as a known limitation.

## A3 (2026-09-28): record notes from preparing the figures

1. **Loss definitions.** Where `eval/gate_g.py` and `eval/gate_g_record.py`
   say "Poisson deviance", they compute mu - y log(mu), the Poisson negative log
   likelihood without the terms that depend only on y. A paired difference in
   this loss is exactly half the paired difference in deviance, so every sign,
   standard-error ratio and p-value in the FREEZE and A2 records stands. The PG
   gate's relative figure for shots on goal (-0.45%) is relative to this loss,
   not to deviance. The 1.0 player-game confirmation used the true deviance.
2. **Ladder provenance.** R6e and R6L ran before the master tensor was rebuilt
   on official power-play opportunities. Their parent on the earlier tensor was
   the R6 run at 0.67299, not the 0.67301 of the rebuilt R6 in the CANDIDATE
   table. Against it R6e is +0.00031 and R6L -0.000004, so both rejections
   stand.
3. **R6e configuration.** The ledger's hash for R6e matches `r6e.json` as
   committed when it ran. At that commit the Elo anchor was always on in the
   model code. The next commit made the anchor optional and added
   `"elo_anchor": true` to `r6e.json` to keep the same behaviour, which is why
   the committed file's hash differs from the ledger.

## A4 (2026-09-28, before any 2026-27 game): in-season shot records when MoneyPuck lags

**Finding (inputs only, before any live game).** The nightly ingest takes
2026-27 shot records from MoneyPuck's season file, which returns 404 until
MoneyPuck publishes it. A game without MoneyPuck coverage was deferred for up
to two days and then ingested degraded. A sandbox replay of the first eight
days of 2025-26 without the file showed that degraded games enter player,
goalie and team state with zero expected goals, not as missing values. After a
week:
- team expected-goal rates (d95) were 13-14% low;
- skater individual expected goals were 12% low;
- NeurHL-G's win probabilities (Elo and NeurHL-G stack) moved 0.9 points on
  average, and up to 4.7.

**Rule.** A game MoneyPuck does not cover takes its shot records from the
league's own play-by-play and shift chart, built by
`neurhl/live/nhl_api_shots.py` with MoneyPuck's definitions for every field the
frozen xG model reads. The game is re-ingested from MoneyPuck once MoneyPuck
covers it, and the source of each game is recorded in the ingest state.
`--no-api-fallback` restores the previous rule. If both sources fail for a
game, it is still ingested degraded, as before, and logged.

**Validation on 2025-26, before the season**
(`neurhl/configs/nhl_api_shots_parity.json`), against an adoption threshold set
before the study ran (team-game correlation at least 0.98, bias within 2%):
- 99.5% of MoneyPuck's regular-season shots match a play-by-play shot.
- Every model input agrees exactly on at least 98.8% of matched rows.
- With the frozen xG model, team-game expected goals correlate 0.989 (bias
  -0.3%) on the same covariates, and 0.985 (bias +0.8%) through the full live
  path.

**Unchanged.** The xG model, NeurHL-G's weights and stack, and the way every
win probability is computed. Only the source of shot records for games
MoneyPuck has not yet covered changes.

## A5 (2026-09-29, before the first 2026-27 game and before the sealed seasons are read): the seal spent descriptively; the engine refit on all seasons becomes the live engine

**Owner directive (2026-09-29).** Spend the sealed seasons, refit the engine
on everything, and run the refit as the single replacement engine.

**1. Descriptive seal.**
- `eval/seal_g.py --descriptive` runs the SEAL procedure exactly as declared
  (section SEAL): configuration g1, snapshots 2025 (seasons <= 2024) and 2026
  (seasons <= 2025), five seeds, the walk-forward stack, S1, S2 and the
  secondary family.
- Only the S-STOP precondition is waived. The FREEZE-on-origin check, the
  config hash and the sealed-input hashes are all still checked. The 37
  sealed inputs were verified unchanged before this amendment.
- Because S-STOP failed, the result is reported as **descriptive, not
  confirmatory**, whatever it shows. It is written to
  `output/g_seal_result.json` with `"mode": "descriptive"`, and it is
  reported in full, including a null.
- The seasons are then spent for good.

**2. Refit.** The same run trains the live bundle `g2027_v2`:
- configuration g1, seasons <= 2026, five seeds;
- the stack refit on out-of-sample predictions of 2011-2026 (fallback Elo +
  G where NeurHL-H is missing), as `train/train_live_g.build` declares.

**3. Replacement.**
- `configs/live_models.json` points to `g2027_v2`. From the first
  forecast after this switch, every game-day forecast (morning and pregame)
  uses `g2027_v2` alone. The files record the bundle name and sha, so every
  scored row states its engine.
- The LIVE scoring rules (section LIVE) are unchanged. In the season-end
  scorecard, the engine's rows are the `g2027_v2` rows.
- `g2027_v1` stays hashed in the repository and no longer forecasts.

**4. Goal level (A1).** A1's procedure is run once with `g2027_v2`, as the
switch happens, before the first game:
- m0 = L/M on fallback lineups for the first 14 days of the schedule;
- `configs/live_goal_calibration.json` is the new state;
- the `g2027_v1` state is kept as `configs/live_goal_calibration_g2027_v1.json`.

**5. Season forecast.** The NeurHL 1.0 season model (`sim/unified_2027.py`) is
rerun with `g2027_v2` on the same post-deadline rosters, draws, seeds and
settings.
- It is issued as the dated file set `output/neurhl_1_0/v2_20260929/`, with
  the same consistency check.
- That set becomes the primary 2026-27 season forecast.
- The frozen `g2027_v1` files (PLAN_NeurHL_1_0 FREEZE) stay unchanged and are
  scored beside it as the earlier freeze.

## A6 (2026-09-29, before the first 2026-27 game): A5 carried out

**1. Descriptive seal.** `output/g_seal_result.json`, 2,624 games of 2025 and
2026, mode descriptive:

| Comparison | Log-loss difference (SE) | Other |
|---|---|---|
| S1: NeurHL-G - Elo | -0.00691 (0.00189) | p = 0.0003; 2025 -0.00778, 2026 -0.00603 |
| S2: NeurHL-G - NeurHL-H | -0.00214 (0.00088) | p = 0.015 |

- All nine secondary tests favour NeurHL-G after Holm correction: team shots,
  xG and goals; skater TOI, shots, ixG, goals and assists.
- Under the SEAL rules this would read PASS. It stays descriptive, because
  S-STOP failed on the gate window and was waived.

**2. Refit.** `g2027_v2`: configuration g1, seasons <= 2026, five seeds.
- bundle.json sha256 `cdac2d4e81275e2a553a785cd31a9186ce030e3edab3fdef365650a6af25d7c7`.
- Stack: Elo 0.128, NeurHL-G 0.349, NeurHL-H 0.544 (fallback Elo 0.531,
  G 0.512).

**3. Replacement.** `configs/live_models.json` points to `g2027_v2` from
2026-09-29 18:50 UTC. Every forecast after that uses it.

**4. Goal level.** A1's procedure run with `g2027_v2`:

| Engine | m0 | Mean raw projection M |
|---|---|---|
| `g2027_v2` | 0.9722 | 3.0884 |
| `g2027_v1` | 0.9229 | 3.2534 |

The refit on the newer seasons removes most of the drift A1 corrected. The
state for `g2027_v1` is kept as `configs/live_goal_calibration_g2027_v1.json`.

**5. Season forecast.** `output/neurhl_1_0/v2_20260929/` is the primary 2026-27
season forecast. It passes all 41 consistency checks; the cross-path check
is skipped because the opening preview came from `g2027_v1`.

It differs from A5 in three declared ways (PLAN_NeurHL_1_1 A15):
- the 2026-09-29 roster snapshot, which includes the Marchenko-Knies trade
  that the 2026-09-28 snapshot missed;
- MoneyPuck's injury list, with researched statuses for undated entries;
- rookies' goals and assists from their translated pre-NHL records.

Its team points correlate 0.988 with the frozen `g2027_v1` files. Those
files stay unchanged and are scored beside it.

| File | SHA-256 |
|---|---|
| `neurhl/output/neurhl_1_0/v2_20260929/games_2027.csv` | `953203ba129ddac89cbd5d53d8dcc0df86c84fdf0b9a06191fc28f268fc3a1f6` |
| `neurhl/output/neurhl_1_0/v2_20260929/teams_2027.csv` | `76318fc6afa61240aedcfdcc32615c95d3872bd63175f13f8c7a7b1c7dafa613` |
| `neurhl/output/neurhl_1_0/v2_20260929/team_points_quantiles_2027.csv` | `532c38d9941d145eef2e522978b46c84dc2c6ebbfb089ee916554040eaa95fc4` |
| `neurhl/output/neurhl_1_0/v2_20260929/skaters_2027.csv` | `d5ece29ac9da07151a5752806b6fd842369b81674c0a7f78e07ec8e05ff671f2` |
| `neurhl/output/neurhl_1_0/v2_20260929/goalies_2027.csv` | `552022c16ba5a2aec383b571c54a997b15073ce9d916236a44486c8743f0dc0c` |
| `neurhl/output/neurhl_1_0/v2_20260929/player_games_2027.csv.gz` | `50b570e355e37c4fa8ae2fd8606c24b4da5e0595123a2fa30535fb1a609c627f` |
| `neurhl/output/neurhl_1_0/v2_20260929/consistency_2027.json` | `c57172be9209cfce9f77ccf579be71f3e0cf631346b1099b3e953f4483b8d7ce` |
| `neurhl/output/neurhl_1_0/v2_20260929/checks_2027.json` | `15fe41a29dc0a59e5b16a97a4668a1b67ba97a766f8d6441b15c9ba6beb66406` |
| `neurhl/output/neurhl_1_0/v2_20260929/run_2027.json` | `1f2dfbe891ae182a936a061355c8ab1c2ebfcfcee856fe5e4f5036c512de969b` |
