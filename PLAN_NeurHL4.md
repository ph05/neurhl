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
- In season, m = (A + k m0) / (P + k) with k = 300 team-games, where A and P are
  the actual and unscaled-predicted regulation goals over completed 2026-27
  games that have a primary forecast. Walk-forward: a forecast uses only games
  already finished.
NeurHL-G's weights, stack and win probabilities are untouched, and no 2025 or
2026 game is scored or used to fit anything; the seal stays intact.
