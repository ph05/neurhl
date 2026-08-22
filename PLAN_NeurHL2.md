# PLAN NeurHL-2 — generative hockey simulation engine, prereg

STATUS: COMMITTED BEFORE ANY NeurHL-2 NUMBER EXISTS (house commit-before-results
convention, as PLAN_V3→V6 and PLAN_NeurHL honoured).

NeurHL-2 replaces the v1 discriminative game model with a **simulation engine
running from the puck up**: events → players → games → seasons. One calibrated
generative process yields game outcomes, player props, standings and playoff odds
coherently, instead of a separate model per question.

## S. Scope and standing constraints (carried verbatim from PLAN_NeurHL)

- Deliverable: the **upcoming season only** (2026-27). No 2027-28 outputs.
- **2026-27 production scoring remains v1/v4/HOWE in every branch.** NeurHL-2 is a
  parallel report-only track and never replaces the live holdout.
- P10 market firewall unchanged: **no odds data in any feature, target or
  selection step.** Market appears only as a validation ceiling and in forward CLV
  logging.
- P3 unchanged: externally model-derived quantities fit on full-sample data
  (MoneyPuck xG, NST derived rates) are excluded as inputs. Raw recorded fields
  from those sources are admissible.
- P9 determinism unchanged: fixed recorded seeds; all gated numbers from **CPU
  inference over saved checkpoints**; committed prediction artifacts re-derivable
  to `max|Δp| ≤ 1e-6`.

## W. Windows (three, not two) — `neurhl/windows.py`

| Window | Seasons | Use |
|---|---|---|
| **DEV** | 2009–2011 | architecture, model size, calibration choices |
| **TUNE** | 2012–2017 | **SPENT** by v1's ~10 configs — development signal only |
| **CONFIRM** | 2018–2026 | **untouched**; scored ONCE |

The v1 build had only TUNE and CONFIRM, so architecture decisions had nowhere to
live and leaked into the spent window. DEV costs nothing (those seasons were
training-only) and protects both.

**A4 carried verbatim — structurally broken seasons TRAIN but never SCORE.**
`NO_SCORE = {2013, 2021}`: 2012-13 lockout (48 games, conference-only, no
preseason) and 2020-21 COVID (56 games, division-only, no crowds). 2020 IS scored
— it stopped early but was structurally normal. Criterion is schedule structure,
fixed by history, never by an observed result.

## C1 — the ONE confirmatory test

> On regular-season games of {2018, 2019, 2020, 2022, 2023, 2024, 2025, 2026}
> (**n ≈ 10,184**), the mean paired per-game log-loss difference NeurHL-2 − v1 is
> negative with **two-sided p < 0.05**, and the season-clustered test agrees in
> direction. Reported with 95% CIs, overall and per season.

Run **once**, guarded in code (`neurhl/eval/confirm_2018_2026.py` refuses to run
twice, and refuses to run before the DEV/TUNE gates are recorded).

Sensitivity analyses reported **alongside, never as substitutes**: wild cluster
bootstrap on season, block bootstrap at game level, sign test across 8 seasons.

**Multiplicity:** all ~10 v1 configs were evaluated on 2012–2017; CONFIRM has
never been touched, which is the cleanest available control. Within DEV/TUNE, the
NeurHL-2 family is capped at **12 configurations**, all logged in
`neurhl/configs/search_ledger_v2.csv`, with Holm-adjusted p reported beside raw.

## O. Objective — game-by-game predictive accuracy is PRIMARY

The individual game is the atomic prediction: a model that gets single games right
yields correct season aggregates as a consequence, while the converse fails
(season totals can look right through regression to the mean).

- **Primary:** per-game log loss (headline), plus Brier, reliability, ECE, and the **Murphy decomposition** — a log-loss win with no *resolution* gain is reported as recalibration, not information.
- **Reported per segment:** favourite/underdog, rest advantage, back-to-backs, divisional, early/late season, home/away.
- **Secondary (diagnostic):** season standings MAE, CRPS, playoff Brier, points-SD realism. Correct per-game hazards that aggregate to wrong season spread indicate a missing-uncertainty bug.
- **Baseline ladder, always reported in full:** constant 0.6898 → v1 Elo 0.67585 → Tier-0 0.67692 → v1 hierarchical 0.67314 → HOWE → market closing line (~0.66) as practical ceiling. Report the fraction of the achievable gap closed.

Acknowledged in advance: the game-level skill window is ~0.03 nats, so genuine
gains are small in absolute terms. That is a power problem, addressed by the
10,184-game confirmatory window, not a reason to soften the target.

## A. Architecture

- **S0** unified event + line-change stream (~9M tokens, 2008–2026).
- **S1** neural marked temporal point process (Transformer-Hawkes family; Zuo et al. 2020) emitting `p(Δt)·p(type)·p(actor | on-ice)·p(location)·p(outcome)`. **Starts at ~5M parameters** — events within a game are heavily correlated so effective n is far below 9M — scaled only where held-out likelihood earns it. 5-seed deep ensemble.
- **S2** player effects as **hazard modulation** (Thomas et al. 2013), anchored on stint-level ridge **RAPM** (Macdonald 2011): `effect = rapm + s·tanh(δ)` with the output layer **zero-initialised**, so at step 0 the engine reproduces the validated RAPM solution exactly. Hierarchical shrinkage to position × role × TOI-tier priors, **posterior variance retained**.
- **S2b** GNN over teammate/opponent/goalie edges — **deferred behind a pair screen**; cut without hesitation if it does not improve out-of-season stint MSE.
- **S3** deployment (line-change) process.
- **S4** game outcomes by **semi-analytic integration of competing hazards**, NOT Monte Carlo rollout (rollouts compound error; the integral is also differentiable, so hazards can be fine-tuned directly against per-game log loss). Monte Carlo reserved for joint quantities.
- **S4b** goalies as first-class: EB-shrunk GSAx, plus a **starter-prediction model**. **S4c** explicit score-effect conditioning.
- **S5** season simulation with **posterior sampling of player effects**, emitting the `howe.rebuild_sim` dict contract. **S6** 2026-27 roster construction with wide-variance cold-start priors.

## D. Coverage groups and sparse data — `neurhl/registry.py`

G0 core events (2008+) · G1 coordinates (2012+) · G2 shot distance (2008+) ·
G3 shifts/on-ice (2008+) · G4 scratches/referees/coaches (2012+) ·
**G5 NHL EDGE (2022+)** · G6 declared lineups (2027) · G7 careers/bios (static).

Sparse groups are **used, not discarded** — the 2026-27 season being projected has
all of them. Four mechanisms:
1. **Pattern-aware missingness**: explicit availability mask + learned "absent" embedding, never zero-fill (zero conflates unobserved with observed-as-zero).
2. **Group dropout** during training (DropIn / modality dropout) so the model never becomes *dependent* on a group absent for most of history.
3. **Zero-initialised residual paths** for G5/G6: with the group masked, output is bit-identical to the base model — a sparse source can only ever add.
4. **Auxiliary supervision**: an EDGE-era head predicts tracking quantities from event history, distilling them into the shared representation for eras with no EDGE.

**Headline C1 uses only always-available groups (G0–G4)** so it is comparable
across the whole window. EDGE's contribution is a **separate pre-specified
secondary analysis** on the EDGE era with its own stated n. All results reported
stratified by availability regime.

## G. Gates (pre-specified; recorded with `rule_eval` in `params_neurhl2.json`)

| Gate | n | Rule |
|---|---|---|
| **X1** xG skill | 1.65M shots | beat distance+angle logistic by ≥0.002 nats in ≥15/19 vantages |
| **X2** xG earns its place | team-games | team-aggregated xG differential beats Corsi differential out of sample — **if X2 fails, delete L0 and use Corsi** |
| **X3** xG calibration | 20 bins | \|observed − predicted\| ≤ 0.005 per decile |
| ~~**R1** RAPM repeatability~~ | ~~900k player-games~~ | ~~split-half correlation exceeds that of raw on-ice CF%~~ **RETIRED — see A5** |
| **R1′** RAPM transfer | players changing team | prior rating predicts realised on-ice results in a NEW team/linemate context better than raw on-ice rate does |
| **R2** RAPM value | 600k held-out stints | beats raw rate and team-fixed-effects |
| **P-screen** | 600k stints | pair-interaction terms improve out-of-season MSE → GNN go/no-go |
| **E1** event realism | held-out games | simulated goals/shots/PP%/penalties per game inside the observed historical band, **checked per period** |
| **E2** score effects | held-out | simulated score-state-conditional shot rates match observed |
| **G-STOP** | DEV+TUNE | if the per-game effect vs v1 is weaker than −0.0015, **do not spend CONFIRM** |
| **C1** | 10,184 games | the confirmatory test above |

**Ablations required before any causal claim:** remove EDGE (G5), graph edges,
goalie module, score effects, scratches.

### A8 — TWO leaks found in S1; all P5 results before this point are VOID

*Recorded because the failure matters more than the fix: the preregistered gates
passed with both leaks present, and would have passed forever.*

**Leak 1 — `home_next`.** The on-ice encoder chose its "for"/"against" pooling
using a flag equal to the owning team of the **next** event — which is the team
component of the `(type, team)` target. Agreement with the target's team:
**1.0000**, zero off-diagonal.

**Leak 2 — `on_next`, and worse.** The trunk conditioned on the on-ice set at the
**next** event. Line changes happen *at stoppages*, so a change in on-ice
composition **is** the announcement of the faceoff that caused it:

| Δ on-ice → next event | P(faceoff) | lift |
|---|---|---|
| −1 skater | 0.986 | 4.96 |
| −2 skaters | 0.992 | 4.99 |
| next event has no on-ice data | 0.995 | 5.00 |
| **baseline** | **0.199** | — |

and Δ+1 skater lifted P(penalty) 2.76×. About 3,000 of 62,094 held-out steps had
a near-deterministic answer supplied to them.

**Why the gates did not catch either.** E1 and E2 score the model using the same
inputs the model was trained on, so a leaked input is *structurally invisible* to
them. Realism gates test whether the model reproduces reality given its inputs;
they cannot test whether its inputs are legitimate. **Leakage requires either a
downstream consumer that must supply every input itself, or a direct causality
test.** Leak 1 surfaced only because S4 had to fabricate the flag and produced
λ_home = 0.39 g/60 against λ_away = 3.30. Leak 2 surfaced only because that
prompted an actual audit.

**Fix.** Orientation is fixed to home slots 0-6 / away slots 7-13, and the trunk
conditions on the **current** on-ice set. This is also the correct generative
order: condition on current personnel → sample the event → let S3 change
personnel at the resulting stoppage. The actor head now chooses from current
personnel, lowering its ceiling honestly — the next actor is already on the ice
for **0.5959** of steps, against the 0.827 the leaked version enjoyed.

**New standing requirement — `neurhl/tests/audit_leakage.py` must pass before any
S1 number is reported.** Three checks:
1. **Causality** (decisive): corrupt every event after a cut point, rebuild the
   batch, and require every model input before the cut to be bit-identical. 22
   inputs × 3 games. Static source analysis is *not* sufficient — it tells you
   which slice a line reads, not whether the value reaches the model.
2. **Target identity**: no input may agree with any target above 98%, nor recover
   the target's team.
3. **Empirical**: the exempted input (personnel, legitimately supplied by S3)
   must carry no lift on the next event's type.

**All P5 gate results reported before this amendment are VOID.** The leaked
checkpoints are retained under `neurhl/checkpoints/leaked/` so the void numbers
stay reproducible and clearly labelled, and every seed has been retrained on
audited inputs.

### A7 — X3 failed; calibration method corrected (chosen on DEV/TUNE, never on CONFIRM)

*Declared before the corrected walk-forward is run. The original X3 failure
stands recorded: 0/18 vantages within tolerance, pooled max decile deviation
0.02224.*

**Two distinct causes, both diagnosed from the training side.**

**(1) The calibrator was correcting the wrong model.** `CalibratedClassifierCV(…,
ensemble=False)` fits the isotonic map on out-of-fold predictions produced by
models trained on 2/3 of the data, then applies it to a final model trained on
all of it. The deployed model is better calibrated than the folds, so the
correction is misapplied. Measured on DEV V=2011, max decile deviation:
cv-isotonic **0.0088**, uncalibrated **0.0061** — calibration was making it
*worse*.

**(2) A recording-regime shift from 2023, not a hockey shift.** Share of shots
recorded within 10 ft was stable at 7.3-8.7% (2010-2021) and then runs 11.9% /
12.8% / 12.5% / **14.5%** (2023-2026), while the goal rate on those same close
shots *falls* from 0.166-0.179 to 0.149 / 0.143 / 0.140 / **0.132**. Rebound
share rises 5.1-5.6% → 8.1%; the rush flag collapses 0.19% → 0.06%. A model
fitted on the old regime reads "8 ft" and predicts the old ~17% conversion, so it
over-predicts exactly where the top decile lives. This is precisely the "meta
shift" hazard the project was told to guard against, in its most insidious form —
the *measurement* changed, not the sport.

**Corrected method.** Fit the GBM on seasons **< V-1**; fit isotonic on that same
model's predictions for season **V-1**; apply to V. The base model is consistent
with the one being corrected, the calibrator sits on the most recent available
season so it tracks drift automatically, and nothing from V is touched. Cost is
one season of training data.

**Window discipline.** The method was selected on DEV V=2011 (0.0088 → **0.0037**,
a pass) and confirmed on TUNE V=2016 as development signal (0.0104 → **0.0060**).
CONFIRM vantages were NOT consulted in choosing it; the drift evidence above is
entirely a property of training-side seasons and needs no test data to see.

### A6 — R1′ INCONCLUSIVE on DEV; team-season fixed effects added to the RAPM design

*Declared after the A5 replacement gate returned a negative result, before the
corrected estimator is run. The negative result stands recorded.*

**Measured on all three DEV folds at fixed λ=51,200** (2008→2009, 2008-09→2010,
2008-10→2011), target = realised on-ice net rate in the test season demeaned by
the player's new team:

| pooled | n | RAPM | raw | raw (team-demeaned) | p vs raw | p vs raw-dev |
|---|---|---|---|---|---|---|
| movers | 485 | 0.3626 | 0.3226 | **0.3709** | 0.151 | 0.793 |
| stayers | 1346 | 0.4586 | 0.3645 | **0.4933** | <0.001 | 0.018 |

RAPM beats *plain* raw decisively, so it is removing real confounding. But a
one-line within-team demeaning matches or beats it, significantly so for stayers
— **in the wrong direction**. R1′ is therefore **INCONCLUSIVE**, not passed. The
single-fold 0.3924 reported earlier was λ selected on that same fold and is
optimistically biased; it is withdrawn.

**Diagnosis — a real identification defect, not a target artefact.** Within a
season a player's design column is nearly collinear with his team's roster,
because he plays essentially every shift for one team. Ridge cannot cleanly
separate "this player is good" from "this player's team is good", so the fitted
coefficients retain team level. The transfer target is a *within-team* deviation,
which is exactly the component the estimator fails to isolate — and which
demeaning gets for free.

**Correction:** add **team-season fixed effects** to the design (unpenalised,
alongside intercept and home). Player coefficients then estimate performance
*relative to their own team*, which is the identified and transferable quantity.
This is a standard RAPM variant, and it is a fix to the estimator, not a change
to the test.

**Anti-forking-path condition, binding:** the corrected estimator is evaluated on
**the same three pre-specified folds**, and reported against **both** targets —
team-demeaned *and* undemeaned realised on-ice rate — because the demeaned target
structurally favours within-team predictors and the undemeaned one favours
team-level predictors. Reporting only the flattering one is the failure mode this
condition exists to prevent. If the corrected estimator still fails to beat
team-demeaned raw, **RAPM is dropped as the anchor and the team-demeaned raw rate
becomes the prior the hazard model is anchored on** — the anchor is chosen on
evidence, and the plan's residual `effect = anchor + s·tanh(δ)` form is unchanged
either way.

### A5 — R1 retired and replaced; λ no longer selected on stint MSE

*Declared after seeing the DEV result, before running the replacement. The
original R1 result stands recorded as a failure of the gate as written.*

**Measured on DEV (train 2008-2010, test 2011, 742,015 stints, 882 players):**

| split-half r | offence | defence |
|---|---|---|
| RAPM | 0.656 | 0.505 |
| raw on-ice CF% | 0.789 | 0.714 |

R1 as written therefore **FAILS**. But the gate was mis-specified, and the result
is what a correct RAPM *should* produce: **raw on-ice rate is more repeatable
precisely because it is contaminated.** A player skates with the same linemates,
takes the same zone starts and draws the same quality of competition in both
random halves, so those confounders repeat and carry the correlation with them.
RAPM removes exactly those terms, so it necessarily repeats less. Reliability is
not validity — a thermometer stuck at 72°F is perfectly repeatable and useless.
Ranking rating systems by split-half correlation actively rewards confounding,
so the original R1 would have selected the *worse* estimator.

**R1′ (replacement): predictive transfer across a change of context.** For
players whose team changes between the fit window and the held-out season,
correlate the prior rating with realised on-ice results in the new team. A rating
that measures the PLAYER transfers to new linemates; a rating that is really
measuring his old linemates does not. PASS requires RAPM to beat raw on-ice rate
on transfer.

**λ selection moves to the same criterion.** On DEV the stint-level wMSE path is
monotone to the grid edge (λ=6400, i.e. shrink players to nearly zero) and the
entire grid spans 0.12%. That is a noise-dominated objective, not a preference
for heavy shrinkage: the median stint is 3-9 seconds carrying 0 or 1 attempts, so
irreducible variance swamps player signal at that resolution. λ is instead chosen
to maximise held-out transfer correlation. R2 is retained unchanged as a
sanity floor, with its margin reported honestly (−0.083%) rather than as support.

## H. Boundaries and engineering guarantees

- Touches only `neurhl/` + this file. Nothing in `src/`, `output/`, or `data/processed/` is modified; `src/` never imports `neurhl/` (battery-enforced).
- **Artifact lineage (`neurhl/manifest.py`)**: every derived file carries source fingerprints, code SHA and config hash; consumers call `require_fresh()` and **refuse stale inputs**. This exists because two v1 bugs — mixed model generations in `proj_team_*`, and playoff contamination of Layer-1 — were both invisible staleness.
- **Feature registry (`neurhl/registry.py`)**: coverage group and vantage rule declared once per feature; availability masks and leakage checks generated from it. Guards: no OUTCOME feature as an input; no feature used outside its true coverage window.
- **Window guards (`neurhl/windows.py`)**: `assert_scorable` refuses to score a broken season or cross a window boundary.

## FINAL STATUS (written after every gate ran)

| phase | outcome |
|---|---|
| P0 prereg, manifests, registry, windows | done |
| P1 shifts + stints, 2008-2026 | done; 18.9M shifts, 7.1M stints, 19/19 seasons validated |
| P2 MoneyPuck corpus (P3 allowlist) | done; 2.08M shots, pre-2012 coordinate hole closed |
| P3 L0 xG | **X1 PASS** (17/18), **X2 PASS**, **X3 FAIL** (recorded) |
| P4 RAPM | **R1 retired (A5)**, **R1′ INCONCLUSIVE (A6)** |
| P5 S1 event simulator | **E1 PASS, E2 PASS** — after TWO leaks found and fixed (A8) |
| P6 S3 deployment, S4b goalie/starter | **NOT BUILT** |
| P7 S4 game outcomes | **FALLBACK INVOKED** — Elo unbeaten |
| P7 S5 season simulation | done; **beats both house benchmarks** |
| P8 confirmatory run C1 | **NOT RUN — G-STOP fired. CONFIRM unspent.** |
| P9 2026-27 projection + player props | done |
| S2b GNN | **NOT BUILT** — pair screen never reached |

**The headline result is negative and is reported as such.** The engine does not
improve game-by-game prediction, the metric the project was directed to optimise:

| TUNE, n=4,920 | log loss |
|---|---|
| constant | 0.68942 |
| **Elo** | **0.67668** |
| simulator alone | 0.68416 |
| Elo+simulator blend | 0.67763 |
| v1 hierarchical (incumbent) | 0.67314 |

G-STOP therefore fired and **CONFIRM (2018-2026) was never touched**. It remains
unspent for any future attempt, which was the whole reason for reserving it.

**Where the engine does win is the season layer**, exactly the split the
preregistered fallback anticipated:

| pooled DEV+TUNE | NeurHL-2 | house |
|---|---|---|
| standings MAE | **9.61** | HOWE 10.36 |
| CRPS | **6.93** | 7.10 / 7.77 |
| 80% coverage | **0.794** | nominal 0.80 |
| player points MAE | **6.66** | league-mean 16.27 |

**Four leaks/gaps found, all by building a consumer that could not cheat rather
than by any gate.** E1 and E2 both PASSED with two leaks present, because a gate
that scores a model on the inputs it trained on is structurally blind to leakage.
`neurhl/tests/audit_leakage.py` is now a standing requirement.

**Known limitations, unfixed:** X3 calibration (top decile, driven by a 2023
recording-regime shift); RAPM not shown better than team-demeaned raw; assists
absent for 2008-2011 in the HTM era; no deployment model, so player props do not
use S1's actor head.

## Acceptance

`neurhl/tests/review_tests_neurhl2.py` re-asserts every recorded gate decision
against these rules mechanically, verifies artifact freshness, re-derives
committed predictions on CPU to `max|Δp| ≤ 1e-6`, and confirms CONFIRM was
touched exactly once. `NOTES.md` gains a NeurHL-2 section recording ships AND
nulls.
