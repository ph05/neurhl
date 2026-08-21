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

## Acceptance

`neurhl/tests/review_tests_neurhl2.py` re-asserts every recorded gate decision
against these rules mechanically, verifies artifact freshness, re-derives
committed predictions on CPU to `max|Δp| ≤ 1e-6`, and confirms CONFIRM was
touched exactly once. `NOTES.md` gains a NeurHL-2 section recording ships AND
nulls.
