# PLAN NeurHL-3 — player game/season projection layers + game-outcome reattempt, prereg

STATUS: COMMITTED BEFORE ANY NeurHL-3 NUMBER EXISTS (house commit-before-results
convention, as PLAN_V3→V6, PLAN_NeurHL and PLAN_NeurHL2 honoured).

NeurHL-3 adds the layers PLAN_NeurHL2 left unbuilt — deployment (S3), goalie
starter/quality (S4b) — and builds on them the two hard deliverables: a
**player GAME-level projection layer** and a **player SEASON-level projection
v2**, plus one disciplined game-outcome reattempt carrying only information no
prior attempt used. Every standing constraint carries verbatim: P3 allowlist,
P10 market firewall, P9 determinism (CPU-derived gated numbers, ≤1e-6
re-derivation), A4/NO_SCORE {2013, 2021}, `src/` untouched, production scoring
stays v1/v4/HOWE. G6 declared-lineup snapshots are live-2027 inputs ONLY and
never enter any backtest (PLAN_NeurHL §P1 rule, carried).

## W. Windows (mechanical, `neurhl/windows.py`)

| Layer | Develop (iterate freely) | Declared eval (spent precedent) | One-shot confirm (untouched) |
|---|---|---|---|
| player-game (PG), goalie | PLAYER_DEV {2011, 2012} · PLAYER_TUNE {2014-2017} | PLAYER_EVAL {2022-2026} (A10 spend) | **PLAYER_CONFIRM {2018, 2019, 2020}** (~3,624 games ≈ 130k skater-games), spent ONCE by `neurhl/eval/confirm_player_2018_2020.py` |
| player-season (PS) | vantages {2011, 2012, 2014-2017} | {2022-2026} | joins the same one spend event |
| game (N5) | pre-gate window: DEV∩{proj_team exists} + TUNE_SCORED (n≈8,610) | — | **C1** = CONFIRM_SCORED, n≈10,184, once, behind G-STOP (PLAN_NeurHL2 C1 as declared) |
| xG (X3/A11) | selection on DEV/TUNE vantages only | pooled over all 18 vantages (X-gate practice) | — |

All scoring passes `assert_scorable` / `assert_scorable_player`; 2021 is
NO_SCORE and belongs to no player window. The player one-shot and the game C1
are separate events, each preceded by its own config-freeze commit; the player
spend runs first.

## M. Modules

N1 availability/absence (`neurhl/data/build_absences.py`) · N2 deployment
(`neurhl/data/build_usage.py` + usage heads of the PG model) · N3 goalie
(`neurhl/data/build_goalie_games.py`, `neurhl/models/goalie_start.py`) · N4
player event rates (`neurhl/models/player_game.py`,
`neurhl/train/train_player_game.py`, `neurhl/eval/backtest_player_game.py`) ·
N5 game blend (`neurhl/eval/backtest_game_v3.py`,
`neurhl/eval/confirm_2018_2026.py`) · N6 season v2
(`neurhl/sim/player_season_v2.py`, `neurhl/eval/backtest_player_season.py`).
Data stages D1 mp_extra, D2 absences, D3 HTM RO/GS, D4 snapshot extension,
D5 usage, D6 goalie games, D7 NST/EDGE (deferred behind declared screens).

## PG. Player GAME-level layer

**Targets** per skater-game (regular season, skaters only): `toi_share`
(player TOI / team dressed-skater TOI — the recorded H4 definition), `shots`
(SOG, Poisson), `p_goal` = P(goals≥1), `p_assist` = P(assists≥1). Conditioning
is the dressed 18+2 and announced starters — identical to the recorded H4
protocol, so the recorded baseline is comparable; live 2026-27 measures the
same conditioning variable via G6 snapshots.

**Model**: four walk-forward HistGradientBoosting models chained
opportunity → volume → conversion (stage-1 usage predictions feed stage-2
shots; both feed the binaries). Feature blocks, declared: B0 self-form (the
baseline's own inputs verbatim, plus α∈{0.02, 0.3} EWMAs, last-1/3/5 windows —
the model NESTS the baseline); B1 usage structure from D5 (ev/pp/sh TOI EWMAs,
5v5 line-rank + 3-vs-25-game delta, PP-unit membership); B2 opportunity from
D2 (vacated EWMA-TOI of same-position absent regulars, absent-regular count,
above-me-absent flag, return ramp); B3 context (opponent elo_logit,
shot-suppression form, penalties-taken rate, opposing starter gq/GSAx, rest,
b2b, games_last_7d, toi_last_7d, signed tz, home, era columns); B4 line/coach
patterns (modal-linemate quality, linemate churn, TOI concentration,
coach-change flag; mp_extra situational cross-features 2008-2024 under
availability masks). Calibration: binaries get isotonic fit on season V−1
out-of-sample predictions of the deployed model (A7 method); `toi_share`
renormalised within (team, game) to Σ=1 — conservation exact. Sparse blocks
enter NaN-coded under registry availability masks (PLAN_NeurHL2 §D mechanism,
tree-native).

**Declared pre-config diagnostic** (costs no config): on PLAYER_DEV, decompose
the EWMA residual variance of toi_share by B1/B2 covariates to quantify
reachable headroom. Expected outcome, written in advance: modest — the EWMA is
strong at 0.00619 MAE; the reachable signal is role changes, absences and
deployment shifts, which is exactly what B1/B2 measure.

### PG gates (the recorded H4 null is the bar: own-EWMA won toi_share
0.00619 vs 0.00934 and shots 1.080 vs 1.148; H4 won the binaries)

| Gate | Rule | Baseline | Instrument |
|---|---|---|---|
| **PG1** toi_share | model MAE < baseline MAE with game-clustered paired p<0.05 AND season-clustered direction agrees | player's own EWMA(α=0.1) renormalised over dressed skaters — the exact code at `eval/backtest_player.py:41-54,99-107` | paired per-row diff, cluster-robust SE by game_id; season-clustered t (df = seasons−1) per `eval/backtest_hier.py:97-111` |
| **PG2** shots | Poisson deviance better, same significance; MAE reported alongside | own EWMA(α=0.1) shots | same |
| **PG3** p_goal | log loss better, same significance | walk-forward SHRUNK EWMA: p̂=(n·ewma+k·base)/(n+k), k∈{5,10,20,40,80} chosen per vantage on season V−1 only | same |
| **PG4** p_assist | log loss better, same significance | same shrunk-EWMA construction | same |
| **PG-ALL** | PG1-PG4 all pass, Holm-corrected across the 4 heads | — | — |
| **PG-STOP** | PLAYER_CONFIRM is spent ONLY if PG-ALL passes on PLAYER_EVAL | — | pre-gate, mirrors G-STOP |

Gate population = rows where the baseline is defined (coverage reported).
Rookie/no-history rows are scored separately against positional base rates as a
non-gated capability metric.

**Pre-committed ladder** per failing head, in order, one ledger config each:
(1) declared feature-set extension (the α-variants, window variants,
situational EWMAs); (2) residualised target (predict target − EWMA, add back);
(3) convex blend w·model+(1−w)·EWMA with w fitted on season V−1 only;
(4) DATA rung — mp_extra situational features if not yet entered, then NST
individual tables (D7), then official RO scratches (D3 tranche 2). A residual
failure is recorded WITH the next data hypothesis pre-registered — never a
terminal null.

## GS/GQ. Goalie module

| Gate | Rule | Baseline | Instrument |
|---|---|---|---|
| **GS1** starter | starter log loss beats BOTH heuristics, season-clustered | (a) most-recent-starter-repeats; (b) season starts-share, both walk-forward | paired, season-clustered t |
| **GQ1** GSAx earns its place | walk-forward GSAx predicts next-season sv% better than `gq` (EWMA-shrunk sv%, `models/baseline_gbm.py:50-73`) | `gq` | Steiger's z (`train/rapm_folds.py:39-56`) |

If GQ1 fails, N5 uses `gq` only and GSAx stays a diagnostic — declared now.

## PS. Player SEASON v2

Path A = incumbent `models/player_proj.py` (recorded anchors: 2026-vantage
MAE 9.39 at ≥40GP; pooled 2022-2026 9.54 vs league-mean 18.54). Path B = the
PG layer aggregated over the real schedule with an N1 availability hazard
(P(dressed) from gp_share history, absence spells, age), Monte Carlo over
availability draws (fixed seed), totals mapped through the existing
`to_totals` conservation code.

| Gate | Rule |
|---|---|
| **PS1** ship rule (declared before results) | B ships iff pooled ≥40GP points-MAE < A on the 11 vantages AND B wins ≥6/11; else A ships; the 50/50 A+B blend ships only if it beats both pooled (computed in the same run). Whichever ships must also beat the 9.39/9.54 anchors — a regression blocks shipping regardless. |
| **PS2** conservation (mechanical, hard fail) | per-team-game Σ toi_share = 1 exactly; positional minutes = budgets {F 181.0, D 116.9}·G; league Σ exp_gp within 0.5% of 18·G·32·0.968; no exp_gp > 78·(G/82); TOI ceilings hold. |

Season CRPS/coverage (`eval/backtest_season.py:47-54`) reported as secondary.
Goalie season projection (starts × quality) ships as a labelled secondary
gated only by GS1/GQ1.

## GB. Game-outcome reattempt (N5)

Form: the incumbent v1-H thin head extended in place — per predict-season T,
StandardScaler + LogisticRegression (CS grid chosen on train seasons < T) on
`[elo_logit, proj_diff, proj_cl_diff, rest_diff]` + NEW columns, exactly as
`eval/backtest_hier.py:39,72-96` fits the incumbent, so v1-H (0.67314) is
nested and is the floor by construction. New-information blocks, ranked, no
prior attempt used any of them:
1. `repl_delta_diff` — replacement-strength delta from per-game absences (D2 ×
   walk-forward player value: rapm_prior net where present, else EWMA pts/60 ×
   toi_share);
2. goalie block — `gq_diff` (announced starters; Tier-0 precedent) +
   `gsax_diff` (D6, only if GQ1 passes); ablation with N3-predicted starters
   reported to bound the announcement advantage;
3. schedule-density — `games7d_diff`, team `toi_last_7d` diff,
   `travel_km_7d_diff`, `dtz_signed_diff` (signed from arena longitudes; the
   v6 null tested unsigned only);
4. `sched_strength_todate_diff` — mean opponent pre-game Elo faced to date;
5. EDGE (G5) — NOT in the headline; separate pre-specified secondary scored
   only inside C1 on 2023-2026 with its own stated n, identical
   hyperparameters, no extra configs (PLAN_NeurHL2 §D headline restriction to
   G0-G4 respected).

**Config budget**: game level stays in `configs/search_ledger_v2.csv`, cap 12
total; two retroactive rows record the spent S4-v2 evaluations, leaving 10.
Allocation: g-01 base+block1 · g-02 +block2 · g-03 +block3 · g-04 +block4 ·
g-05 HistGBM on identical columns (run only if DEV residuals show curvature) ·
g-06 interaction/regularisation variant · g-07/g-08 diagnosed repairs only
(each must cite a measured cause, A6-style) · g-09/g-10 reserved, may go
unused. Exactly ONE candidate is named in a freeze commit before G-STOP.

| Gate | Rule |
|---|---|
| **GB1** dev/tune | candidate pooled log loss < v1-H pooled on the same games AND season-clustered direction agrees |
| **G-STOP** | mean paired diff candidate − v1-H ≤ −0.0015 on the pre-gate window (n≈8,610) AND clustered agreement, else **C1 is not spent** and the ladder continues with the next data stage — recorded, never terminal |
| **C1** | as declared in PLAN_NeurHL2 (paired per-game diff vs **v1 Elo** negative, two-sided p<0.05, season-clustered agrees) **plus co-primary: the same test vs v1-H**. Beating Elo but not v1-H is the pre-declared partial branch and is reported as partial. Run once by `eval/confirm_2018_2026.py` (run-once guard; refuses without a recorded G-STOP pass; sensitivity battery: wild cluster bootstrap on season, game-level block bootstrap, sign test; Murphy decomposition — a win with no resolution gain is reported as recalibration). No re-spend, ever. |

Power, stated in advance: at the incumbent-like effect (−0.0028) C1 has >90%
power; at exactly −0.0015 it is ~55-60% — accepted.

## X3. Amendment A11 ladder (declared in PLAN_NeurHL2 §A11; cap 4 configs)

Rung 1: house walk-forward per-rink distance-offset covariate. Rung 2:
cross-source NHL-xy↔MoneyPuck disagreement index per rink-season (2012+,
NaN-masked before). Rung 3: trailing rush-share + trailing realised
close-shot conversion covariates. X1/X2 must not regress below pass on any
rung. Constants fixed from DEV vantages only. Residual failure after all rungs
stands recorded with the per-rink coordinate re-registration study
pre-registered as the next hypothesis.

## L. Ledgers and caps

`configs/search_ledger_v3.csv` (run_id, date, layer, config_delta, window,
metric, value, notes): player-game cap **10**, player-season **4**, goalie
**4**, xG/A11 **4**. Game level: ledger v2, cap 12 (10 remaining). Exceeding a
cap voids that layer's version. Holm-adjusted p beside raw wherever a family
exceeds one test.

## Q. Leakage and reproducibility (extended standing requirements)

`neurhl/tests/audit_leakage_tabular.py` applies the A8 causality standard to
EVERY D-stage builder: corrupt all rows after a cut date, rebuild features,
require values before the cut to be bit-identical — must pass before any gated
number is reported from a builder's output. `neurhl/tests/review_tests_neurhl3.py`
re-asserts every gate decision against these rules mechanically, verifies
ledger caps, re-derives committed prediction artifacts on CPU to ≤1e-6, and
confirms each one-shot window was spent at most once. Library versions pinned
in `configs/env.json`. `NOTES.md` records ships AND nulls.

## Deliverables

(1) Per-game 2026-27 player projection artifact (per game: each dressed
skater's E[TOI share], E[SOG], P(goal), P(assist)), produced pre-game from G6
snapshot inputs. (2) Regenerated season `player_proj_2027.csv` from the PS1
winner, with Monte-Carlo percentiles. (3) Goalie starters + quality secondary.
(4) Game-outcome: report-track only, whatever C1 records. All report-only;
2026-27 production scoring remains v1/v4/HOWE in every branch.
