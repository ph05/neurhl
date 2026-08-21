# PLAN NeurHL — neural model track, prereg

STATUS: LOCKED 2026-08-20. All baseline numbers below were computed by
eval/baselines_harness.py + models/baseline_gbm.py on TUNE windows only,
BEFORE any NeurHL model number existed on any gated window (the only NeurHL
executions so far are mechanical smoke tests on 400-game subsets with RANDOM
embeddings, recorded in the ledger as non-runs). THIS FILE IS COMMITTED BEFORE
ANY GATE RUNS — the v4/v6 commit-before-results guarantee.

CORPUS DECISION (A1, decided by prereg rule, eda/eda_07_htm.md): PASSED —
all 5 HTM seasons reconcile with official totals (GPG/SOG rel. err. <= 0.07%);
2012 HTM-vs-JSON gold standard: 100% per-game event-count agreement on all 9
core types, 6,545/6,545 goals matched to the second with 100% scorer-ID
agreement, 97.2% exact on-ice sets. Therefore pretrain corpus = 2008-2026
(~7.56M events); tune vantages = predict seasons 2012-2017 (P2 branch A).
Exclusion ledger: game 2012020660 (degenerate, <200 plays); 3 unparseable 2009
PL reports; 5 HTM games (of 4,920) without a games_ctx label join.

NeurHL ("NeurHL" exactly — all prose, reports, and output model/column names)
is the repo's first neural model: a hierarchical network over the event-scale
corpora, simulating game, player, and season outcomes. It lives in neurhl/ and
this file; it changes nothing in src/, output/, or data/processed/.

## S. Scope

- Headline deliverable: the UPCOMING season only — 2026-27 preseason projection
  (points/percentiles/playoff-cup probabilities over the real 2026-27 schedule
  and rosters) plus in-season game predictions. NO 2027-28 outputs (scope
  directive 2026-08-19).
- 2026-27 live-holdout scoring of v1/v4/HOWE is UNTOUCHED. NeurHL is a parallel
  report-only track there; its own 2026-27 projections become preregistered live
  predictions scored as the season unfolds. No mid-holdout swap, ever.
- v6 (PLAN_V6.md) is a separate in-flight track; NeurHL neither reads its
  pending outputs nor blocks on it.

## D. Data acquisition (NeurHL-owned fetchers, neurhl/data/fetch/)

A1 NHL HTM reports (www.nhl.com/scores/htmlreports): PL/ES/TH/TV, season_end
   2008-2012 backfill + TH/TV for the 57 shift-missing 2024-25 games. Extends
   the event corpus to 2008+ and completes shift coverage. Raw gitignored.
A2 Natural Stat Trick season tables (team 5v5/sva/pp/pk/all; skater std/oi;
   goalie), season_end 2008-2026. Cross-check + Tier-0 context only. Raw HTML
   archived; >=12 s politeness.
A3 NHL stats-rest player reports (skater summary/realtime/faceoffwins/shooting/
   timeonice; goalie summary/advanced), 2006-2026, reg+playoffs. Official
   player aggregates covering the pre-2012 era.
A4 MoneyPuck shots 2008-2011 RAW fields only (already on disk) as the pre-2012
   shot stream companion to A1 (see P3).
A5 Daily lineup/injury/cap snapshot logger (DailyFaceoff, PuckPedia) — archival
   forward-looking infrastructure only, the v6 D7 precedent; can never feed a
   backtest.

Integrity bar: every acquired corpus passes a cross-source reconciliation
(>98% per season vs official reports / MP shots) before tensorization; failing
seasons are excluded and documented as nulls.

## P. Protocol (anti-leakage / anti-era-bias)

P1 Vantage rule: any quantity used for a season-V prediction (pretrain events,
   careers, vocab, calibration, arena coordinate normalizers, feature scalers)
   is computed from season_end <= V-1 only; in-season (h2-style) predictions may
   additionally use season-V games strictly before the predicted game's date.
P2 Pretrain snapshots: expanding-window event-LM snapshots per vantage. Tune
   vantages: 2012-2017 if A1 passes integrity, else 2014-2017. Restatement
   vantages 2018-2026 fine-tune forward one season at a time.
P3 MoneyPuck: all MP model-derived columns (xGoal*, xRebound, arena-adjusted
   coords, win prob) are EXCLUDED everywhere — they were fit by MP on
   full-sample data and leak the future. Raw recorded fields (coords, type,
   times, players) are admissible, source-flagged.
P4 fastRhockey: never a model input. Used once as the cross-source check of our
   own shifts->PBP on-ice join (>=99% 5v5 agreement asserted on the overlap).
P5 Era conditioning, not averaging: era covariates must be walk-forward
   computable (league GPG to date, PP rate to date, 3v3-OT flag, SO/loser-point
   flags, COVID flags, scaled season index). No season one-hots; no full-sample
   normalization anywhere.
P6 COVID: 2020 and 2021 train with games-played-aware loss weights, are
   excluded from per-82 MAE denominators (restated per-82 as house convention),
   and are always broken out in evaluation slices.
P7 Window policy (strict house rule): all tuning/architecture/hyperparameter
   selection and all gates run ONLY on tune vantages (P2) with rolling-origin
   CV at game granularity inside them. 2018-2026 is touched exactly once, by
   eval/restate_2018_2026.py, after the gate lock (P8 governs the outcome).
   Search budget: <= 40 configurations, each logged in
   neurhl/configs/search_ledger.csv and committed — exceeding the budget voids
   the version (documented null).
P8 Restatement decision rule (pre-committed): on the one-shot 2018-2026
   restatement, (i) NeurHL ships as a standalone report track iff it beats the
   HOWE restatement on >=2 of 3 primary metrics (h1 deviation MAE, Spearman,
   season CRPS) with no single-era carry (must not lose in >=2 consecutive
   pre-2022 seasons while winning overall); (ii) else the 50/50 HOWE(+)NeurHL
   Elo-space blend ships as the report track iff it beats HOWE; (iii) else
   NeurHL is a documented null and remains a diagnostic. 2026-27 production
   scoring stays v1/v4/HOWE in all branches.
P9 Determinism: all seeds fixed and recorded in neurhl/output/params_neurhl.json.
   Torch-MPS training nondeterminism is accepted; trained checkpoints are the
   ground truth (SHA256s recorded), ALL gated and reported numbers come from
   CPU inference over saved checkpoints, and per-game prediction artifacts are
   committed CSVs that tests/review_tests_neurhl.py re-derives (max|dp| <= 1e-6).
   Sim seeds 711 (h1 report) / 722 (secondary).
P10 Market firewall: no odds/market data in any feature, target, calibration,
   or model-selection step. Market contact is limited to eval reports and the
   existing forward CLV logging.

## A. Architecture (summary; full configs locked in neurhl/configs/)

Tier 0 (mandatory falsification rung): xgboost + small MLP on engineered
  per-game features (Elo diff, rest/b2b/travel, goalie EWMA, vantage-safe roster
  aggregates, era covariates). Built and evaluated BEFORE the prereg number-lock.
Tier 1: (a) event-LM — ~3M-param transformer over per-game event sequences
  (factorized token embeddings incl. player identities and own-join on-ice
  slots), masked+next-event multi-task objective; product = player embeddings
  per vantage snapshot. (b) career encoder (GRU over all-league career rows,
  distilled onto event-LM embeddings); rookie blend w = gp/(gp+40); no-data
  players -> position mean (counted).
Tier 2: roster-aware multi-task game net (~3.5M params): set-attention roster
  pooling + context vector, FiLM era conditioning; heads = 4-class outcome
  (reg/OT-SO x home/away), bivariate-Poisson score, aux 5v5 SAT, player-game
  rates (TOI share, shots, P(goal), P(assist)). 5-seed deep ensemble;
  temperature calibration fit inside the train window only.
Tier 3: ratings bridge inverts the engine outcome model to Elo-equivalent
  ratings + per-game d_adj offsets, drives src/engine.simulate_season, and emits
  the howe.rebuild_sim dict contract so all downstream tooling works unchanged.

## G. Gates (LOCKED with numbers, 2026-08-20)

Anchors (neurhl/output/baselines_tune.json, tune windows only): v1 walk-forward
game log loss 0.67585 pooled predict-seasons 2012-2017 (0.67548 on 2014-2017;
constant-home 0.68865); Tier-0 0.67692 pooled 2014-2017 (Tier-0 = stronger of
{standardized logistic, early-stopped xgboost} on engineered features — the
initial fixed xgboost at 0.69812 was mis-specified and was replaced BEFORE any
NeurHL number existed, ledger t0-001/t0-002; the replacement only raises the
bar); v1 h1 season MAE/82 9.571 (deviation 9.580), Spearman 0.535, sim CRPS
6.774 mean (uniform MAE 11.581, regressed 10.287).

All NeurHL numbers: CPU inference, 5-seed ensemble, temperature-calibrated on
the val season only (P1):
G1 game:        pooled game log loss on predict seasons 2012-2017
                <= 0.67385 (= v1 0.67585 - 0.002) AND beats v1 in >=4 of 6
                seasons.
G2 season:      h1 deviation MAE (mean over 2012-2017) <= 9.42
                (= 9.571 - 0.15) AND Spearman >= 0.545 (= 0.535 + 0.01).
G3 calibration: reliability slope in [0.90, 1.10] AND ECE <= 0.015 on
                P(home win), pooled tune seasons.
G4 big-data:    pooled 2014-2017 game log loss <= 0.67392 (= Tier-0 0.67692
                - 0.003; same window Tier-0 was computed on). Failing G4 while
                passing G1-G3 downgrades the claim (documented), it does not
                block shipping.
G5 CRPS:        season sim CRPS (mean 2012-2017) <= 6.674 (= 6.774 - 0.10).
G6 repro:       committed prediction artifacts reproduce on CPU
                (max|dp| <= 1e-6). Hard fail.
Ship-to-restatement requires G1-G3 + G6; then P8 governs. Per-season and
per-era breakdowns are mandatory in every gate report.

## A1. AMENDMENT 1 (2026-08-21) — team-form context, committed BEFORE its run

House precedent: PLAN_V3/V4/V5/V6 are successive amendments, each committed
before its gates ran. This amendment follows that pattern. Gate THRESHOLDS
G1-G6 are UNCHANGED; only the model's inputs and capacity change.

### A1.1 What config #1 (the architecture as first prereg'd) produced

Recorded in neurhl/output/archive/config1/params_neurhl.json. Game-level
home-win log loss by predict season, NeurHL vs v1:

  2012 0.7432 / 0.6779   2013 0.6806 / 0.6748   2014 0.6867 / 0.6767
  2015 0.6851 / 0.6709   2016 0.6909 / 0.6839   2017 0.6851 / 0.6705
  pooled 2012-2017 0.6966 (G1 bar 0.67385) -> G1 FAIL
  pooled 2014-2017 0.6873 (G4 bar 0.67392) -> G4 FAIL

Constant-home on the same games is 0.6898, so config #1 is at best level with
knowing nothing. Temperature calibration is a no-op (tau 0.91-1.08): the
ensemble is correctly SCALED but not DISCRIMINATING.

### A1.2 The diagnostic that motivated this amendment

Committed as neurhl/eda/eda_09_signal_location.md. Regularized logistic
regression, trained <=2013, tested on 2015 (1,230 games):

  roster embeddings (mean-pooled) + context   0.6888   ~= constant-home 0.6898
  team rolling form alone (2 features)        0.6780
  team rolling form + context                 0.6760
  form + context + embeddings                 0.6838   (embeddings HURT)

Two team-history features beat the entire 1.2M-parameter network. Mean-pooled
player embeddings carry essentially no game-outcome signal. Read together with
eda_08 (position decodable at 0.952, pts/60 R^2 0.54), the finding is that the
event-LM encodes what KIND of player someone is far better than how GOOD their
team is — and every NHL roster has a similar mix of roles, so pooled rosters
barely separate teams. The binding constraint is INFORMATION, not capacity.

### A1.3 Changes (config #2 of the <=40 budget)

C1 ctx gains 6 rolling team-form features (goals for/against and points per
   game over each team's previous 25 games): in-season updating for game-level
   evaluation (mirrors how house Elo updates within a season), and frozen at
   the end of season T-1 for preseason/h1 projection. Strictly pre-game, P1-clean.
C2 capacity reduced ~11x (d 128->48, encoder layers 2->1, trunk 512x2->128,
   dropout 0.1->0.3, weight decay 0.01->0.1), addressing the overfitting also
   observed in config #1 (every seed peaked ~epoch 5, then degraded).
C3 training corpus starts at 2008 rather than 2009 (uses all available data).

A capacity-only config is deliberately NOT run: the A1.2 probe shows even an
optimally-regularized linear model on embeddings reaches only 0.6888, so
shrinking alone cannot clear the bar. Predicting that and skipping it is
recorded here rather than spent from the budget.

### A1.4 What this does to the scientific claim (stated plainly)

NeurHL is no longer "event data alone beats Elo" — it becomes "event-derived
player representations PLUS team form". The honest test of whether the event
data contributes anything is therefore G4 vs Tier-0, which already contains
Elo diff, team form, travel, goalie and roster-aggregate features. If NeurHL
does not beat Tier-0 by 0.003, the big-data hypothesis nulls at game level and
will be reported as such. G1 (vs v1) remains the ship gate.

## H. Boundaries

NeurHL touches only neurhl/ + this file + .gitignore additions. Nothing in
src/ imports neurhl (battery-enforced). The 2026-27 holdout scoring, v5/v6
outputs, and all params_v*.json are untouched. Heavy artifacts (tensors,
checkpoints) gitignored; prediction CSVs, EDA reports, params_neurhl.json,
search ledger committed.

## Acceptance

eval/gates.py writes full gate records into neurhl/output/params_neurhl.json;
tests/review_tests_neurhl.py mechanically re-asserts every recorded decision
against the rules above, re-derives committed prediction artifacts from hashed
checkpoints, verifies the sim dict against a dry-run of the market tooling
loader, and greps src/ for neurhl imports. NOTES.md gains a NeurHL section
recording ships AND nulls.
