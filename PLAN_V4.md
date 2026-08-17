# v4 — bug resolution + five reviewed improvements (commissioned 2026-08-17)

Source: independent model review of 2026-08-16 (47/48 checks passed; findings below).
Discipline UNCHANGED: all tuning/screens on train (<= 2017). The 2018-2026 preseason
windows are SPENT and no decision in v4 may touch them. Three REPORT-ONLY diagnostics
(validation-window CRPS restatement, market comparison, historical ensemble numbers) are
computed ONCE, after this file is committed, and are constitutionally incapable of
changing any ship decision (all decisions are gate-based on train, or locked in this
file). The live 2026-27 season is the pre-registered out-of-sample test for everything.

## B. Bugs (decision-free fixes, no gates needed)

B1. engine.py standings key computed in float32 silently dropped the ROW tiebreaker
    (ULP at 1e10 ~ 1024 > any ROW value; bound at the playoff cutoff in ~3.8% of sims).
    Fix: float64 key. Also add total wins W as the 4th key (NHL order: PTS, RW, ROW, W)
    — derivable from tracked arrays at zero cost.
B2. analytic_xpts `missing` diagnostic: operator precedence made the assert MESSAGE
    wrong (condition was correct). Fix.
B3. Reproducibility debt: the two shipped constants that came from uncommitted ad-hoc
    analyses get committed scripts:
      - src/repro_b2b.py — the rest/b2b measurement behind B2B_ELO=38.
        Acceptance: home-on-b2b / away-on-b2b deltas within ±1.5% of the documented
        -6%/+6%, |t| >= 4, Elo equivalent in [28, 48]. Constant unchanged if reproduced.
      - overlay.py grows the v3.1 partial-regression amendment as code (raw gate AND
        amended path, flags preserved). Acceptance: rho_partial reproduces 1.011 ± 0.15
        under the original spec (TOI-share arrival weights).
B4. Overlay estimation/application weight mismatch (review finding): the mechanism was
    estimated with realized-TOI arrival weights but applied at full weight. Fix by
    re-estimating rho_partial on train with PRODUCTION-CONSISTENT full arrival weights;
    rho* = clip(0.5 * rho_full, 0, 0.75) replaces 0.506 whatever it turns out to be.
    Cap ±25 and zero-sum recentering unchanged. (Train-only re-estimation; train is
    not a single-use window.)
B5. live.py rebuilt to (a) the VALIDATED estimator — blend prior with history-carried
    Elo (full games.csv + fetched 2026-27 games through run_elo), not the prior-seeded
    Elo it currently runs — and (b) the full promised outputs: rest-of-season sims with
    b2b adjustments + gated noise layers, nightly live_odds_<date>.csv + ratings
    snapshot. Parity test: on a synthetic mid-season cutoff of a past season, live path
    reproduces the replay-engine ratings to < 0.5 Elo.
B6. h2 (2027-28) simulations use h2 ages (reference season_end 2028) for availability;
    v3 reused the h1 closure.

## I1. Goalie game layer (starter rotation in the simulator)

Model: per team, tandem (G1, G2) by projected workload; s1 = G1 start share, EB-shrunk
gp-share history toward the causal league starter mean (shrink n0_g tuned on train grid
{2, 4, 8} by weighted MSE of next-season share). Per simulated game the starter is drawn
Bernoulli(p1); p1 = s1 normally, p1 = beta_b2b * s1 on second-of-b2b nights.
beta_b2b = 0.5 is DECLARED, not fit (no game-level start data in the repo; public
starter-usage patterns put backup share of b2b second games near 60-70%); sensitivity at
{0.3, 0.7} reported. Per-game Elo delta = k * 30 * (theta_started − theta_tandem_mean),
theta_tandem_mean = s1*theta1 + (1−s1)*theta2 — ZERO-MEAN by construction, so the mean
goalie effect stays with the (already-gated) tandem_gsax feature; this layer adds
variance shape + the b2b/backup-quality interaction only.

GATE G (train replays 2012-2017, h1; sigma_c refit with every shipped noise source ON):
  (a) 80% central coverage of actual points in [0.72, 0.90];
  (b) coverage spread across tandem-gap tertiles <= spread without the layer + 0.02;
  (c) max |mean simulated points shift| per team < 0.4 pts/82 (zero-mean verification).
Ship iff (a) AND (b) AND (c).

## I4. Availability 2.0

(a) Per-player persistence: player availability prior = EB blend of own past GP shares
    (recency-weighted) toward age-bucket mean; weight n0_a tuned on train grid
    {4, 8, 16, 32} by MSE of next-season GP share (targets <= 2017).
    SCREEN S1: within-bucket year-over-year corr of GP share on train >= 0.10, else
    persistence is dropped (age-only retained) and documented as null.
(b) Zero-GP mass: P0(bucket) = fraction of regulars with ZERO next-season NHL games,
    measured on train; availability draw becomes the mixture P0·δ(0) + (1−P0)·BetaBin,
    centered at the mixture mean (stays zero-mean; fattens the catastrophic tail the
    review showed was excluded).
(c) Goalie slot: each team's G1 joins the draw list with value_goals =
    (theta1 − theta2) * 2500 (replacement is the backup, not a replacement skater);
    goalie availability params fit on clear starters (gp share >= 0.45) with the same
    bucket machinery. SCREEN S2: goalie next-season share overdispersion vs binomial
    >= 5x on train, else the goalie slot is dropped and documented.
(d) h2 closures use h2 ages (B6).

GATE A2 (same machinery as v3 Gate A, train 2012-2017, sigma_c refit): 80% coverage in
[0.72, 0.90] AND fragility-tertile coverage spread <= v3-availability spread + 0.02.
Components failing their screens are dropped individually; the gate applies to whatever
survives. If the gate fails, v3 availability is retained unchanged.

## I5. Prospect pipeline (two-season-ahead structural prior)

Join drafts 2009-2019 to skater panels by (normalized name, F/D position group).
QUALITY BAR: >= 80% of picks 1-60 from drafts 2009-2015 who logged >= 82 NHL GP by
2019 must match a panel playerId; below the bar, the feature is abandoned (declared
draft_cap decay retained) and the join failure documented.
Ramp: expected points-above-baseline by (pick bucket {1-10, 11-30, 31-60, 61+},
years-since-draft 1..8), fit on drafts <= 2013 with outcomes <= 2017 (train only).
Feature prospect_pipeline(V, h) = Σ over own not-yet-established draftees (career NHL
TOI < 1500 min at V, draft years V-7..V) of ramp(bucket, (V+h) − draft_year).
draft_cap remains a candidate; the ledger arbitrates redundancy.

GATE F2 (v3 Gate F rules, evaluated separately at h1 and h2 on train LOSO):
add-one-in dMAE <= +0.02 AND sign stability >= 0.67 at that horizon; the feature enters
production only at horizons where it passes; final-set LOSO <= incumbent-set + 0.05
else full fallback.

## I2. Scoring axes (evaluation infrastructure; report-only, no model decisions)

CRPS of team season-point distributions from sim draws (estimator verified against the
closed-form Gaussian CRPS in tests). Reported on train alongside gates as context
(NEVER a gate), and restated once on the already-published validation-window
projections after this commit (labeled REPORT-ONLY). Market comparison: loader for
data/market/nhl_totals_<season_end>.csv (schema: team, total, over_price, under_price,
source, retrieved). An automated-fetch attempt is permitted; if no trustworthy source
is obtained the comparison is BLOCKED and documented (no synthetic lines). live.py logs
nightly xPts + playoff/Cup odds so closing-line-value vs any hand-recorded market is
computable later.

## I3. Ensemble (decision LOCKED at this commit, before any computation)

ENS = Elo-space equal blend per team: r_ens = 1505 + 0.5*(r_v1 − 1505) + 0.5*(r_v4 − 1505).
r_v1: frozen v1 params (project_ratings with causal beta; production beta/om through
2026). r_v4: v4 ridge ratings + amended overlay. ENS is simulated with the full v4
simulator and ships as a CO-HEADLINE sheet REGARDLESS of any subsequently computed
report-only historical number. Rationale requiring no new window data: v2 confirm
(2018-2021) favored the feature model, v2 final (2022-2026) favored v1, both inside
noise — under relative-skill uncertainty the equal-weight combination is the robust
choice (forecast-combination literature; equal weights are hard to beat). Live 2026-27
scores v1, v4, ENS on rest-of-season MAE, CRPS, playoff Brier — pre-registered.

## Deliverables

engine.py fixes; src/goalie_game.py, src/availability2.py, src/prospects.py,
src/scoring.py, src/repro_b2b.py; overlay.py amendment-as-code; src/backtest4.py
(writes its prereg block to params_v4_gates.json BEFORE computing; runs all gates);
src/report4.py (v4 + ENS production, h2-age fix); rebuilt src/live.py; extended
review_tests.py; NOTES.md appended with every gate number; params_v4.json.
v1/v2/v3 outputs remain untouched (v1 stays hash-frozen).
