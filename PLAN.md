# NHL 2027–28 Season Projection Model — Design Spec

**Date:** 2026-08-12 · **Author:** ph05
**Deliverable:** A spreadsheet containing the aggregate expected returns of the 2027–28 NHL season.

## 1. Objective and interpretation

Project the 2027–28 NHL regular season and playoffs at the team level, from today's vantage
point (August 2026). "Expected returns" is interpreted as the full distribution of season
outcomes, aggregated per team: expected standings points, spread (percentiles), playoff
probability, division/conference/Presidents' Trophy probability, and Stanley Cup probability —
plus fair-odds columns (decimal/American) so the projections can be read as break-even prices
against any futures market.

**Key structural facts (verified 2026-08-12):**
- The last completed season is 2025–26. The 2026–27 season has not been played. A 2027–28
  projection is therefore a **two-season-ahead** forecast; uncertainty must widen accordingly.
- New CBA: **84-game season starting 2026–27** — 28 intra-division (4× each of 7 rivals),
  24 intra-conference non-division (3× each of 8), 32 inter-conference (2× each of 16);
  42 home / 42 road. 2027–28 point totals are on an 84-game scale (league mean ≈ 94–95 pts,
  not ≈ 92).
- **32 teams**, no expansion approved for 2027–28 (Houston tracked for ~2029–30).
- Playoff format unchanged: top 3 per division + 2 wild cards per conference, fixed bracket,
  best-of-7 (2-2-1-1-1).

## 2. Vantage-point discipline (no look-ahead)

The model may use only information available at forecast time. Backtests mirror the real task:
predicting season T uses data through season T−2 (two-ahead) or T−1 (one-ahead diagnostics).
Roster moves of summer 2026 and 2027 are unknowable to the team-level engine; they are handled
statistically (mean reversion + inflated variance), not by hand-tweaking ratings.
(Nate Silver: uncertainty is information; don't fake precision you don't have.)

## 3. Approaches considered

1. **Player-level bottom-up** (Luszczyszyn-style GSVA: project every player, aggregate rosters).
   Most granular, but rosters for 2027–28 don't exist yet, player projection two years out is
   noisy, and the build cost is very high. Rejected for this scope.
2. **Pure results-based Elo** (FiveThirtyEight-style). Simple, robust, easy to backtest, few
   parameters. But ignores shot-quality information that is known (Tulsky) to be more repeatable
   than goal results in limited samples.
3. **Hybrid: MOV-adjusted Elo blended with an xG-share rating, projected forward with an AR(1)
   mean-reversion process, then Monte Carlo over a synthetic CBA-correct schedule.** ← **Chosen.**
   Captures both results and process, stays parsimonious (≈7 free parameters), and the AR(1)
   carryover gives a principled two-season-ahead distribution.

## 4. Architecture

- `src/fetch_*.py` — data acquisition with on-disk caching (`data/raw/`), polite rate limits.
- `src/build_dataset.py` — clean unified tables: `games.csv` (one row/game, franchise-mapped),
  `team_seasons.csv` (points, GF/GA, xG% from MoneyPuck).
- `src/eda.py` — exploratory statistics → `NOTES.md` (reliability/repeatability, home ice, OT
  rates, scoring environment, parity). EDA outputs set parameter priors.
- `src/engine.py` — Elo loop (MOV multiplier, home ice, season carryover), xG rating, blend,
  AR(1) projection, OT/shootout submodel, schedule generator, vectorized season+playoff
  simulator with standings tiebreakers.
- `src/backtest.py` — walk-forward tuning and frozen validation vs baselines.
- `src/report.py` — final run + `output/nhl_2027_28_projections.xlsx` (+ CSV).

## 5. Model detail

**Elo:** init 1505 mean; expansion teams start low (VGK lesson noted); franchise continuity
ATL→WPG (2011), ARI→UTA (2024). Update per game incl. playoffs:
`R' = R + K · mov_mult(gd, elo_diff) · (S − E)`, home-ice bonus H Elo points,
`E = 1/(1+10^(−Δ/400))`. OT/SO wins treated as 1-goal margins. Between seasons, revert toward
1505 by carryover φ_s.

**xG rating:** season 5v5-or-all-situations xG share → Elo-equivalent points via an empirical
goals-per-game ↔ Elo mapping; blended rating `R_blend = w·R_elo + (1−w)·R_xg` used only for
*preseason* projection (in-season updates stay pure Elo).

**Two-ahead projection:** end-of-2025–26 blended ratings → season-strength AR(1):
`r_{T+2} = φ²·r_T + ε`, with φ and innovation σ estimated from historical year-over-year rating
regressions (and the φ² implication checked against the直接 measured two-lag correlation).
Per-simulation draw of true strength ⇒ honest widening of two-year-out distributions.

**Game model:** P(reaches OT) empirical (≈23–25%, checked vs |Δ| dependence); regulation winner
from Elo; OT/SO winner from a compressed-Elo logistic fit on historical OT outcomes. Points:
W=2, OTL=1, L=0. Regulation wins tracked for tiebreakers.

**Simulation:** synthetic 84-game schedule per CBA matrix; 10,000 season sims; standings with
points → regulation wins → total wins → random tiebreak; playoff bracket sims through the Cup.

## 6. Tuning & anti-overfitting protocol

- Data: 2007–08 → 2025–26 (MoneyPuck xG coverage begins 2007–08). 2007–08/2008–09 = Elo burn-in.
- **Train:** seasons ≤2016–17 (K, H, MOV form, φ_s via in-season log loss; w, strength-σ via
  preseason MAE walk-forward inside train).
- **Freeze, then validate once:** 2017–18 → 2025–26, one-ahead and two-ahead, vs baselines:
  (a) all-teams-equal, (b) prior points regressed 50% to mean ("Marcel"-lite, Tango),
  (c) raw prior points. Metrics: per-82-normalized points MAE/RMSE, rank correlation, playoff
  Brier + calibration, per-game log loss vs constant-home-prior.
- Few parameters, coarse grids, prefer plateaus over sharp optima (Silver: simple beats clever
  out of sample). COVID-shortened seasons (2019–20, 2020–21) included for rating updates,
  validation metrics reported with and without them. Shortened seasons evaluated per-game.
- Variance calibration: simulated points SD must match realized historical SD (fat, honest
  distributions — the classic failure is sims that are too narrow).
- Optional trajectory/age term: tested on train only; kept **only** if it clearly improves
  two-ahead validation; dropped otherwise and reported as a null result (Tulsky-style
  skepticism about small-sample stories).

## 7. Deliverable spec (`output/nhl_2027_28_projections.xlsx`)

Sheets: `README` (methodology, caveats, parameter table, teachers' principles applied);
`Projections_2027_28` (headline aggregate: xPts mean/SD/percentiles, division/playoff/conference/
Cup probabilities, fair odds); `Projections_2026_27` (one-ahead bonus); `Sim_Distributions`
(percentile grid, P(≥100 pts) etc.); `Ratings` (Elo/xG/blend end-2025–26 → projected mean
2027–28); `Backtest` (per-season model vs baselines, calibration table, frozen parameters).
Plus a CSV copy of the headline sheet.

## 8. Risks / honesty notes

- Two-year-ahead team forecasts have thin skill margins over regressed baselines — expected,
  and reported rather than hidden.
- Offseason 2026/2027 transactions, aging cliffs, goalie volatility ("goalies are voodoo") are
  in the error term, not the point estimate.
- 84-game era has no historical data; per-game modeling + scale-up is assumed adequate.
- Schedule is synthetic (real 2027–28 schedule unpublished); order/rest effects ignored.
