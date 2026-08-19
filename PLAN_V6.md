# PLAN v6 — second data expansion ("all 8"), prereg

Preregistered 2026-08-19. THIS FILE IS COMMITTED BEFORE ANY FETCH COMPLETES OR ANY
GATE RUNS (restoring the v4-style commit-before-results guarantee that PLAN_V5's
same-session lock lacked). Train-only discipline unchanged: tune/gate on
2012-2017 predictions, 2018-2026 only for post-lock report-only restatement,
live 2026-27 = untouched holdout scoring v1/v4/HOWE.

## D. Data acquisition (8 items, as scoped in session discussion)

D1 NHL shift charts (api.nhle.com stats/rest shiftcharts), regular season,
   season_end 2010-2026 (~19.5k games; earlier seasons kept if the endpoint has
   them, missing years tolerated -> league-average fill). Raw gitignored.
D2 Player career stats (api-web player/{id}/landing) for every draft pick with a
   joined playerId in draft_join.csv — junior/AHL/European season totals for
   production-based prospect ramps. Raw gitignored, derived committed.
D3 Coaching records: hockey-reference NHL_<yyyy>_coaches.html, 2006-2026
   (polite >=4.5s). Derived coaches.csv committed.
D4 Travel/timezone: NO new fetch — computed from games.csv + a static arena
   coordinate/timezone table (franchise-era aware) committed as data (this table
   is reference data authored in-repo, cross-checked against known arena moves).
D5 Playoff play-by-play: same fetcher as D2(v5) with gameType=3, 2012-2026
   (~1.3k games). Report-only diagnostics this round (playoff vs regular outcome
   environment); no playoff-model change ships from this plan.
D6 NHL EDGE player-level (skater-detail), seasons 2022-2026, skaters with GP>=20
   in mp_skaters that season. Report-only by construction (as PLAN_V5 D4) PLUS a
   snapshot archive: EDGE history is not guaranteed to remain served; we log it.
D7 Odds snapshot logger: src/log_odds.py appends a dated cross-book Cup futures
   snapshot (vegasinsider table, the source of data/market/nhl_cup_2027_best_price)
   to data/market/odds_log/. Seeded once now; intended for daily/regular runs.
   Forward-looking infrastructure only — nothing here can affect any backtest.
D8 Player absence spells: NO new fetch — derived from rosterSpots already inside
   the 2012-2026 PBP corpus (40 dressed players per game). Per player-season:
   games dressed, missed, and missed-game STREAK structure (long spells ~ injury,
   scattered ~ scratches).

## F. Candidates (locked; nothing added later inside this plan)

Ridge candidates (added to the v5 shipped sets; walk-forward safe, measured in
season_end <= V):
  c1 line_cont   share of season-V forward-pair TOI (from shift overlap) that
                 already played together in season V-1
  c2 toi_hhi_f   Herfindahl concentration of season-V forward TOI
  c3 coach_new   head-coach change during season V or between V-1 and V (both
                 knowable at vantage; the coach FOR season T is deliberately NOT
                 used — midseason-change leak). Plus tenure length as the same
                 candidate's continuous form; whichever of indicator/tenure gates
                 better is the single c3 entrant (never both).
  c4 prospect_prod  production-weighted prospect pipeline: draft-pick value scaled
                 by league-adjusted pre-NHL points pace (D2 careers), same ramp
                 framework as v4's prospect_pipeline. Collinearity rule (as v5):
                 {prospect_pipeline, prospect_prod} — at most one per horizon,
                 add-one-in decides, replacement variant also tested.
Game-level candidate (NOT ridge; mirrors the v4 B3 b2b protocol):
  c5 travel      logistic of home win on Elo diff + b2b flags + away-team travel
                 terms (timezone crossings in prior 48h; km traveled prior 3 days),
                 fit on 2010-2017 regular season only. Ships into d_adj ONLY if
                 |t| >= 4 AND mean |Elo equivalent| >= 10 AND same sign in both
                 halves of the train window (2010-2013 / 2014-2017). Anything less
                 -> documented measurement, no ship.
Screen (diagnostic only this plan):
  S3 spell-based injury propensity (from D8 streak structure — information v4's
     failed S1 persistence test did not have): EB player-propensity vs the shipped
     bucket-only availability on the same MSE protocol as v4 S1. This plan only
     RECORDS the screen result; any availability rebuild is a future plan.

## G. Gates

GATE F4 for c1-c4, per horizon: identical to v5's F3 (add-one-in train LOSO
  dMAE < 0.00 strict AND sign stability >= 0.67; greedy forward joint assembly in
  dMAE order; final set must beat the v5 incumbent or the candidate set is null).
GATE T1 for c5: the three-part bar declared above, decided entirely on 2010-2017.
EDGE (D6), playoffs (D5), odds (D7), spells (D8): report-only/infrastructure —
  ineligible to ship anything in this plan, declared now.

## H. Boundaries (unchanged from PLAN_V5 H)

Live 2026-27 scoring stays v1/v4/HOWE. If F4/T1 pass, v6 + HOWE6 = 0.5*v1+0.5*v6
become the 2027-28 co-headline with fresh seeds (611/622); v5 outputs untouched;
2026-27 v6 numbers are report-only context. Post-lock 2018-2026 restatement
follows the v4-I3/v5 precedent.

## Acceptance

backtest6.py writes output/params_v6.json with full gate records; battery gains a
SECTION 9 (shift/coach/travel/spell table integrity + cross-checks: shift-derived
team TOI vs 60min*GP identity, coach table covers every team-season, travel table
matches schedule count, spells reconcile with panel GP); determinism: any shipped
report rerun hash-identical.
