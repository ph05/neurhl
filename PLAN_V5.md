# PLAN v5 — data expansion (hockey-reference, MoneyPuck shots, NHL PBP, NHL EDGE)

Preregistered 2026-08-19 BEFORE any gate computation (this file is written before
backtest5.py exists or runs). Same discipline as PLAN_V4: train-only tuning,
2018-2026 untouched except locked report-only restatements, live 2026-27 = holdout.

## D. Data acquisition (raw -> data/raw/, derived -> data/processed/)

D1 MoneyPuck shot-level files, seasons 2007-2025 (start-year keys; season_end
   2008-2026): moneypuck.com shots_<year>.zip. Raw zips are NOT committed
   (gitignored, ~300MB); derived team-season aggregates are committed.
D2 NHL play-by-play: api-web.nhle.com gamecenter play-by-play JSON, regular season,
   season_end 2012-2026 (game IDs from api.nhle.com/stats/rest game index). Raw
   gzipped JSON NOT committed (gitignored, multi-GB); derived per-team-season event
   aggregates are committed.
D3 NHL stats-rest team reports (summary, faceoffpercentages, penalties, realtime,
   summaryshooting, powerplay, penaltykill), season_end 2006-2026 — official
   aggregate cross-check for D2 and pre-2008 coverage where MoneyPuck is absent.
D4 NHL EDGE tracking (skating speed/distance, shot speed, zone time), available
   2021-22+ only: too short for the 2012-2017 train window -> REPORT-ONLY/EDA and
   live-season diagnostics; ineligible to ship as a v5 feature by construction.
D5 hockey-reference season pages 2006-2026 (standings incl. SRS/SOS + team stats):
   polite scrape (>=4s between requests, cached). Used for cross-validation of D2/D3
   and EDA (SRS, one-goal-game records); HR is already the source of the games table.
D6 sportsdataverse fastRhockey-data NHL bulk parquet (requested addition,
   2026-08-19): team_box, player_box, schedules, processed PBP as published
   (2010-2024 at fetch time). Processed cross-check for D2/D3 and game-level box
   aggregates; same eligibility rules as the source it mirrors.

Integrity bar: every derived table cross-checked against an independent source where
one exists (PBP-derived faceoff/penalty aggregates vs NHL stats-rest vs MoneyPuck;
shots-derived xG vs mp_teams). Mismatches > 2% are investigated before use.

## E. EDA (exploratory; descriptive on full sample per the NOTES v1 precedent,
##    anything that informs SELECTION restricted to the train window)

For each new team-season metric: lag-1/lag-2 repeatability, corr with t+1 pts_pct,
era stability, and — train window only (predict-seasons 2012-2017) — correlation with
v4 walk-forward residuals (does it add what the incumbent misses?). Written to
NOTES.md by src/eda_v5.py (auto-generated section).

## F. Candidate features (locked list; anything else found later goes to the next
##    plan, not this one)

All walk-forward-safe at vantage V (use season_end <= V only), deviation-space,
within-season centered, added to the v4 19-feature incumbent:

  f1 fo_dev        faceoff win% deviation (all situations; MP/PBP/stats-rest agree)
  f2 pen_diff      penalties drawn - taken per game, deviation
  f3 hd_share      high-danger xG share of total xGF (5v5), deviation
  f4 flurry_xg_dev flurryScoreVenueAdjusted 5v5 xG% deviation (upgrade candidate over
                   the shipped scoreVenueAdjusted xg_dev; if both survive, keep one —
                   whichever gates better — never both, they are near-collinear)
  f5 corsi_dev     score-adjusted Corsi% deviation (volume separated from quality)
  f6 rush_xg_dev   rush-shot xG% deviation from shot-level data (shots within 4s of a
                   last event in neutral/defensive zone), 5v5

## G. Gates (stricter than v4 F2 because these are empirically mined, not
##    theory-first; multiple-testing discount is the point)

GATE F3 per candidate, per horizon (h1 and h2 gated independently, same as v4):
  add-one-in train LOSO (predict 2012-2017): dMAE < 0.00 (strict improvement — the
  v4 tolerance of +0.02 does NOT apply here) AND sign stability >= 0.67.
GATE F3-joint: the final v5 feature set must beat the v4 incumbent set's LOSO MAE
  (dMAE < 0) at the shipped horizon(s); ties or losses -> candidate set does not ship
  and v5 is documented as a null result. Assembly procedure (declared before any
  gate is computed): greedy forward addition of gate-passers in add-one-in dMAE
  order, each accepted only if joint train LOSO improves; replacement variants
  (candidate swapped for xg_dev) evaluated only under the collinearity rule below.
Ridge lambda: re-tuned on the same train grid as v4 (no new grids).
Collinearity rule (prereg): among f4 vs incumbent xg_dev, and f5 vs f4/xg_dev, ship
  at most the single best gater; add-one-in evaluated against the incumbent WITH the
  other candidates absent.

## H. What v5 may and may not touch

- The locked 2026-27 holdout is NOT restated as a headline: v1/v4/HOWE live scoring
  (PLAN_V4 B5/I3) continues untouched. v5's 2026-27 numbers are REPORT-ONLY context.
- v5 ships (if gates pass) as the 2027-28 co-headline: projections_2027_28_v5 and
  report-only HOWE5 = 0.5*v1 + 0.5*v5. The production HOWE (v1+v4) remains the
  entity being scored live; no mid-holdout swap.
- If no candidate gates in: v5 = documented null, data + EDA remain as
  infrastructure, and this plan closes.

## Acceptance

- backtest5.py prints per-candidate gate lines and writes output/params_v5.json
  with the full gate record (mirrors params_v4 gates).
- review battery extended with: derived-aggregate cross-source checks (<2% mismatch),
  and (if shipped) v5 output consistency checks mirroring 7.6.
