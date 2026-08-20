# EDA-06 — cross-source label integrity

Join: 19,272/19,272 exact matches — full coverage.


## Final-score agreement

Compared: 19,272 games. Score mismatches: **0** (0.00000%).

## OT/SO flag agreement (regular season)

went_ot disagreements: **0** (0.00000%); went_so disagreements: **0**.

## Shift-chart TOI identity (team on-ice player-hours per game)

Expected ≈6.0 h/game (6 on-ice incl. goalie × 60 min; PK time subtracts, OT adds). Observed: mean 5.949, min 5.638, max 6.114 across 490 team-seasons.

Team-seasons outside [5.5, 6.3]: **0**


Verdict: checks above graduate to review_tests_neurhl.py with thresholds — score mismatch = hard fail; date±1 handled in the tensorizer join; OT-flag disagreements enumerated as exclusions.
