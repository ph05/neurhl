# ORR vs NeurHL: preregistered 2026-27 evaluation

*Committed on 2026-10-02, when ORR's results file held 8 of the 1,344 regular-season games and no ORR daily forecast had yet been scored. This file fixes how ORR is judged against NeurHL at the end of the 2026-27 regular season. `orr/evaluate_2027.py` implements it exactly; its dry run is `orr/output/evaluation_2027.json`. Nothing here may change after this commit. Any later analysis is labelled exploratory.*

## Forecast files (SHA-256 recorded by `evaluate_2027.py` at this commit)

- **ORR preseason:** `orr/output/freeze_2027/{teams,games,skaters,goalies}_2027.csv`. These are ORR 1.0, built from information dated before the 2026-09-29 cutoff and published 2026-10-01 00:37 UTC.
- **NeurHL 1.0:** `neurhl/output/neurhl_1_0/{teams,games,skaters,goalies}_2027.csv`, published 2026-09-29 02:12 UTC.
- **NeurHL 1.1:** `neurhl/output/neurhl_1_1/season/{teams,games,skaters}_2027.csv`, published 2026-09-29 20:53 UTC.
- **ORR daily:** `orr/output/live/<date>/games_<date>.csv`, the latest forecast created before each game's puck drop.
- **NeurHL pregame:** NeurHL's committed `neurhl/output/live/2027/<date>/pregame_<game>.csv` files (G stack, H and Elo).

## Rules for eligibility

- **A forecast counts for a game only if it was published before that game's scheduled puck drop.** Times come from the NHL schedule; publication time is the commit time on GitHub.
- **ORR's preseason file** counts only for games after its publication time.
- **NeurHL 1.0 and 1.1** count for every game, since they were published before the first puck drop.

## Primary comparisons

Negative differences favour ORR. Every primary comparison uses a paired bootstrap with 2,000 resamples, seed 7, and a 95% CI. Units are games for game metrics, teams for team metrics, and players for player metrics.

| # | Comparison | Metric | Population |
|---|---|---|---|
| P1 | ORR daily vs NeurHL-G pregame | log loss of P(home win) | games where both are eligible |
| P2 | ORR preseason file vs NeurHL 1.1 file | log loss | games after ORR's publication |
| P3 | ORR preseason teams vs NeurHL 1.1 teams | final points MAE | all 32 teams |
| P4 | ORR preseason skaters vs NeurHL 1.1 skaters | points MAE | skaters projected by both with 40+ GP |

## Verdict rule

For each comparison:
- **ORR better:** the 95% CI lies entirely below 0.
- **NeurHL better:** the CI lies entirely above 0.
- **No difference shown:** otherwise.

There is no overall winner by majority. The table of verdicts is the result.

## Secondary (reported, not judged)

- NeurHL 1.0 in place of 1.1 for P2, P3 and P4.
- P4 over every skater projected by both, whatever his games played.
- For P3:
  - CRPS of final points, from a normal with each file's mean and the sd implied by its 10th to 90th percentile range, so both files are scored alike;
  - each file's 80% interval coverage.
- Brier score difference beside every game comparison.
- Goalie season save % MAE, for goalies with 1,000+ shots against, against NeurHL 1.0 (NeurHL 1.1 has no goalie file).
- ORR daily against NeurHL-H pregame and against Elo pregame.

## SHA-256 of the evaluated files at commit time

| File | SHA-256 |
|---|---|
| `orr/output/freeze_2027/teams_2027.csv` | `16b35fecbba1c28ad43c514eed4172d2d955744456fdd85585e442b57c838328` |
| `orr/output/freeze_2027/games_2027.csv` | `e16c3f807c0b16d12589124b32a1dba412c1e47286c00f90b724fef8006541f3` |
| `orr/output/freeze_2027/skaters_2027.csv` | `925684ccf7b88963b6c81e8a3244a88a8d31b9c51f133fecdc7dc3778a7e9f7b` |
| `orr/output/freeze_2027/goalies_2027.csv` | `4fe8ad7e80537ad607b5f79e229485234f99e44db8cdbd5f53b43431ec85f89d` |
| `neurhl/output/neurhl_1_0/teams_2027.csv` | `969526a2592b85b5843d0360693da18569904413ac582231962865652921dd86` |
| `neurhl/output/neurhl_1_0/games_2027.csv` | `0e6ec35ca83c8919ecf6f90906d0f47137cf1385ba7d5bdf55c46ee38068fff1` |
| `neurhl/output/neurhl_1_0/skaters_2027.csv` | `946ea41d4b85f615710dfd457ee81a233a6b8a70ab9e40809bb1b77873cdf753` |
| `neurhl/output/neurhl_1_0/goalies_2027.csv` | `046ce38daa7cad3aec9f5be4a3133f8a29c31dac9f176bfacbc9d679c6999ed2` |
| `neurhl/output/neurhl_1_1/season/teams_2027.csv` | `a29c652e2d0ad1bd38bc366c7f2058849f521d2c3639b7eeba2ce07f9633b626` |
| `neurhl/output/neurhl_1_1/season/games_2027.csv` | `b2de83d093170b4db7e5fc362efd9f596edacfb8c930f96ea8c2748071bfdc3f` |
| `neurhl/output/neurhl_1_1/season/skaters_2027.csv` | `3ff97bc121508dca443c5891781497aa97aab59b3e99cc967aa97d351245db45` |

At the final evaluation, `evaluate_2027.py` recomputes these hashes. A mismatch means a file changed, and the result for that file is then void.
