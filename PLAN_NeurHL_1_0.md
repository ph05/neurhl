# PLAN NeurHL 1.0: one model for the 2026-27 season

STATUS: COMMITTED 2026-09-28, AFTER THE NHL ROSTER DEADLINE AND BEFORE THE
UNIFIED MODEL'S 2026-27 PREDICTIONS ARE FROZEN. The freeze section below is
appended once they are, before the first regular-season game
(2026-09-29 21:00 UTC).

## Decision

NeurHL 1.0 is one interconnected model that predicts player-games, games and
season totals for 2026-27. The levels are constrained to agree, and that
agreement is itself a check. Two decisions were taken with the project owner
on 2026-09-28:

1. **New dated freeze, primary.** The unified model's predictions are frozen
   as new hash-locked files and are the NeurHL 1.0 forecast everywhere. The
   predictions frozen on 2026-09-25 (PLAN_NeurHL_LIVE.md) are unchanged and
   are scored as preregistered, as the earlier freeze.
2. **No retraining.** The core is the validated game engine: NeurHL-G v1
   (PLAN_NeurHL4 FREEZE; bundle `g2027_v1`, trained on seasons through 2024).
   Additional statistics enter the stat sheets and season totals, not the
   engine. A retrained engine is a later version, tested on the sealed
   2025-26 seasons.

## The model

**Engine.** For any game and dressed lineups, the engine gives:
- every skater's ice time by strength, shots, attempts, individual xG, goals,
  assists and on-ice xG;
- the team totals, which equal the sums of those skater outputs by
  construction;
- the goalies' shots and goals against;
- the four-way outcome, from its hazard integration.

The final home-win probability is the engine's logistic stack over the Elo,
engine and NeurHL-H logits (coefficients 0.21, 0.29 and 0.58), the same stack
the game-day forecasts use. The four-way outcome is rescaled to it. Goal means
on the stat sheet use the A1 multiplier (PLAN_NeurHL4 A1), frozen on the
post-deadline rosters.

**Lineups** (`neurhl/sim/availability_2027.py`). Each team's post-deadline
roster, with:
- players the status file marks unavailable removed;
- players on injured reserve (DailyFaceoff's list, off the NHL roster) out for
  15 games, then back;
- the club's call-up pool (players dropped from the previous snapshot, and
  its 2025-26 depth players) used only to fill a lineup the roster cannot.

Each healthy skater has a per-game dressing probability. It comes from a
binomial logistic model fitted on every opening-night skater of 2012-2024
(2013 and 2021 excluded; n = 6,980), using his shrunk share of games dressed
over three seasons, exposure, age, position and recent ice time. The depth
chart comes from recent ice time. Absences come in spells: 87% of missed games
fall in multi-game spells averaging 30 games, tuned on 2022-23.

Goalie start shares come from recent starts, clipped to [0.50, 0.72]. The
starter's share is multiplied by 0.54 in the second game of a back-to-back,
tuned on 2022-23.

Validated on 2023-24 with the model refit on seasons through 2023 (643
opening-night skaters):
- dressing share: MAE 0.162, against 0.208 for last season's share;
- mean games played by regulars: 69.6 simulated, 69.9 real;
- top-six forwards dress 0.884 of games simulated, 0.889 real.

The model draws K lineup samples for the whole schedule.

**Player-games and games.** The engine runs on all 1,344 games for each of
the K draws. A player-game's expectation is its average over the draws, and a
player who does not dress in a draw contributes zero. So games played is a
sum of dressing probabilities, and the team totals remain the sums of the
players in every draw.

**Season.** The model simulates 20,000 seasons from the per-game outcomes:
- one team-strength shock per team and season, standard deviation 0.07 on the
  log goal rate. This is `TEAM_SIGMA` from NeurHL-2, calibrated there on
  2011-2017 and carried over unchanged.
- The shock acts through the engine's own sensitivity of the win logit to the
  goal-rate ratio.
- Standings use the NHL's division format (three per division plus two wild
  cards per conference). Tiebreakers are points, regulation wins, regulation
  and overtime wins, total wins, then a coin flip from a dedicated generator.
- Playoff series are best of seven with home ice to the better record. Game
  probabilities come from ratings fitted to the stacked game probabilities,
  plus the same season shock.

**Totals.** Team and player season totals are sums of the game and
player-game expectations. The additional statistics are per-60 rates from
`neurhl/data/build_player_rates.py` times the engine's ice time: hits, blocked
shots, giveaways, takeaways, faceoffs, penalty minutes, and penalties taken
and drawn. Faceoffs are balanced so that each team takes every faceoff of a
game and the two teams' wins sum to them. The rates are validated on 2023-2024
(`player_rates_validation.json`).

**Preseason convention.** No 2026-27 game has been played. Every game is
therefore evaluated with the state as of opening night, and days-into-season
set to its opening value. The state builders count every appended row as a
game played, so each skater, goalie and team state is built once and every
game is assembled from those rows plus its own schedule context: rest,
back-to-back, travel and time zones. NeurHL-H's term uses each team's
opening-night projection from its expected lineup.

## Consistency, the acceptance check

`neurhl/tests/check_neurhl_1_0.py` reads only the output files and must pass
before the freeze. It verifies:
- **Games:** probabilities in (0, 1); outcome4 sums to 1 and matches the
  home-win probability; league scoring, overtime share and home-win rate
  inside historical bands.
- **Teams:** wins, losses and overtime losses add to 84; league points equal
  2 x games plus overtime games in every simulated season; playoff,
  division, Presidents' Trophy and bracket probabilities sum to their slot
  counts.
- **Players:** each team's skater goals, shots and additional statistics sum
  to the team totals; 18 skater-games and one goalie start per team-game;
  goalie goals against and wins sum to the team's.
- **Player-games:** skater goals per team-game equal the game file's; ice time
  inside the physical budget; faceoffs balanced; season totals equal the sums
  of the player-game rows.

## Scoring

Scoring runs after the last regular-season game (2027-04-10), with one
inference. Interim scorecards are descriptive.

- **Games:** per-game log loss of the frozen home-win probability, paired on
  the same games against the preseason Elo reference and against the
  2026-09-25 NeurHL probability, with a week-block bootstrap interval
  (9,999 draws, seed 711).
- **Standings:** team points MAE and CRPS against the final table, with
  coverage of the 10th-90th percentile interval (nominal 0.80). Compared with
  the 2026-09-25 projection and with HOWE.
- **Players:** skater points MAE for skaters with at least 40 games, against
  the 2026-09-25 blend. Also reported: goals, assists, shots, games played
  and ice time; goalie starts and save percentage.
- **Game-day forecasts:** the morning and pregame forecasts remain scored
  under PLAN_NeurHL4 LIVE. They are the same model at game level, with that
  day's lineups and every game already played.

## Expectations, stated before any game

- **Games.** Every preseason probability uses opening-night state, so the
  edge over the preseason Elo reference should be smaller than the in-season
  engine's gate result (-0.0056 against Elo). One season's standard error is
  about 0.002-0.003, so only a gap of about 0.006 or more is likely to be
  resolved.
- **Standings.** Nothing yet shows this season layer is better than the
  earlier one. The shock size was calibrated for a different game model, so
  the coverage of the 80% intervals may miss nominal.
- **Players.** Games played depends on the availability model, and a star's
  long absence or a trade will dominate individual errors.

## Known limitations

- The season shock is carried over, not recalibrated for this engine.
- The starter named from last season's starts takes too many starts:
  0.577 of his team's starts simulated against 0.496 real in 2023-24.
- The sampler has no trades or outside call-ups: 26.8 distinct skaters per
  team simulated, 31.1 real.
- NeurHL-H's term does not respond to sampled absences.
- Trades, call-ups and in-season injuries are unknown at the freeze.
- The stat sheet's goal level depends on the A1 multiplier.

## A1 (2026-09-28, before the freeze): goalie start shares

In validation, the named starter took too many starts: 0.593 of his team's
starts predicted against 0.496 realised in 2023-24. His recent start share is
now shrunk toward an even split:

p_start = clip(0.5 + 0.65 x (share - 0.5), 0.50, 0.72)

- `share` = (starts in 2025-26 + 0.5 x starts in 2024-25) / (games dressed in
  2025-26 + 0.5 x games dressed in 2024-25).
- The factor 0.65 was chosen on 2022-23 only, from a grid of 0.00 to 1.50 in
  steps of 0.05. The error curve is flat between 0.55 and 0.75.
- The backup takes the rest, and the back-to-back factor of 0.54 still
  applies.

| Named starter's share of team starts | Predicted | Realised | MAE |
|---|---|---|---|
| 2022-23 (tuning), new rule | 0.556 | 0.545 | 0.130 |
| 2022-23, previous rule | 0.588 | 0.545 | 0.136 |
| 2023-24 (validation), new rule | 0.555 | 0.496 | 0.141 |
| 2023-24, previous rule | 0.593 | 0.496 | 0.151 |

The remaining over-prediction comes mostly from misnamed starters: in 11 of
32 teams in 2023-24, the goalie named starter took fewer starts than the
backup. No share rule fixes that. Skater lineups are unchanged: the same
random draws give the same skaters.

## FREEZE (2026-09-29): the NeurHL 1.0 predictions for 2026-27

Run on the post-deadline rosters of 2026-09-28 with 64 lineup
draws and 20,000 simulated seasons (seed 711); season shock sd
0.07; stat-sheet goal multiplier 0.9229; engine bundle
`g2027_v1` (sha256 a43965e13ff54796...); lineups from sim/availability_2027.py;
code at commit `9c78a40`. The consistency check passed 42 of 42.

| File | SHA-256 |
|---|---|
| `neurhl/output/neurhl_1_0/games_2027.csv` | `0e6ec35ca83c8919ecf6f90906d0f47137cf1385ba7d5bdf55c46ee38068fff1` |
| `neurhl/output/neurhl_1_0/teams_2027.csv` | `969526a2592b85b5843d0360693da18569904413ac582231962865652921dd86` |
| `neurhl/output/neurhl_1_0/team_points_quantiles_2027.csv` | `7a1a7b606265499b0008067f0e95ce0252d253acd4a307e8743a6891d288f729` |
| `neurhl/output/neurhl_1_0/skaters_2027.csv` | `946ea41d4b85f615710dfd457ee81a233a6b8a70ab9e40809bb1b77873cdf753` |
| `neurhl/output/neurhl_1_0/goalies_2027.csv` | `046ce38daa7cad3aec9f5be4a3133f8a29c31dac9f176bfacbc9d679c6999ed2` |
| `neurhl/output/neurhl_1_0/player_games_2027.csv.gz` | `23f7c873fbdb1429a9b1a8191d6c6bce03e23cf96dd28c81cca0617c65353f1e` |
| `neurhl/output/neurhl_1_0/consistency_2027.json` | `a3835c1989cb4de1a8e5b23fe9652483fb850757a59e73ee1f3c8383e6ceb77d` |
| `neurhl/output/neurhl_1_0/checks_2027.json` | `ac389097708eb9cdf85c9f29dfd1d70d824f72b4a2fef4c10e81636ab9e1b097` |
| `neurhl/output/neurhl_1_0/run_2027.json` | `3bedbc7d069e3480b7a0882f99c283cbe7973ad47bc40ea136900b89b1ce7ac2` |
| `neurhl/output/neurhl_1_0/player_rates_2027.csv` | `6875cc0850961c5b82f93d7f179cdf6f038b55263846e37bbe03adcc4e785190` |
| `neurhl/output/neurhl_1_0/availability_2027.csv` | `0c28bef2206dd06e235a2ab06fa3ea72c053ccf4006bf51211f454bbe9f881fe` |

Highest projected points: CAR 114.0, COL 111.7, TBL 106.1, VGK 105.3, DAL 102.2. Highest Cup odds: CAR 17.4%, COL 14.6%, VGK 7.5%, TBL 7.2%, DAL 5.0%.

These files are never edited. A correction is issued as a new, dated file set
beside them, and both are scored.
