# HatTrick

HatTrick is an NHL forecasting system built to compete with NeurHL. It was designed from the review in [`review/REVIEW_NeurHL.md`](../review/REVIEW_NeurHL.md), and it produces three things from one model:

- **player projections**: every skater's and goalie's season line, with intervals;
- **game predictions**: every one of the 1,344 games of 2026-27, frozen before the season, plus a daily in-season forecast;
- **season standings**: points distributions, playoff, division, Presidents' Trophy and Cup odds.

The 2026-27 preseason freeze uses only information that existed before **2026-09-29 17:00 ET**, the first puck drop.

## The design, in one paragraph

Start from the strongest cheap signal, then add what it misses, and never be more confident than the backtests allow:

- **Standings** blend the de-vigged sportsbook points line with a regressed team-history view and a bottom-up roster view. Blend weights are fitted leave-one-season-out on the seasons where all three exist.
- The line was recorded on 2026-08-17. Everything that happened between then and the cutoff is priced by the roster model, since the market had not seen it:
  - trades and signings;
  - camp injuries;
  - suspensions;
  - the Hellebuyck standoff.
- Blended **points targets** are turned into offence/defence ratings on the actual schedule. Every game is then simulated with **one scoring model**: a Poisson base plus an explicit late-game (pulled-goalie) layer, with era-specific overtime and shootout models, rest/travel effects and starting-goalie effects.
- **Player lines** come from exposure-regressed, age-adjusted per-60 rates, and ice time is conserved within each team. Player goals are reconciled to team goals, so the three outputs agree.

## Pipeline

```
 pre-cutoff git snapshot (snapshot.py)
   ├── data.py ─────────────┬──────────────────────────┬──────────────────────────┐
   │                        │                          │                          │
 players.py / goalies.py   gametable.py              teams.py                   market lines
 (rates, aging, NHLe)      (every game: rest, travel, (top-down team history,    (de-vigged,
   │                        shots, starters, ids)      style GF/GA)               84-game)
 deploy.py (GP, TOI,             │                          │                          │
 injuries, conservation)   gamemodel.py (scoring       blend (LOSO weights) + news adjustment
   │                        model) + ratings.py              │
 team_components.py ──────► (preseason / in-season) ──► calibrate.py: targets → (o, d)
                                                              │
                                                     season.py: 20k+ seasons, strength drawn
                                                     from its posterior + in-season drift,
                                                     exact standings rules, playoff bracket
                                                              │
                                   freeze.py → output/freeze_2027/ (teams, games, skaters, goalies,
                                                 ratings, manifest with SHA-256 of every input)
                                   inseason.py → output/live/<date>/ (daily games + re-run season)
                                   score.py → scorecard against every NeurHL release
```

## What is taken from NeurHL, and what is not

The review lists its strengths; HatTrick keeps them:

- conservation of ice time and goals;
- walk-forward state;
- a causality audit, extended here to reading every input from the pre-cutoff git tree;
- dated forecasts with provenance;
- availability modelling;
- an exact playoff bracket;
- published quantiles.

HatTrick drops these NeurHL practices:

- backtests that know the actual lineup, starter or season team;
- a neural stack that ties Elo;
- post-hoc level multipliers;
- a fixed team shock borrowed from an older layer;
- goalie start caps;
- repeated releases after the season began.

## Running it

```sh
pip install numpy "pandas>=2" pyarrow scipy scikit-learn
python3 -m hattrick.snapshot                 # materialise the pre-cutoff inputs, print their provenance
python3 -m hattrick.freeze                   # the 2026-27 preseason freeze
python3 -m hattrick.inseason --date 2026-10-01 --results results.csv [--goalies starters.csv]
python3 -m hattrick.score                    # scorecard against NeurHL
python3 -m hattrick.tests.test_season        # tests (also test_players, test_gamemodel)
```

Backtests: `python3 -m hattrick.backtest.players_bt`, `hattrick.backtest.goalies_bt`, `hattrick.backtest.games_bt`, `hattrick.backtest.teams_bt`. Results and the head-to-head with NeurHL are in [`RESULTS.md`](RESULTS.md).

## Data

Every repository input is read from commit `9580606`, the last commit before the cutoff:

- MoneyPuck season summaries, 2008-09 to 2025-26;
- game results, 2005-06 to 2025-26;
- the 2026-27 schedule;
- rosters as of 2026-09-29;
- raw injury fields (MoneyPuck and DailyFaceoff) recorded 2026-09-28;
- the sportsbook points lines and Cup odds recorded 2026-08-17.

Added for HatTrick:

- **Per-game box scores**, 2010-11 to 2023-24, from the sportsdataverse/fastRhockey-data mirror of the NHL API. These cover the starting goalie, ice time and shots.
- **Historical preseason points lines**, 2018-19 to 2025-26 (222 team-seasons), hand-collected from Hockey-Reference and cross-checked (`data_market_history.csv`).
- **Injury, suspension and holdout timelines** for 46 players, from reports dated on or before 2026-09-29 (`data_injuries_2027.csv`, one source URL per row).
