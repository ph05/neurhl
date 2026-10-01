# ORR

*ORR (Odds, Ratings & Rosters) was developed under the working name HatTrick and renamed on 2026-10-01, after the 2026-27 forecast was built. Files written before the rename record paths under `hattrick/`: the freeze's `manifest_2027.json` and `fitted_inputs_2027.json`, and the commits up to `8b0fc4b`. Some backtest JSON keys and player-output columns (`g_hattrick`, `"hattrick"`) keep the old name so that existing outputs stay readable.*

ORR is an NHL forecasting system built to compete with NeurHL. It was designed from the review in [`review/REVIEW_NeurHL.md`](../review/REVIEW_NeurHL.md), which was finished and committed before any ORR code. One model produces three things:

- **player projections**: every skater's and goalie's season line, with intervals, plus daily per-game player lines in-season;
- **game predictions**: all 1,344 games of 2026-27 from preseason information, plus a daily in-season forecast;
- **season standings**: points distributions, and playoff, division, Presidents' Trophy and Cup odds.

**Information cutoff.** The 2026-27 forecast uses only information dated before **2026-09-29 17:00 ET (21:00 UTC)**, the first puck drop. Every repository input is read from the git tree of the last commit before that time (`snapshot.py`). The files ORR adds are dated sources (see Data).

The forecast was built on 2026-09-30 and 2026-10-01, after the season had started. The scorer therefore counts it only for games after its publication time; see [`RESULTS.md`](RESULTS.md).

## The design

Start from the strongest cheap signal, add what it misses, and never be more confident than the backtests allow.

- **Standings target.** A convex blend of two views:
  - the de-vigged sportsbook points line, with the news it had not seen priced in;
  - a regressed team-history view.

  The weights are 0.75 / 0.25, chosen by leave-one-season-out over a pre-declared list of view sets on the seasons where every view exists. Convex weights keep the blend's spread no wider than its inputs'.
- **News.** The line was recorded on 2026-08-17. Everything between then and the cutoff is priced by the player model as an event counterfactual: the same rosters with each move undone and each later-reported absence healed. This covers:
  - trades and signings;
  - contract terminations;
  - camp injuries;
  - suspensions;
  - the Hellebuyck standoff: a 10% chance he plays for Winnipeg, and no other team credited with him.
- **Ratings.** Points targets become offence/defence log-rate ratings on the actual schedule.
- **Rating uncertainty.** The SD comes from the blend's out-of-sample error, net of the game luck the scoring model implies, plus in-season drift from the rating filter's process noise.
- **One scoring model** for every game: a Conway-Maxwell-Poisson base and a mean-preserving late-game (pulled-goalie) layer, with era-specific overtime and shootout models and with rest/travel and starting-goalie effects.
  - Preseason game probabilities are averaged over joint draws of both teams' ratings.
  - The season simulator applies the exact NHL tiebreakers and playoff bracket.
- **Player lines** come from exposure-regressed, age-adjusted per-60 rates, a depth-chart model of who dresses, and ice time conserved within each team. Player goals are reconciled to team goals, and goalie goals-against to team goals-against, so the three outputs agree.
- **In-season.** A Kalman-style filter updates the ratings after every result, from goals, and from shots when the results file carries them. Its home-ice estimate replaces the preseason one, and its uncertainty is anchored to the freeze's on day 0. The daily forecast uses plug-in probabilities (the backtested rule) and known starting goalies where given.
- **In-season, ORR 1.1** (from 2026-10-01; `--model 1.0` restores the original loop). It adds the one pre-registered 1.1 item that passed its rule, X1 (`orr/PLAN_1_1.md`, `orr/lineups.py`):
  - every game's dressed skaters enter both the update and the forecast, through their projected 5v5 on-ice xG value and ice time, measured against the team's expected lineup;
  - so do its starting goalies.

  Lineups and starters come from NeurHL's committed pregame lineup files. The other eight items failed their rules and are not in the model; see RESULTS.

## Pipeline

```
 pre-cutoff git snapshot (snapshot.py)  +  added data (fetch_boxes.py, data_*.csv)
   ├── data.py ─────────────┬──────────────────────────┬──────────────────────────┐
   │                        │                          │                          │
 players.py / goalies.py   gametable.py              teams.py                   market lines
 (rates, aging, NHLe,      (every game: rest, travel, (top-down team history,    (de-vigged,
  draft-slot rookies)       shots, starters, ids)      style GF/GA)               84-game)
 deploy.py (GP, TOI,             │                          │                          │
 injuries, conservation)   gamemodel.py (scoring       convex LOSO blend  ◄── news priced by
   │                        model) + ratings.py              │                 team_components.py
 team_components.py ──────► (preseason / in-season) ──► calibrate.py: targets → (o, d)
                                                              │
                                                     season.py: simulated seasons, strength drawn
                                                     from its posterior + in-season drift,
                                                     exact standings rules, playoff bracket
                                                              │
                                   freeze.py → output/freeze_2027/ (teams, games, skaters, goalies,
                                                 ratings, manifest with SHA-256 of inputs and code)
                                   ingest.py → output/live/results_2027.csv
                                   inseason.py → output/live/<date>/ (games, players, standings)
                                   score.py → scorecard against every NeurHL release
```

## What is taken from NeurHL, and what is not

The review lists NeurHL's strengths, and ORR keeps them:

- conservation of ice time and goals;
- walk-forward state;
- a causality audit, extended here to reading every input from the pre-cutoff git tree;
- dated forecasts with provenance;
- availability modelling;
- an exact playoff bracket;
- published quantiles.

It drops these NeurHL practices:

- backtests that rely by default on each player's actual season team or dressed lineup. ORR uses the season team only on 2025-26, where no box scores exist, and reports known-starter game results as a separate variant;
- a neural stack that ties Elo;
- post-hoc level multipliers;
- a fixed team shock borrowed from an older layer;
- goalie start caps;
- repeated releases after the season began.

## Running it

```sh
pip install numpy "pandas>=2" pyarrow scipy scikit-learn requests

# inputs
python3 -m orr.fetch_boxes          # fastRhockey box scores, verified against the freeze manifest
python3 -m orr.snapshot             # materialise the pre-cutoff inputs and print their provenance

# rebuild the intermediate files (committed; only needed after code changes)
python3 -m orr.team_components --history   # bottom-up team components by season
python3 -m orr.roster_delta                # roster-change history
python3 -m orr.ratings                     # game-model parameters for 2026-27

# the 2026-27 forecast
python3 -m orr.freeze --sims 40000  # refuses to run with uncommitted model code unless --allow-dirty

# in-season, each day
python3 -m orr.ingest               # NHL API, with shots; or --add "id,date,home,away,hg,ag,REG|OT|SO[,source,shots_h,shots_a]"
python3 -m orr.inseason --date 2026-10-02 [--goalies starters.csv] [--model 1.1|1.0] [--lineup-dir DIR]
python3 -m orr.score                # scorecard against NeurHL

# tests
for t in test_season test_players test_gamemodel test_freeze test_inseason; do python3 -m orr.tests.$t; done
```

**Shots matter in-season.** Run `ingest` where the NHL API is reachable, so that results carry shots on goal. These are the backtest results:

| Updated on | ORR 1.1 | NeurHL-G | Verdict |
|---|---|---|---|
| Goals and shots | 0.6600 | 0.6601 | Tie |
| Goals alone | 0.6622 | 0.6601 | Worse by 0.0021, not significant |

ORR 1.0 scored 0.6634 on goals alone, significantly worse than NeurHL-G. See RESULTS.

The game table (`orr/cache/gametable.parquet`) is built on first use.

- **Backtests:** `python3 -m orr.backtest.{players_bt,goalies_bt,games_bt,teams_bt,gamefile_bt,inseason_bt}`. The ORR 1.1 hindcast, `orr.backtest.inseason_bt_1_1`, was run once and refuses to overwrite its output. The nine pre-registered 1.1 experiments are in `orr/experiments/<ID>/`.
- **Website (GitHub Pages):** `python3 -m orr.site.build_orr` and `python3 -m orr.site.build_dashboard` write `docs/orr/index.html` and `docs/orr/compare.html`. Once on the Pages branch they are served at [ph05.github.io/neurhl/orr/](https://ph05.github.io/neurhl/orr/), beside NeurHL's site.

Results and the head-to-head with NeurHL are in [`RESULTS.md`](RESULTS.md).

## Data

Every repository input is read from commit `9580606` (2026-09-29 16:53 EDT), the last commit before the cutoff:

- MoneyPuck season summaries, 2008-09 to 2025-26;
- game results, 2005-06 to 2025-26;
- the 2026-27 schedule;
- rosters and roster changes up to 2026-09-29;
- raw injury fields (MoneyPuck and DailyFaceoff) recorded 2026-09-28;
- the sportsbook points lines and Cup odds recorded 2026-08-17.

Added for ORR:

- **Per-game box scores**, 2010-11 to 2023-24, from the sportsdataverse/fastRhockey-data mirror of the NHL API (`fetch_boxes.py`; SHA-256 in the freeze manifest). They cover the starting goalie, ice time and shots.
- **Historical preseason points lines**, 2018-19 to 2025-26 (222 team-seasons). Hand-collected from Hockey-Reference and cross-checked (`data_market_history.csv`).
- **Injury, suspension and holdout timelines** for 46 players, from reports dated on or before 2026-09-29 (`data_injuries_2027.csv`, one source URL per row).

2026-27 results are read only by `ingest.py`, `inseason.py` (for dates after them) and `score.py`.
