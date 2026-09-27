# neurhl/live: roster, lineup and publishing plumbing for NeurHL-G (PLAN_NeurHL4 D, LIVE)

Run everything from the repo root with
`uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow --with requests python ...`.

| Module | What it does | CLI |
|---|---|---|
| `fetch_rosters.py` | Pulls the 32 rosters from `api-web.nhle.com/v1/roster/{TEAM}/20262027` (browser UA, 0.5 s between calls). Writes `data/raw/rosters/<date>/nhl_roster_{TEAM}.json`, `rosters.csv` (team, player_id, first, last, pos C/L/R/D/G, sweater, birthdate) and `moves_vs_<prev>.csv` (added/removed against the previous dated snapshot, or against the frozen August files, labelled `2026-08`). The August files `data/raw/nhl_roster_*_20262027.json` are read only. | `python neurhl/live/fetch_rosters.py [--date YYYY-MM-DD] [--offline]` |
| `status.py` | Reads `data/manual/player_status_2027.csv`. `unavailable(date)` returns the player ids whose latest entry with `effective_from <= date` is SUSPENDED_NOT_REPORTING, IR, LTIR, WAIVED or HOLDOUT. A later ACTIVE entry clears the player. | `python neurhl/live/status.py [YYYY-MM-DD]` |
| `parse_dailyfaceoff.py` | Parses the daily DailyFaceoff snapshots in `data/raw/lineup_snapshots/<date>/` through their `__NEXT_DATA__` JSON. `parse_lines` (32 teams: EV f1-f4/d1-d3/g, pp1-2, pk1-2, ir), `parse_injuries` (injury news feed, with NHL ids), `parse_goalies` (starting goalies; empty on days without regular-season games; fails soft on schema drift) and `check_lines` (1 g1, >= 12 EV F, >= 6 EV D per team). `SLUG_TO_NHL` is the explicit slug-to-NHL-abbreviation table. | `python neurhl/live/parse_dailyfaceoff.py [--date D] [--csv OUTDIR]` |
| `id_map.py` | `map_ids(lines, rosters)` maps DailyFaceoff rows to NHL ids. The order is: alias (`data/manual/df_alias.csv`), then (team, jersey) with a name check, then exact normalized name on the team, then a league-unique name, then fuzzy first-name matching. Players missing from the current roster fall back to the August files and then to `data/raw/mp_lookup.csv`. Returns the mapped frame (`player_id, match_method, roster_team, roster_name, on_roster`) and the unmatched rows. | `python neurhl/live/id_map.py [--date D] [--rosters D] [--out CSV]` |
| `lineup_resolver.py` | `resolve(game_id, date, home, away, snapshot_dir=None)` returns, per side, 12 F + 6 D NHL ids (`forwards`, `defense`, `skaters`), one starting `goalie`, `lineup_source` (NHL_API, DF_CONFIRMED, DF_PROJECTED, FALLBACK), `goalie_source` (NHL_API, DF_CONFIRMED, DF_LIKELY, DF_LINES, FALLBACK) and `notes` (why each step was or was not used). Priority: the game's boxscore once it lists the lineup; then the latest DailyFaceoff lines snapshot for the team updated within 36 h of the start (goalie from the starting-goalies feed if Confirmed/Likely, else DailyFaceoff g1/g2); then FALLBACK (skaters dressed in the team's last 2026-27 game, or for game 1 the latest roster snapshot ranked by 2025-26 TOI, minus `status.unavailable(date)` and DailyFaceoff IR/out players, topped up by prior TOI; goalie by 2026-27 starts to date, prior-season starts for game 1). Never raises; degrades to FALLBACK. | `python neurhl/live/lineup_resolver.py [--date D] [--game GID] [--snapshot-dir DIR] [--rosters D] [--no-api] [--json OUT] [--ids]` |
| `publish.sh` | Publishes files through the bot clone (default `/Users/ph/Development/nhl-2026-2027-models-live`): `git pull --rebase`, copies the given repo-relative files/dirs, stages only `neurhl/output/live/`, `data/manual/`, `data/raw/rosters/`, `docs/`, commits with a plain message (`PUBLISH_MSG`; trailers/attribution are refused, and undone if a hook adds one), pushes to main with 3 retries, then checks with `git ls-remote` that the commit is on origin/main. Last stdout line is the sha; nonzero exit on any failure (2 bad path, 3 clone/pull, 4 commit, 5 push, 6 not on origin). | `neurhl/live/publish.sh [BOT_CLONE] FILE...` |
| `deadline.py` | `check(start_utc, sha)` via `gh api`: committer date (`.commit.committer.date`) before the start, commit on main (compare API), and the earliest `stamp.yml` run for the sha (server-side time). `precedes(start, sha)` is the plain committer-date test. For the acceptance tests. | `python neurhl/live/deadline.py --sha SHA (--start ISO \| --game GID) [--require-stamp]` (exit 0/1/2) |
| `forecast.py` | PLACEHOLDER entry point for `--mode morning` / `--mode pregame`; only logs. | `python neurhl/live/forecast.py --mode M` |
| `ingest_2027.py` | Nightly ingest of completed 2026-27 regular-season games into the `*_2027` tables that `data/build_g_state.py` and `sim/g_live.py` read (see "Nightly ingest" below). | `python neurhl/live/ingest_2027.py [--no-fetch] [--force] [--through D]` |
| `test_ingest_parity.py` | Sandbox parity and dry-run tests for the ingest (see below). Never writes to the real tables. | `python neurhl/live/test_ingest_parity.py {dryrun\|parity\|parity2013\|live\|all}` |
| `run_morning.sh`, `run_nightly.sh` | launchd wrappers: morning calls `forecast.py --mode morning`; nightly runs `score_live_2027.py`, `ingest_2027.py`, then `build_g_state.py` (see below). | |

Intraday snapshots: `python neurhl/data/fetch/snapshot_lineups.py --intraday` writes
`data/raw/lineup_snapshots/<date>/<HHMM>/` (local time) with `df_goalies.html.gz` and the
`df_lines_<slug>.html.gz` pages of the teams playing that date only (teams from
`api-web.nhle.com/v1/score/<date>`, else the local schedule files). It never skips and writes
nothing on days without games. The default daily mode (no flag, launchd `com.neurhl.snapshot`
at 17:30) is unchanged. `parse_dailyfaceoff.parse_lines(dir)` reads one directory
non-recursively, so the daily directory and each intraday subdirectory parse on their own; the
resolver picks the newest file for the team across both, taken no later than min(now, start)
(intraday time from the directory name, daily time from the file mtime).

## Scheduling (launchd)

Plists live in `neurhl/live/launchd/`; `install.sh` copies them to `~/Library/LaunchAgents`
and loads them (`launchctl bootstrap gui/<uid>`), `uninstall.sh` unloads and removes them.
Both take optional labels and never touch `com.neurhl.snapshot`. Times are the Mac's local
time (ET). Each job runs under `/usr/bin/caffeinate -i`, has PATH
`/opt/homebrew/bin:~/.local/bin:...` (uv is `/opt/homebrew/bin/uv`), runs from the repo root
and logs to `data/raw/lineup_snapshots/launchd_<name>.log`.

| Label | When | Runs |
|---|---|---|
| `com.neurhl.snapshot` (existing) | 17:30 daily | `snapshot_lineups.py` (daily mode) |
| `com.neurhl.intraday` | every 30 min, 09:00-23:00 | `snapshot_lineups.py --intraday` |
| `com.neurhl.morning` | 11:00 | `neurhl/live/run_morning.sh` -> `forecast.py --mode morning` |
| `com.neurhl.pregame` | every 10 min, 11:00-23:50 | `forecast.py --mode pregame` (uv with numpy, pandas<3, pyarrow, requests; add `--with` deps to the plist if forecast.py needs more) |
| `com.neurhl.nightly` | 04:30 | `neurhl/live/run_nightly.sh` (results, ingest, G state) |

Forecasts are computed in this repo (the resolver reads the gitignored
`neurhl/data/tensors/` and `data/raw/lineup_snapshots/`) and only the outputs go through the
bot clone with `publish.sh`.

Server-side evidence: `.github/workflows/stamp.yml` runs on every push to main that touches
`neurhl/output/live/**`, prints the run's UTC time, the head sha and the sha256 of each changed
file (also in the run summary). `deadline.py --require-stamp` uses that run's `created_at`.

## Nightly ingest (`run_nightly.sh`, `ingest_2027.py`)

`run_nightly.sh` (uv with numpy, pandas<3, pyarrow, requests, numba, scikit-learn==1.9.1, scipy;
scikit-learn is pinned to the version the frozen xG checkpoint was pickled with, and the ingest
refuses a mismatch)
runs three steps and logs to `data/raw/lineup_snapshots/launchd_nightly.log` (launchd
redirects there; run any other way, the output is also appended). It exits nonzero if any step
failed; a lock directory (`data/raw/lineup_snapshots/.nightly.lock`, with the holder's pid; a
stale lock is removed) stops two runs overlapping.

1. `neurhl/eval/score_live_2027.py`: completed games into `neurhl/output/live/results_2027.csv`
   plus the interim scorecard. If it fails, the ingest still runs on the cached file.
2. `neurhl/live/ingest_2027.py` (below).
3. `neurhl/data/build_g_state.py`, only when one of `games_ctx, player_games, usage,
   onice_rates, goalie_games, pgx, tgx` `_2027` is newer than `gst_tm_2027.parquet` (or that
   file is missing). It is skipped when the ingest fails. build_g_state rewrites `gst_*` for
   every season; rows of earlier seasons depend only on earlier games, so they do not change
   when 2027 is added (checked in the sandbox), unless an earlier-season input has been rebuilt
   since the last run.

`ingest_2027.py` builds, for season S = 2027 (`--season`), the same tables the historical
pipeline built for 2008-2026, by calling the builders' own functions:

| Table | Builder code used |
|---|---|
| `events_S` | `tensorize_events.tensorize_game` |
| `shifts_S`, `stints_S` | `build_shifts._json_game` (goalies from the game's `rosterSpots`), `build_stints._game` |
| `games_ctx_S`, `player_games_S` | `tensorize_games.roster_toi_one` + its `main()` body. Labels (goals, OT/SO) come from the final play-by-play, rest/travel from `sim/schedule_context.build` over the completed regular-season games (build_travel's rules), and the era vector from `tensorize_games.era_table` (the prior-season construction for a season not yet in games.csv) |
| `mp_shots_S` | `parse_mp_shots.parse_season` on MoneyPuck `shots_{S-1}.zip` (re-downloaded nightly, ingested games only) |
| `xg_shots_S` | frozen GBM + `train_xg.seq_calibrate` (below) |
| `goalie_games_S` | `build_goalie_games.build`, with the walk-forward carry replayed over 2008..S-1 |
| `stream_S`, `pgx_S`, `tgx_S` | `build_stream.build_season`, `build_xg_games.build_player/build_team` |
| `usage_S`, `onice_rates_S`, `absences_S` | `build_usage.build`, `build_onice_rates.main()` body, `build_absences.build` |
| `rr_S` | `build_right_rail.parse` over `data/raw/right_rail/S/` (fetched for each ingested game). `tgx_S.pp_opps` becomes the official count (as `build_right_rail.py` does) only if `tgx_{S-1}` carries `pp_opps_stint`, so `pp_opps` keeps one definition across seasons. Today `tgx_2026` does not |

**Walk-forward xG (A9).** The GBM for V = 2027 is fit once on seasons <= 2025 (train_xg's
`GBM` settings), with the isotonic seed from its 2026 predictions, and saved to
`neurhl/checkpoints/xg_live_v2027.pkl` (with a `.sha256` sidecar that is checked on every
load). It is never refit. Each night every 2027 shot is scored in `(date, game_id)` order by
`train_xg.seq_calibrate`, which refits the isotonic map every 25 league games on the trailing
120k out-of-sample shots strictly before the block. So a game's xG uses only earlier games,
and the values for games already ingested do not change as the season goes on. Freeze the
checkpoint before the season (`ingest_2027.py --freeze-xg-only`); the driver refuses to freeze
once 2027 shots exist. The reason: `models/xg.era_covariates` and `regime_covariates` sort the
one 2008 training game without a games_ctx date (2007020003) to the end of the covariate
table, so its covariates depend on where the table ends. Pre-season the table ends in 2026,
exactly as in `train_xg.main`.

**Readiness, deferral, idempotence.** The completed games are the rows of `results_2027.csv`.
A game is ingested once (a) its play-by-play has `gameState` OFF/FINAL, (b) its shift chart
covers >= 5.4 player-equivalents per side (2026 minimum: 5.68), and (c) MoneyPuck has >= 90% of
its pbp unblocked attempts (2026 minimum: 97%). An unmet condition defers the game for
`--max-defer-days` (2) after its date. After that it is ingested degraded with a WARNING line,
recorded in `neurhl/data/tensors/_ingest_2027/state.json`, and the missing source is re-checked
(shift charts re-fetched) on every later run. Raw files go to `data/raw/{pbp,shifts,right_rail}/2027/`
and `data/raw/mp_shots/shots_2026.zip`; nothing older is written. Per-game work runs only for
new or re-checked games and is merged into the season files. The season-level tables are then
rebuilt from those files (about 40 s at season end), so walk-forward quantities are always
recomputed in date order. With no new game and an unchanged MoneyPuck zip the run is a no-op
("no new 2027 games"). With no completed game at all it prints "no completed 2027 games" and
exits 0. If a run dies partway through, the next run redoes its games and the season tables
(`dirty`/`inflight` in state.json). In the real tensors directory the driver refuses to write
any season <= 2026 and any symlink.

**Tests** (`test_ingest_parity.py`, sandboxes under `neurhl/data/tensors/_sandbox_ingest/`,
symlinks to the real history, driver refuses to write through them):
`parity` rebuilds 2026 (regular season + playoffs) from the raw files and compares every
table with the real one. All 14 match (12 byte-identical including row order; `games_ctx`,
`tgx` differ only in row order). `parity2013` does the same for a right-rail season with the
excluded game 2012020660. `live` treats 2026 as the live season: 15 days, then 30 days
incrementally, then a no-op run. It compares the 30-day tables with the real 2026 rows of
those games (all equal, xG included), runs build_g_state in the sandbox, and checks the
`gst_*_2026` rows against the real ones. It also checks recovery from a crashed run. `dryrun`
checks the real 2027 path with an empty results file.

## Gotchas

- **Imports.** `common.py` puts `src/` first on `sys.path`, and `src/live.py` then shadows this package. Never write `import live.x`. Import it as `neurhl.live.x` from the repo root, or run the files as scripts. The modules handle both cases.
- **Current roster vs. DailyFaceoff.** The NHL roster endpoint drops injured players and players sent to the minors (for example Bedard on IR and Milic assigned). DailyFaceoff still lists them, so `on_roster=False` marks them. Camp rosters are inflated until the roster deadline (2026-09-28 17:00 ET).
- **DailyFaceoff fields.** `slot` is `positionIdentifier` without the line number (`lw`, `c`, `rw`, `ld`, `rd`, `g1`, `g2`, `sk1`-`sk5`, `ir1`...). The line comes from `group`. `updated_at` is the time DailyFaceoff last edited that team's lines, not the time of the snapshot. Pages can be stale: VGK still showed its July "2026 Offseason (Projected)" lines on 2026-09-25.
- **Short names.** DailyFaceoff's `sortedTeams` short names (LA, MON, NAS, NJ, SJ, TB, VEG, WAS) are not NHL codes. Key on the slug.
- **Starting goalies.** The schema was checked against the archived page `/starting-goalies/2026-04-09`. Archived pages seem to attach goalies to their current teams, so only the snapshots taken each day count as evidence.
- **Manual files.** Entries in `data/manual/player_status_2027.csv` change inputs only. Each one must be committed before the first prediction that uses it.
- **Lineup sources.** DailyFaceoff lines pages have no "confirmed" flag. DF_CONFIRMED is a heuristic (updated within 12 h of the start and a source name without "project", "offseason" or "camp"); check it against the real regular-season `sourceName` values once games start. The injury news feed is not used for exclusions (it is news, not a list); the lines pages' `ir` group and `out`/`ir` statuses are. Day-to-day players in the lines stay in.
- **Unavailable players.** `status.unavailable(date)` and DailyFaceoff IR players are removed on every path except NHL_API (the boxscore is the actual lineup). A goalie feed naming an unavailable goalie is ignored (Hellebuyck: WPG resolves to Skinner).
- **Automation needs.** The bot clone must exist (`git clone https://github.com/ph05/neurhl.git /Users/ph/Development/nhl-2026-2027-models-live`). Git pushes use the `osxkeychain` credential helper, and the login keychain locks on sleep and after 1 h idle, so a push from launchd can fail while it is locked; use a credential that does not need the keychain for the bot clone (a fine-grained token in that clone's own `credential.helper store`, or an SSH deploy key), or change the keychain settings. The Mac sleeps after 1 min idle: launchd runs a missed calendar job once on wake, so keep the machine awake during game days (`sudo pmset -c sleep 0`, or a `pmset repeat wake` before 09:00).

## Operating requirements for the 2026-27 season

The scheduled jobs (neurhl/live/launchd/) need three things from the machine,
which the operator provides (decided 2026-09-27):

1. **GitHub push credentials usable without a UI.** Git's credential helper is
   the macOS keychain; the login keychain locks after sleep and after an hour
   idle (`security show-keychain-info` shows `lock-on-sleep timeout=3600s`), and a
   push from launchd then fails with `failed to get: -25320`. Any of: keep the
   session unlocked on game days, give the bot clone its own credential, or
   relax the keychain lock settings.
2. **The Mac awake on game days** from 10:30 ET to the last pregame freeze
   (about 22:30 ET) and at 04:30 for the nightly ingest. `caffeinate -i` in each
   job only holds while that job runs; a job missed during sleep runs late on
   wake, and a pregame forecast that lands after puck drop counts as MISSED.
3. **Network access** for the NHL API, DailyFaceoff and MoneyPuck.

Known data risk: 2026-27 shot xG needs MoneyPuck's season shot file
(`shots_2026.zip`), which returns 404 until MoneyPuck publishes it (usually
within days of opening night). Until then the ingest defers each game up to
2 days, then ingests it without xG and flags it degraded; it is re-ingested
once the file appears. Forecasts are unaffected on opening night (state comes
from completed seasons).
