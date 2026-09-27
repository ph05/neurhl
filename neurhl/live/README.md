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
| `run_morning.sh`, `run_nightly.sh` | launchd wrappers: morning calls `forecast.py --mode morning`; nightly is a stub that logs. | |

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
| `com.neurhl.nightly` | 04:30 | `neurhl/live/run_nightly.sh` (stub) |

Forecasts are computed in this repo (the resolver reads the gitignored
`neurhl/data/tensors/` and `data/raw/lineup_snapshots/`) and only the outputs go through the
bot clone with `publish.sh`.

Server-side evidence: `.github/workflows/stamp.yml` runs on every push to main that touches
`neurhl/output/live/**`, prints the run's UTC time, the head sha and the sha256 of each changed
file (also in the run summary). `deadline.py --require-stamp` uses that run's `created_at`.

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
