# neurhl/live: roster and lineup plumbing for NeurHL-G (PLAN_NeurHL4 D, LIVE)

Run everything from the repo root with
`uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow --with requests python ...`.

| Module | What it does | CLI |
|---|---|---|
| `fetch_rosters.py` | Pulls the 32 rosters from `api-web.nhle.com/v1/roster/{TEAM}/20262027` (browser UA, 0.5 s between calls). Writes `data/raw/rosters/<date>/nhl_roster_{TEAM}.json`, `rosters.csv` (team, player_id, first, last, pos C/L/R/D/G, sweater, birthdate) and `moves_vs_<prev>.csv` (added/removed against the previous dated snapshot, or against the frozen August files, labelled `2026-08`). The August files `data/raw/nhl_roster_*_20262027.json` are read only. | `python neurhl/live/fetch_rosters.py [--date YYYY-MM-DD] [--offline]` |
| `status.py` | Reads `data/manual/player_status_2027.csv`. `unavailable(date)` returns the player ids whose latest entry with `effective_from <= date` is SUSPENDED_NOT_REPORTING, IR, LTIR, WAIVED or HOLDOUT. A later ACTIVE entry clears the player. | `python neurhl/live/status.py [YYYY-MM-DD]` |
| `parse_dailyfaceoff.py` | Parses the daily DailyFaceoff snapshots in `data/raw/lineup_snapshots/<date>/` through their `__NEXT_DATA__` JSON. `parse_lines` (32 teams: EV f1-f4/d1-d3/g, pp1-2, pk1-2, ir), `parse_injuries` (injury news feed, with NHL ids), `parse_goalies` (starting goalies; empty on days without regular-season games; fails soft on schema drift) and `check_lines` (1 g1, >= 12 EV F, >= 6 EV D per team). `SLUG_TO_NHL` is the explicit slug-to-NHL-abbreviation table. | `python neurhl/live/parse_dailyfaceoff.py [--date D] [--csv OUTDIR]` |
| `id_map.py` | `map_ids(lines, rosters)` maps DailyFaceoff rows to NHL ids. The order is: alias (`data/manual/df_alias.csv`), then (team, jersey) with a name check, then exact normalized name on the team, then a league-unique name, then fuzzy first-name matching. Players missing from the current roster fall back to the August files and then to `data/raw/mp_lookup.csv`. Returns the mapped frame (`player_id, match_method, roster_team, roster_name, on_roster`) and the unmatched rows. | `python neurhl/live/id_map.py [--date D] [--rosters D] [--out CSV]` |

## Gotchas

- **Imports.** `common.py` puts `src/` first on `sys.path`, and `src/live.py` then shadows this package. Never write `import live.x`. Import it as `neurhl.live.x` from the repo root, or run the files as scripts. The modules handle both cases.
- **Current roster vs. DailyFaceoff.** The NHL roster endpoint drops injured players and players sent to the minors (for example Bedard on IR and Milic assigned). DailyFaceoff still lists them, so `on_roster=False` marks them. Camp rosters are inflated until the roster deadline (2026-09-28 17:00 ET).
- **DailyFaceoff fields.** `slot` is `positionIdentifier` without the line number (`lw`, `c`, `rw`, `ld`, `rd`, `g1`, `g2`, `sk1`-`sk5`, `ir1`...). The line comes from `group`. `updated_at` is the time DailyFaceoff last edited that team's lines, not the time of the snapshot. Pages can be stale: VGK still showed its July "2026 Offseason (Projected)" lines on 2026-09-25.
- **Short names.** DailyFaceoff's `sortedTeams` short names (LA, MON, NAS, NJ, SJ, TB, VEG, WAS) are not NHL codes. Key on the slug.
- **Starting goalies.** The schema was checked against the archived page `/starting-goalies/2026-04-09`. Archived pages seem to attach goalies to their current teams, so only the snapshots taken each day count as evidence.
- **Manual files.** Entries in `data/manual/player_status_2027.csv` change inputs only. Each one must be committed before the first prediction that uses it.
