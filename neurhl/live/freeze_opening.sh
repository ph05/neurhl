#!/bin/bash
# NeurHL: post-deadline freeze for opening night (run after 2026-09-28 17:00 ET).
#
#   1. dated roster snapshot (the rosters teams filed at the deadline)
#   2. refreshed 1.0 preseason files + PLAN_NeurHL_LIVE_U1_20260928.md (hashes)
#   3. a fresh DailyFaceoff snapshot (lines posted after the deadline)
#   4. NeurHL-G goal-level calibration m0 frozen on final rosters (PLAN_NeurHL4 A1)
#   5. opening-night preview forecasts for 2026-09-29 (descriptive; the scored
#      morning and pregame forecasts follow on game day)
#   6. NeurHL 1.0 (PLAN_NeurHL_1_0.md): the unified season projection on the final
#      rosters, its consistency check (must pass, and must agree with the preview
#      on opening night), and the FREEZE record with hashes
#   7. site rebuild, acceptance batteries, one commit
#   8. push (needs GitHub credentials usable without a UI; see neurhl/live/README.md)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
UV=(uv run -q --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow
    --with numba --with torch --with scipy --with scikit-learn==1.9.1 --with requests --with openpyxl)
DATE=${FREEZE_DATE:-2026-09-28}
DRY=${DRY_RUN:-0}          # 1: rehearsal, restores the plan and stages, commits and pushes nothing
log() { echo "[freeze_opening $(date '+%H:%M:%S')] $*"; }

log "1. roster snapshot $DATE"
if [ "$DRY" = 1 ] && [ -f "data/raw/rosters/$DATE/rosters.csv" ]; then
  log "dry run: using the existing snapshot (a committed snapshot is never re-fetched)"
else
  "${UV[@]}" python neurhl/live/fetch_rosters.py --date "$DATE" || exit 1
fi
"${UV[@]}" python -c "
import pandas as pd
r = pd.read_csv('data/raw/rosters/$DATE/rosters.csv')
n = r.groupby('team').size()
print('roster sizes: min', n.min(), 'max', n.max(), '| over 23:', ', '.join(f'{t} {k}' for t, k in n[n > 23].items()) or 'none')
"
log "2. dated update U1 of the 2026-09-25 files"
"${UV[@]}" python neurhl/live/refresh_preseason.py --date "$DATE" || exit 1
log "3. DailyFaceoff snapshot"
"${UV[@]}" python neurhl/data/fetch/snapshot_lineups.py || log "snapshot failed (continuing with the latest one)"
log "4. goal-level calibration m0"
"${UV[@]}" python neurhl/live/goal_calibration.py --freeze || exit 1
log "5. preview forecasts for 2026-09-29 (the check compares them with NeurHL 1.0)"
"${UV[@]}" python neurhl/live/forecast.py --mode preview --date 2026-09-29 --dry-run || exit 1
log "6. NeurHL 1.0: unified season projection, consistency check, freeze record"
U1D=neurhl/output/neurhl_1_0
rm -f "$U1D"/games_2027.csv "$U1D"/teams_2027.csv "$U1D"/skaters_2027.csv "$U1D"/goalies_2027.csv \
      "$U1D"/player_games_2027.csv.gz "$U1D"/consistency_2027.json "$U1D"/checks_2027.json "$U1D"/run_2027.json \
      "$U1D"/team_points_quantiles_2027.csv
"${UV[@]}" python neurhl/data/build_player_rates.py --validate || exit 1
"${UV[@]}" python neurhl/sim/availability_2027.py --rosters-date "$DATE" --summary > /dev/null || exit 1
"${UV[@]}" python neurhl/tests/test_unified_canonical.py --rosters-date "$DATE" | tail -1 | grep -q PASS || { log "canonical assembly differs from the live path"; exit 1; }
"${UV[@]}" python neurhl/sim/unified_2027.py --rosters-date "$DATE" --draws 64 --sims 20000 || exit 1
"${UV[@]}" python neurhl/tests/check_neurhl_1_0.py || exit 1
[ "$DRY" = 1 ] && cp PLAN_NeurHL_1_0.md "${TMPDIR:-/tmp}/plan_neurhl_1_0_backup.md"
"${UV[@]}" python neurhl/sim/freeze_1_0.py || exit 1
if [ "$DRY" = 1 ]; then
  tail -12 PLAN_NeurHL_1_0.md
  cp "${TMPDIR:-/tmp}/plan_neurhl_1_0_backup.md" PLAN_NeurHL_1_0.md && log "dry run: plan restored"
fi
log "7. site, batteries, commit"
"${UV[@]}" python neurhl/site/build_site.py || exit 1
for t in review_tests_neurhl3 review_tests_neurhl4; do
  "${UV[@]}" python "neurhl/tests/$t.py" | tail -1
done
if [ "$DRY" = 1 ]; then
  log "dry run complete: nothing staged, committed or pushed"
  exit 0
fi
# one path at a time: a single missing path must not leave the commit empty
for p in data/raw/rosters/"$DATE" neurhl/output/*_20260928.csv PLAN_NeurHL_LIVE_U1_20260928.md \
         neurhl/configs/playoff_tiebreak_check_20260928.json neurhl/configs/live_goal_calibration.json \
         neurhl/output/live/2027 docs neurhl/configs/acceptance_neurhl3.json \
         neurhl/configs/acceptance_neurhl4.json neurhl/output/live/scorecard_2027.json \
         "$U1D" PLAN_NeurHL_1_0.md README.md EVIDENCE.md; do
  if [ -e "$p" ]; then git add -- "$p"; else log "not found, not staged: $p"; fi
done
git commit -q -m "Opening-night freeze: NeurHL 1.0 predictions for 2026-27 on the post-deadline rosters, dated update U1, goal calibration m0, preview forecasts" \
  && log "committed $(git rev-parse --short HEAD)"
log "8. push"
git push -q origin main && log "pushed" || log "PUSH FAILED: unlock the keychain (or configure credentials) and run: git push origin main"
