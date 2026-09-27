#!/bin/bash
# NeurHL: post-deadline freeze for opening night (run after 2026-09-28 17:00 ET).
#
#   1. dated roster snapshot (the rosters teams filed at the deadline)
#   2. refreshed 1.0 preseason files + PLAN_NeurHL_LIVE_U1_20260928.md (hashes)
#   3. a fresh DailyFaceoff snapshot (lines posted after the deadline)
#   4. NeurHL-G goal-level calibration m0 frozen on final rosters (PLAN_NeurHL4 A1)
#   5. opening-night preview forecasts for 2026-09-29 (descriptive; the scored
#      morning and pregame forecasts follow on game day)
#   6. site rebuild, acceptance batteries, one commit
#   7. push (needs GitHub credentials usable without a UI; see neurhl/live/README.md)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
UV=(uv run -q --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow
    --with numba --with torch --with scipy --with scikit-learn --with requests --with openpyxl)
DATE=2026-09-28
log() { echo "[freeze_opening $(date '+%H:%M:%S')] $*"; }

log "1. roster snapshot $DATE"
"${UV[@]}" python neurhl/live/fetch_rosters.py --date "$DATE" || exit 1
log "2. refreshed 1.0 files"
"${UV[@]}" python neurhl/live/refresh_preseason.py --date "$DATE" || exit 1
log "3. DailyFaceoff snapshot"
"${UV[@]}" python neurhl/data/fetch/snapshot_lineups.py || log "snapshot failed (continuing with the latest one)"
log "4. goal-level calibration m0"
"${UV[@]}" python neurhl/live/goal_calibration.py --freeze || exit 1
log "5. preview forecasts for 2026-09-29"
"${UV[@]}" python neurhl/live/forecast.py --mode preview --date 2026-09-29 --dry-run || exit 1
log "6. site, batteries, commit"
"${UV[@]}" python neurhl/site/build_site.py || exit 1
for t in review_tests_neurhl3 review_tests_neurhl4; do
  "${UV[@]}" python "neurhl/tests/$t.py" | tail -1
done
git add data/raw/rosters/"$DATE" neurhl/output/*_20260928.csv PLAN_NeurHL_LIVE_U1_20260928.md \
        neurhl/configs/playoff_tiebreak_check_20260928.json neurhl/configs/live_goal_calibration.json \
        neurhl/output/live/2027 docs neurhl/configs/acceptance_neurhl3.json \
        neurhl/configs/acceptance_neurhl4.json 2>/dev/null
git commit -q -m "Opening-night freeze: post-deadline rosters, dated 1.0 update U1, goal calibration m0, preview forecasts" \
  && log "committed $(git rev-parse --short HEAD)"
log "7. push"
git push -q origin main && log "pushed" || log "PUSH FAILED: unlock the keychain (or configure credentials) and run: git push origin main"
