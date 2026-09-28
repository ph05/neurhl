#!/bin/bash
# NeurHL LIVE: nightly job (launchd com.neurhl.nightly, 04:30 ET).
#   1. neurhl/eval/score_live_2027.py   completed 2026-27 results -> neurhl/output/live/results_2027.csv
#                                       (+ interim scorecard of the 2026-09-25 freeze), then
#      neurhl/eval/score_neurhl_1_0.py  the interim scorecard of the NeurHL 1.0 freeze
#   2. neurhl/live/ingest_2027.py       new completed games -> the *_2027 tables in neurhl/data/tensors/
#   3. neurhl/data/build_g_state.py     pre-game state (gst_*), only when a *_2027 input table is newer
#                                       than gst_tm_2027.parquet (or that file is missing)
#   4. git fetch origin main, then neurhl/eval/score_live_g_2027.py: the NeurHL-G live scorecard
#                                       (PLAN_NeurHL4 LIVE). Forecasts are committed by the publishing
#                                       clone, so the fetch brings their commits into origin/main here;
#                                       if it fails the scorer is skipped and the last scorecard stands
#   5. neurhl/site/build_site.py, then neurhl/live/publish.sh: results, scorecards and site, once
#                                       2026-27 games have been played
# Log: data/raw/lineup_snapshots/launchd_nightly.log (launchd redirects stdout/stderr there; run any
# other way, the output is also appended to it). Exit status is nonzero if any step failed.
# A failed results fetch does not stop the ingest (it then works from the cached results file);
# a failed ingest skips build_g_state.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
LOG="$REPO/data/raw/lineup_snapshots/launchd_nightly.log"
mkdir -p "$(dirname "$LOG")"
# under launchd (com.neurhl.nightly) the plist already sends stdout/stderr to $LOG
if [ "${XPC_SERVICE_NAME:-}" != "com.neurhl.nightly" ]; then
  exec > >(tee -a "$LOG") 2>&1
fi
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

LOCK="$REPO/data/raw/lineup_snapshots/.nightly.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  pid="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "[run_nightly $(ts)] another nightly run (pid $pid) holds $LOCK; exiting"
    exit 1
  fi
  echo "[run_nightly $(ts)] removing stale lock (pid ${pid:-?} not running)"
  rm -rf "$LOCK"; mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

UV=(uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow
    --with requests --with numba --with scikit-learn==1.9.1 --with scipy python)
# scikit-learn is pinned to the version neurhl/checkpoints/xg_live_v2027.pkl was frozen with
# (ingest_2027.py refuses a mismatch rather than score with a differently-unpickled model)
T="$REPO/neurhl/data/tensors"
status=0

step() {  # step NAME CMD...
  local name="$1"; shift
  echo "[run_nightly $(ts)] $name: start"
  local t0=$SECONDS
  "$@"
  local rc=$?
  echo "[run_nightly $(ts)] $name: exit $rc ($((SECONDS - t0))s)"
  return $rc
}

echo "[run_nightly $(ts)] start ($REPO)"

step "score_live_2027" "${UV[@]}" neurhl/eval/score_live_2027.py || status=1
step "score_neurhl_1_0" "${UV[@]}" neurhl/eval/score_neurhl_1_0.py || status=1

if step "ingest_2027" "${UV[@]}" neurhl/live/ingest_2027.py; then
  need=0
  if [ -f "$T/games_ctx_2027.parquet" ]; then
    if [ ! -f "$T/gst_tm_2027.parquet" ]; then
      need=1
    else
      for f in games_ctx player_games usage onice_rates goalie_games pgx tgx; do
        if [ "$T/${f}_2027.parquet" -nt "$T/gst_tm_2027.parquet" ]; then need=1; fi
      done
    fi
  fi
  if [ "$need" = 1 ]; then
    step "build_g_state" "${UV[@]}" neurhl/data/build_g_state.py || status=1
  else
    echo "[run_nightly $(ts)] build_g_state: skipped (no new 2027 inputs)"
  fi
else
  status=1
  echo "[run_nightly $(ts)] build_g_state: skipped (ingest failed)"
fi

# 4. NeurHL-G scorecard, after the ingest so its stat-sheet actuals are current
if step "fetch" git -C "$REPO" fetch --quiet origin main; then
  step "score_live_g_2027" "${UV[@]}" neurhl/eval/score_live_g_2027.py || status=1
else
  status=1
  echo "[run_nightly $(ts)] score_live_g_2027: skipped (fetch failed; a stale origin/main would mark forecasts MISSED)"
fi

# 5. publish results, scorecards and site once 2026-27 games have been played
RES="$REPO/neurhl/output/live/results_2027.csv"
if [ -f "$RES" ] && [ "$(wc -l < "$RES")" -gt 1 ]; then
  if step "build_site" "${UV[@]}" neurhl/site/build_site.py; then
    PUB=(neurhl/output/live/results_2027.csv neurhl/output/live/scorecard_2027.json)
    # publish.sh refuses a missing path, so the NeurHL-G scorecard goes only once it exists
    [ -f neurhl/output/live/scorecard_g_2027.json ] && PUB+=(neurhl/output/live/scorecard_g_2027.json)
    [ -f neurhl/output/live/scorecard_1_0_2027.json ] && PUB+=(neurhl/output/live/scorecard_1_0_2027.json)
    PUBLISH_MSG="live: results and scorecard through $(date -v-1d '+%Y-%m-%d')" \
      step "publish" bash neurhl/live/publish.sh - "${PUB[@]}" docs || status=1
  else
    status=1
  fi
else
  echo "[run_nightly $(ts)] publish: skipped (no completed 2026-27 games)"
fi

echo "[run_nightly $(ts)] exit $status"
exit $status
