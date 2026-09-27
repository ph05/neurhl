#!/bin/bash
# NeurHL LIVE: forecast wrapper for launchd (com.neurhl.morning / com.neurhl.pregame).
# usage: neurhl/live/run_forecast.sh {morning|pregame|preview} [extra forecast.py args]
# Loads every dependency the forecast path needs (NeurHL-G: torch, numba;
# NeurHL-H: scikit-learn, scipy) and logs start and exit. scikit-learn is
# pinned to the version run_nightly.sh uses.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
MODE="${1:-morning}"; shift || true
echo "[run_forecast $MODE $(date '+%Y-%m-%d %H:%M:%S %Z')] start"
uv run -q --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow \
  --with requests --with numba --with torch --with scipy --with scikit-learn==1.9.1 --with openpyxl \
  python neurhl/live/forecast.py --mode "$MODE" "$@"
rc=$?
echo "[run_forecast $MODE $(date '+%Y-%m-%d %H:%M:%S %Z')] exit $rc"
exit $rc
