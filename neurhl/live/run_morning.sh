#!/bin/bash
# NeurHL LIVE: morning forecast (launchd com.neurhl.morning, 11:00 ET).
# Stub wrapper: runs the forecast entry point in morning mode. forecast.py is a placeholder
# until the lead implements it (it will also call neurhl/live/publish.sh).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
echo "[run_morning $(date '+%Y-%m-%d %H:%M:%S %Z')] start"
uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow --with requests \
  python neurhl/live/forecast.py --mode morning
rc=$?
echo "[run_morning $(date '+%Y-%m-%d %H:%M:%S %Z')] exit $rc"
exit $rc
