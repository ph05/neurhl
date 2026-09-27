#!/bin/bash
# NeurHL LIVE: nightly job (launchd com.neurhl.nightly, 04:30 ET).
# STUB: intended for results ingestion, scoring, roster refresh and publishing; not implemented yet.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
echo "[run_nightly $(date '+%Y-%m-%d %H:%M:%S %Z')] stub: nothing to do yet"
exit 0
