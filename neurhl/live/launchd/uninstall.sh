#!/bin/bash
# Unload and remove the NeurHL LIVE launchd agents. Never touches com.neurhl.snapshot.
# usage: neurhl/live/launchd/uninstall.sh [label ...]   (default: all four)
set -uo pipefail
DEST="$HOME/Library/LaunchAgents"
LABELS=("$@")
[ "${#LABELS[@]}" -gt 0 ] || LABELS=(com.neurhl.intraday com.neurhl.morning com.neurhl.pregame com.neurhl.nightly)
for label in "${LABELS[@]}"; do
  [ "$label" != "com.neurhl.snapshot" ] || { echo "skip $label"; continue; }
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null && echo "unloaded $label" || echo "$label was not loaded"
  rm -f "$DEST/$label.plist"
done
launchctl list | grep com.neurhl || true
