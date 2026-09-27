#!/bin/bash
# Install and load the NeurHL LIVE launchd agents (intraday, morning, pregame, nightly).
# Leaves the existing com.neurhl.snapshot agent (daily 17:30 snapshot) alone.
# usage: neurhl/live/launchd/install.sh [label ...]   (default: all four)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HOME/Library/LaunchAgents"
LABELS=("$@")
[ "${#LABELS[@]}" -gt 0 ] || LABELS=(com.neurhl.intraday com.neurhl.morning com.neurhl.pregame com.neurhl.nightly)
mkdir -p "$DEST"
for label in "${LABELS[@]}"; do
  [ "$label" != "com.neurhl.snapshot" ] || { echo "skip $label (managed separately)"; continue; }
  src="$HERE/$label.plist"
  [ -f "$src" ] || { echo "no such plist: $src" >&2; exit 1; }
  plutil -lint "$src" >/dev/null
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  cp "$src" "$DEST/$label.plist"
  launchctl bootstrap "gui/$(id -u)" "$DEST/$label.plist"
  echo "loaded $label"
done
launchctl list | grep com.neurhl || true
