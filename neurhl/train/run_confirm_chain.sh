#!/usr/bin/env bash
# NeurHL-H confirmation chain (PLAN_NeurHL amendment A5).
#
# Walk-forward artifacts for the one-shot 2018-2026 confirmation:
#   1. event-LM snapshots v2019..v2026, each warm-started from the previous
#      vantage (PLAN_NeurHL P2: restatement vantages fine-tune forward one
#      season at a time), then the career encoder and the blended player
#      embeddings for the same vantage;
#   2. Layer 1 (player model) for every predict-season 2010..2026, rebuilt
#      from the current tensors so the tune-window re-derivation and the
#      confirmation share one consistent artifact set.
#
# Resumable: every step is skipped when its output already exists. No game
# outcome is scored here; scoring is eval/restate_hier.py, which runs once.
#
# Usage: nohup caffeinate -i bash neurhl/train/run_confirm_chain.sh > chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=(uv run --no-project --python 3.12 --with torch --with numpy
    --with "pandas<3" --with pyarrow)
CK=neurhl/checkpoints
TN=neurhl/data/tensors

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

for V in $(seq 2019 2026); do
  if [ ! -f "$CK/event_lm_v$V.json" ]; then
    echo "[$(stamp)] event-LM v$V (init from v$((V - 1)))"
    "${UV[@]}" python neurhl/train/pretrain_events.py --vantage "$V" \
      --init-from "$CK/event_lm_v$((V - 1)).pt"
  fi
  if [ ! -f "$CK/career_v$V.pt" ]; then
    echo "[$(stamp)] career encoder v$V"
    "${UV[@]}" python neurhl/train/train_career.py --vantage "$V"
  fi
  if [ ! -f "$TN/embeddings_v$V.npz" ]; then
    echo "[$(stamp)] embeddings v$V"
    "${UV[@]}" python neurhl/train/build_embeddings.py --vantage "$V"
  fi
done

for T in $(seq 2018 2026) $(seq 2010 2017); do
  if [ ! -f "$TN/proj_team_$T.parquet" ]; then
    echo "[$(stamp)] Layer 1, predict-season $T"
    "${UV[@]}" python neurhl/train/train_player.py --predict-season "$T"
  fi
done

echo "[$(stamp)] chain complete"
