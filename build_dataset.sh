#!/usr/bin/env bash
# Stage 1: build the canonical dataset from the infos pkl.
#   infos pkl -> NuScenes DB (v1.0-carla) -> ego-state cache -> conversations
#
# Stage 0 (raw episodes -> infos pkl) lives in the sibling data-gen harness and
# is NOT run here:
#   cd ~/projects/parking_data_gen && source venv/bin/activate
#   python scripts/build_infos_pkl.py \
#     --raw_dir  <repo>/data_carla/raw \
#     --out      <repo>/data_carla/processed/parking_infos_temporal.pkl
#
# Steps 2-4 default all paths via scripts/paths.py (no args needed).
# Run from the repo root in the model venv.

set -euo pipefail
cd "$(dirname "$0")"

echo "=== [1/3] NuScenes DB tables (v1.0-carla) + absolute camera paths ==="
python scripts/build_carla_nusc_tables.py

echo "=== [2/3] Ego-state cache (speeds, history, reverse-aware command) ==="
python scripts/generate_cached_nuscenes_info.py

echo "=== [3/3] Conversations (per-frame token refs; content filled at train time) ==="
python scripts/build_carla_conversations.py

echo "=== Dataset build complete. Next: bash run_pipeline.sh ==="
