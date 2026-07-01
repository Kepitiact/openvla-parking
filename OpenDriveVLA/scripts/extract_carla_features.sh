#!/usr/bin/env bash
# Pre-compute UniAD BEV features for all CARLA frames.
# Must be run from the OpenDriveVLA/ directory.
# Output: data_carla/processed/uniad_features/<token>.pth for every frame.

set -e

CKPT_PATH=${1:-"../checkpoints/OpenDriveVLA-0.5B"}
NUM_WORKERS=${NUM_WORKERS:-0}

REPO_ROOT="$(cd .. && pwd)"
export CARLA_DATA_ROOT="${CARLA_DATA_ROOT:-${REPO_ROOT}/data_carla}"
export NUSC_DATA_ROOT="${NUSC_DATA_ROOT:-${REPO_ROOT}/data/nuscenes}"
export CACHED_DATA_PATH="${CACHED_DATA_PATH:-${CARLA_DATA_ROOT}/processed/cached_parking_info.pkl}"
export PYTHONPATH="$(pwd)/third_party/mmdetection3d_1_0_0rc6:${PYTHONPATH}"

# NOTE: the nvcc shim is handled inside drivevla/_bootstrap.py, conditional on a
# real nvcc being absent (so HAL's real CUDA toolkit is used as-is).

# Optional overrides for per-round DAgger extraction:
#   CONV      conversations json to stamp uniad_pth into (default: base)
#   ANN_FILE  infos pkl to iterate (default: the config's; use a dagger-only pkl
#             to extract just one round's tokens)
CONV=${CONV:-"${CARLA_DATA_ROOT}/processed/carla_conversations.json"}
EXTRACT_ARGS=(
  --model-path "${CKPT_PATH}"
  --uniad-config projects/configs/stage1_track_map/carla_parking.py
  --conversations "${CONV}"
  --out-dir "${CARLA_DATA_ROOT}/processed/uniad_features"
  --num-workers "${NUM_WORKERS}"
)
[[ -n "${ANN_FILE:-}" ]] && EXTRACT_ARGS+=(--ann-file "${ANN_FILE}") || true

python drivevla/extract_uniad_features.py "${EXTRACT_ARGS[@]}"
