#!/usr/bin/env bash
# Pre-compute UniAD features + decoded detections for CARLA frames.
# Must be run from the OpenDriveVLA/ directory.
# Output: <CARLA_DATA_ROOT>/processed/uniad_features/<token>.pth for every frame.
#
# UniAD is now built DIRECTLY from (config, checkpoint) — no LLaVA wrapper, no
# OpenDriveVLA-0.5B, no tokenizer, no cached_parking_info.pkl. See
# drivevla/extract_uniad_features.py for why (the wrapper's baked nuScenes vision-tower
# weights silently clobbered the CARLA checkpoint).
#
# Usage:
#   UNIAD_CKPT=/abs/path/to/uniad_carla_trained.pth bash scripts/extract_carla_features.sh
#   # subset:
#   UNIAD_CKPT=... ANN_FILE=../data_carla/processed/subset.pkl bash scripts/extract_carla_features.sh

set -e

REPO_ROOT="$(cd .. && pwd)"
export CARLA_DATA_ROOT="${CARLA_DATA_ROOT:-${REPO_ROOT}/data_carla}"
export NUSC_DATA_ROOT="${NUSC_DATA_ROOT:-${REPO_ROOT}/data/nuscenes}"
export PYTHONPATH="$(pwd)/third_party/mmdetection3d_1_0_0rc6:$(pwd):${PYTHONPATH}"

# The trained CARLA UniAD checkpoint. REQUIRED and never defaulted: the filename
# `uniad_base_track_map.pth` used to name TWO different models (the nuScenes warm-start
# and the CARLA-trained one), and picking the wrong one silently reverts the detector to
# nuScenes weights — the exact bug this refactor exists to kill.
if [[ -z "${UNIAD_CKPT:-}" ]]; then
  echo "ERROR: set UNIAD_CKPT to the trained CARLA UniAD checkpoint (absolute path)." >&2
  echo "  e.g. UNIAD_CKPT=\$(pwd)/checkpoints/uniad_carla_trained.pth bash scripts/extract_carla_features.sh" >&2
  exit 1
fi

# CONFIG must be the config UniAD was TRAINED with — it supplies the BEV geometry as
# well as the model. (The vision tower asserts this too.)
CONFIG=${CONFIG:-projects/configs/stage1_track_map/carla_parking_stage1.py}
CONV=${CONV:-"${CARLA_DATA_ROOT}/processed/carla_conversations.json"}

EXTRACT_ARGS=(
  --config "${CONFIG}"
  --checkpoint "${UNIAD_CKPT}"
  --conversations "${CONV}"
  --out-dir "${CARLA_DATA_ROOT}/processed/uniad_features"
  --num-workers "${NUM_WORKERS:-1}"
)
[[ -n "${ANN_FILE:-}" ]] && EXTRACT_ARGS+=(--ann-file "${ANN_FILE}") || true
[[ -n "${MAX_SAMPLES:-}" ]] && EXTRACT_ARGS+=(--max-samples "${MAX_SAMPLES}") || true

python drivevla/extract_uniad_features.py "${EXTRACT_ARGS[@]}"
