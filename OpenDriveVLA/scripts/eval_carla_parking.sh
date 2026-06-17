#!/usr/bin/env bash
# Run OpenDriveVLA inference on the CARLA parking dataset.
# Usage: bash scripts/eval_carla_parking.sh [CKPT_PATH] [NUM_GPU]
# Defaults: CKPT_PATH=../checkpoints/OpenDriveVLA-0.5B  NUM_GPU=1
# Must be run from the OpenDriveVLA/ directory.

CKPT_PATH=${1:-"../checkpoints/OpenDriveVLA-0.5B"}
NUM_GPU=${2:-1}
# Optional 3rd arg: a conversations subset (e.g. a small reverse-check set).
# Defaults to the full dataset.
DATA_FILE=${3:-"../data_carla/processed/carla_conversations.json"}

# mmdet3d is not pip-installed in the venv, but the source tree works when on PYTHONPATH.
# Do NOT add third_party/mmcv_1_7_2 — the compiled ops come from the venv's mmcv package.
export PYTHONPATH="$(pwd)/third_party/mmdetection3d_1_0_0rc6:${PYTHONPATH}"

# Point the LLaVA dataset to the CARLA cache instead of the nuScenes one.
export CACHED_DATA_PATH="../data_carla/processed/cached_parking_info.pkl"

export INFERENCE_EXTRA_ARGS="--uniad-config projects/configs/stage1_track_map/carla_parking.py \
  --data ${DATA_FILE}"

bash scripts/eval_drivevla.sh "${CKPT_PATH}" "${NUM_GPU}"
