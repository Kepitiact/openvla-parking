#!/usr/bin/env bash
# Fine-tune OpenDriveVLA on CARLA parking data.
# Must be run from the OpenDriveVLA/ directory.
#
# Prerequisites:
#   1. Run bash scripts/extract_carla_features.sh  (once per dataset)
#   2. data_carla/processed/carla_conversations.json has uniad_pth paths
#
# Outputs: checkpoints + train.log saved to ../checkpoints/OpenDriveVLA-0.5B-carla/
# After training, merge LoRA and run inference:
#   python drivevla/merge_lora.py --base ... --lora .../epoch_010 --out .../merged
#   bash scripts/eval_carla_parking.sh .../merged
#
# Env knobs (all optional):
#   NUM_WORKERS   dataloader workers            (default 4)
#   BATCH_SIZE    micro-batch per GPU           (default 1; UniAD requires 1)
#   GRAD_ACCUM    gradient accumulation steps   (default 8)
#   SEED          RNG seed                      (default 42)
#   BF16          1 to use the bf16 path        (default off -> fp16)
#   RESUME        1 to resume from saved state  (default off)
#   NPROC         GPUs for torchrun multi-GPU   (default 1 -> single-GPU path)

set -e

CKPT_PATH=${1:-"../checkpoints/OpenDriveVLA-0.5B"}
NUM_EPOCHS=${2:-10}
# Optional 3rd arg: output dir (use a distinct one to avoid overwriting a prior run).
OUTPUT_DIR=${3:-"../checkpoints/OpenDriveVLA-0.5B-carla"}
# Optional 4th arg: learning rate. Warm-start/DAgger refinement runs use a lower
# LR than the original 2e-4 fit (additive refinement, not a fresh train).
LR=${4:-"2e-4"}

NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SAVE_STEPS=${SAVE_STEPS:-0}
SEED=${SEED:-42}
NPROC=${NPROC:-1}

# Portable data roots consumed by the UniAD config (carla_parking.py). Derived
# from the repo root (parent of the OpenDriveVLA/ dir this runs from); override
# by exporting them before calling.
REPO_ROOT="$(cd .. && pwd)"
export CARLA_DATA_ROOT="${CARLA_DATA_ROOT:-${REPO_ROOT}/data_carla}"
export NUSC_DATA_ROOT="${NUSC_DATA_ROOT:-${REPO_ROOT}/data/nuscenes}"
export CACHED_DATA_PATH="${CACHED_DATA_PATH:-${CARLA_DATA_ROOT}/processed/cached_parking_info.pkl}"
export PYTHONPATH="$(pwd)/third_party/mmdetection3d_1_0_0rc6:${PYTHONPATH}"

# Conversations source: a single json, a {base,dagger_r1,...}.json glob, or a
# yaml manifest (non-destructive DAgger rounds). Override with CONV=...
CONV=${CONV:-"${CARLA_DATA_ROOT}/processed/carla_conversations.json"}

# Reduce allocator fragmentation — important on the shared 8GB desktop GPU
# (~2.2GB is taken by the display, leaving ~5.4GB for training).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Unbuffered stdout so progress/loss lines appear live instead of being held in
# Python's block buffer.
export PYTHONUNBUFFERED=1

# NOTE: the nvcc shim is now handled inside drivevla/_bootstrap.py, conditional
# on a real nvcc being absent (so HAL's real CUDA toolkit is used as-is).

TRAIN_ARGS=(
  --model-path "${CKPT_PATH}"
  --uniad-config projects/configs/stage1_track_map/carla_parking.py
  --conversations "${CONV}"
  --output-dir "${OUTPUT_DIR}"
  --num-epochs "${NUM_EPOCHS}"
  --lr "${LR}"
  --lora-rank 64
  --lora-alpha 128
  --grad-accum "${GRAD_ACCUM}"
  --save-every 1
  --save-steps "${SAVE_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
)
[[ "${BF16:-0}" == "1" ]] && TRAIN_ARGS+=(--bf16)
[[ "${RESUME:-0}" == "1" ]] && TRAIN_ARGS+=(--resume)

if [[ "${NPROC}" -gt 1 ]]; then
  echo ">>> Multi-GPU training via torchrun (nproc=${NPROC})"
  torchrun --nproc_per_node="${NPROC}" drivevla/train_drivevla.py "${TRAIN_ARGS[@]}" --distributed
else
  python drivevla/train_drivevla.py "${TRAIN_ARGS[@]}"
fi
