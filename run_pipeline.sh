#!/usr/bin/env bash
# Stage 2: the full fine-tune pipeline on the current dataset.
#   build (optional) -> UniAD feature extraction -> LoRA train -> merge LoRA
#
# Assumes the dataset is already built (bash build_dataset.sh). Pass BUILD=1 to
# run build_dataset.sh first.
#
# Env knobs (all optional):
#   BUILD=1        run build_dataset.sh first          (default off)
#   BASE           base checkpoint      (default ../checkpoints/OpenDriveVLA-0.5B)
#   OUT            output run dir        (default ../checkpoints/OpenDriveVLA-0.5B-carla)
#   EPOCHS         training epochs                      (default 3)
#   LR             learning rate                        (default 2e-4)
#   SKIP_EXTRACT=1 skip UniAD feature extraction        (default off)
#   NPROC          GPUs for multi-GPU torchrun          (default 1)
#   NUM_WORKERS, GRAD_ACCUM, SAVE_STEPS, BF16, RESUME   forwarded to training
#
# Run from the repo root in the model venv.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

BASE=${BASE:-"../checkpoints/OpenDriveVLA-0.5B"}
OUT=${OUT:-"../checkpoints/OpenDriveVLA-0.5B-carla"}
EPOCHS=${EPOCHS:-3}
LR=${LR:-"2e-4"}

if [[ "${BUILD:-0}" == "1" ]]; then
  bash build_dataset.sh
fi

cd OpenDriveVLA

if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
  echo "=== [1/3] UniAD feature extraction ==="
  bash scripts/extract_carla_features.sh "${BASE}"
else
  echo "=== [1/3] Skipping feature extraction (SKIP_EXTRACT=1) ==="
fi

echo "=== [2/3] LoRA fine-tune (${EPOCHS} epochs) ==="
# Training knobs (NPROC/NUM_WORKERS/GRAD_ACCUM/SAVE_STEPS/BF16/RESUME) are read
# from the environment by train_carla_parking.sh.
bash scripts/train_carla_parking.sh "${BASE}" "${EPOCHS}" "${OUT}" "${LR}"

echo "=== [3/3] Merge LoRA -> standalone model ==="
LAST_EPOCH=$(printf "epoch_%03d" "${EPOCHS}")
python drivevla/merge_lora.py --base "${BASE}" --lora "${OUT}/${LAST_EPOCH}" --out "${OUT}/merged"

echo "=== Pipeline complete. Merged model: ${OUT}/merged ==="
