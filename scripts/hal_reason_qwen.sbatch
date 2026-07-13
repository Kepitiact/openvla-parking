#!/bin/bash
# Reasoning-trace generation with the Qwen2.5-32B teacher, on a HAL GPU node, inside
# the Singularity container. --teacher qwen is fixed here; pass --source gt (GT-grounded
# bridge pass) OR --source uniad (real perception; also pass --uniad-features-dir).
#
# Submit from anywhere on HAL (args pass through to run_generate):
#   # GT-grounded pass:
#   sbatch scripts/hal_reason_qwen.sbatch \
#     --source gt \
#     --infos  <staging>/data_carla/processed/parking_infos_train.pkl \
#     --out    <staging>/data_carla/processed/reasoning/v1_qwen_gt --limit 200
#   # UniAD-grounded pass (after extract_uniad_features.py has written the .pth dir):
#   sbatch scripts/hal_reason_qwen.sbatch \
#     --source uniad --uniad-features-dir <staging>/data_carla/processed/uniad_features \
#     --infos  <staging>/data_carla/processed/parking_infos_train.pkl \
#     --out    <staging>/data_carla/processed/reasoning/v2_qwen_uniad
# Watch:
#   squeue --me ; tail -f /workspaces/s0002438/openvla-parking/reason_qwen_<jobid>.log
#SBATCH --job-name=reason_qwen
#SBATCH --partition=ztest
#SBATCH --gres=gpu:2
#SBATCH --time=00:45:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/reason_qwen_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking

# Qwen2.5-32B-Instruct weights live on staging (NEVER in the ~100 GB workspace quota).
# gpu:2 shards the fp16 32B across 2x40 GB via device_map=auto. To fit ONE 40 GB A100
# instead, set --gres=gpu:1 above and QWEN_4BIT=1 below (needs bitsandbytes in the SIF).
export QWEN_MODEL_PATH="$STAGING/models/Qwen2.5-32B-Instruct"
# export QWEN_4BIT=1

# Generated reasoning data MUST land on staging (workspace quota crashes running jobs);
# pass --out under $STAGING. --source gt --teacher qwen are fixed here; --infos/--out/
# --limit come from the sbatch args ("$@").
#
# --pwd $REPO       : cwd = repo root so `python -m reasoning_data_gen...` resolves
# PYTHONPATH        : repo root (reasoning_data_gen adds scripts/ to sys.path itself)
# LD_LIBRARY_PATH   : container libs first (kept for parity with the other HAL jobs)
srun singularity exec --nv \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO" \
  --env PYTHONPATH="$REPO" \
  --env QWEN_MODEL_PATH="$QWEN_MODEL_PATH" \
  --env QWEN_4BIT="${QWEN_4BIT:-}" \
  --env TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  python -m reasoning_data_gen.run_generate \
    --teacher qwen --qwen-model-path "$QWEN_MODEL_PATH" "$@"

# First fit-check: add --limit 50 to confirm the teacher loads + produces clean traces
# on ztest (45 min). For the full ~62k-frame run use a zprod partition (no time limit;
# checkpoint/resume-friendly since run_generate writes a self-contained versioned dir).
