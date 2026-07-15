#!/bin/bash
# SMOKE TEST for the training path -- the first time a REAL batch runs end to end.
#
# V4/V7 only proved a synthetic step. This drives the real thing: the collator, loading
# UniAD .pth features off staging, the align-stage freeze, a real forward+backward on real
# batch shapes, DDP across 2 ranks, and a checkpoint save. --max-steps 30 stops it in a few
# minutes. Trajectory-only (REASONING_TRACES unset) so it needs nothing from the teacher run
# still generating on zprod -- it isolates the expensive, risky plumbing. The reason+gate
# path gets its own smoke once v1_uniad_epoch6/traces.jsonl lands, before the full finetune.
#
# ztest + gpu:2, so it runs in parallel with the teacher shards on zprod (no contention).
#
# Submit:
#   sbatch scripts/hal_train_smoke.sbatch
# PASS = it reaches "--max-steps 30 reached: smoke test done" with a finite, decreasing loss
# and writes a checkpoint, with no collator/feature/gate crash.
#SBATCH --job-name=train_smoke
#SBATCH --partition=ztest
#SBATCH --gres=gpu:2
#SBATCH --time=00:45:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/train_smoke_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking
PROC=$REPO/data_carla/processed

UNIAD_CKPT=$REPO/checkpoints/stage1_carla_full/epoch_6.pth
MODEL=$REPO/checkpoints/DriveVLA-Qwen2.5-3B-init
OUT=$REPO/checkpoints/DriveVLA-3B-smoke

# Trajectory-only: REASONING_TRACES unset, gate off. This smoke is about the plumbing, not
# the reasoning path.
echo "[smoke] align stage, trajectory-only, 30 steps, 2 GPUs"

srun singularity exec --nv \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO/OpenDriveVLA" \
  --env PYTHONPATH="$REPO/OpenDriveVLA" \
  --env UNIAD_CKPT="$UNIAD_CKPT" \
  --env CACHED_DATA_PATH="$PROC/cached_parking_info.pkl" \
  --env CARLA_DATA_ROOT="$STAGING/data_carla" \
  --env NUSC_DATA_ROOT="$REPO/data/nuscenes" \
  --env TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  torchrun --nproc_per_node=2 --master_port=29507 \
    drivevla/train_drivevla.py \
      --model-path "$MODEL" \
      --stage align \
      --conversations "$PROC/carla_conversations_train.json" \
      --output-dir "$OUT" \
      --num-epochs 1 --max-steps 30 \
      --bf16 --distributed \
      --grad-accum 8 --batch-size 1 --num-workers 4 --save-every 1

echo "[smoke] done -> $OUT"
