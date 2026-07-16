#!/bin/bash
# Train the reasoning-VLA: our Qwen2.5-3B backbone + frozen UniAD.
#
# The model emits its reasoning BEFORE the trajectory:
#   <reason_start>...<reason_end><traj_start>[(x,y,h)x6]<traj_end>
# and (with REASON_GATE=1) the trajectory tokens cannot see the perception tokens, so the
# only path from perception to waypoints runs THROUGH the reasoning.
#
# Two stages -- run align first, then finetune from its output:
#   sbatch scripts/hal_train_vla.sbatch align
#   sbatch scripts/hal_train_vla.sbatch finetune
#
#   align     LLM frozen, no LoRA. Only the projectors and the two NEW reasoning-token
#             embeddings train, so UniAD's features learn to land in Qwen's residual
#             stream before any LLM weight is perturbed.
#   finetune  add LoRA on the attention; projectors + embeddings keep training.
#
# UniAD is ALWAYS frozen (it is not in the trainable groups) and is loaded ONLY from
# UNIAD_CKPT -- our backbone carries no baked tower weights, by construction.
#SBATCH --job-name=train_vla
#SBATCH --partition=zprod
# 4 GPUs, not 8: zprod is saturated and a full 8-GPU node had a ~3-day queue estimate,
# while 4 backfills onto a partially-used node almost immediately. ~2x wall time (align
# ~3.4h, finetune ~12h) is a good trade vs waiting days. Bump back to gpu:8 if zprod frees.
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/checkpoints/train_vla_%j.log

set -euo pipefail

STAGE=${1:?usage: sbatch scripts/hal_train_vla.sbatch <align|finetune> [epochs]}
EPOCHS=${2:-}

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking
PROC=$REPO/data_carla/processed

# The trained CARLA UniAD. Never defaulted to a bare filename: `uniad_base_track_map.pth`
# once named two different models, and picking the wrong one silently reverted the
# detector to nuScenes weights.
UNIAD_CKPT=${UNIAD_CKPT:-$REPO/checkpoints/stage1_carla_full/epoch_6.pth}

# The reasoning traces the model is trained to emit. Unset this and it trains
# trajectory-only (the pre-reasoning baseline) -- which is exactly the A/B we want.
REASONING_TRACES=${REASONING_TRACES:-$STAGING/data_carla/processed/reasoning/v1_uniad_epoch6/traces.jsonl}

# The reason gate. Since counterfactual training is deferred to v2, this is the ONLY
# mechanism forcing the trajectory to route through the reasoning. Set 0 for the A/B.
REASON_GATE=${REASON_GATE:-1}

case "$STAGE" in
  align)    MODEL=${MODEL:-$REPO/checkpoints/DriveVLA-Qwen2.5-3B-init}
            OUT=$REPO/checkpoints/DriveVLA-3B-align
            DEFAULT_EPOCHS=1 ;;
  finetune) MODEL=${MODEL:-$REPO/checkpoints/DriveVLA-3B-align}
            OUT=$REPO/checkpoints/DriveVLA-3B-reason
            DEFAULT_EPOCHS=3 ;;
  *) echo "stage must be align|finetune" >&2; exit 1 ;;
esac
EPOCHS=${EPOCHS:-$DEFAULT_EPOCHS}

echo "[train] stage=$STAGE  model=$MODEL  ->  $OUT"
echo "[train] uniad=$UNIAD_CKPT (FROZEN)"
echo "[train] traces=$REASONING_TRACES"
echo "[train] REASON_GATE=$REASON_GATE  epochs=$EPOCHS"

srun singularity exec --nv \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO/OpenDriveVLA" \
  --env PYTHONPATH="$REPO/OpenDriveVLA" \
  --env UNIAD_CKPT="$UNIAD_CKPT" \
  --env REASONING_TRACES="$REASONING_TRACES" \
  --env REASON_GATE="$REASON_GATE" \
  --env CACHED_DATA_PATH="$PROC/cached_parking_info.pkl" \
  --env CARLA_DATA_ROOT="$REPO/data_carla" \
  --env NUSC_DATA_ROOT="$REPO/data/nuscenes" \
  --env TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  torchrun --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" --master_port=29501 \
    drivevla/train_drivevla.py \
      --model-path "$MODEL" \
      --stage "$STAGE" \
      --conversations "$PROC/carla_conversations_train.json" \
      --output-dir "$OUT" \
      --num-epochs "$EPOCHS" \
      --bf16 --distributed \
      --lora-rank 64 --lora-alpha 128 \
      --grad-accum 8 --batch-size 1 --num-workers 4 --save-every 1

echo "[train] done -> $OUT"
