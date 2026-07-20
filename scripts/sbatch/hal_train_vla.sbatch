#!/bin/bash
# Train the reasoning-VLA: our Qwen2.5-3B backbone + frozen UniAD.
#
# The model emits its reasoning BEFORE the trajectory:
#   <reason_start>...<reason_end><traj_start>[(x,y,h)x6]<traj_end>
# and (with REASON_GATE=1) the trajectory tokens cannot see the perception tokens, so the
# only path from perception to waypoints runs THROUGH the reasoning.
#
# Two stages -- run align first, then finetune from its output:
#   sbatch scripts/sbatch/hal_train_vla.sbatch align
#   sbatch scripts/sbatch/hal_train_vla.sbatch finetune
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
# 8 GPUs on zprod: the queue wait is driven by the zprodhigh priority tier holding the
# nodes, not by GPU count (4 vs 8 gave the same estimate), and zprodhigh is reserved for
# production/urgent work -- not ours to use. So wait on zprod and keep 8 GPUs for the
# shorter run (align ~1.7h, finetune ~6h) once it schedules. --nproc_per_node below tracks
# whatever is allocated, so this is the only line to change for a different GPU count.
#SBATCH --gres=gpu:8
# Request only what the run needs. Slurm BACKFILLS short jobs into gaps ahead of big
# reservations -- a 48h request (the old value) fits in no gap, so an align job needing ~2h
# sat behind a ~3-week estimate. 12h covers the longest real run (finetune, 3 epochs ~6h)
# with margin. Override per stage at submit: --time=04:00:00 for align backfills even better.
#SBATCH --time=12:00:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/checkpoints/train_vla_%j.log

set -euo pipefail

STAGE=${1:?usage: sbatch scripts/sbatch/hal_train_vla.sbatch <align|finetune> [epochs]}
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
            OUT=${OUT:-$REPO/checkpoints/DriveVLA-3B-align}
            DEFAULT_EPOCHS=1 ;;
  finetune) MODEL=${MODEL:-$REPO/checkpoints/DriveVLA-3B-align}
            # OVERRIDABLE and it matters: the gate A/B runs finetune twice. Without
            # distinct OUT dirs both arms write to DriveVLA-3B-reason and the second
            # silently clobbers the first -- destroying the comparison.
            OUT=${OUT:-$REPO/checkpoints/DriveVLA-3B-reason}
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
      --grad-accum 8 --batch-size 1 --num-workers 4 --save-every 1 \
      --save-steps "${SAVE_STEPS:-2000}"

echo "[train] done -> $OUT"
