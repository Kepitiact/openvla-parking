#!/bin/bash
# UniAD Stage-1 (track+map) full training on HAL: 1 node x 8 A100, inside Singularity.
#
# queue_length is passed as the FIRST script argument (default 5). Do NOT use
# --export=ALL,QL=... on this cluster: it triggers "user env retrieval failed".
# Memory fit check first (free 45-min queue), queue_length=5:
#   sbatch -p ztest -t 00:40:00 scripts/hal_train_stage1.sbatch 5
#   # watch a few iters; if CUDA OOM, resubmit with arg 3.
# Real run (zprod, resumable across preemption):
#   sbatch scripts/hal_train_stage1.sbatch 5      # or: ... 3  (if 40GB needs it)
#
#SBATCH --job-name=uniad_stage1
#SBATCH --partition=zprod
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=72:00:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/checkpoints/stage1_carla_full/train_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
CONFIG=projects/configs/stage1_track_map/carla_parking_stage1.py
WORKDIR=$REPO/checkpoints/stage1_carla_full     # persistent on NFS -> survives preemption
QL=${1:-5}                                      # queue_length via arg1 (5 full; 3 to fit 40GB)

mkdir -p "$WORKDIR"

# Single node, 8 procs via torch.distributed.launch (inside dist_train.sh).
# --auto-resume picks up the latest checkpoint in WORKDIR on requeue.
# --no-validate: train cleanly + checkpoint; evaluate as a separate step afterwards
#   (the eval/inference forward_test path is validated on its own, so a val-hook bug
#   can't crash a multi-hour training job at an epoch boundary).
# --cfg-options sets queue_length on the actual data pipelines without editing configs.
srun singularity exec --nv \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO/OpenDriveVLA" \
  --env PYTHONPATH="$REPO/OpenDriveVLA" \
  --env TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  torchrun --nproc_per_node=8 --master_port=29500 \
    third_party/mmdetection3d_1_0_0rc6/tools/train.py \
    "$CONFIG" --launcher pytorch --seed 0 \
    --work-dir "$WORKDIR" \
    --auto-resume \
    --no-validate \
    --cfg-options data.train.queue_length=$QL data.val.queue_length=$QL \
                  checkpoint_config.max_keep_ckpts=2 \
                  checkpoint_config.by_epoch=False checkpoint_config.interval=1000
