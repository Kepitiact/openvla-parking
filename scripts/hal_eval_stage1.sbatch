#!/bin/bash
# UniAD Stage-1 evaluation on a HAL GPU, inside the Singularity container.
# Open-loop perception eval (no simulator): runs the trained model over val frames
# and reports det mAP / track AMOTA / map IoU.
#
# Dry-run on the first N val frames (fits the 45-min ztest queue):
#   sbatch -p ztest -t 00:40:00 scripts/hal_eval_stage1.sbatch \
#       /workspaces/s0002438/openvla-parking/checkpoints/stage1_carla_full/epoch_2.pth 150
# Full val (after training; ~1-2 h single-GPU, so NOT ztest — use zprod/ztestpreemp):
#   sbatch scripts/hal_eval_stage1.sbatch <abs/epoch_N.pth>
#SBATCH --job-name=uniad_eval
#SBATCH --partition=ztest
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/checkpoints/stage1_carla_full/eval_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
CONFIG=projects/configs/stage1_track_map/carla_parking_stage1.py
CKPT=${1:?usage: sbatch scripts/hal_eval_stage1.sbatch <abs-checkpoint.pth> [max_samples]}
MAX=${2:-}
MAXARG=""
[ -n "$MAX" ] && MAXARG="--max-samples $MAX"

# Dump to node-local /tmp (off the /workspaces quota) and score in the same job.
PREDS="/tmp/preds_${SLURM_JOB_ID}.pkl"
srun singularity exec --nv \
  --bind /workspaces/s0002438 \
  --pwd "$REPO/OpenDriveVLA" \
  --env PYTHONPATH="$REPO/OpenDriveVLA" \
  --env TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  "$SIF" bash -lc "
    python ../scripts/uniad_stage1_eval.py --config '$CONFIG' --checkpoint '$CKPT' $MAXARG --dump-out '$PREDS' &&
    python ../scripts/uniad_stage1_metrics.py --preds '$PREDS' --infos '$REPO/data_carla/processed/parking_infos_val.pkl'
  "
