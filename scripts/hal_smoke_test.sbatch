#!/bin/bash
# UniAD Stage-1 smoke test on a HAL GPU, inside the Singularity container.
# Runs the full forward+backward (1 step) against the small CARLA dataset.
#
# Submit from anywhere on HAL:
#   sbatch scripts/hal_smoke_test.sbatch
# Watch:
#   squeue --me ; tail -f /workspaces/s0002438/openvla-parking/smoke_<jobid>.log
#SBATCH --job-name=uniad_smoke
#SBATCH --partition=ztest
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/smoke_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif

# --nv           : expose the host GPU/driver to the container
# --bind         : make the workspace (repo + data + checkpoint) visible
# --pwd          : CWD=OpenDriveVLA so the config's _REPO resolves to repo root
# PYTHONPATH     : make projects.mmdet3d_plugin / llava / drivevla importable (mounted, live)
# TRITON_CACHE_DIR: node-local scratch, off NFS (avoids DeepSpeed/Triton exit hang)
# LD_LIBRARY_PATH: put the container's own libs first so cv2's libGLdispatch resolves
#   to the glibc-2.35-compatible one, not the newer host lib --nv injects (GLIBC_2.38
#   mismatch). /.singularity.d/libs stays on the path for the host libcuda.
srun singularity exec --nv \
  --bind /workspaces/s0002438 \
  --pwd "$REPO/OpenDriveVLA" \
  --env PYTHONPATH="$REPO/OpenDriveVLA" \
  --env TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  python ../scripts/uniad_stage1_smoke_test.py \
    --config projects/configs/stage1_track_map/carla_parking_stage1.py "$@"

# Extra args pass through to the smoke test, e.g.:
#   sbatch scripts/hal_smoke_test.sbatch --tiny
# The full config (queue_length=5) needs >40 GB; --tiny (queue_length=1, 0.5x image)
# fits the 40 GB ztest A100 and validates the full fwd/bwd. For the real HAL config
# on 40 GB cards you'll need memory relief (grad checkpointing / 80 GB nodes / smaller
# queue_length) — a training-scale concern, separate from this env smoke test.
