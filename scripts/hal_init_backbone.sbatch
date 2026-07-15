#!/bin/bash
# Build the DriveVLA-Qwen2.5-3B-init backbone ON HAL.
#
# checkpoints/ is gitignored (a ~6 GB model has no business in git), so the backbone built
# locally by init_drivevla_qwen3b.py never reached HAL -- the smoke crashed loading it. Build
# it here from the Qwen2.5-3B-Instruct weights already on staging. CPU only: it loads the
# base weights, adds the 22 special tokens + 3 fresh projectors, and saves. No GPU.
#
# Submit:
#   sbatch scripts/hal_init_backbone.sbatch
# Produces: checkpoints/DriveVLA-Qwen2.5-3B-init  (then re-run the smoke)
#SBATCH --job-name=init_backbone
#SBATCH --partition=zprodcpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/workspaces/s0002438/openvla-parking/init_backbone_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking

srun singularity exec \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO" \
  --env PYTHONPATH="$REPO/OpenDriveVLA" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  "$SIF" \
  python scripts/init_drivevla_qwen3b.py \
    --base "$STAGING/models/Qwen2.5-3B-Instruct" \
    --out  "$REPO/checkpoints/DriveVLA-Qwen2.5-3B-init"

echo "[init] backbone built -> $REPO/checkpoints/DriveVLA-Qwen2.5-3B-init"
ls -la "$REPO/checkpoints/DriveVLA-Qwen2.5-3B-init" 2>&1 | head
