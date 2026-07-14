#!/bin/bash
# Full UniAD feature extraction on HAL (train + val), with the TRAINED CARLA checkpoint.
#
# Builds UniAD directly from (config, checkpoint) -- no LLaVA wrapper, no OpenDriveVLA-0.5B,
# so the baked-nuScenes-weights bug cannot recur. UNIAD_CKPT is REQUIRED (never defaulted:
# `uniad_base_track_map.pth` once named two different models).
#
# Submit (arg1 = infos pkl, arg2 = output dir).
# NB the asymmetry, and it is not a typo:
#   infos  live in the REPO      (/workspaces/.../data_carla/processed) -- ~5 GB, fine there
#   output goes to STAGING       (78 GB would blow the ~100 GB workspace quota)
#   sbatch scripts/hal_extract_features.sbatch \
#     /workspaces/s0002438/openvla-parking/data_carla/processed/parking_infos_train.pkl \
#     /staging/short_range_perception/s0002438/openvla_parking/data_carla/processed/uniad_features
#
# SIZE: 1.27 MB/frame -> train 61,659 = 78 GB, val 9,454 = 12 GB. This MUST land on
# /staging: /workspaces has a ~100 GB quota and a full workspace has killed a run before.
#
# TIME: ~0.6-1.5 s/frame single-GPU -> train alone is ~10-24 h. zprod has no time limit.
# Extraction is DETERMINISTIC (verified: two identical runs -> byte-identical detections),
# so a re-run reproduces this exactly.
#
# NOTE ON SHARDING: extract_uniad_features.py supports --num-shards/--shard-idx (split by
# scene). It is ~8x faster but NOT output-identical: it changes ~1.3% of traces, because the
# dataset builds each frame's temporal queue by index, so filtering data_infos perturbs the
# first frames of every scene. Left OFF deliberately -- we extract once and reuse for months.
#SBATCH --job-name=uniad_extract
#SBATCH --partition=zprod
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/checkpoints/stage1_carla_full/extract_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking

INFOS=${1:?usage: sbatch scripts/hal_extract_features.sbatch <infos.pkl> <out-dir> [ckpt]}
OUTDIR=${2:?usage: sbatch scripts/hal_extract_features.sbatch <infos.pkl> <out-dir> [ckpt]}
CKPT=${3:-$REPO/checkpoints/stage1_carla_full/epoch_6.pth}   # the final trained model
CONFIG=projects/configs/stage1_track_map/carla_parking_stage1.py

case "$OUTDIR" in
  /staging/*) ;;
  *) echo "REFUSING: out-dir must be on /staging (90 GB would blow the /workspaces quota)." >&2
     exit 1 ;;
esac

mkdir -p "$OUTDIR"
echo "[extract] infos=$INFOS"
echo "[extract] ckpt =$CKPT"
echo "[extract] out  =$OUTDIR"

srun singularity exec --nv \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO/OpenDriveVLA" \
  --env PYTHONPATH="$REPO/OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6:$REPO/OpenDriveVLA" \
  --env CARLA_DATA_ROOT="$STAGING/data_carla" \
  --env NUSC_DATA_ROOT="$REPO/data/nuscenes" \
  --env TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  python drivevla/extract_uniad_features.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --ann-file "$INFOS" \
    --out-dir "$OUTDIR" \
    --conversations "$STAGING/data_carla/processed/carla_conversations.json" \
    --num-workers 4

# find, not `ls *.pth`: 61k filenames overflow ARG_MAX, so the glob dies with
# "Argument list too long" and the count silently prints 0 on a run that fully succeeded.
echo "[extract] done: $(find "$OUTDIR" -name '*.pth' 2>/dev/null | wc -l) feature files"
