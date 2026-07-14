#!/bin/bash
# Do the perception streams carry information? Linear probe, per stream, on the FULL
# extraction. This is what decides whether REASON_GATE=track,map,scene is a real
# experiment or a coin flip -- and it costs minutes, versus a full training run to find
# out the hard way.
#
# The 371-frame local dry run could not answer it: 10 episodes / 7 weather classes, with
# 5 classes having exactly ONE episode, so episode-grouped CV had zero training examples
# for half the held-out frames. Needs the 1508-episode extraction.
#
# CPU only. No GPU, no training, and no label here ever enters the model.
#
# Submit (after extraction finishes):
#   sbatch scripts/hal_probe_streams.sbatch
#SBATCH --job-name=probe_streams
#SBATCH --partition=zprodcpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/workspaces/s0002438/openvla-parking/probe_streams_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking

FEATS=${1:-$STAGING/data_carla/processed/uniad_features}
# Every 5th frame: still every episode (~12k frames, ~150 episodes/weather class), but a
# fifth of the 78 GB read. Neighbouring frames of one episode are near-duplicates.
STRIDE=${2:-5}

srun singularity exec \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO" \
  --env PYTHONPATH="$REPO" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  "$SIF" \
  python scripts/probe_perception_streams.py \
    --features-dir "$FEATS" \
    --raw-dir "$STAGING/data_carla/raw" \
    --stride "$STRIDE"
