#!/bin/bash
# UniAD->reasoning CONVERSION sanity check on real decoded outputs (CPU-only).
#
# Validates that SceneRecord.from_uniad correctly turns real UniAD detections into
# scenes/facts/traces, using the deterministic MockVerbalizer (no GPU, no Qwen). Run
# this on a small subset (e.g. 10 episodes) with the epoch4 UniAD weights BEFORE the
# GPU Qwen pass — the decoded-detection format is the same as the final model, so a
# clean run here means the whole UniAD path is wired.
#
# PREREQUISITE (GPU, separate job): produce the per-frame decoded .pth first, e.g.
#   run drivevla/extract_uniad_features.py with the epoch4 UniAD weights over the
#   subset infos -> <staging>/data_carla/processed/uniad_features/<token>.pth
#   (each .pth already carries result_track.detections; see extract_uniad_features.py).
#
# Then submit (args pass through to run_generate):
#   sbatch scripts/sbatch/hal_reason_uniad_convert.sbatch \
#     --infos <staging>/data_carla/processed/parking_infos_val.pkl \
#     --uniad-features-dir <staging>/data_carla/processed/uniad_features \
#     --out   <staging>/data_carla/processed/reasoning/v0_uniad_convert --limit 400
#SBATCH --job-name=reason_uniad_convert
#SBATCH --partition=zprodcpu
#SBATCH --time=00:30:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/reason_uniad_convert_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif

# CPU-only: from_uniad + fact rules + mock teacher need no GPU. The reconcile
# histogram in the manifest reports how often perception (UniAD) agrees with GT
# (ok / perception_limited / contradiction) — the key signal for this check.
srun singularity exec \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO" \
  --env PYTHONPATH="$REPO" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  "$SIF" \
  python -m reasoning_data_gen.run_generate \
    --source uniad --teacher mock "$@"

# Inspect afterwards:
#   cat <out>/manifest.json   # reconcile_histogram, uniad.missing_pth_frames, rates
#   head <out>/traces.jsonl   # decision + trace grounded in UniAD detections
