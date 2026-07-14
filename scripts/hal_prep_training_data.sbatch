#!/bin/bash
# CPU prep for VLA training: the two files training CANNOT run without.
#
#   cached_parking_info.pkl   per-token ego state, past-2s history, 6-step GT trajectory,
#                             maneuver + ego-local slot. build_llava_conversation reads
#                             EVERY field of the prompt and the GT answer from this.
#   carla_conversations.json  the training index (one entry per frame). Feature paths
#                             (uniad_pth) get stamped into it after extraction.
#
# Neither exists on HAL. No GPU needed. Run it WHILE the feature extraction is going.
#
# Submit:
#   sbatch scripts/hal_prep_training_data.sbatch
# Then AFTER extraction finishes, stamp the feature paths in:
#   sbatch scripts/hal_prep_training_data.sbatch --stamp-only
#SBATCH --job-name=prep_train_data
#SBATCH --partition=zprodcpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
# MEMORY: this is the whole reason the first attempt died. The nuScenes devkit parses the
# ENTIRE v1.0-carla DB into Python objects -- sample_annotation.json alone is 1.4 GB
# (~3.4M annotations) and JSON->objects costs 3-5x -- on top of the 2.6 GB infos pkl.
# The Slurm default (1856 MB) OOM-killed it in 51 seconds.
#
# 40G, not more: zprodcpu nodes have only 48 GB TOTAL (hal-cn[03-04]), so a bigger request
# is rejected outright ("Requested node configuration is not available"). 40G is safe --
# the same job completes on a 30 GB laptop with ~24 GB free, so the true need is under that.
#SBATCH --mem=40G
#SBATCH --output=/workspaces/s0002438/openvla-parking/checkpoints/stage1_carla_full/prep_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking
PROC=$REPO/data_carla/processed
FEATS=$STAGING/data_carla/processed/uniad_features

STAMP_ONLY=0
[ "${1:-}" = "--stamp-only" ] && STAMP_ONLY=1

run() {
  srun singularity exec \
    --bind /workspaces/s0002438,/staging/short_range_perception \
    --pwd "$REPO" \
    --env PYTHONPATH="$REPO/scripts:$REPO" \
    --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
    "$SIF" "$@"
}

if [ "$STAMP_ONLY" -eq 0 ]; then
  # 1. cached info: ego state + past-2s history + 6-step future trajectory, from the
  #    nuScenes-format DB. This is the source of every number in the prompt.
  echo "[1/2] cached_parking_info.pkl (train+val infos, from the v1.0-carla DB) ..."
  run python scripts/generate_cached_nuscenes_info.py \
      --infos "$PROC/parking_infos_temporal.pkl" \
      --dataroot "$REPO/data/nuscenes" \
      --version v1.0-carla \
      --output "$PROC/cached_parking_info.pkl"

  # 2. the training index.
  echo "[2/2] carla_conversations.json ..."
  run python scripts/build_carla_conversations.py \
      --infos "$PROC/parking_infos_temporal.pkl" \
      --out "$PROC/carla_conversations.json"
fi

# 3. Stamp uniad_pth into the conversations. The extraction skips this when the
#    conversations file does not exist yet (which it didn't), so do it here.
#    Safe to re-run; only stamps tokens whose feature file actually exists.
echo "[stamp] pointing conversations at $FEATS ..."
run python - "$PROC/carla_conversations.json" "$FEATS" <<'PY'
import json, os, sys
conv_path, feat_dir = sys.argv[1], sys.argv[2]
with open(conv_path) as f:
    convs = json.load(f)
hit = 0
for e in convs:
    tok = e.get("sample_id") or e.get("qa_id", "").removesuffix("_trajectory")
    p = os.path.join(feat_dir, f"{tok}.pth")
    if os.path.exists(p):
        e["uniad_pth"] = p
        hit += 1
with open(conv_path, "w") as f:
    json.dump(convs, f)
print(f"  stamped {hit}/{len(convs)} conversations with uniad_pth")
if hit == 0:
    print("  WARNING: no features found yet -- re-run with --stamp-only after extraction.")
PY

echo "[prep] done."
ls -lh "$PROC/cached_parking_info.pkl" "$PROC/carla_conversations.json" 2>/dev/null || true
