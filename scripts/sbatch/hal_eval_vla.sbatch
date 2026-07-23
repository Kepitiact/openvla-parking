#!/bin/bash
# Eval the reasoning-VLA end to end: merge LoRA -> inference on VAL -> score trajectory +
# reasoning. This is the eval RUNNER that was missing (only the scorers existed).
#
# ⚠️ UNPROVEN until its first ztest run. Verify with LIMIT=20 on ztest BEFORE trusting it on
# the full val set — the config's val-vs-combined selection and the val uniad_pth wiring are
# the parts most likely to need a fix (same class as the training-path bugs).
#
# Submit (env overridable):
#   # fast verification on ztest (~minutes):
#   LORA=<...>/DriveVLA-3B-reason-gated/epoch_003 LIMIT=20 \
#     sbatch --partition=ztest --gres=gpu:2 --time=00:40:00 scripts/sbatch/hal_eval_vla.sbatch
#   # full val:
#   LORA=<...>/DriveVLA-3B-reason-gated/epoch_003 \
#     sbatch --gres=gpu:4 --time=04:00:00 scripts/sbatch/hal_eval_vla.sbatch
#SBATCH --job-name=eval_vla
#SBATCH --partition=zprod
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/workspaces/s0002438/openvla-parking/checkpoints/eval_vla_%j.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking
PROC=$REPO/data_carla/processed
STG=$STAGING/data_carla/processed

# The finetune adapter to evaluate, and the align model it was trained on top of (its base).
LORA=${LORA:?set LORA=<...>/DriveVLA-3B-reason-gated/epoch_00N}
BASE=${BASE:-$(ls -d "$REPO"/checkpoints/DriveVLA-3B-align/epoch_* 2>/dev/null | sort | tail -1)}
GATE=${REASON_GATE:-track}          # eval with the SAME gate the model was trained with
LIMIT=${LIMIT:-0}
MERGED="$LORA/merged"
OUT_DIR="$LORA/eval"
TRACES=${TRACES:-$STG/reasoning/v1_uniad_epoch6/traces.jsonl}

# Point inference at the VAL split only (not the combined infos). carla_parking.py resolves
# ann_file from CARLA_INFOS_VAL when set, else CARLA_DATA_ROOT/processed/...temporal.pkl.
export CARLA_INFOS_VAL=$PROC/parking_infos_val.pkl

echo "[eval] base=$BASE"
echo "[eval] lora=$LORA  gate=$GATE  limit=$LIMIT"

run() { srun singularity exec --nv \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO/OpenDriveVLA" \
  --env PYTHONPATH="$REPO/OpenDriveVLA" \
  --env UNIAD_CKPT="$REPO/checkpoints/stage1_carla_full/epoch_6.pth" \
  --env REASON_GATE="$GATE" \
  --env CARLA_DATA_ROOT="$REPO/data_carla" \
  --env CARLA_INFOS_VAL="$CARLA_INFOS_VAL" \
  --env NUSC_DATA_ROOT="$REPO/data/nuscenes" \
  --env CACHED_DATA_PATH="$PROC/cached_parking_info.pkl" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  "$SIF" "$@"; }

# 1. merge the LoRA adapter onto the align base -> a loadable full model
if [ ! -f "$MERGED/config.json" ]; then
  echo "[eval] 1/3 merging LoRA -> $MERGED"
  run python drivevla/merge_lora.py --base "$BASE" --lora "$LORA" --out "$MERGED"
else
  echo "[eval] 1/3 merged model exists, skipping merge"
fi

# 2. inference on val -> planning_conversations_val.json (answer=traj span, reasoning kept)
echo "[eval] 2/3 inference on val (limit=$LIMIT)"
# --data selects the frames: with use_uniad_pth the dataset runs on this conversation list
# and reads each frame's features from its stamped uniad_pth (val -> uniad_features_val).
# --uniad-config carla_parking.py (NOT the default base_track_map.py, which points ann_file
# at a nuScenes val pkl that does not exist here -- the first-run crash).
run torchrun --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" --master_port=29503 \
  drivevla/inference_drivevla.py \
    --model-path "$MERGED" --use-uniad-pth --bf16 \
    --uniad-config projects/configs/stage1_track_map/carla_parking.py \
    --data "$PROC/carla_conversations_val.json" \
    --output "$OUT_DIR/planning_conversations_val.json" \
    --limit "$LIMIT"

# 3. score both halves: trajectory (final-pose) and reasoning (faithfulness)
echo "[eval] 3/3 scoring"
run python scripts/score_trajectory.py \
      --student "$OUT_DIR/planning_conversations_val.json" --traces "$TRACES"
run python scripts/score_reasoning_faithfulness.py \
      --student "$OUT_DIR/planning_conversations_val.json" --traces "$TRACES"

echo "[eval] done -> $OUT_DIR"
