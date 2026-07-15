#!/bin/bash
# FULL 62k reasoning-trace generation with the Qwen2.5-32B teacher, EPISODE-SHARDED.
#
# One array task per shard, all parallel on zprod. ~10k frames/shard at ~2.7 s/frame -> each
# shard ~7-8 h; 6 shards in parallel -> ~8 h wall, vs ~46 h for a single un-resumable job.
# Sharding is by whole episode, so prev_reverse (gear-change) and the fallback/validator
# counts are never computed on a fragment. A failed shard re-runs alone.
#
# Submit (SHARDS must match --array range: 0-5 => 6 shards):
#   sbatch scripts/hal_reason_qwen_full.sbatch
# Watch:
#   squeue --me
#   tail -f /workspaces/s0002438/openvla-parking/reason_full_<jobid>_<shard>.log
# When ALL shards are COMPLETED, merge + verify (disjoint, complete):
#   python scripts/merge_reasoning_shards.py \
#     --shard-dirs <staging>/reasoning/v1_uniad_epoch6/shard0{0..5} \
#     --out        <staging>/reasoning/v1_uniad_epoch6
#SBATCH --job-name=reason_full
#SBATCH --partition=zprod
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/workspaces/s0002438/openvla-parking/reason_full_%A_%a.log

set -euo pipefail

REPO=/workspaces/s0002438/openvla-parking
SIF=/workspaces/s0002438/openvla.sif
STAGING=/staging/short_range_perception/s0002438/openvla_parking

NUM_SHARDS=6
SHARD_IDX=${SLURM_ARRAY_TASK_ID}
STG=$STAGING/data_carla/processed
OUT=$STG/reasoning/v1_uniad_epoch6/shard0${SHARD_IDX}

# temp 0.3 (the QwenVerbalizer default): 0 fallbacks in the fit-check = 100% real teacher
# traces. Higher temp trades faithfulness for cosmetic diversity and injects mock templates
# via fallbacks -- worse for the final dataset. Rich manner facts are on by default.
export QWEN_MODEL_PATH="$STAGING/models/Qwen2.5-32B-Instruct"

echo "[reason] shard ${SHARD_IDX}/${NUM_SHARDS} -> $OUT"

srun singularity exec --nv \
  --bind /workspaces/s0002438,/staging/short_range_perception \
  --pwd "$REPO" \
  --env PYTHONPATH="$REPO" \
  --env QWEN_MODEL_PATH="$QWEN_MODEL_PATH" \
  --env TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}" \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  python -m reasoning_data_gen.run_generate \
    --teacher qwen --qwen-model-path "$QWEN_MODEL_PATH" \
    --source uniad --uniad-features-dir "$STG/uniad_features" \
    --infos "$REPO/data_carla/processed/parking_infos_train.pkl" \
    --num-shards "$NUM_SHARDS" --shard-idx "$SHARD_IDX" \
    --out "$OUT"

echo "[reason] shard ${SHARD_IDX} done"
