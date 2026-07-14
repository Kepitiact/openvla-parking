#!/usr/bin/env bash
# One non-destructive DAgger round:
#   ingest -> dagger-only feature extraction -> per-round conversations
#   -> rebuild conversation manifest -> warm-start LoRA train -> merge LoRA
#
# Each round ADDS files (dagger_r0N_infos.pkl, dagger_r0N_conversations.json) and
# appends to data_carla/processed/conversations.yaml; the base dataset files are
# never rewritten. Re-running a round is idempotent up to the training step.
#
# The DAgger package (dagger_cached.pkl / dagger_infos.pkl / dagger_conversations.json
# under data_carla/processed/) must already be produced by the data-gen harness.
#
# Usage:
#   bash run_dagger_round.sh <round>
# Env knobs:
#   WARM_START   checkpoint to warm-start from (default ../checkpoints/OpenDriveVLA-0.5B)
#   EPOCHS       refinement epochs             (default 2)
#   LR           lower LR for refinement       (default 1e-4)
#   OUT          output run dir                (default ../checkpoints/OpenDriveVLA-0.5B-carla-dagger<round>)
#   NPROC, NUM_WORKERS, GRAD_ACCUM, SAVE_STEPS, BF16, RESUME  forwarded to training

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

ROUND=${1:?"usage: bash run_dagger_round.sh <round>"}
RID=$(printf "%02d" "${ROUND}")
PROC="${REPO_ROOT}/data_carla/processed"
WARM_START=${WARM_START:-"../checkpoints/OpenDriveVLA-0.5B"}
EPOCHS=${EPOCHS:-2}
LR=${LR:-"1e-4"}
OUT=${OUT:-"../checkpoints/OpenDriveVLA-0.5B-carla-dagger${ROUND}"}

echo "=== [1/5] Ingest DAgger round ${ROUND} (merge cached+infos, register DB, normalize version) ==="
python scripts/ingest_dagger.py --round "${ROUND}"

echo "=== [2/5] Per-round conversations + dagger-only feature extraction ==="
# Fresh copy of the round's conversations; the extractor stamps uniad_pth into it.
cp -f "${PROC}/dagger_conversations.json" "${PROC}/dagger_r${RID}_conversations.json"
cd OpenDriveVLA
CONV="${PROC}/dagger_r${RID}_conversations.json" \
  ANN_FILE="${PROC}/dagger_r${RID}_infos.pkl" \
  bash scripts/extract_carla_features.sh "${WARM_START}"
cd ..

echo "=== [3/5] Rebuild conversation manifest (base + all rounds) ==="
python scripts/build_conversation_manifest.py

echo "=== [4/5] Warm-start LoRA fine-tune on the manifest ==="
cd OpenDriveVLA
CONV="${PROC}/conversations.yaml" \
  bash scripts/train_carla_parking.sh "${WARM_START}" "${EPOCHS}" "${OUT}" "${LR}"

echo "=== [5/5] Merge LoRA -> standalone model ==="
LAST_EPOCH=$(printf "epoch_%03d" "${EPOCHS}")
python drivevla/merge_lora.py --base "${WARM_START}" --lora "${OUT}/${LAST_EPOCH}" --out "${OUT}/merged"
cd ..

echo "=== DAgger round ${ROUND} complete. Merged model: ${OUT}/merged ==="
