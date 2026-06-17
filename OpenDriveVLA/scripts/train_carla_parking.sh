#!/usr/bin/env bash
# Fine-tune OpenDriveVLA on CARLA parking data.
# Must be run from the OpenDriveVLA/ directory.
#
# Prerequisites:
#   1. Run bash scripts/extract_carla_features.sh  (once per dataset)
#   2. data_carla/processed/carla_conversations.json has uniad_pth paths
#
# Outputs: checkpoints saved to ../checkpoints/OpenDriveVLA-0.5B-carla/
# After training, run inference with:
#   bash scripts/eval_carla_parking.sh ../checkpoints/OpenDriveVLA-0.5B-carla/epoch_010

set -e

CKPT_PATH=${1:-"../checkpoints/OpenDriveVLA-0.5B"}
NUM_EPOCHS=${2:-10}

export PYTHONPATH="$(pwd)/third_party/mmdetection3d_1_0_0rc6:${PYTHONPATH}"
export CACHED_DATA_PATH="../data_carla/processed/cached_parking_info.pkl"
# Reduce allocator fragmentation — important on the shared 8GB desktop GPU
# (~2.2GB is taken by the display, leaving ~5.4GB for training).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Unbuffered stdout so progress/loss lines appear live in the log instead of
# being held in Python's 8KB block buffer.
export PYTHONUNBUFFERED=1

# DeepSpeed checks for nvcc even when we don't use it for training.
if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDA_HOME}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    CUDA_VER=$(python -c "import torch; v=torch.version.cuda or '12.1'; print('.'.join(v.split('.')[:2]))")
    SHIM="$(pwd)/.cache/fake_cuda"
    mkdir -p "${SHIM}/bin"
    cat > "${SHIM}/bin/nvcc" <<EOF
#!/usr/bin/env bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release ${CUDA_VER}, V${CUDA_VER}.0"
EOF
    chmod +x "${SHIM}/bin/nvcc"
    export CUDA_HOME="${SHIM}"
    export PATH="${SHIM}/bin:${PATH}"
    echo ">>> nvcc shim created at ${SHIM}"
fi

python drivevla/train_drivevla.py \
  --model-path "${CKPT_PATH}" \
  --uniad-config projects/configs/stage1_track_map/carla_parking.py \
  --conversations ../data_carla/processed/carla_conversations.json \
  --output-dir ../checkpoints/OpenDriveVLA-0.5B-carla \
  --num-epochs "${NUM_EPOCHS}" \
  --lr 2e-4 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --grad-accum 8 \
  --save-every 1 \
  --num-workers 0
