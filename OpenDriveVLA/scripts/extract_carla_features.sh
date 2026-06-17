#!/usr/bin/env bash
# Pre-compute UniAD BEV features for all CARLA frames.
# Must be run from the OpenDriveVLA/ directory.
# Output: data_carla/processed/uniad_features/<token>.pth for every frame.

set -e

CKPT_PATH=${1:-"../checkpoints/OpenDriveVLA-0.5B"}

export PYTHONPATH="$(pwd)/third_party/mmdetection3d_1_0_0rc6:${PYTHONPATH}"
export CACHED_DATA_PATH="../data_carla/processed/cached_parking_info.pkl"

# DeepSpeed checks for nvcc even when we don't use it for training.
# Create a shim if nvcc is not on PATH.
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

python drivevla/extract_uniad_features.py \
  --model-path "${CKPT_PATH}" \
  --uniad-config projects/configs/stage1_track_map/carla_parking.py \
  --conversations ../data_carla/processed/carla_conversations.json \
  --out-dir ../data_carla/processed/uniad_features \
  --num-workers 0
