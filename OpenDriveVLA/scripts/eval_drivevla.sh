CKPT_PATH=$1
NUM_GPU=$2

# DeepSpeed import checks CUDA_HOME/bin/nvcc even in inference-only mode.
# If nvcc is not available on this host, provide a lightweight local shim so
# runtime dependency checks can proceed.
ensure_cuda_home_for_deepspeed() {
    if [[ -n "${CUDA_HOME}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        return
    fi
    if command -v nvcc >/dev/null 2>&1; then
        return
    fi

    local torch_python
    torch_python=$(command -v python)
    local cuda_ver
    cuda_ver=$(${torch_python} - <<'PY'
import torch
v = torch.version.cuda or "12.1"
print(".".join(v.split(".")[:2]))
PY
)

    local shim_home
    shim_home="$(pwd)/.cache/fake_cuda"
    mkdir -p "${shim_home}/bin"
    cat > "${shim_home}/bin/nvcc" <<EOF
#!/usr/bin/env bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release ${cuda_ver}, V${cuda_ver}.0"
EOF
    chmod +x "${shim_home}/bin/nvcc"

    export CUDA_HOME="${shim_home}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo ">>> CUDA_HOME not set and nvcc not found; using local nvcc shim at ${CUDA_HOME}" | tee -a "${EVAL_LOG_FILE}"
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_PATH_CLEAN=$(basename ${CKPT_PATH})
LOG_RESULT_DIR="output/${CKPT_PATH_CLEAN}/${TIMESTAMP}"

mkdir -p ${LOG_RESULT_DIR}/log
mkdir -p ${LOG_RESULT_DIR}/results
EVAL_LOG_FILE="${LOG_RESULT_DIR}/log/eval.log"

echo ">>> Experiment Configuration:" | tee -a ${EVAL_LOG_FILE}
echo "- LOG_RESULT_DIR: ${LOG_RESULT_DIR}" | tee -a ${EVAL_LOG_FILE}
echo "- CKPT_PATH: ${CKPT_PATH}" | tee -a ${EVAL_LOG_FILE}
echo "----------------------------------------" | tee -a ${EVAL_LOG_FILE}

# ----------------------------------------------------------------

echo ">>> Inference ${CKPT_PATH}" | tee -a ${EVAL_LOG_FILE}

ensure_cuda_home_for_deepspeed

PLAN_CONV_PATH="${LOG_RESULT_DIR}/results/plan_conv.json"
INFERENCE_EXTRA_ARGS=${INFERENCE_EXTRA_ARGS:-}
USE_BF16=${USE_BF16:-0}
PRECISION_ARG=""
if [[ "${USE_BF16}" == "1" ]]; then
    PRECISION_ARG="--bf16"
fi

    # -m debugpy --listen 6000 --wait-for-client \
PYTHONPATH="$(pwd)":$PYTHONPATH \
torchrun --nproc_per_node=${NUM_GPU} \
    drivevla/inference_drivevla.py \
    --num-workers 4 \
    ${PRECISION_ARG} \
    --model-path ${CKPT_PATH} \
    --output ${PLAN_CONV_PATH} \
    ${INFERENCE_EXTRA_ARGS} \
    2>&1 | tee -a ${EVAL_LOG_FILE}

echo ">>> Evaluating ${PLAN_CONV_PATH}..." | tee -a ${EVAL_LOG_FILE}

python drivevla/eval_drivevla.py \
    --output ${PLAN_CONV_PATH} \
    2>&1 | tee -a ${EVAL_LOG_FILE}
