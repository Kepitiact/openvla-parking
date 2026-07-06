# OpenDriveVLA CARLA / UniAD Stage-1 environment for the HAL cluster.
#
# Env-ONLY image: it freezes the painful compiled stack (torch 2.1.2+cu121,
# mmcv-full 1.7.2, mmdet 2.26.0, mmseg 0.29.1, mmengine 0.9.0, mmdet3d 1.0.0rc6
# with its CUDA ops compiled) exactly per requirements.lock.txt. The fast-changing
# OpenDriveVLA package (llava/drivevla/projects) + data + checkpoint are NOT baked;
# they are bind-mounted at runtime and made importable via PYTHONPATH.
#
# Build (from repo root):
#   docker build -t openvla-parking:cu121 .
# Convert to SIF on HAL:
#   docker save openvla-parking:cu121 -o openvla.tar   # then scp to HAL
#   singularity build openvla.sif docker-archive://openvla.tar
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0+PTX" \
    MAX_JOBS=4 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: Python 3.10 (Ubuntu 22.04 default), git (for transformers-from-git
# and any VCS pins), build toolchain + ninja (mmdet3d CUDA ops), and the runtime
# libs opencv/open3d need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python-is-python3 \
        git build-essential ninja-build \
        libgl1 libglib2.0-0 libgomp1 \
        ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade "pip<25" setuptools wheel

# 1) torch first, from the cu121 index (must precede mmcv / mmdet3d builds).
RUN python -m pip install \
        torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

# 2) mmcv-full: prebuilt wheel matching torch2.1/cu121 (no source compile).
RUN python -m pip install mmcv-full==1.7.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 3) The rest of the frozen env, straight from the lockfile — minus the lines we
#    handle specially: the two editable installs (machine-specific paths), and
#    torch/torchvision/mmcv-full (already installed above). transformers-from-git
#    stays (git is installed). --extra-index-url keeps any cu121 wheels resolvable.
#    --no-deps: the lockfile IS a full pip freeze (every transitive dep pinned), so
#    we install the exact set verbatim. This reproduces the env faithfully AND avoids
#    the modern resolver rejecting benign version skew that the working env tolerates
#    (e.g. requests==2.28.1 vs jupyterlab-server's requests>=2.31).
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN grep -vE '^\s*(-e |#|torch==|torchvision==|mmcv-full==)' /tmp/requirements.lock.txt \
        > /tmp/req.filtered.txt && \
    python -m pip install --no-deps -r /tmp/req.filtered.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121

# 4) mmdet3d 1.0.0rc6 from THIS repo's vendored source, editable so it matches the
#    lockfile. Lives at /opt (never mounted over), so its compiled .so files persist.
#    mmdet3d rc6's setup.py `import torch`s at top level; modern setuptools' editable
#    path re-spawns a build-ISOLATED sub-build with no torch, which fails. So we pin
#    the mmlab-compatible setuptools 59.5.0 and build via the classic `setup.py
#    develop` (in-place, uses the global env where torch is present). FORCE_CUDA +
#    arch list compile the deformable-attention / voxel ops for the A100 (sm_80).
COPY OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6 /opt/mmdetection3d_1_0_0rc6
RUN cd /opt/mmdetection3d_1_0_0rc6 && \
    python -m pip install "setuptools==59.5.0" && \
    python setup.py develop --no-deps

# 5) Fail the build now if the load-bearing imports don't resolve. deepspeed's
#    import-time nvcc probe is satisfied by the real CUDA toolkit + CUDA_HOME (no
#    GPU needed here). This is the in-image version of the ENV_SETUP smoke check.
RUN python -c "import torch, mmcv, mmdet, mmseg, mmdet3d; \
from mmcv.ops import MultiScaleDeformableAttention; import deepspeed; \
print('torch', torch.__version__, '| cuda', torch.version.cuda, \
'| mmcv', mmcv.__version__, '| mmdet3d', mmdet3d.__version__)"

WORKDIR /workspace
CMD ["/bin/bash"]
