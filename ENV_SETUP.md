# Environment setup (OpenDriveVLA CARLA pipeline)

Reproducing the Python environment for training/inference. This is a **build
recipe + lockfile**, not a container (the container is a separate deliverable).

- Frozen lockfile: [`requirements.lock.txt`](requirements.lock.txt) (`pip freeze` of the working venv).
- Python: 3.10, CUDA 12.1 wheels (`torch==2.1.2+cu121`).

## The painful stack

These four pins are the ones that break most often; install them in this order:

| Package | Version | Notes |
|---|---|---|
| `torch` | `2.1.2+cu121` | Install first, from the cu121 index. |
| `mmcv-full` | `1.7.2` | Must match the torch/CUDA ABI; installs compiled ops. |
| `mmdet` | `2.26.0` | Pulls `mmengine`, `mmsegmentation==0.29.1`. |
| `mmdet3d` | `1.0.0rc6` | Editable from `OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6`. |

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmengine==0.9.0
pip install -e OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6
pip install -e OpenDriveVLA            # llava + drivevla deps
pip install accelerate==0.29.3 peft==0.4.0 deepspeed==0.14.2
```

> The `requirements.lock.txt` line for mmdet3d is an editable install and records a
> machine-specific absolute path. Re-run the `pip install -e` above against this
> repo's `third_party/` instead of trusting that path.

## nvcc shim (conditional)

DeepSpeed probes for `nvcc` at import time. The drivevla entry scripts bootstrap
through [`OpenDriveVLA/drivevla/_bootstrap.py`](OpenDriveVLA/drivevla/_bootstrap.py),
which creates a fake `nvcc` **only when a real one is absent**:

- On a desktop without the CUDA toolkit → a shim under `OpenDriveVLA/.cache/fake_cuda/`.
- **On HAL / any host with a real `nvcc` on `PATH` (or a valid `CUDA_HOME`) the shim
  is NOT created** — real nvcc-compiled ops build normally.

No manual step is needed either way.

## Multi-GPU

Training uses HuggingFace Accelerate under the hood when launched distributed:

```bash
# single GPU (default, unchanged)
bash scripts/train_carla_parking.sh
# multi-GPU on one node
NPROC=4 bash scripts/train_carla_parking.sh          # -> torchrun --nproc_per_node=4 ... --distributed
```
