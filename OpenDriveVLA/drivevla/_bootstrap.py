"""Shared standalone bootstrap for the drivevla CLI scripts.

Import this module *before* any llava / mmdet3d import so that
`python drivevla/<script>.py` works without a shell wrapper:
  - puts the OpenDriveVLA root and third_party/mmdet3d on sys.path
  - satisfies DeepSpeed's import-time nvcc probe.

The nvcc shim is created ONLY when a real CUDA toolkit is absent (B8). On HAL,
where a real nvcc is on PATH (or CUDA_HOME points at a real toolkit), nothing is
faked so nvcc-compiled ops build normally.
"""

import os
import pathlib
import sys
from shutil import which

_ODV = pathlib.Path(__file__).resolve().parent.parent
for _p in (str(_ODV / "third_party" / "mmdetection3d_1_0_0rc6"), str(_ODV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _have_real_nvcc() -> bool:
    if which("nvcc"):
        return True
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    return bool(cuda_home and os.path.isfile(os.path.join(cuda_home, "bin", "nvcc")))


def ensure_nvcc_shim() -> None:
    """Fake nvcc only if no real CUDA toolkit is available."""
    if _have_real_nvcc():
        return
    try:
        import torch
        cuda_ver = torch.version.cuda or "12.1"
    except Exception:
        cuda_ver = "12.1"
    cuda_ver = ".".join(cuda_ver.split(".")[:2])
    shim = _ODV / ".cache" / "fake_cuda"
    nvcc = shim / "bin" / "nvcc"
    if not nvcc.exists():
        (shim / "bin").mkdir(parents=True, exist_ok=True)
        nvcc.write_text(
            "#!/usr/bin/env bash\n"
            'echo "nvcc: NVIDIA (R) Cuda compiler driver"\n'
            f'echo "Cuda compilation tools, release {cuda_ver}, V{cuda_ver}.0"\n'
        )
        os.chmod(nvcc, 0o755)
    os.environ.setdefault("CUDA_HOME", str(shim))
    os.environ["PATH"] = f"{shim / 'bin'}:{os.environ.get('PATH', '')}"


ensure_nvcc_shim()
