"""
Merge LoRA adapter weights back into the base model for inference.

Usage (from OpenDriveVLA/ directory):
  python drivevla/merge_lora.py \
    --base ../checkpoints/OpenDriveVLA-0.5B \
    --lora ../checkpoints/OpenDriveVLA-0.5B-carla/epoch_010 \
    --out  ../checkpoints/OpenDriveVLA-0.5B-carla/merged
"""

import argparse
import os
import pathlib
import sys

# Bootstrap paths so `python drivevla/merge_lora.py` works without a shell wrapper:
# llava + projects live at the OpenDriveVLA root, mmdet3d in third_party, and
# DeepSpeed checks for nvcc on import (reuse the shim the other scripts create).
_ODV = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ODV / "third_party" / "mmdetection3d_1_0_0rc6"))
sys.path.insert(0, str(_ODV))
_SHIM = _ODV / ".cache" / "fake_cuda"
if not (_SHIM / "bin" / "nvcc").exists():
    (_SHIM / "bin").mkdir(parents=True, exist_ok=True)
    (_SHIM / "bin" / "nvcc").write_text(
        '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.1, V12.1.0"\n')
    os.chmod(_SHIM / "bin" / "nvcc", 0o755)
os.environ.setdefault("CUDA_HOME", str(_SHIM))
os.environ["PATH"] = f"{_SHIM / 'bin'}:{os.environ.get('PATH', '')}"

import torch
from peft import PeftModel
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to base OpenDriveVLA checkpoint")
    ap.add_argument("--lora", required=True, help="Path to LoRA checkpoint directory")
    ap.add_argument("--out",  required=True, help="Path to save merged model")
    return ap.parse_args()


def main():
    args = parse_args()

    print("Loading base model...")
    disable_torch_init()
    overwrite_config = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}
    tokenizer, model, _, _ = load_pretrained_model(
        args.base,
        model_base=None,
        model_name="llava_qwen",
        device_map="cpu",
        multimodal=True,
        attn_implementation="eager",
        overwrite_config=overwrite_config,
    )
    model = model.to(torch.float16)

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, args.lora, torch_dtype=torch.float16)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.out} ...")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("Done. Run inference with:")
    print(f"  bash scripts/eval_carla_parking.sh {args.out}")


if __name__ == "__main__":
    main()
