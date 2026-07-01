"""
Merge LoRA adapter weights back into the base model for inference.

Usage (from OpenDriveVLA/ directory):
  python drivevla/merge_lora.py \
    --base ../checkpoints/OpenDriveVLA-0.5B \
    --lora ../checkpoints/OpenDriveVLA-0.5B-carla/epoch_010 \
    --out  ../checkpoints/OpenDriveVLA-0.5B-carla/merged
"""

import argparse
import json
import os
import pathlib
import sys

import _bootstrap  # noqa: F401  (sys.path + conditional nvcc shim; must precede llava/mmdet3d)

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

    # Guard against merging a LoRA adapter onto the wrong base checkpoint: the
    # adapter records the base it was trained on. Mismatched bases merge silently
    # but produce garbage weights, so fail clearly here.
    adapter_cfg_path = os.path.join(args.lora, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            trained_base = json.load(f).get("base_model_name_or_path")
        if trained_base:
            same = (os.path.basename(os.path.normpath(trained_base))
                    == os.path.basename(os.path.normpath(args.base)))
            try:
                same = same or os.path.samefile(trained_base, args.base)
            except OSError:
                pass
            if not same:
                raise SystemExit(
                    f"--base mismatch: adapter was trained on "
                    f"'{trained_base}' but --base is '{args.base}'.\n"
                    "Pass the same base checkpoint used for training, "
                    "or fix the adapter's base_model_name_or_path."
                )

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
