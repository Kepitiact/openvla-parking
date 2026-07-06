"""Extract the UniAD track+map perception weights from a merged OpenDriveVLA
checkpoint into a standalone `uniad_base_track_map.pth` for Stage-1 warm-start.

The fine-tuned OpenDriveVLA model bundles the full UniAD model as its vision
tower under `model.vision_tower.vision_tower.vision_model.*` (img_backbone,
img_neck, pts_bbox_head=track/det, seg_head=map/lot, query_interact, memory_bank,
...). Stage-1 training (`load_from = ckpts/uniad_base_track_map.pth`) expects a
plain UniAD state_dict, so we strip that prefix and save in mmcv checkpoint form
({'state_dict': ..., 'meta': {...}}).

Usage (openvla .venv):
  .venv/bin/python scripts/extract_uniad_from_opendrivevla.py \
      --merged checkpoints/OpenDriveVLA-0.5B-carla/merged/model.safetensors \
      --out OpenDriveVLA/ckpts/uniad_base_track_map.pth
"""
import argparse
import pathlib

import torch
from safetensors import safe_open

PREFIX = "model.vision_tower.vision_tower.vision_model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged model.safetensors")
    ap.add_argument("--out", required=True, help="output uniad_base_track_map.pth")
    args = ap.parse_args()

    state = {}
    with safe_open(args.merged, framework="pt") as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                state[k[len(PREFIX):]] = f.get_tensor(k)

    if not state:
        raise SystemExit(f"No tensors with prefix {PREFIX!r} in {args.merged}")

    # Report submodule coverage.
    tops = {}
    for k in state:
        t = k.split(".")[0]
        tops[t] = tops.get(t, 0) + 1
    print(f"Extracted {len(state)} UniAD tensors:")
    for t, c in sorted(tops.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": state,
        "meta": {
            "note": "UniAD track+map weights extracted from OpenDriveVLA merged "
                    "checkpoint for CARLA Stage-1 warm-start.",
        },
    }
    torch.save(ckpt, str(out))
    print(f"\nSaved warm-start checkpoint -> {out} ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
