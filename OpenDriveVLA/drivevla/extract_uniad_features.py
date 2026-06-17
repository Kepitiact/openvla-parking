"""
Pre-compute UniAD BEV features for all CARLA frames and save as .pth files.

This is a prerequisite for fine-tuning. Running UniAD once per frame during
feature extraction is far cheaper than running it on every training step.

After this script:
  data_carla/processed/uniad_features/<token>.pth  for every frame
  data_carla/processed/carla_conversations.json updated with uniad_pth paths

Usage (from OpenDriveVLA/ directory):
  bash scripts/extract_carla_features.sh
  # or directly:
  PYTHONPATH=$(pwd):$PYTHONPATH python drivevla/extract_uniad_features.py
"""

import argparse
import json
import os
import pathlib
import sys

# Bootstrap paths so this runs standalone (llava/projects live at the repo root,
# mmdet3d in third_party) and DeepSpeed's nvcc check passes (reuse the shared shim).
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
import torch.distributed as dist

from tqdm import tqdm
from mmengine import Config
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.train.train import DataArguments

from data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset
from data_utils.nuscenes_llava_distributed_sampler import ContinuousSceneDistributedSampler
from torch.utils.data import DataLoader, SequentialSampler


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="../checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--uniad-config", default="projects/configs/stage1_track_map/carla_parking.py")
    ap.add_argument("--conversations", default="../data_carla/processed/carla_conversations.json")
    ap.add_argument("--out-dir", default="../data_carla/processed/uniad_features")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (same as inference)
    disable_torch_init()
    overwrite_config = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        model_base=None,
        model_name="llava_qwen",
        device_map=str(device),
        multimodal=True,
        attn_implementation="eager",
        overwrite_config=overwrite_config,
    )
    model.eval()
    vision_tower = model.get_vision_tower()

    # Dataset (test mode, no uniad_pth — we're generating them)
    uniad_cfg = Config.fromfile(args.uniad_config)
    data_args = DataArguments(
        data_path=args.conversations,
        lazy_preprocess=True,
        frames_upbound=32,
    )
    dataset = LLaVANuScenesDataset(
        tokenizer,
        data_args,
        uniad_cfg.data.test,
        llava_test_mode=True,
        use_uniad_pth=False,
    )

    # Must process frames in scene order (UniAD is temporally stateful)
    sampler = ContinuousSceneDistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=False,
        drop_last=False,
    )
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collator)

    skipped = 0
    saved = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for batch in tqdm(loader, desc="Extracting UniAD features"):
                sample_id = batch["id"][0] if isinstance(batch["id"], list) else batch["id"]
                token = str(sample_id).removesuffix("_trajectory")
                out_path = out_dir / f"{token}.pth"

                if out_path.exists():
                    skipped += 1
                    continue

                uniad_data = batch.get("uniad_data")
                if uniad_data is None:
                    continue

                # Move data to device
                def _to_device(x):
                    if isinstance(x, torch.Tensor):
                        return x.to(device)
                    if isinstance(x, dict):
                        return {k: _to_device(v) for k, v in x.items()}
                    if isinstance(x, list):
                        return [_to_device(v) for v in x]
                    return x

                uniad_data = _to_device(uniad_data)
                results_for_vlm = vision_tower(uniad_data)

                # Move result to CPU before saving to avoid GPU OOM when accumulating
                def _to_cpu(x):
                    if isinstance(x, torch.Tensor):
                        return x.cpu()
                    if isinstance(x, dict):
                        return {k: _to_cpu(v) for k, v in x.items()}
                    if isinstance(x, list):
                        return [_to_cpu(v) for v in x]
                    return x

                results_cpu = _to_cpu(results_for_vlm)
                # Keep only the fields training reads (see
                # llava_arch.encode_vision_tower_result). The full result carries
                # bev_embed (~41MB) plus detection/tracking outputs that the LLM
                # never consumes — saving them all needs ~46MB/frame (~2.3TB for
                # the dataset). The slim dict is ~1.2MB/frame (~58GB total).
                rt = results_cpu.get("result_track", {})
                rs = results_cpu.get("result_seg", {})
                slim = {
                    "scene_token": results_cpu.get("scene_token"),
                    "sample_token": results_cpu.get("sample_token"),
                    "result_track": {
                        "track_query_embeddings": rt.get("track_query_embeddings"),
                        "img_feat_2D": rt.get("img_feat_2D"),
                        "track_gt_inds_to_embed_idx": rt.get("track_gt_inds_to_embed_idx"),
                    },
                    "result_seg": {
                        "chosen_output_query_things": rs.get("chosen_output_query_things"),
                        "output_query_stuff": rs.get("output_query_stuff"),
                    },
                    "planning_gt": results_cpu.get("planning_gt"),
                }
                torch.save(slim, out_path)
                saved += 1

    print(f"\nDone. Saved {saved} new feature files, skipped {skipped} existing.")

    # Update conversations.json with uniad_pth paths
    print(f"Updating {args.conversations} with uniad_pth paths...")
    with open(args.conversations) as f:
        convs = json.load(f)

    out_dir_abs = out_dir.resolve()
    updated = 0
    for entry in convs:
        token = entry.get("sample_id") or entry.get("qa_id", "").removesuffix("_trajectory")
        pth_path = out_dir_abs / f"{token}.pth"
        if pth_path.exists():
            entry["uniad_pth"] = str(pth_path)
            updated += 1

    with open(args.conversations, "w") as f:
        json.dump(convs, f)

    print(f"Updated {updated}/{len(convs)} conversations with uniad_pth paths.")
    print(f"Ready for training. Run: bash scripts/train_carla_parking.sh")


if __name__ == "__main__":
    main()
