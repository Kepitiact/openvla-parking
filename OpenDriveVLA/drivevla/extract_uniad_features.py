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

import _bootstrap  # noqa: F401  (sys.path + conditional nvcc shim; must precede llava/mmdet3d)

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
    ap.add_argument("--ann-file", default=None,
                    help="Override the UniAD config's data.test.ann_file (the infos pkl). "
                         "Use to extract only a subset, e.g. dagger_infos.pkl.")
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

    # CRITICAL: OpenDriveVLA-0.5B was trained with `mm_tunable_parts` including the
    # vision tower, so its checkpoint bakes the ORIGINAL nuScenes UniAD weights. The
    # load_pretrained_model call above restores those over the CARLA-trained epoch_4
    # that the tower loaded at build time, i.e. extraction would silently run nuScenes
    # UniAD on CARLA images (near-field detection collapses). Force our checkpoint back
    # into the UniAD backbone so the extracted features reflect the model we trained.
    from mmcv.runner import load_checkpoint as _load_uniad_ckpt
    _uniad_ckpt = os.environ.get("UNIAD_CKPT", "checkpoints/uniad_base_track_map.pth")
    _load_uniad_ckpt(vision_tower.vision_tower.vision_model, _uniad_ckpt, map_location="cpu")
    print(f"[fix] reloaded CARLA UniAD weights into vision tower from {_uniad_ckpt}")

    # Dataset (test mode, no uniad_pth — we're generating them)
    uniad_cfg = Config.fromfile(args.uniad_config)
    if args.ann_file:
        uniad_cfg.data.test.ann_file = args.ann_file
    data_args = DataArguments(
        data_path=args.conversations,
        lazy_preprocess=True,
        # nuScenes multi-frame cap; unused for single-frame CARLA parking (D4).
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
                # llava_arch.encode_vision_tower_result), plus the decoded
                # detections needed to build reasoning data. We still drop the
                # big bev_embed (~41MB/frame). track_query_embeddings + img_feat
                # are ~1.2MB/frame; the decoded boxes add only a few KB.
                rt = results_cpu.get("result_track", {})
                rs = results_cpu.get("result_seg", {})

                # Decoded objects UniAD perceives this frame, as plain CPU tensors
                # (so loading doesn't require mmdet3d box classes). track_bbox_results
                # is [(boxes_3d, scores, labels, bbox_index, mask)]; boxes_3d.tensor
                # is [N, 9] = ego-frame (x,y,z,w,l,h,yaw,vx,vy). None when no detections.
                det = None
                try:
                    tbr = rt.get("track_bbox_results")
                    if tbr:
                        boxes_3d, scores, labels = tbr[0][0], tbr[0][1], tbr[0][2]
                        box_t = boxes_3d.tensor if hasattr(boxes_3d, "tensor") else boxes_3d
                        det = {
                            "boxes": box_t.detach().cpu(),
                            "scores": scores.detach().cpu(),
                            "labels": labels.detach().cpu(),
                        }
                except Exception as e:
                    print(f"  warn: could not extract detections for {token}: {e}")

                # Raw per-frame detections (boxes_3d_det, ~300 queries, pre-tracking) —
                # denser than the confirmed tracks; lets us compare/feed raw vs tracked.
                det_raw = None
                try:
                    if rt.get("boxes_3d_det") is not None:
                        bt = rt["boxes_3d_det"]
                        bt = bt.tensor if hasattr(bt, "tensor") else bt
                        det_raw = {
                            "boxes": bt.detach().cpu(),
                            "scores": rt["scores_3d_det"].detach().cpu(),
                            "labels": rt["labels_3d_det"].detach().cpu(),
                        }
                except Exception as e:
                    print(f"  warn: no raw detections for {token}: {e}")

                slim = {
                    "scene_token": results_cpu.get("scene_token"),
                    "sample_token": results_cpu.get("sample_token"),
                    "result_track": {
                        "track_query_embeddings": rt.get("track_query_embeddings"),
                        "img_feat_2D": rt.get("img_feat_2D"),
                        "track_gt_inds_to_embed_idx": rt.get("track_gt_inds_to_embed_idx"),
                        "detections": det,
                        "detections_raw": det_raw,
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
