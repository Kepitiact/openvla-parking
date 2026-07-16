"""Pre-compute UniAD features + decoded detections for CARLA frames -> per-frame .pth.

Builds UniAD DIRECTLY from (config, checkpoint) — the same way
scripts/uniad_stage1_eval.py does — instead of going through the LLaVA/OpenDriveVLA
wrapper.

WHY (this is the fix for a real, silent bug):
  The old path called llava.model.builder.load_pretrained_model() to instantiate a
  1.46 GB LLaVA-Qwen model and then used ONLY its vision tower. But OpenDriveVLA-0.5B
  ships with `mm_tunable_parts` including `mm_vision_tower`, so its model.safetensors
  bakes 1742 `vision_model.*` weights — the ORIGINAL nuScenes UniAD. Loading it
  restored those weights OVER the CARLA-trained checkpoint the tower had just loaded,
  so extraction silently ran nuScenes UniAD on CARLA parking images (proved by weight
  signature: query_embedding.weight sum 1274.58 (nuScenes) vs 1321.52 (CARLA epoch_4)).
  Near-field recall was 0.04 instead of 1.00.

  Building the model straight from (config, checkpoint) makes that class of bug
  impossible: BOTH things the detector depends on — the weights and the BEV geometry —
  are now explicit at the call site. There is no LLM, no tokenizer, no collator, and
  no baked state to clobber anything.

  The LLaVA wrapper adds nothing to the perception path: the model inputs from both
  paths were verified byte-identical (same img sum, lidar2img, l2g, timestamp).

The output .pth schema is UNCHANGED — reasoning_data_gen.SceneRecord.from_uniad and
llava_arch.encode_vision_tower_result both depend on it, and must need zero edits.

Usage (from OpenDriveVLA/):
  python drivevla/extract_uniad_features.py \
      --config projects/configs/stage1_track_map/carla_parking_stage1.py \
      --checkpoint /abs/path/to/trained_uniad.pth \
      --conversations ../data_carla/processed/carla_conversations.json \
      --out-dir ../data_carla/processed/uniad_features \
      [--ann-file ../data_carla/processed/subset.pkl] [--max-samples N]

NOTE: --checkpoint is REQUIRED and has no default, on purpose. Two different models
have historically shared the filename `uniad_base_track_map.pth` (the 200 MB nuScenes
warm-start vs the CARLA-trained model); defaulting to a path is how you silently get
the wrong weights. Pass it explicitly.
"""

import argparse
import importlib
import json
import os
import pathlib
import sys

import _bootstrap  # noqa: F401  (sys.path + conditional nvcc shim; must precede mmdet3d)

import torch
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="projects/configs/stage1_track_map/carla_parking_stage1.py",
                    help="UniAD config — the SAME one training used. Supplies both the "
                         "model and the BEV geometry.")
    ap.add_argument("--checkpoint", required=True,
                    help="Trained CARLA UniAD checkpoint (absolute path). Required: "
                         "no default, so the wrong weights can never be picked up.")
    ap.add_argument("--conversations",
                    default="../data_carla/processed/carla_conversations.json")
    ap.add_argument("--ann-file", default=None,
                    help="Override the config's data.test.ann_file (the infos pkl), "
                         "e.g. to extract only a subset.")
    ap.add_argument("--out-dir", default="../data_carla/processed/uniad_features")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    # Shard BY SCENE across GPUs. Safe because UniAD's temporal state is per-scene
    # (uniad_e2e.py resets prev_bev whenever scene_token changes), so whole scenes are
    # independent. Frame order WITHIN a scene is preserved, which is what matters.
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    return ap.parse_args()


def _select_shard(data_infos, num_shards, shard_idx):
    """Keep only the scenes belonging to this shard, preserving frame order.

    Scenes are assigned longest-first to the currently-emptiest shard (LPT), because
    episodes vary a lot (14-147 frames) and naive round-robin leaves one shard running
    long after the others finish.
    """
    if num_shards <= 1:
        return data_infos
    order, lengths = [], {}
    for i in data_infos:
        s = i["scene_token"]
        if s not in lengths:
            lengths[s] = 0
            order.append(s)
        lengths[s] += 1

    loads = [0] * num_shards
    owner = {}
    for s in sorted(order, key=lambda s: (-lengths[s], s)):   # deterministic
        k = min(range(num_shards), key=lambda j: (loads[j], j))
        owner[s] = k
        loads[k] += lengths[s]

    mine = [i for i in data_infos if owner[i["scene_token"]] == shard_idx]
    print(f"[shard {shard_idx}/{num_shards}] {len(mine)} frames, "
          f"{sum(1 for s in owner.values() if s == shard_idx)} scenes "
          f"(shard loads: {loads})")
    return mine


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    return x


def _decode(rt, keys, token, what):
    """Pull a (boxes, scores, labels) triple out of the track result as plain CPU
    tensors, so loading a .pth needs no mmdet3d box classes."""
    try:
        bk, sk, lk = keys
        b = rt.get(bk)
        if b is None:
            return None
        b = b.tensor if hasattr(b, "tensor") else b
        return {"boxes": b.detach().cpu(),
                "scores": rt[sk].detach().cpu(),
                "labels": rt[lk].detach().cpu()}
    except Exception as e:  # pragma: no cover - defensive, mirrors previous behaviour
        print(f"  warn: no {what} for {token}: {e}")
        return None


def main():
    args = parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    import torch.distributed as dist

    # Some UniAD collect/reduce ops assume a process group exists; make a 1-proc one.
    # NEVER hardcode the port: sharded extraction runs several of these concurrently on
    # ONE node, and they would all fight over the same port ("Address already in use").
    # world_size=1, so any free port is fine — let the OS pick.
    if not dist.is_initialized():
        import socket

        def _free_port():
            s = socket.socket()
            s.bind(("", 0))
            port = s.getsockname()[1]
            s.close()
            return str(port)

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _free_port())
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(int(os.environ.get("SHARD_GPU", 0)))
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    cfg = Config.fromfile(args.config)
    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        importlib.import_module(os.path.dirname(plugin_dir).replace("/", "."))

    from mmdet.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model

    from drivevla.utils.remove_mmlab_datacontainer import remove_datacontainer
    from drivevla.utils.tensor_utils import move_data_to_device

    # The geometry the detector will actually use, printed so it can never be a
    # silent surprise again (a wrong point_cloud_range is exactly how the last bug hid).
    print(f"[cfg] {args.config}")
    print(f"[cfg] point_cloud_range = {cfg.point_cloud_range}")
    print(f"[cfg] patch_size        = {cfg.patch_size}")
    print(f"[ckpt] {args.checkpoint}")

    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file

    print("[1/3] Building dataset ...")
    dataset = build_dataset(cfg.data.test)
    dataset.data_infos = _select_shard(dataset.data_infos,
                                       args.num_shards, args.shard_idx)
    if args.max_samples is not None:
        dataset.data_infos = dataset.data_infos[:args.max_samples]
    # shuffle=False keeps frames in scene order — UniAD is temporally stateful, so the
    # sequence must not be permuted (and frames must not be skipped; see below).
    loader = build_dataloader(dataset, samples_per_gpu=1,
                              workers_per_gpu=args.num_workers,
                              dist=False, shuffle=False)
    print(f"    {len(dataset)} frames")

    print("[2/3] Building UniAD + loading checkpoint ...")
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda().eval()

    print("[3/3] Extracting ...")
    saved = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for i, data in enumerate(tqdm(loader, desc="Extracting UniAD features")):
                token = dataset.data_infos[i]["token"]

                # Same unwrap contract as projects/.../uniad/apis/test.py
                data = remove_datacontainer(data)
                data["img_metas"] = data["img_metas"][0]
                data["img"] = data["img"][0]
                data = move_data_to_device(data, "cuda")

                # forward_test returns (result, results_for_vlm) — the vision tower used
                # exactly this call; we just skip the wrapper around it.
                _result, results_for_vlm = model(return_loss=False, rescale=True, **data)

                results_cpu = _to_cpu(results_for_vlm)
                rt = results_cpu.get("result_track", {})
                rs = results_cpu.get("result_seg", {})

                # Confirmed tracks. track_bbox_results is [(boxes_3d, scores, labels,
                # bbox_index, mask)]; boxes_3d.tensor is [N,9] ego (x,y,z,w,l,h,yaw,vx,vy).
                det = None
                try:
                    tbr = rt.get("track_bbox_results")
                    if tbr:
                        boxes_3d, scores, labels = tbr[0][0], tbr[0][1], tbr[0][2]
                        box_t = boxes_3d.tensor if hasattr(boxes_3d, "tensor") else boxes_3d
                        det = {"boxes": box_t.detach().cpu(),
                               "scores": scores.detach().cpu(),
                               "labels": labels.detach().cpu()}
                except Exception as e:
                    print(f"  warn: could not extract detections for {token}: {e}")

                # Raw per-frame detections (pre-tracking, ~300 queries) — denser.
                det_raw = _decode(rt, ("boxes_3d_det", "scores_3d_det", "labels_3d_det"),
                                  token, "raw detections")

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
                torch.save(slim, out_dir / f"{token}.pth")
                saved += 1

    print(f"\nDone. Saved {saved} feature files -> {out_dir}")
    # NOTE: every frame is (re)processed. We deliberately do NOT skip existing files:
    # UniAD carries temporal state across frames within a scene, so skipping a frame
    # would feed the next one a stale prev_bev and silently corrupt its features.

    if args.num_shards > 1:
        # Every shard would read-modify-write the same JSON concurrently and corrupt it.
        # Stamp once, after all shards finish (see scripts/sbatch/hal_extract_features.sbatch).
        print(f"[shard {args.shard_idx}] skipping conversations stamping (sharded run); "
              "stamp once after all shards complete.")
    elif args.conversations and os.path.exists(args.conversations):
        print(f"Updating {args.conversations} with uniad_pth paths...")
        with open(args.conversations) as f:
            convs = json.load(f)
        out_abs = out_dir.resolve()
        updated = 0
        for entry in convs:
            token = entry.get("sample_id") or entry.get("qa_id", "").removesuffix("_trajectory")
            p = out_abs / f"{token}.pth"
            if p.exists():
                entry["uniad_pth"] = str(p)
                updated += 1
        with open(args.conversations, "w") as f:
            json.dump(convs, f)
        print(f"Updated {updated}/{len(convs)} conversations with uniad_pth paths.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
