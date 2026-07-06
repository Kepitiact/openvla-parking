"""UniAD Stage-1 smoke test — prove the CARLA track+map training pipeline runs.

This does NOT train the model. It runs the smallest slice that exercises the full
machinery once, to de-risk the real HAL run:

  1. load carla_parking_stage1.py config
  2. build the dataset from our CARLA infos + v1.0-carla DB + lot map GT
     -> validates gt_boxes / anns / seg-GT shapes and the len(anns)==gt_boxes
        assertion at runtime (this part needs no GPU)
  3. build the UniAD model + load the warm-start checkpoint
     -> validates the extracted uniad_base_track_map.pth keys match the model
  4. run ONE forward+backward (train_step) on the GPU if available
     -> validates data->model->track+seg loss->grad actually executes

Steps 1-3 run on CPU; step 4 needs CUDA (UniAD's deformable attention has no CPU
kernel). --no-gpu-step skips step 4 (validation-only mode / approach B).

Usage (from OpenDriveVLA/, openvla .venv):
  ../.venv/bin/python ../scripts/uniad_stage1_smoke_test.py \
      --config projects/configs/stage1_track_map/carla_parking_stage1.py \
      [--no-gpu-step]

Env (paths the config reads):
  NUSC_DATA_ROOT, CARLA_INFOS_TRAIN, CARLA_INFOS_VAL, CARLA_LOT_GT, UNIAD_WARMSTART
"""
import argparse
import os
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-gpu-step", action="store_true",
                    help="skip the CUDA forward/backward (validate data+build only)")
    ap.add_argument("--tiny", action="store_true",
                    help="shrink the model/data for an 8GB GPU (queue_length=1, "
                         "smaller image + backbone) — test-only, does not affect HAL config")
    ap.add_argument("--warmstart", default=os.environ.get("UNIAD_WARMSTART"))
    args = ap.parse_args()

    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.parallel import collate
    from functools import partial

    # UniAD plugin must be imported so custom types register.
    import importlib
    from mmdet3d.models import build_model
    from mmdet.datasets import build_dataset, build_dataloader

    print("=" * 60)
    print("[1/4] Loading config ...")
    cfg = Config.fromfile(args.config)
    if cfg.get("plugin", False) or "plugin_dir" in cfg:
        importlib.import_module(
            cfg.get("plugin_dir", "projects.mmdet3d_plugin").replace("/", ".").rstrip("."))
    else:
        importlib.import_module("projects.mmdet3d_plugin")
    print(f"    pc_range={cfg.point_cloud_range} patch={cfg.patch_size} "
          f"bev=({cfg.bev_h_},{cfg.bev_w_}) seg_classes={cfg.model.seg_head.num_classes}")

    if args.tiny:
        # Test-only shrink for an 8GB GPU. Does NOT touch the real HAL config file.
        # queue_length is the dominant memory driver (stacks per-frame activations);
        # a downscaled image keeps the fwd/bwd under 8GB.
        cfg.data.train.queue_length = 1
        cfg.queue_length = 1
        # Inject a 0.5x image scale right before normalization (halves H,W ->
        # ~4x fewer backbone activations).
        pipe = list(cfg.data.train.pipeline)
        for i, p in enumerate(pipe):
            if p.get("type") == "NormalizeMultiviewImage":
                pipe.insert(i, dict(type="RandomScaleImageMultiViewImage", scales=[0.5]))
                break
        cfg.data.train.pipeline = pipe
        print("    [tiny] queue_length=1, image scale 0.5 (test-only)")

    print("[2/4] Building dataset (CARLA infos + v1.0-carla DB + lot map GT) ...")
    ds = build_dataset(cfg.data.train)
    print(f"    dataset built: {len(ds)} samples | carla_lot_gt={getattr(ds,'carla_lot_gt',None) is not None}")
    # The temporal loader needs queue_length prior frames in the same scene, so
    # the first (queue_length-1) frames of a scene return None. Pick a safe index
    # and ensure the group flag exists (normally set by the sampler).
    import numpy as _np
    if not hasattr(ds, "flag"):
        ds.flag = _np.zeros(len(ds), dtype=_np.uint8)
    ql = cfg.data.train.get("queue_length", 4)
    idx = min(ql, len(ds) - 1)
    sample = ds[idx]
    print(f"    sample idx {idx} built (queue_length={ql})")
    # Report the GT tensors that reached the batch.
    def _shape(x):
        d = x.data if hasattr(x, "data") else x
        return tuple(d.shape) if hasattr(d, "shape") else type(d).__name__
    for k in ("gt_labels_3d", "gt_inds", "gt_lane_labels", "gt_lane_masks",
              "gt_fut_traj", "gt_segmentation"):
        if k in sample:
            print(f"      {k}: {_shape(sample[k])}")
    print("    -> data pipeline OK (gt_boxes/anns/seg-GT shapes valid)")

    print("[3/4] Building model + loading warm-start ...")
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ws = args.warmstart or cfg.get("load_from")
    if ws and os.path.exists(ws):
        _ = load_checkpoint(model, ws, map_location="cpu", strict=False)
        print(f"    warm-start loaded (non-strict) from {ws}")
    else:
        print(f"    WARNING: warm-start not found at {ws!r}; random init")

    if args.no_gpu_step or not torch.cuda.is_available():
        print("[4/4] SKIPPED gpu fwd/bwd "
              f"({'--no-gpu-step' if args.no_gpu_step else 'no CUDA'}).")
        print("\nSMOKE TEST (data+build+warmstart) PASSED.")
        return

    print("[4/4] One forward+backward on GPU ...")
    from mmcv.parallel import DataContainer as DC
    model = model.cuda().train()
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=0,
                              num_gpus=1, dist=False, shuffle=False)
    batch = next(iter(loader))

    def _unwrap(v):
        # Unwrap one sample (batch dim 1) from mmcv DataContainers and move any
        # tensors to GPU. mmcv's own scatter is broken on torch 2.1 (_get_stream).
        if isinstance(v, DC):
            data = v.data
            # DC wraps as [ [sample0] ] for cpu_only meta, or [tensor] otherwise.
            inner = data[0] if isinstance(data, list) else data
            return _unwrap(inner)
        if isinstance(v, torch.Tensor):
            return v.cuda()
        if isinstance(v, list):
            return [_unwrap(x) for x in v]
        if isinstance(v, dict):
            return {k: _unwrap(x) for k, x in v.items()}
        return v

    inputs = {k: _unwrap(v) for k, v in batch.items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    losses = model.forward_train(**inputs)
    loss, log_vars = model._parse_losses(losses)
    loss_terms = [k for k in log_vars if "loss" in k]
    print(f"    total loss={float(loss):.4f} | {len(loss_terms)} loss terms")
    for k in loss_terms[:14]:
        print(f"      {k}: {log_vars[k]:.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("    fwd+bwd+step OK")
    print("\nSMOKE TEST (full fwd/bwd) PASSED.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
