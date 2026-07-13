"""UniAD Stage-1 inference: run the trained model over val frames and DUMP predictions.

Open-loop, no simulator, no nuScenes eval toolkit. Scoring is a separate CPU step
(scripts/uniad_stage1_metrics.py) that matches these predictions to GT from the val
infos. This keeps the expensive GPU inference separate from cheap, iterable scoring.

Run from OpenDriveVLA/ (like training), so the config paths resolve:
  python ../scripts/uniad_stage1_eval.py --config <cfg> --checkpoint <abs/epoch_N.pth> \
      --dump-out ../checkpoints/stage1_carla_full/preds_epoch2.pkl [--max-samples N]
"""
import argparse
import importlib
import os
import pickle
import sys
import traceback


def _np(x):
    import numpy as np
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dump-out", required=True, help="output path for the predictions pkl")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="only run the first N val frames (dry-run subset)")
    ap.add_argument("--source", choices=["track", "det"], default="track",
                    help="'track' = confirmed tracked boxes (boxes_3d); "
                         "'det' = raw per-frame detections (boxes_3d_det, ~300/frame)")
    args = ap.parse_args()
    BK, SK, LK = ({"track": ("boxes_3d", "scores_3d", "labels_3d"),
                   "det": ("boxes_3d_det", "scores_3d_det", "labels_3d_det")}[args.source])

    import numpy as np
    import torch
    import torch.distributed as dist
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    # Some collection/reduce ops assume a process group; init a trivial 1-proc group.
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29599")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    cfg = Config.fromfile(args.config)
    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        importlib.import_module(os.path.dirname(plugin_dir).replace("/", "."))

    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.uniad.apis.test import custom_single_gpu_test

    print("[1/3] Building val dataset ...")
    dataset = build_dataset(cfg.data.test)
    if args.max_samples is not None:
        dataset.data_infos = dataset.data_infos[:args.max_samples]
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1,
                              dist=False, shuffle=False)
    class_names = list(cfg.class_names)
    print(f"    {len(dataset)} val frames | classes={class_names}")

    print("[2/3] Building model + loading checkpoint ...")
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda().eval()

    print("[3/3] Running inference ...")
    results = custom_single_gpu_test(model, loader)
    # custom_single_gpu_test returns a dict of result streams; the per-frame
    # detection/track predictions are under 'bbox_results'.
    if isinstance(results, dict):
        results = results.get("bbox_results", results)

    # Print the full structure of one result (1 level of nesting) so extraction is verifiable.
    def _struct(d, pre="    ", depth=0):
        if isinstance(d, dict):
            for k, v in d.items():
                shp = getattr(getattr(v, "tensor", v), "shape", None)
                print(f"{pre}{k}: {type(v).__name__}{'' if shp is None else ' ' + str(tuple(shp))}")
                if isinstance(v, dict) and depth < 1:
                    _struct(v, pre + "    ", depth + 1)
    print("[struct] results[0]:")
    if results:
        _struct(results[0])

    def _find_boxes(det):
        # boxes may sit at top level or under 'pts_bbox'
        if isinstance(det, dict):
            if BK in det:
                return det
            if isinstance(det.get("pts_bbox"), dict) and BK in det["pts_bbox"]:
                return det["pts_bbox"]
        return None

    print(f"[source] scoring '{args.source}' boxes ({BK})")
    preds = []
    for i, det in enumerate(results):
        d = _find_boxes(det)
        token = dataset.data_infos[i]["token"]
        if d is None:
            preds.append(dict(token=token, boxes=np.zeros((0, 7), np.float32),
                              scores=np.zeros((0,), np.float32),
                              labels=np.zeros((0,), np.int64),
                              track_ids=np.full((0,), -1, np.int64)))
            continue
        boxes = _np(d[BK].tensor)[:, :7].astype(np.float32)
        scores = _np(d[SK]).astype(np.float32)
        labels = _np(d[LK]).astype(np.int64)
        tk = None
        if args.source == "track":
            for key in ("track_ids", "instance_ids", "obj_idxes", "track_scores"):
                if key in d:
                    tk = _np(d[key]).reshape(-1).astype(np.int64)
                    break
        track_ids = tk if (tk is not None and len(tk) == len(scores)) \
            else np.full(len(scores), -1, np.int64)
        preds.append(dict(token=token, boxes=boxes, scores=scores,
                          labels=labels, track_ids=track_ids))

    os.makedirs(os.path.dirname(os.path.abspath(args.dump_out)), exist_ok=True)
    with open(args.dump_out, "wb") as f:
        pickle.dump(dict(class_names=class_names, preds=preds), f)
    print(f"\nDumped predictions for {len(preds)} frames -> {args.dump_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
