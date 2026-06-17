"""
Rewrite pre-extracted UniAD feature files to the minimal set training reads.

The full UniAD output carries bev_embed (~41MB) plus detection/tracking tensors
the language model never consumes (see llava_arch.encode_vision_tower_result).
This shrinks each file from ~46MB to ~1.2MB. Safe to re-run: already-slim files
(below --slim-threshold-mb) are skipped, and writes are atomic (temp + rename)
so an interruption cannot corrupt a file.

Run from the repo root:
  python scripts/slim_uniad_features.py
"""

import argparse
import os
import pathlib
import sys

import paths

# OpenDriveVLA internals are needed to unpickle the saved objects.
_ODV = paths.PROJECT_ROOT / "OpenDriveVLA"
sys.path.insert(0, str(_ODV / "third_party" / "mmdetection3d_1_0_0rc6"))
sys.path.insert(0, str(_ODV))
# DeepSpeed checks for nvcc on import; reuse the shim the extraction run created.
_SHIM = _ODV / ".cache" / "fake_cuda"
os.environ.setdefault("CUDA_HOME", str(_SHIM))
os.environ["PATH"] = f"{_SHIM / 'bin'}:{os.environ.get('PATH', '')}"

import torch  # noqa: E402  (after sys.path / env setup)


def slim_dict(d):
    """Keep only the fields read by llava_arch.encode_vision_tower_result."""
    rt = d.get("result_track", {}) or {}
    rs = d.get("result_seg", {}) or {}
    return {
        "scene_token": d.get("scene_token"),
        "sample_token": d.get("sample_token"),
        "result_track": {
            "track_query_embeddings": rt.get("track_query_embeddings"),
            "img_feat_2D": rt.get("img_feat_2D"),
            "track_gt_inds_to_embed_idx": rt.get("track_gt_inds_to_embed_idx"),
        },
        "result_seg": {
            "chosen_output_query_things": rs.get("chosen_output_query_things"),
            "output_query_stuff": rs.get("output_query_stuff"),
        },
        "planning_gt": d.get("planning_gt"),
    }


def main():
    ap = argparse.ArgumentParser(description="Slim pre-extracted UniAD feature .pth files in place.")
    ap.add_argument("--features-dir", type=pathlib.Path, default=paths.FEATURES_DIR)
    ap.add_argument("--slim-threshold-mb", type=float, default=5.0,
                    help="Files already smaller than this are assumed slim and skipped.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = all).")
    args = ap.parse_args()

    files = sorted(args.features_dir.glob("*.pth"))
    if args.limit:
        files = files[:args.limit]
    threshold = args.slim_threshold_mb * 1e6

    freed = 0
    slimmed = 0
    skipped = 0
    failed = 0
    for i, f in enumerate(files, 1):
        before = f.stat().st_size
        if before < threshold:
            skipped += 1
            continue
        try:
            d = torch.load(f, map_location="cpu")
            slim = slim_dict(d)
            tmp = f.with_suffix(".pth.tmp")
            torch.save(slim, tmp)
            os.replace(tmp, f)  # atomic
            freed += before - f.stat().st_size
            slimmed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED {f.name}: {e}")
        if i % 200 == 0:
            print(f"  {i}/{len(files)}  slimmed={slimmed} skipped={skipped} freed={freed/1e9:.1f}GB")

    print(f"\nDone. slimmed={slimmed} skipped={skipped} failed={failed} "
          f"freed={freed/1e9:.1f}GB")


if __name__ == "__main__":
    main()
