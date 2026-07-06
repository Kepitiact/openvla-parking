"""Standalone test: CarlaVectorMap -> UniAD preprocess_map -> BEV seg mask PNG.

Verifies approach-B lot map GT rasterizes correctly in UniAD's exact BEV canvas,
using a real CARLA ego pose from the infos pkl. No training / GPU needed.

Usage (openvla .venv):
  .venv/bin/python scripts/test_carla_seg_gt.py \
      --lot_gt /home/.../data/processed/lot_map_gt_Town04_Opt.json \
      --infos /tmp/test_infos.pkl --frame 0 --out /tmp/seg_gt.png
"""
import argparse
import os
import pathlib
import pickle
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "OpenDriveVLA"))

from carla_vector_map import CarlaVectorMap

# Import UniAD's rasterize module directly by file path — going through the
# mmdet3d_plugin package __init__ pulls in mmdet3d (heavy / GPU stack).
import importlib.util as _ilu
_rast_path = (_HERE.parent / "OpenDriveVLA" / "projects" / "mmdet3d_plugin" /
              "datasets" / "data_utils" / "rasterize.py")
_spec = _ilu.spec_from_file_location("uniad_rasterize", str(_rast_path))
_rast = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rast)
preprocess_map = _rast.preprocess_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lot_gt", required=True)
    ap.add_argument("--infos", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="/tmp/seg_gt.png")
    args = ap.parse_args()

    patch_size = (102.4, 102.4)
    canvas_size = (200, 200)
    num_classes = 3
    thickness = 2
    angle_class = 36

    vm = CarlaVectorMap(args.lot_gt, patch_size=patch_size, canvas_size=canvas_size)
    infos = pickle.load(open(args.infos, "rb"))["infos"]
    info = infos[args.frame]
    ego_t = info["ego2global_translation"]
    ego_r = info["ego2global_rotation"]

    vectors = vm.gen_vectorized_samples("carla", ego_t, ego_r)
    by_type = {}
    for v in vectors:
        by_type[v["type"]] = by_type.get(v["type"], 0) + 1
    print(f"ego xy={np.round(ego_t[:2],1)} | vectors: {len(vectors)} | by type: {by_type}")

    semantic, instance, _, _ = preprocess_map(
        vectors, patch_size, canvas_size, num_classes, thickness, angle_class)
    print(f"semantic_masks shape: {semantic.shape} | nonzero per class: "
          f"{[int((semantic[c]!=0).sum()) for c in range(num_classes)]}")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, num_classes + 1, figsize=(4 * (num_classes + 1), 4))
    names = ["divider(0)", "ped_crossing(1)", "boundary(2)"]
    for c in range(num_classes):
        axs[c].imshow(semantic[c], cmap="gray", origin="lower")
        axs[c].set_title(f"{names[c]}: {int((semantic[c]!=0).sum())} px")
        axs[c].plot(canvas_size[1] / 2, canvas_size[0] / 2, "r+", ms=12)  # ego centre
    combo = np.zeros((*canvas_size, 3))
    combo[..., 0] = semantic[2]      # boundary -> red (slot+lot outlines)
    combo[..., 1] = semantic[0]      # divider  -> green (aisle)
    axs[-1].imshow(combo, origin="lower")
    axs[-1].plot(canvas_size[1] / 2, canvas_size[0] / 2, "w+", ms=12)
    axs[-1].set_title("combined (ego=+)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
