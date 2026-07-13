"""BEV overlay of what UniAD PERCEIVES vs ground truth, per frame, per episode.

Answers "how good is UniAD's detection?" directly: GT boxes (gray outline) vs the
decoded UniAD detections (colored by class, alpha by confidence) in the ego BEV
frame (up = forward, right = right). Reads the decoded .pth from
extract_uniad_features.py — no GPU, no re-running UniAD.

Usage (from repo root):
  .venv/bin/python scripts/visualize_uniad_detections.py \
      --episode episode_0000 \
      --infos data_carla/processed/parking_infos_epoch4test.pkl \
      --uniad-features-dir data_carla/processed/uniad_features \
      --score-thr 0.3
  # -> outputs/uniad_det_views/episode_0000/frame_XXXX.png (+ a filmstrip)
"""
import argparse
import pathlib
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

import paths

CLASS_COLORS = {"car": "tab:blue", "truck": "tab:orange", "pedestrian": "tab:red"}
UNIAD_CLASS_NAMES = ["car", "truck", "construction_vehicle", "bus", "trailer",
                     "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]


def _corners_plot(cx_fwd, cy_left, w, l, yaw):
    """Box (lidar x=fwd,y=left,w,l,yaw) -> 4 BEV plot corners (x=right, y=forward)."""
    along = np.array([np.cos(yaw), np.sin(yaw)])      # lidar (x,y) along length
    across = np.array([-np.sin(yaw), np.cos(yaw)])
    c = np.array([cx_fwd, cy_left])
    pts = []
    for a in (1, -1):
        for b in (1, -1):
            p = c + a * (l / 2) * along + b * (w / 2) * across
            pts.append([-p[1], p[0]])                 # (right, forward)
    return np.array([pts[0], pts[1], pts[3], pts[2]])  # ordered rectangle


def _draw_frame(ax, info, det, score_thr):
    gt = np.asarray(info["gt_boxes"], dtype=float).reshape(-1, 7)
    gt_names = [str(n) for n in info.get("gt_names", [])]
    for i in range(len(gt)):
        x, y, _z, w, l, h, yaw = gt[i]
        ax.add_patch(Polygon(_corners_plot(x, y, w, l, yaw), closed=True,
                             fill=False, edgecolor="0.6", lw=1.0, zorder=1))
    n_det = 0
    if det is not None and det.get("boxes") is not None:
        boxes = np.asarray(det["boxes"], dtype=float).reshape(-1, 9)
        scores = np.asarray(det["scores"], dtype=float).reshape(-1)
        labels = np.asarray(det["labels"]).reshape(-1).astype(int)
        for i in range(len(boxes)):
            if scores[i] < score_thr:
                continue
            n_det += 1
            x, y, _z, w, l, h, yaw = boxes[i, :7]
            cls = UNIAD_CLASS_NAMES[labels[i]] if 0 <= labels[i] < len(UNIAD_CLASS_NAMES) else "car"
            ax.add_patch(Polygon(_corners_plot(x, y, w, l, yaw), closed=True,
                                 fill=True, alpha=0.25 + 0.5 * min(1.0, float(scores[i])),
                                 facecolor=CLASS_COLORS.get(cls, "tab:green"),
                                 edgecolor=CLASS_COLORS.get(cls, "tab:green"), lw=1.2, zorder=2))
    # ego + slot
    ax.plot(0, 0, marker="^", color="black", ms=10, zorder=3)
    ts = info.get("target_slot", {})
    ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_aspect("equal")
    ax.axhline(0, color="0.9", lw=0.5); ax.axvline(0, color="0.9", lw=0.5)
    ax.set_title(f"{info['token']}  GT={len(gt)}  UniAD={n_det}", fontsize=8)
    return len(gt), n_det


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--infos", default=str(paths.PROCESSED_DIR / "parking_infos_epoch4test.pkl"))
    ap.add_argument("--uniad-features-dir", default=str(paths.PROCESSED_DIR / "uniad_features"))
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--source", choices=["track", "det"], default="track",
                    help="'track' = confirmed tracks; 'det' = raw detections (needs "
                         "features re-extracted with detections_raw)")
    ap.add_argument("--out-dir", default=str(paths.PROJECT_ROOT / "outputs" / "uniad_det_views"))
    ap.add_argument("--filmstrip-cols", type=int, default=6)
    args = ap.parse_args()
    DET_KEY = "detections_raw" if args.source == "det" else "detections"

    with open(args.infos, "rb") as f:
        infos = [i for i in pickle.load(f)["infos"] if i["scene_token"] == args.episode]
    if not infos:
        raise SystemExit(f"no frames for {args.episode} in {args.infos}")
    infos.sort(key=lambda x: x["frame_idx"])
    feat = pathlib.Path(args.uniad_features_dir)
    out = pathlib.Path(args.out_dir) / args.episode
    out.mkdir(parents=True, exist_ok=True)

    totals = []
    for info in infos:
        p = feat / f"{info['token']}.pth"
        det = None
        if p.exists():
            det = (torch.load(p, map_location="cpu", weights_only=False)
                   .get("result_track", {}) or {}).get(DET_KEY)
        fig, ax = plt.subplots(figsize=(5, 5))
        g, d = _draw_frame(ax, info, det, args.score_thr)
        totals.append((g, d))
        fig.savefig(out / f"{info['token']}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # filmstrip contact sheet
    n = len(infos); cols = args.filmstrip_cols; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax, info in zip(np.array(axes).ravel(), infos):
        p = feat / f"{info['token']}.pth"
        det = ((torch.load(p, map_location="cpu", weights_only=False)
                .get("result_track", {}) or {}).get(DET_KEY)) if p.exists() else None
        _draw_frame(ax, info, det, args.score_thr)
    for ax in np.array(axes).ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "_filmstrip.png", dpi=90); plt.close(fig)

    g = np.array([t[0] for t in totals]); d = np.array([t[1] for t in totals])
    print(f"{args.episode}: {n} frames -> {out}")
    print(f"  GT boxes/frame  mean={g.mean():.1f}  | UniAD dets/frame mean={d.mean():.1f} "
          f"({100*d.mean()/max(1,g.mean()):.0f}% of GT count)")
    print(f"  legend: gray outline = GT, colored (blue car/orange truck/red ped, alpha=conf) = UniAD")


if __name__ == "__main__":
    main()
