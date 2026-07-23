"""
Stitch one episode's predicted vs ground-truth trajectories into two views:

  episode_overview.png  - whole episode in a common (global) frame: the car's
                          actual driven path (green) with the model's predicted
                          future overlaid at each step (red).
  episode_filmstrip.png - per-frame strip: front camera image (top) + the
                          ego-local GT/pred trajectory plot (bottom).

Reuses the data/coordinate conventions from inspect_opendrivevla_sample.py.

Usage (from repo root):
  python scripts/visualize_episode.py --episode episode_0000 \
    --predictions OpenDriveVLA/output/merged/<TS>/results/plan_conv_rank0.json
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import paths
from inspect_opendrivevla_sample import (
    load_infos,
    load_prediction_map,
    resolve_predictions_path,
    resolve_image_path,
    build_ego_trajectories,
    quaternion_to_yaw,
)


def local_to_global(local_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Inverse of inspect_opendrivevla_sample.global_to_local (x=right, y=forward)."""
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([math.sin(yaw), -math.cos(yaw)])
    local_xy = np.atleast_2d(local_xy)
    return origin_xy + local_xy[:, 0:1] * right + local_xy[:, 1:2] * forward


def episode_frames(infos: list[dict], episode: str) -> list[dict]:
    frames = [i for i in infos if i["token"].startswith(episode + "_")]
    frames.sort(key=lambda i: int(i["frame_idx"]))
    return frames


def make_overview(frames, token_to_info, pred_map, stride, out_path):
    fig, ax = plt.subplots(figsize=(9, 9))

    # Actual driven path = the ego positions through the episode (global frame).
    path = np.array([f["ego2global_translation"][:2] for f in frames], dtype=float)
    ax.plot(path[:, 0], path[:, 1], "-", color="tab:green", linewidth=2.5,
            label="Actual path (GT)", zorder=2)
    ax.scatter(*path[0], s=120, marker="o", color="green", zorder=4, label="start")
    ax.scatter(*path[-1], s=160, marker="*", color="black", zorder=4, label="end (parked)")

    # Model's predicted future at each (strided) frame, transformed to global.
    labeled = False
    for f in frames[::stride]:
        entry = pred_map.get(f["token"], {})
        pred = entry.get("trajectory", [])
        if not pred:
            continue
        origin = np.asarray(f["ego2global_translation"][:2], dtype=float)
        yaw = quaternion_to_yaw(f["ego2global_rotation"])
        g = local_to_global(np.asarray(pred, dtype=float), origin, yaw)
        ax.plot(g[:, 0], g[:, 1], "--", color="tab:red", linewidth=1.0, alpha=0.6,
                zorder=3, label=None if labeled else "Predicted future (per step)")
        labeled = True

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("global x (m)")
    ax.set_ylabel("global y (m)")
    ax.set_title(f"Episode overview: {frames[0]['token'].rsplit('_', 1)[0]}  ({len(frames)} frames)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_filmstrip(frames, token_to_info, pred_map, data_root, max_frames, out_path):
    # Sample evenly so the strip stays readable.
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        sel = [frames[i] for i in idx]
    else:
        sel = frames

    n = len(sel)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, f in enumerate(sel):
        # Top: front camera (real GT image)
        ax_img = axes[0, col]
        img_path = resolve_image_path(data_root, f["cams"]["CAM_FRONT"]["data_path"])
        try:
            ax_img.imshow(Image.open(img_path).convert("RGB"))
        except Exception as e:
            ax_img.text(0.5, 0.5, f"no image\n{e}", ha="center", va="center", fontsize=6)
        rev = " R" if f.get("reverse") else ""
        ax_img.set_title(f"f{int(f['frame_idx']):04d}{rev}", fontsize=9)
        ax_img.axis("off")

        # Bottom: ego-local GT vs predicted
        ax_p = axes[1, col]
        hist, fut, _, _ = build_ego_trajectories(f, token_to_info, history_steps=4, future_steps=6)
        if len(hist):
            ax_p.plot(hist[:, 0], hist[:, 1], "o-", color="tab:blue", ms=3, lw=1.5)
        if len(fut):
            ax_p.plot(fut[:, 0], fut[:, 1], "o-", color="tab:green", ms=3, lw=1.5)
        entry = pred_map.get(f["token"], {})
        pred = entry.get("trajectory", [])
        if pred:
            p = np.asarray(pred, dtype=float)
            ax_p.plot(p[:, 0], p[:, 1], "o--", color="tab:red", ms=3, lw=1.5)
        ax_p.scatter([0], [0], marker="*", color="black", s=60)
        ax_p.set_aspect("equal", adjustable="box")
        ax_p.grid(True, alpha=0.2)
        ax_p.tick_params(labelsize=6)

        # The model's REASONING for this frame, wrapped under the trajectory plot -- the
        # whole point of a reasoning-VLA is to read this beside the motion it produced.
        reasoning = entry.get("reasoning", "")
        if reasoning:
            wrapped = textwrap.fill(reasoning, width=34)
            ax_p.set_xlabel(wrapped, fontsize=6, wrap=True, color="tab:red")

    axes[0, 0].set_ylabel("front cam", fontsize=9)
    axes[1, 0].set_ylabel("blue=hist  green=GT  red=pred\n(red text = model reasoning)", fontsize=7)
    fig.suptitle(f"Filmstrip: {sel[0]['token'].rsplit('_', 1)[0]}  ({n} of {len(frames)} frames)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize one episode's predicted vs GT trajectories.")
    ap.add_argument("--episode", required=True, help="Episode id, e.g. episode_0000")
    ap.add_argument("--predictions", type=Path, required=True, help="plan_conv(.json/_rank0.json) file")
    ap.add_argument("--info-pkl", type=Path, default=paths.INFOS_PKL)
    ap.add_argument("--data-root", type=Path, default=paths.NUSC_ROOT)
    ap.add_argument("--out-dir", type=Path, default=paths.PROJECT_ROOT / "outputs" / "episode_views")
    ap.add_argument("--overview-stride", type=int, default=2, help="Plot prediction every Nth frame in overview")
    ap.add_argument("--filmstrip-frames", type=int, default=10, help="Max frames in the filmstrip")
    args = ap.parse_args()

    pred_path = resolve_predictions_path(args.predictions)
    infos, token_to_info = load_infos(args.info_pkl)
    pred_map = load_prediction_map(pred_path)

    frames = episode_frames(infos, args.episode)
    if not frames:
        raise SystemExit(f"No frames found for episode '{args.episode}'")
    have_pred = sum(1 for f in frames if pred_map.get(f["token"], {}).get("trajectory"))
    print(f"{args.episode}: {len(frames)} frames, {have_pred} with predictions")
    if have_pred == 0:
        print("WARNING: none of this episode's frames are in the predictions file.")

    out_dir = args.out_dir / args.episode
    out_dir.mkdir(parents=True, exist_ok=True)
    overview = out_dir / "episode_overview.png"
    filmstrip = out_dir / "episode_filmstrip.png"

    make_overview(frames, token_to_info, pred_map, args.overview_stride, overview)
    make_filmstrip(frames, token_to_info, pred_map, args.data_root, args.filmstrip_frames, filmstrip)
    print(f"wrote:\n  {overview}\n  {filmstrip}")


if __name__ == "__main__":
    main()
