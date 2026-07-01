#!/usr/bin/env python3
"""Verify trajectory continuity across frames.

Each frame's prompt (history + future) is in that frame's OWN ego-local frame,
which rotates every step — so the same physical point has different numbers in
different frames and can't be compared by eye. This tool lifts each frame's
history/future back into a common GLOBAL frame and overlays them, so you can SEE
that one frame's GT future traces the same path as a later frame's history.

Example (reproduces the f0049-future vs f0052-history check):
  python scripts/check_trajectory_continuity.py \
    --episode episode_1333 --future-frame 49 --history-frame 52
"""
import argparse
import pickle
import re

import numpy as np
from pyquaternion import Quaternion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths


def frame_index(token):
    m = re.search(r"_f(\d+)$", token)
    return int(m.group(1)) if m else None


def ego_local_to_global(pts_rf, origin_xy, rot):
    """Inverse of generate_cached_nuscenes_info.global_to_local_xy:
    ego-local (right, forward) -> global (x, y)."""
    R = rot.rotation_matrix
    origin = np.array([origin_xy[0], origin_xy[1], 0.0])
    out = []
    for right, forward, *_ in pts_rf:  # future waypoints carry a 3rd heading column
        local = np.array([forward, -right, 0.0])  # undo [-l[1], l[0]]
        out.append((R @ local + origin)[:2])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="episode_1333")
    ap.add_argument("--future-frame", type=int, default=49,
                    help="overlay this frame's GT future (in global)")
    ap.add_argument("--history-frame", type=int, default=52,
                    help="overlay this frame's history (in global)")
    ap.add_argument("--infos", default=str(paths.INFOS_PKL))
    ap.add_argument("--cache", default=str(paths.CACHED_INFO))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.infos, "rb") as f:
        infos = pickle.load(f)["infos"]
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)

    # Per-frame global pose for this episode.
    pose = {}
    for x in infos:
        if x["token"].startswith(args.episode + "_f"):
            pose[frame_index(x["token"])] = (
                np.array(x["ego2global_translation"][:2], float),
                Quaternion(x["ego2global_rotation"]),
                x["token"],
            )
    if not pose:
        raise SystemExit(f"no frames found for {args.episode}")
    idxs = sorted(pose)

    # Actual driven path (global).
    path = np.array([pose[i][0] for i in idxs])

    # Future of one frame, history of another — both lifted to global.
    fo, ho = args.future_frame, args.history_frame
    t_f, R_f, tok_f = pose[fo]
    fut_global = ego_local_to_global(cache[tok_f]["gt_ego_fut_trajs"], t_f, R_f)
    t_h, R_h, tok_h = pose[ho]
    his_global = ego_local_to_global(cache[tok_h]["gt_ego_his_trajs"], t_h, R_h)

    # ---- numeric proof: future point t+k must equal actual global of frame fo+k ----
    print(f"{args.episode}: frames {idxs[0]}..{idxs[-1]}")
    print(f"\nf{fo:04d} GT future vs ACTUAL global poses (must match):")
    for k, g in enumerate(fut_global):
        j = fo + k
        act = pose[j][0] if j in pose else None
        tag = f"actual f{j:04d}=({act[0]:.2f},{act[1]:.2f})" if act is not None else "(past episode end)"
        print(f"  t+{0.5*k:.1f}s: future=({g[0]:.2f},{g[1]:.2f})   {tag}")
    print(f"\nf{ho:04d} history vs ACTUAL global poses (must match):")
    for k, g in enumerate(his_global):
        j = ho - (len(his_global) - 1 - k)
        act = pose[j][0] if j in pose else None
        tag = f"actual f{j:04d}=({act[0]:.2f},{act[1]:.2f})" if act is not None else "(before episode start)"
        print(f"  t{0.5*(k-(len(his_global)-1)):+.1f}s: hist=({g[0]:.2f},{g[1]:.2f})   {tag}")

    # ---- plot in global frame ----
    plt.figure(figsize=(7, 8))
    plt.plot(path[:, 0], path[:, 1], "-", color="0.7", label="actual driven path")
    plt.scatter(path[:, 0], path[:, 1], s=12, color="0.7")
    plt.plot(fut_global[:, 0], fut_global[:, 1], "-o", color="tab:red",
             label=f"f{fo:04d} GT future")
    plt.plot(his_global[:, 0], his_global[:, 1], "-o", color="tab:blue",
             label=f"f{ho:04d} history")
    plt.scatter(*pose[fo][0], marker="*", s=220, color="tab:red", zorder=5)
    plt.scatter(*pose[ho][0], marker="*", s=220, color="tab:blue", zorder=5)
    plt.gca().set_aspect("equal")
    plt.xlabel("global x (m)"); plt.ylabel("global y (m)")
    plt.title(f"{args.episode}: f{fo:04d} future vs f{ho:04d} history (global frame)")
    plt.legend(); plt.grid(True, alpha=0.3)
    out = args.out or f"outputs/continuity_{args.episode}_f{fo}_f{ho}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
