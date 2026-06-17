"""
CARLA trajectory evaluator — compares OpenDriveVLA predictions against GT poses.

Usage:
  python scripts/eval_carla_predictions.py
  python scripts/eval_carla_predictions.py \
    --plan-conv OpenDriveVLA/output/OpenDriveVLA-0.5B/<timestamp>/results/plan_conv.json \
    --info-pkl data_carla/processed/parking_infos_temporal.pkl
"""

import argparse
import json
import math
import pathlib
import pickle
import re

import numpy as np

import paths

_ROOT = paths.PROJECT_ROOT

# Timesteps at 2 Hz: index 0=0.5s, 1=1.0s, 2=1.5s, 3=2.0s, 4=2.5s, 5=3.0s
EVAL_AT = {1: 1, 2: 3, 3: 5}   # seconds → 0-based waypoint index


def quaternion_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def global_to_local(future_xy, origin_xy, yaw):
    delta = future_xy - origin_xy
    lx = delta[0] * math.sin(yaw) - delta[1] * math.cos(yaw)
    ly = delta[0] * math.cos(yaw) + delta[1] * math.sin(yaw)
    return lx, ly


def parse_trajectory(text):
    pairs = re.findall(r"[\[(]([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)[\])]", text)
    return [(float(x), float(y)) for x, y in pairs]


def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            token = rec.get("id", "").removesuffix("_trajectory")
            answer = rec.get("answer", [])
            text = answer[0] if isinstance(answer, list) and answer else str(answer)
            waypoints = parse_trajectory(text)
            if waypoints:
                preds[token] = waypoints
    return preds


def compute_gt_waypoints(info, token_to_info, n=6):
    origin = np.array(info["ego2global_translation"][:2], dtype=float)
    yaw = quaternion_to_yaw(info["ego2global_rotation"])
    waypoints = []
    cur = info
    for _ in range(n):
        nxt = token_to_info.get(cur.get("next", ""))
        if nxt is None:
            waypoints.append(waypoints[-1] if waypoints else (0.0, 0.0))
            continue
        xy = np.array(nxt["ego2global_translation"][:2], dtype=float)
        waypoints.append(global_to_local(xy, origin, yaw))
        cur = nxt
    return waypoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-conv", type=pathlib.Path,
                    default=_ROOT / "OpenDriveVLA/output/OpenDriveVLA-0.5B/20260612_151523/results/plan_conv.json")
    ap.add_argument("--info-pkl", type=pathlib.Path, default=paths.INFOS_PKL)
    args = ap.parse_args()

    with open(args.info_pkl, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    token_to_info = {i["token"]: i for i in infos}

    preds = load_predictions(args.plan_conv)
    print(f"Loaded {len(preds)} predictions, {len(infos)} GT frames")

    errors = {s: [] for s in EVAL_AT}
    matched = 0
    reverse_errors = {s: [] for s in EVAL_AT}
    forward_errors = {s: [] for s in EVAL_AT}

    for info in infos:
        token = info["token"]
        if token not in preds:
            continue
        pred = preds[token]
        gt = compute_gt_waypoints(info, token_to_info)
        if len(pred) < 6 or len(gt) < 6:
            continue
        matched += 1
        is_reverse = info.get("reverse", False)
        for sec, idx in EVAL_AT.items():
            px, py = pred[idx]
            gx, gy = gt[idx]
            l2 = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            errors[sec].append(l2)
            (reverse_errors if is_reverse else forward_errors)[sec].append(l2)

    print(f"\nMatched {matched} frames\n")
    print(f"{'':20s} {'1s':>8} {'2s':>8} {'3s':>8}")
    print("-" * 44)
    for label, err_dict in [("All", errors), ("Reverse/park", reverse_errors), ("Forward", forward_errors)]:
        row = f"{label:20s}"
        for sec in (1, 2, 3):
            vals = err_dict[sec]
            row += f"  {np.mean(vals):6.2f}m" if vals else f"  {'N/A':>6}"
        print(row)

    print("\nL2 displacement error (lower = better). Baseline: pretrained model prediction quality.")


if __name__ == "__main__":
    main()
