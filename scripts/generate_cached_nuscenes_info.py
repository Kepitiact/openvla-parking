#!/usr/bin/env python3
import argparse
import os
import pickle

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import paths


def load_info_tokens(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj
    tokens = [x["token"] for x in infos if "token" in x]
    # Also extract per-token reverse flag if present (CARLA data).
    reverse_flags = {x["token"]: bool(x.get("reverse", False))
                     for x in infos if "token" in x}
    # Per-token maneuver labels (maneuver-level command). Optional — older data lacks them.
    maneuver_info = {
        x["token"]: {
            "maneuver_type": x.get("maneuver_type"),
            "side": x.get("side"),
            "target_slot": x.get("target_slot"),
        }
        for x in infos if "token" in x
    }
    return tokens, reverse_flags, maneuver_info


def get_sample_pose(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    translation = np.array(ego_pose["translation"], dtype=np.float64)
    rotation = Quaternion(ego_pose["rotation"])
    return sample, translation, rotation


def global_to_local_xy(global_xy, origin_xyz, origin_rot):
    point_xyz = np.array([global_xy[0], global_xy[1], 0.0], dtype=np.float64)
    local_xyz = origin_rot.rotation_matrix.T @ (point_xyz - origin_xyz)
    # Ego frame: x=forward, y=left. GPT-Driver cache convention: x=right, y=forward.
    return np.array([-local_xyz[1], local_xyz[0]], dtype=np.float64)


def _normalize_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def slot_to_local(target_slot, cur_xyz, cur_rot):
    """Global target-slot pose -> ego-local [right, forward, dheading], matching the
    trajectory convention (x=right, y=forward). Returns None if no slot is available."""
    if not target_slot or "pose" not in target_slot:
        return None
    pose = target_slot["pose"]
    sx, sy = float(pose["translation"][0]), float(pose["translation"][1])
    right, forward = global_to_local_xy([sx, sy], cur_xyz, cur_rot)
    slot_yaw = Quaternion(pose["rotation"]).yaw_pitch_roll[0]
    ego_yaw = cur_rot.yaw_pitch_roll[0]
    dheading = _normalize_angle(slot_yaw - ego_yaw)
    return np.array([right, forward, dheading], dtype=np.float32)


def collect_history_local(nusc, sample_token, cur_xyz, cur_rot, history_steps=4):
    # Build [t-2.0, t-1.5, t-1.0, t-0.5, t] with 0.5s nuScenes cadence.
    hist_tokens = []
    tok = sample_token
    for _ in range(history_steps):
        prev_tok = nusc.get("sample", tok)["prev"]
        if not prev_tok:
            break
        hist_tokens.append(prev_tok)
        tok = prev_tok
    hist_tokens = hist_tokens[::-1]

    local_points = []
    for tok in hist_tokens:
        _, xyz, _ = get_sample_pose(nusc, tok)
        local_points.append(global_to_local_xy(xyz[:2], cur_xyz, cur_rot))

    if local_points:
        while len(local_points) < history_steps:
            local_points.insert(0, local_points[0].copy())
    else:
        local_points = [np.zeros(2, dtype=np.float64) for _ in range(history_steps)]

    local_points.append(np.zeros(2, dtype=np.float64))
    return np.array(local_points, dtype=np.float32)


def collect_future_local(nusc, sample_token, cur_xyz, cur_rot, future_steps=6):
    # Build [t, t+0.5, ..., t+3.0] in current ego frame.
    fut_points = [np.zeros(2, dtype=np.float64)]
    tok = sample_token
    for _ in range(future_steps):
        next_tok = nusc.get("sample", tok)["next"]
        if not next_tok:
            fut_points.append(fut_points[-1].copy())
            continue
        _, xyz, _ = get_sample_pose(nusc, next_tok)
        fut_points.append(global_to_local_xy(xyz[:2], cur_xyz, cur_rot))
        tok = next_tok
    return np.array(fut_points, dtype=np.float32)


def infer_future_command(fut_traj, is_reverse=False):
    # fut_traj is (right, forward). [right, left, forward, reverse] 4-element vector.
    if is_reverse:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # reverse
    final_x = float(fut_traj[-1, 0])
    if final_x >= 2.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # right
    if final_x <= -2.0:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # left
    return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # forward


def build_entry(nusc, sample_token, is_reverse=False, maneuver=None):
    _, cur_xyz, cur_rot = get_sample_pose(nusc, sample_token)

    gt_ego_his_trajs = collect_history_local(nusc, sample_token, cur_xyz, cur_rot)
    gt_ego_his_diff = np.diff(gt_ego_his_trajs, axis=0).astype(np.float32)
    gt_ego_fut_trajs = collect_future_local(nusc, sample_token, cur_xyz, cur_rot)

    dt = 0.5
    # Trajectory is now (right, forward): index 0=right, index 1=forward.
    dx_right = float(gt_ego_fut_trajs[1, 0] - gt_ego_fut_trajs[0, 0])
    dy_fwd = float(gt_ego_fut_trajs[1, 1] - gt_ego_fut_trajs[0, 1])
    vx_true = dy_fwd / dt   # forward velocity (main driving speed)
    vy_true = dx_right / dt  # rightward lateral velocity
    speed_true = float(np.hypot(vx_true, vy_true))

    yaw_now = 0.0
    yaw_next = float(np.arctan2(vy_true, vx_true)) if speed_true > 1e-4 else yaw_now
    yaw_rate = (yaw_next - yaw_now) / dt

    # build_llava_conversation multiplies these by 0.5 before printing.
    # Official GPT-Driver cache stores raw speed here; prompt then shows speed*0.5
    # = "meters per 0.5s step", which is the unit the model was trained on.
    vx_lcf = vx_true
    vy_lcf = vy_true
    speed_lcf = speed_true

    # Keep feature length/layout expected by build_llava_conversation.generate_user_message.
    gt_ego_lcf_feat = np.array(
        [
            vx_lcf,
            vy_lcf,
            float(cur_xyz[0]),
            float(cur_xyz[1]),
            yaw_rate,
            4.5,
            1.8,
            speed_lcf,
            0.0,
        ],
        dtype=np.float32,
    )

    entry = {
        "gt_ego_lcf_feat": gt_ego_lcf_feat,
        "gt_ego_his_trajs": gt_ego_his_trajs,
        "gt_ego_his_diff": gt_ego_his_diff,
        "gt_ego_fut_cmd": infer_future_command(gt_ego_fut_trajs, is_reverse=is_reverse),
        "gt_ego_fut_trajs": gt_ego_fut_trajs,
    }
    # Maneuver-level command (preferred over the per-frame cmd by build_llava_conversation).
    if maneuver:
        entry["maneuver_type"] = maneuver.get("maneuver_type")
        entry["side"] = maneuver.get("side")
        slot_local = slot_to_local(maneuver.get("target_slot"), cur_xyz, cur_rot)
        if slot_local is not None:
            entry["slot_local"] = slot_local
    return entry


def main():
    parser = argparse.ArgumentParser(description="Generate cached_nuscenes_info.pkl for OpenDriveVLA.")
    parser.add_argument("--infos", default=str(paths.INFOS_PKL))
    parser.add_argument("--dataroot", default=str(paths.NUSC_ROOT))
    parser.add_argument("--version", default=paths.NUSC_VERSION)
    parser.add_argument("--output", default=str(paths.CACHED_INFO))
    args = parser.parse_args()

    tokens, reverse_flags, maneuver_info = load_info_tokens(args.infos)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    cached = {}
    for i, token in enumerate(tokens, start=1):
        cached[token] = build_entry(
            nusc, token,
            is_reverse=reverse_flags.get(token, False),
            maneuver=maneuver_info.get(token),
        )
        if i % 50 == 0 or i == len(tokens):
            print(f"Processed {i}/{len(tokens)}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(cached, f)

    print(f"Wrote {len(cached)} entries to {args.output}")


if __name__ == "__main__":
    main()
