from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import paths


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def parse_args() -> argparse.Namespace:
    repo_root = paths.PROJECT_ROOT
    default_info = paths.INFOS_PKL
    default_data_root = paths.NUSC_ROOT

    parser = argparse.ArgumentParser(
        description="Inspect one OpenDriveVLA/nuScenes sample and visualize inputs versus trajectories."
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--sample-token", type=str, help="nuScenes sample token to inspect")
    target_group.add_argument("--sample-index", type=int, default=0, help="Index inside the temporal info file")
    parser.add_argument("--info-pkl", type=Path, default=default_info, help="Path to nuscenes_infos_temporal_*.pkl")
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="Path to the nuScenes data root")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional planning conversation json/jsonl file")
    parser.add_argument("--print-prompt", action="store_true", help="Print selected sample prompt text if available from predictions file")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "sample_inspection", help="Directory for generated artifacts")
    parser.add_argument("--history-steps", type=int, default=4, help="Number of previous ego poses to visualize")
    parser.add_argument("--future-steps", type=int, default=6, help="Number of future ego poses to visualize")
    parser.add_argument("--top-agents", type=int, default=12, help="Number of nearest agents to include in the summary")
    return parser.parse_args()


def resolve_predictions_path(predictions_path: Path | None) -> Path | None:
    if predictions_path is None:
        return None
    if predictions_path.exists():
        return predictions_path
    if predictions_path.is_absolute():
        raise SystemExit(f"Predictions file not found: {predictions_path}")

    cwd_candidate = Path.cwd() / predictions_path
    if cwd_candidate.exists():
        return cwd_candidate

    search_root = paths.PROJECT_ROOT / "OpenDriveVLA" / "output"
    if search_root.exists():
        candidates = list(search_root.glob(f"**/results/{predictions_path.name}"))
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"Resolved --predictions {predictions_path} -> {newest}")
            return newest

    raise SystemExit(
        f"Predictions file not found: {predictions_path} (cwd: {Path.cwd()}). "
        "Use an absolute path or place the file in the current directory."
    )


def load_infos(info_path: Path) -> tuple[list[dict], dict[str, dict]]:
    with info_path.open("rb") as handle:
        data = pickle.load(handle)
    infos = data["infos"]
    token_to_info = {info["token"]: info for info in infos}
    return infos, token_to_info


def load_prediction_map(predictions_path: Path | None) -> dict[str, dict]:
    if predictions_path is None:
        return {}

    predictions: dict[str, dict] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read().strip()

    if not raw_text:
        return predictions

    records: list[dict] = []
    # Supports both plan_conv.json (JSON array) and plan_conv_rank0.json (JSONL).
    if raw_text[0] == "[":
        loaded = json.loads(raw_text)
        if isinstance(loaded, list):
            records = [x for x in loaded if isinstance(x, dict)]
    else:
        for raw_line in raw_text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            parsed = json.loads(raw_line)
            if isinstance(parsed, dict):
                records.append(parsed)

    for record in records:
        raw_id = record.get("id") or record.get("qa_id") or record.get("sample_id") or ""
        if not raw_id:
            continue
        token = str(raw_id).removesuffix("_trajectory")

        answer_value = record.get("answer")
        if isinstance(answer_value, list):
            first_answer = str(answer_value[0]) if answer_value else ""
        elif answer_value is None:
            first_answer = ""
        else:
            first_answer = str(answer_value)

        question = record.get("question")
        reasoning = record.get("reasoning")           # the reasoning-VLA's trace
        if isinstance(reasoning, list):
            reasoning = reasoning[0] if reasoning else ""
        predictions[token] = {
            "trajectory": parse_trajectory_text(first_answer) if first_answer else [],
            "question": question,
            "answer": first_answer,
            "reasoning": reasoning or "",
        }
    return predictions


def parse_trajectory_text(text: str) -> list[tuple[float, float]]:
    # Isolate the trajectory span first. The reasoning-VLA emits
    # <reason_start>…<reason_end><traj_start>[(x,y,h)…]<traj_end>, and the reasoning text
    # contains digits ("about 5.7 m") that the tuple regex would otherwise read as
    # waypoints. If <traj_start> is present, parse ONLY inside it; strip any reason block
    # otherwise. A bare trajectory string (older/trajectory-only output) is unaffected.
    m = re.search(r"<traj_start>(.*?)<traj_end>", text, re.S)
    if m:
        text = m.group(1)
    else:
        text = re.sub(r"<reason_start>.*?<reason_end>", "", text, flags=re.S)
    # Waypoints are (x, y, heading) 3-tuples; the optional heading keeps this
    # backward-compatible with older 2-tuple prediction files. Plots use x, y only.
    pairs = re.findall(
        r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)(?:\s*,\s*[+-]?\d+(?:\.\d+)?)?\s*\)",
        text,
    )
    return [(float(x), float(y)) for x, y in pairs]


def quaternion_to_yaw(quaternion: list[float] | tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def global_to_local(global_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    delta = global_xy - origin_xy
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([math.sin(yaw), -math.cos(yaw)])
    local_x = delta @ right
    local_y = delta @ forward
    return np.stack([local_x, local_y], axis=-1)


def collect_chain(
    current_info: dict,
    token_to_info: dict[str, dict],
    direction: str,
    steps: int,
) -> list[dict]:
    chain: list[dict] = []
    next_token = current_info[direction]
    while next_token and len(chain) < steps:
        next_info = token_to_info.get(next_token)
        if next_info is None:
            break
        chain.append(next_info)
        next_token = next_info[direction]
    return chain


def build_ego_trajectories(
    info: dict,
    token_to_info: dict[str, dict],
    history_steps: int,
    future_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    current_xy = np.asarray(info["ego2global_translation"][:2], dtype=float)
    current_yaw = quaternion_to_yaw(info["ego2global_rotation"])

    history_infos = collect_chain(info, token_to_info, "prev", history_steps)
    history_infos.reverse()
    future_infos = collect_chain(info, token_to_info, "next", future_steps)

    history_tokens = [item["token"] for item in history_infos]
    future_tokens = [item["token"] for item in future_infos]

    history_xy = np.array([item["ego2global_translation"][:2] for item in history_infos], dtype=float) if history_infos else np.zeros((0, 2), dtype=float)
    future_xy = np.array([item["ego2global_translation"][:2] for item in future_infos], dtype=float) if future_infos else np.zeros((0, 2), dtype=float)

    local_history = global_to_local(history_xy, current_xy, current_yaw) if len(history_xy) else history_xy
    local_future = global_to_local(future_xy, current_xy, current_yaw) if len(future_xy) else future_xy
    return local_history, local_future, history_tokens, future_tokens


def build_agent_summary(info: dict, top_agents: int) -> list[dict]:
    boxes = np.asarray(info["gt_boxes"], dtype=float)
    names = info["gt_names"]
    velocities = np.asarray(info["gt_velocity"], dtype=float)
    instance_ids = info["gt_inds"]

    if len(boxes) == 0:
        return []

    distances = np.linalg.norm(boxes[:, :2], axis=1)
    order = np.argsort(distances)[:top_agents]
    summary = []
    for index in order:
        summary.append(
            {
                "name": str(names[index]),
                "instance_id": int(instance_ids[index]),
                "distance_m": round(float(distances[index]), 3),
                "position_xy_m": [round(float(value), 3) for value in boxes[index, :2]],
                "yaw_rad": round(float(boxes[index, 6]), 3),
                "velocity_xy_mps": [round(float(value), 3) for value in velocities[index]],
            }
        )
    return summary


def resolve_image_path(data_root: Path, relative_or_absolute: str) -> Path:
    image_path = Path(relative_or_absolute)
    if image_path.is_absolute():
        return image_path
    return data_root / image_path


def save_camera_grid(info: dict, data_root: Path, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    for axis, camera_name in zip(axes.flatten(), CAMERA_ORDER):
        camera_info = info["cams"][camera_name]
        image_path = resolve_image_path(data_root, camera_info["data_path"])
        axis.imshow(Image.open(image_path).convert("RGB"))
        axis.set_title(camera_name)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


_OBSTACLE_COLOR = {"car": "tab:blue", "truck": "tab:orange", "bus": "tab:orange",
                   "pedestrian": "tab:red", "bicycle": "tab:purple", "motorcycle": "tab:purple"}


def save_trajectory_plot(
    info: dict,
    history_traj: np.ndarray,
    future_traj: np.ndarray,
    predicted_traj: list[tuple[float, float]],
    output_path: Path,
    reasoning_text: str = "",
) -> None:
    figure, axis = plt.subplots(figsize=(8, 8.6))

    # Obstacles (GT boxes), labelled by CLASS -- these are cars/trucks/pedestrians, not
    # "agents". Convert lidar frame (x=forward, y=left) -> model frame (right=-y, forward=x)
    # so they line up with the trajectory, which is drawn in model frame.
    if len(info["gt_boxes"]):
        boxes = np.asarray(info["gt_boxes"], dtype=float)
        names = np.asarray(info["gt_names"])
        obs_right, obs_fwd = -boxes[:, 1], boxes[:, 0]
        for cls in sorted(set(names.tolist())):
            m = names == cls
            axis.scatter(obs_right[m], obs_fwd[m], s=45, alpha=0.5, marker="s",
                         color=_OBSTACLE_COLOR.get(cls, "gray"), label=f"{cls} (obstacle)")

    axis.scatter([0.0], [0.0], s=100, marker="*", color="black", label="Current ego")

    if len(history_traj):
        axis.plot(history_traj[:, 0], history_traj[:, 1], "o-", color="tab:blue", linewidth=2, label="Ego history")

    if len(future_traj):
        axis.plot(future_traj[:, 0], future_traj[:, 1], "o-", color="tab:green", linewidth=2, label="GT future")

    if predicted_traj:
        predicted = np.asarray(predicted_traj, dtype=float)
        axis.plot(predicted[:, 0], predicted[:, 1], "o--", color="tab:red", linewidth=2, label="Predicted future")

    axis.axhline(0.0, color="lightgray", linewidth=1)
    axis.axvline(0.0, color="lightgray", linewidth=1)
    axis.set_xlabel("x right (m)")
    axis.set_ylabel("y forward (m)")
    axis.set_title(f"Trajectory view for sample {info['token']}")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best", fontsize=8)

    # The model's reasoning, in the figure itself, under the plot.
    if reasoning_text:
        figure.subplots_adjust(bottom=0.16)
        figure.text(0.5, 0.03, "reasoning: " + textwrap.fill(reasoning_text, 80),
                    ha="center", va="bottom", fontsize=9, color="tab:red", wrap=True)

    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.predictions = resolve_predictions_path(args.predictions)
    infos, token_to_info = load_infos(args.info_pkl)
    prediction_map = load_prediction_map(args.predictions)

    if args.sample_token:
        info = token_to_info.get(args.sample_token)
        if info is None:
            raise SystemExit(f"Sample token not found in info file: {args.sample_token}")
    else:
        if args.sample_index < 0 or args.sample_index >= len(infos):
            raise SystemExit(f"Sample index out of range: {args.sample_index}")
        info = infos[args.sample_index]

    history_traj, future_traj, history_tokens, future_tokens = build_ego_trajectories(
        info,
        token_to_info,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
    )
    prediction_entry = prediction_map.get(info["token"], {})
    predicted_traj = prediction_entry.get("trajectory", [])
    prompt_text = prediction_entry.get("question")
    answer_text = prediction_entry.get("answer")
    reasoning_text = prediction_entry.get("reasoning")

    output_dir = args.output_dir / info["token"]
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_grid_path = output_dir / "camera_grid.png"
    trajectory_plot_path = output_dir / "trajectory_plot.png"
    summary_path = output_dir / "summary.json"
    prompt_path = output_dir / "prompt.txt"

    save_camera_grid(info, args.data_root, camera_grid_path)
    save_trajectory_plot(info, history_traj, future_traj, predicted_traj, trajectory_plot_path,
                         reasoning_text=reasoning_text or "")

    summary = {
        "sample_token": info["token"],
        "scene_token": info["scene_token"],
        "frame_idx": int(info["frame_idx"]),
        "timestamp_us": int(info["timestamp"]),
        "camera_paths": {
            camera_name: str(resolve_image_path(args.data_root, info["cams"][camera_name]["data_path"]))
            for camera_name in CAMERA_ORDER
        },
        "current_ego_global_translation": [round(float(value), 6) for value in info["ego2global_translation"]],
        "current_ego_global_rotation": [round(float(value), 6) for value in info["ego2global_rotation"]],
        "can_bus": [round(float(value), 6) for value in np.asarray(info["can_bus"], dtype=float)],
        "history_tokens": history_tokens,
        "future_tokens": future_tokens,
        "ego_history_xy_right_forward_m": [[round(float(x), 3), round(float(y), 3)] for x, y in history_traj.tolist()],
        "ego_gt_future_xy_right_forward_m": [[round(float(x), 3), round(float(y), 3)] for x, y in future_traj.tolist()],
        "predicted_future_xy_right_forward_m": [[round(float(x), 3), round(float(y), 3)] for x, y in predicted_traj],
        "prompt_from_predictions": prompt_text,
        "reasoning_from_predictions": reasoning_text,
        "answer_from_predictions": answer_text,
        "agent_count": int(len(info["gt_boxes"])),
        "nearest_agents": build_agent_summary(info, args.top_agents),
        "artifacts": {
            "camera_grid": str(camera_grid_path),
            "trajectory_plot": str(trajectory_plot_path),
            "prompt_text": str(prompt_path) if prompt_text else None,
        },
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if prompt_text:
        with prompt_path.open("w", encoding="utf-8") as handle:
            handle.write(prompt_text)
            handle.write("\n")

    if args.print_prompt and prompt_text:
        print("\n===== Prompt for selected sample =====")
        print(prompt_text)
        print("===== End prompt =====\n")

    if reasoning_text:
        print(f"\n>>> MODEL REASONING: {reasoning_text}")
        print(f">>> MODEL TRAJECTORY: {answer_text}\n")

    print(json.dumps({
        "sample_token": info["token"],
        "camera_grid": str(camera_grid_path),
        "trajectory_plot": str(trajectory_plot_path),
        "summary": str(summary_path),
        "prompt_text": str(prompt_path) if prompt_text else None,
    }, indent=2))


if __name__ == "__main__":
    main()