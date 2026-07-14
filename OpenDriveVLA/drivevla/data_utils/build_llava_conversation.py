import json
import pickle
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from llava.utils import rank0_print

from llava.constants import (
    DEFAULT_SCENE_START_TOKEN,
    DEFAULT_SCENE_TOKEN,
    DEFAULT_SCENE_END_TOKEN,
    DEFAULT_TRAJ_START_TOKEN,
    DEFAULT_TRACK_TOKEN,
    DEFAULT_TRAJ_END_TOKEN,
    DEFAULT_MAP_START_TOKEN,
    DEFAULT_MAP_TOKEN,
    DEFAULT_MAP_END_TOKEN,
    DEFAULT_TRACK_START_TOKEN,
    DEFAULT_TRACK_END_TOKEN,
    DEFAULT_TRAJ_TOKEN,
    DEFAULT_REASON_START_TOKEN,
    DEFAULT_REASON_END_TOKEN,
)


# ── Reasoning traces (Step 5) ────────────────────────────────────────────────
# The assistant turn becomes:
#   <reason_start>{trace}<reason_end><traj_start>[(x,y,h),...]<traj_end>
# so the trajectory is decoded CONDITIONED on the reasoning. Byte-identical to
# reasoning_data_gen.schema.render_assistant_turn (that module produced these traces).
#
# Enabled purely by env, so training with/without reasoning needs no code change:
#   REASONING_TRACES=<...>/reasoning/<version>/traces.jsonl
# Unset -> the assistant turn is trajectory-only, exactly as before.
_REASONING_TRACES = None


def _reasoning_traces():
    global _REASONING_TRACES
    if _REASONING_TRACES is None:
        _REASONING_TRACES = {}
        path = os.environ.get("REASONING_TRACES", "")
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    # Only factual traces train the base behaviour. Counterfactual pairs
                    # are a separate stream (they need a matching synthetic perception
                    # token, which does not exist yet — feeding them in here would teach
                    # the model to stop for obstacles it cannot see).
                    if rec.get("role", "factual") == "factual" and rec.get("trace"):
                        _REASONING_TRACES[rec["token"]] = rec["trace"]
            rank0_print(f"[reasoning] loaded {len(_REASONING_TRACES)} traces from {path}")
        elif path:
            rank0_print(f"[reasoning] WARNING: REASONING_TRACES={path} does not exist; "
                        "training trajectory-only.")
    return _REASONING_TRACES


def build_answer(traj_message: str, sample_token: str) -> str:
    """The assistant turn. Prepends the reasoning block when a trace exists for this
    frame; otherwise emits the trajectory alone (unchanged legacy behaviour)."""
    trace = _reasoning_traces().get(sample_token)
    traj = f"{DEFAULT_TRAJ_START_TOKEN}{traj_message}{DEFAULT_TRAJ_END_TOKEN}"
    if not trace:
        return traj
    return f"{DEFAULT_REASON_START_TOKEN}{trace}{DEFAULT_REASON_END_TOKEN}{traj}"

def get_sample_split(nusc: NuScenes, sample_token: str) -> str:
    """
    Determine the split of a sample given its token.
    
    Parameters:
        nusc: An instance of the NuScenes dataset.
        sample_token: The token of the sample (string).
    
    Returns:
        The split the sample belongs to, e.g., 'train', 'val', or 'test'.
    """
    sample_record = nusc.get("sample", sample_token)
    scene_token = sample_record["scene_token"]
    scene_record = nusc.get("scene", scene_token)
    scene_name = scene_record["name"]
    splits = create_splits_scenes()
    for split_name, scene_tokens in splits.items():
        if scene_name in scene_tokens:
            return split_name
    return "unknown"

def process_traj_data(data, split, nusc):

    # id_to_info = process_train_data(input_path)
    converted_entries = []
    
    for sample_token, value in data.items():
        
        if get_sample_split(nusc, sample_token) != split:
            continue

        converted_entry = {
            "qa_id": sample_token+'_trajectory',
            "sample_id": sample_token,
            "conversations": [
                {
                    "from": "human",
                    "value":"",
                },
                {
                    "from": "gpt",
                    "value":"",
                }
            ]
        }
        
        converted_entries.append(converted_entry)
    rank0_print(f"Loaded total {len(converted_entries)} samples")

    return converted_entries

def generate_user_message(data_dict):
        
    """
    Ego-States. gt_ego_lcf_feat (built by scripts/generate_cached_nuscenes_info.py) is:
        [0] fwd_v      forward velocity  (m/s, signed: negative while reversing)
        [1] right_v    rightward velocity
        [2] global_x   ego position in the MAP     <- deliberately NOT shown, see below
        [3] global_y   ego position in the MAP     <- deliberately NOT shown
        [4] yaw_rate   rad/s
        [5] length     4.5 (constant)
        [6] width      1.8 (constant)
        [7] speed      |v|
        [8] steer      normalized [-1, 1]

    TWO BUGS WERE FIXED HERE:

    1. VELOCITY ORDER. Everything the model reads or writes — the past trajectory, the
       mission-goal slot, and the output waypoints — is (x=RIGHT, y=FORWARD), and the
       system prompt says so. But lcf_feat stores (forward, right), and this function
       printed it verbatim as "Velocity (vx,vy)". So `x` meant "right" everywhere except
       velocity, where it meant "forward".
       Physics check on a reversing frame: measured velocity is forward=-1.80, right=+0.53
       m/s, and the first waypoint (0.5 s later) is (right=+0.17, forward=-0.92).
       Reported as (right, forward) -> (+0.27, -0.90): matches the waypoint.
       Reported as (forward, right) -> (-0.90, +0.27): contradicts it — the prompt told the
       model it was drifting LEFT while the trajectory it had to emit went BACKWARD.

    2. "Can Bus" WAS THE EGO'S ABSOLUTE MAP POSITION. lcf_feat[2:4] is global (x, y), and
       it was being printed as "Can Bus: (287.06, 199.79)". Every episode is in ONE lot, so
       an absolute coordinate is close to a lookup key: the obstacle layout of a fixed lot
       is a FUNCTION of position, which lets the model recall what is around it instead of
       looking at the perception tokens — precisely the shortcut this project exists to
       remove. It is also useless at deployment (meaningless in a new lot). Dropped; the
       acceleration below is real and stays.
    """
    ego_message = ""
    fwd_v = data_dict['gt_ego_lcf_feat'][0] * 0.5
    right_v = data_dict['gt_ego_lcf_feat'][1] * 0.5
    v_yaw = data_dict['gt_ego_lcf_feat'][4]
    # his_diff is already (right, forward) — consistent with the trajectory convention.
    ax = data_dict['gt_ego_his_diff'][-1, 0] - data_dict['gt_ego_his_diff'][-2, 0]
    ay = data_dict['gt_ego_his_diff'][-1, 1] - data_dict['gt_ego_his_diff'][-2, 1]
    vhead = data_dict['gt_ego_lcf_feat'][7] * 0.5
    steeling = data_dict['gt_ego_lcf_feat'][8]
    # (x, y) = (right, forward), the same convention as the waypoints.
    ego_message += f"- Velocity (vx,vy): ({right_v:.2f},{fwd_v:.2f})"
    ego_message += f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})"
    ego_message += f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})"
    ego_message += f" - Heading Speed: ({vhead:.2f})"
    ego_message += f" - Steering: ({steeling:.2f})"

    """
    Historical Trjectory:
        gt_ego_his_trajs: [5, 2] last 2 seconds 
        gt_ego_his_diff: [4, 2] last 2 seconds, differential format, viewed as velocity 
    """
    # his_message = ""
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    # his_message += f"Historical trajectory (last 2 seconds):"
    his_message = f"[({xh1:.2f},{yh1:.2f}),({xh2:.2f},{yh2:.2f}),({xh3:.2f},{yh3:.2f}),({xh4:.2f},{yh4:.2f})]"
    
    """
    Mission goal:
        gt_ego_fut_cmd
    """
    # Prefer the maneuver-level command (maneuver type + side + ego-local target slot)
    # when present; otherwise fall back to the per-frame right/left/forward/reverse command.
    maneuver_type = data_dict.get('maneuver_type')
    if maneuver_type:
        mission_goal = maneuver_type.replace('_', '-') + " park"
        side = data_dict.get('side')
        if side:
            mission_goal += f", {side} side"
        slot = data_dict.get('slot_local')
        if slot is not None:
            mission_goal += f", into slot at ({slot[0]:.2f},{slot[1]:.2f},{slot[2]:.2f})"
    else:
        cmd_vec = data_dict['gt_ego_fut_cmd']
        right, left, forward = cmd_vec[0], cmd_vec[1], cmd_vec[2]
        reverse = cmd_vec[3] if len(cmd_vec) > 3 else 0.0
        if reverse > 0:
            mission_goal = "reverse"
        elif right > 0:
            mission_goal = "turn right"
        elif left > 0:
            mission_goal = "turn left"
        else:
            mission_goal = "keep forward"
    cmd_message = f"{mission_goal}"
    
    """
    Planning trajectory:
        gt_ego_fut_trajs: [7, 3] -> 6 future waypoints (skip index 0 = current frame).
        Each waypoint is (x=right, y=forward, h=heading), where h is the future
        frame's yaw RELATIVE to the current frame (radians, in [-pi, pi]), same
        convention as the mission-goal slot's third component. The heading is an
        emitted output label so the model can express rotation on reverse arcs
        where the positions alone are nearly collinear.
    """
    pts = [
        (data_dict['gt_ego_fut_trajs'][k][0],
         data_dict['gt_ego_fut_trajs'][k][1],
         data_dict['gt_ego_fut_trajs'][k][2])
        for k in range(1, 7)
    ]
    traj_message = "[" + ",".join(f"({x:.2f},{y:.2f},{h:.2f})" for x, y, h in pts) + "]"

    return ego_message, his_message, cmd_message, traj_message


def build_llava_conversation(data_sample, cached_nuscenes_data):

    # Use sample_id directly when present (CARLA tokens contain underscores that
    # split('_')[0] would truncate; nuScenes hex tokens have no underscores so
    # the fallback split is still correct for legacy data).
    raw_id = data_sample.get('id', data_sample.get('qa_id', ''))
    sample_token = data_sample.get('sample_id') or raw_id.split('_')[0]
    value = cached_nuscenes_data.get(sample_token)

    # QA / interpretability probe (test-time): when a sample carries a free-form
    # `question`, build a question-answering prompt instead of the trajectory
    # prompt. Trajectory samples have no `question` key, so the path below is
    # unchanged.
    question = data_sample.get('question')
    if question:
        prompt = (
            f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
            f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
            f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
        )
        if value is not None:
            ego_message, his_message, _, _ = generate_user_message(value)
            prompt += (
                f"Ego states: {ego_message}\n"
                f"Historical trajectory (last 2 seconds): {his_message}\n"
            )
        data_sample['conversations'][0]['value'] = (
            prompt + f"Please answer the following question: {question}"
        )
        if len(data_sample.get('conversations', [])) > 1:
            data_sample['conversations'][1]['value'] = ""
        return data_sample

    if value is None:
        data_sample['conversations'][0]['value'] = (
            f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
            f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
            f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
            "Ego states: unavailable in smoke mode\n"
            "Historical trajectory (last 2 seconds): unavailable in smoke mode\n"
            "Mission goal: keep forward\n"
            f"Planning trajectory: {DEFAULT_TRAJ_TOKEN}"
        )
        if len(data_sample.get('conversations', [])) > 1:
            data_sample['conversations'][1]['value'] = (
                f"{DEFAULT_TRAJ_START_TOKEN}[(0.00,0.00,0.00),(0.00,0.00,0.00),(0.00,0.00,0.00),(0.00,0.00,0.00),(0.00,0.00,0.00),(0.00,0.00,0.00)]{DEFAULT_TRAJ_END_TOKEN}"
            )
        return data_sample

    ego_message, his_message, cmd_message, traj_message = generate_user_message(value)
    data_sample['conversations'][0]['value'] = (
        f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
        f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
        f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
        f"Ego states: {ego_message}\n"
        f"Historical trajectory (last 2 seconds): {his_message}\n"
        f"Mission goal: {cmd_message}\n"
        f"Planning trajectory: {DEFAULT_TRAJ_TOKEN}"
    )
    data_sample['conversations'][1]['value'] = build_answer(traj_message, sample_token)

    return data_sample
