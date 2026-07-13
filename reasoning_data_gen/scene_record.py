"""SceneRecord — the only thing the reasoner is allowed to look at.

A SceneRecord is the *perceivable* scene: detected objects, ego state, the mission
command, and past-2s ego history. In Step 0 it is built from ground truth
(from_gt); in Step 3 it will be built from decoded UniAD outputs (from_uniad) — and
that is the ONLY thing that changes. Nothing here may carry future information; the
future trajectory is a separate output label, not part of the scene.

Coordinate convention (must match the trajectory/slot convention used everywhere):
  ego frame  ->  x = right (+right), y = forward (+forward)
The infos store gt_boxes in the LIDAR/ego frame as (x=forward, y=left, z, w, l, h,
yaw); build_infos_pkl.py documents this. So:
    right   = -gt_boxes[:, 1]      (right = -left)
    forward =  gt_boxes[:, 0]
    v_right = -gt_velocity[:, 1]
    v_fwd   =  gt_velocity[:, 0]
    yaw (object heading rel. to ego forward) is kept as-is (gt_boxes[:, 6]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

from .schema import FactRecord


# UniAD label index -> class name. Matches class_names in
# OpenDriveVLA/projects/configs/stage1_track_map/carla_parking_stage1.py (nuScenes
# 10-class order). CARLA parking only actually contains car/truck/pedestrian; the
# rest are here so a label index never lands out of range.
UNIAD_CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def _norm_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _global_to_ego_rf(global_xy, origin_xyz, origin_rot: Quaternion) -> Tuple[float, float]:
    """Global (x, y) -> ego-local (right, forward). Mirrors
    generate_cached_nuscenes_info.global_to_local_xy (which returns [-left, forward])."""
    p = np.array([global_xy[0], global_xy[1], 0.0], dtype=np.float64)
    loc = origin_rot.rotation_matrix.T @ (p - np.asarray(origin_xyz, dtype=np.float64))
    return float(-loc[1]), float(loc[0])  # (right, forward)


@dataclass
class DetectedObject:
    """One perceivable object in ego frame (right, forward)."""

    cls: str
    right: float
    forward: float
    yaw: float                 # heading relative to ego forward (rad)
    size: Tuple[float, float, float]   # (w, l, h)
    vel: Tuple[float, float]           # (v_right, v_forward) m/s
    track_id: Optional[int] = None

    @property
    def dist(self) -> float:
        return float(np.hypot(self.right, self.forward))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cls": self.cls, "right": self.right, "forward": self.forward,
            "yaw": self.yaw, "size": list(self.size), "vel": list(self.vel),
            "track_id": self.track_id,
        }


@dataclass
class EgoState:
    speed: float
    fwd_v: float
    right_v: float
    yaw_rate: float
    steer: float

    def to_dict(self) -> Dict[str, Any]:
        return {"speed": self.speed, "fwd_v": self.fwd_v, "right_v": self.right_v,
                "yaw_rate": self.yaw_rate, "steer": self.steer}


@dataclass
class Command:
    maneuver_type: Optional[str]
    side: Optional[str]
    reverse: bool
    slot_local: Optional[Tuple[float, float, float]]  # (right, forward, dheading) or None

    def to_dict(self) -> Dict[str, Any]:
        return {"maneuver_type": self.maneuver_type, "side": self.side,
                "reverse": self.reverse,
                "slot_local": list(self.slot_local) if self.slot_local is not None else None}


@dataclass
class SceneRecord:
    """Perceivable scene = detections + ego + command + past-2s history. No future."""

    token: str
    objects: List[DetectedObject]
    ego: EgoState
    command: Command
    history: List[Tuple[float, float]] = field(default_factory=list)  # past ego (right,forward)
    prev_reverse: Optional[bool] = None       # reverse flag of the immediately prior frame
    drivable: Optional[Any] = None            # map/drivable info (None for GT v1)
    scene_token: Optional[str] = None
    frame_idx: Optional[int] = None
    source: str = "gt"

    # ── builders ──────────────────────────────────────────────────────────────
    @staticmethod
    def from_gt(
        info: Dict[str, Any],
        raw: Optional[Dict[str, Any]] = None,
        history: Optional[List[Tuple[float, float]]] = None,
        prev_reverse: Optional[bool] = None,
    ) -> "SceneRecord":
        """Build a SceneRecord from a GT info record (the Step-0 placeholder source).

        `info` is one entry from parking_infos_{train,val}.pkl. `raw` (optional) is
        the matching raw episode dict for map/drivable enrichment — NOT used to add
        privileged planner knowledge (A* path, obstacle ground truth) into the scene,
        because a perceiver would not have it. `history`/`prev_reverse` are past-only
        context the caller (run_generate) supplies from earlier frames.
        """
        objects = _objects_from_gt(info)
        drivable = raw.get("drivable") if isinstance(raw, dict) else None
        return SceneRecord._assemble(info, objects, history, prev_reverse,
                                     source="gt", drivable=drivable)

    @staticmethod
    def from_uniad(
        decoded: Any,
        info: Dict[str, Any],
        history: Optional[List[Tuple[float, float]]] = None,
        prev_reverse: Optional[bool] = None,
        score_thr: float = 0.3,
        class_names: Optional[List[str]] = None,
    ) -> "SceneRecord":
        """Build a SceneRecord from decoded, trained-UniAD outputs — the real
        perception source (validated on HAL; format matches extract_uniad_features.py).

        `decoded` is one per-frame record (the .pth dict saved by
        drivevla/extract_uniad_features.py, or its `result_track.detections`
        sub-dict). Detections are `boxes[N,9]` = ego-frame
        (x,y,z,w,l,h,yaw,vx,vy) + `scores[N]` + `labels[N]` (indices into
        `class_names`). Low-confidence boxes (score < `score_thr`) are dropped so the
        scene reflects what a perceiver would actually trust.

        Ego state, mission command, and slot come from `info` — NOT from perception:
        the car always knows its own speed/gear/goal; only the *objects* are perceived.
        This is exactly why swapping GT→UniAD changes only the object source here.
        """
        objects = _objects_from_uniad(decoded, score_thr,
                                      class_names or UNIAD_CLASS_NAMES)
        return SceneRecord._assemble(info, objects, history, prev_reverse,
                                     source="uniad")

    @staticmethod
    def _assemble(info, objects, history, prev_reverse, source, drivable=None
                  ) -> "SceneRecord":
        """Shared assembly: ego state + mission command + slot, from the info. The
        object list is the only thing that differs between from_gt and from_uniad."""
        ego = EgoState(
            speed=abs(float(info.get("ego_speed", 0.0))),
            fwd_v=float(info.get("ego_fwd_v", 0.0)),
            right_v=float(info.get("ego_right_v", 0.0)),
            yaw_rate=float(info.get("ego_yaw_rate", 0.0)),
            steer=float(info.get("ego_steer", 0.0)),
        )
        slot_local = _slot_to_local(info.get("target_slot"),
                                    info["ego2global_translation"],
                                    info["ego2global_rotation"])
        command = Command(
            maneuver_type=info.get("maneuver_type"),
            side=info.get("side"),
            reverse=bool(info.get("reverse", False)),
            slot_local=slot_local,
        )
        return SceneRecord(
            token=info["token"],
            objects=objects,
            ego=ego,
            command=command,
            history=list(history or []),
            prev_reverse=prev_reverse,
            drivable=drivable,
            scene_token=info.get("scene_token"),
            frame_idx=info.get("frame_idx"),
            source=source,
        )

    # ── convenience ─────────────────────────────────────────────────────────
    def dist_to_slot(self) -> Optional[float]:
        if self.command.slot_local is None:
            return None
        r, f, _ = self.command.slot_local
        return float(np.hypot(r, f))

    def align_err(self) -> Optional[float]:
        if self.command.slot_local is None:
            return None
        return abs(float(self.command.slot_local[2]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token, "scene_token": self.scene_token,
            "frame_idx": self.frame_idx, "source": self.source,
            "objects": [o.to_dict() for o in self.objects],
            "ego": self.ego.to_dict(), "command": self.command.to_dict(),
            "history": [list(h) for h in self.history],
            "prev_reverse": self.prev_reverse,
        }


@dataclass
class Frame:
    """Per-frame bundle passed to the counterfactual synth and eval gates.

    Holds the perceivable scene, the extracted fact, and the OUTPUT trajectory
    label (6 x (right, forward, dheading)). The trajectory lives here — NOT in the
    SceneRecord — precisely because it is future information the reasoner must not
    read; only the counterfactual synth and action-fidelity gate touch it.
    """

    token: str
    scene: SceneRecord
    fact: FactRecord
    trajectory: np.ndarray  # (6, 3) float


def _objects_from_gt(info: Dict[str, Any]) -> List[DetectedObject]:
    """GT gt_boxes (lidar frame x=forward, y=left; [N,7]) -> ego (right, forward)."""
    gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float64).reshape(-1, 7)
    gt_names = list(info.get("gt_names", []))
    gt_vel = np.asarray(info.get("gt_velocity", np.zeros((len(gt_boxes), 2))),
                        dtype=np.float64).reshape(-1, 2)
    gt_inds = list(info.get("gt_inds", [None] * len(gt_boxes)))

    objects: List[DetectedObject] = []
    for i in range(len(gt_boxes)):
        x_fwd, y_left, _z, w, l, h, yaw = gt_boxes[i]
        vr = -float(gt_vel[i, 1]) if i < len(gt_vel) else 0.0
        vf = float(gt_vel[i, 0]) if i < len(gt_vel) else 0.0
        objects.append(DetectedObject(
            cls=str(gt_names[i]) if i < len(gt_names) else "car",
            right=-float(y_left), forward=float(x_fwd), yaw=float(yaw),
            size=(float(w), float(l), float(h)), vel=(vr, vf),
            track_id=int(gt_inds[i]) if i < len(gt_inds) and gt_inds[i] is not None else None,
        ))
    return objects


def _objects_from_uniad(decoded: Any, score_thr: float,
                        class_names: List[str]) -> List[DetectedObject]:
    """UniAD decoded detections -> ego (right, forward). `decoded` is the per-frame
    .pth dict from extract_uniad_features.py (or its result_track.detections). Boxes
    are `[N,9]` = ego-frame (x,y,z,w,l,h,yaw,vx,vy) — same frame as gt_boxes."""
    det = decoded
    if isinstance(decoded, dict) and "boxes" not in decoded:
        det = (decoded.get("result_track") or {}).get("detections")
    if not det or det.get("boxes") is None:
        return []                                  # perception saw nothing this frame

    boxes = np.asarray(det["boxes"], dtype=np.float64).reshape(-1, 9)
    scores = (np.asarray(det["scores"], dtype=np.float64).reshape(-1)
              if det.get("scores") is not None else np.ones(len(boxes)))
    labels = (np.asarray(det["labels"]).reshape(-1).astype(int)
              if det.get("labels") is not None else np.zeros(len(boxes), dtype=int))

    objects: List[DetectedObject] = []
    for i in range(len(boxes)):
        if scores[i] < score_thr:
            continue
        x_fwd, y_left, _z, w, l, h, yaw, vx_fwd, vy_left = boxes[i]
        lbl = int(labels[i]) if i < len(labels) else 0
        cls = class_names[lbl] if 0 <= lbl < len(class_names) else "car"
        objects.append(DetectedObject(
            cls=cls,
            right=-float(y_left), forward=float(x_fwd), yaw=float(yaw),
            size=(float(w), float(l), float(h)),
            vel=(-float(vy_left), float(vx_fwd)),
            track_id=None,
        ))
    return objects


def _slot_to_local(target_slot, origin_xyz, origin_rot_quat):
    """Global target-slot pose -> ego-local (right, forward, dheading). Reuses the
    convention of scripts/generate_cached_nuscenes_info.slot_to_local."""
    if not target_slot or "pose" not in target_slot:
        return None
    pose = target_slot["pose"]
    sx, sy = float(pose["translation"][0]), float(pose["translation"][1])
    rot = Quaternion(origin_rot_quat)
    right, forward = _global_to_ego_rf([sx, sy], origin_xyz, rot)
    slot_yaw = Quaternion(pose["rotation"]).yaw_pitch_roll[0]
    ego_yaw = rot.yaw_pitch_roll[0]
    return (right, forward, _norm_angle(slot_yaw - ego_yaw))
