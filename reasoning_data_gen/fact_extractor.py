"""Deterministic fact extraction: SceneRecord -> FactRecord.

Pure rules, no LLM, no randomness: same SceneRecord always yields the same
FactRecord. Thresholds were calibrated against the training distribution (see
reasoning_data_gen/README.md for the numbers behind each value).

Decision precedence (first match wins) — PERCEPTION DOES NOT GATE THE DECISION.
A detection can never flip the maneuver label (a perception error must not turn
`reverse` into `stop_yield`); it only supplies the *causal factors* that justify it.
  1. shift_gear    reverse flag differs from the previous frame (a gear-change frame)
  2. stop_yield    ~stopped AND a detected object sits in the motion corridor
  3. complete_park ~stopped AND close to the slot AND small heading error
  4. reverse       reverse gear engaged (backing into the slot)
  5. creep         forward, crawling, with a nearby in-path object (small clearance)
  6. align         forward, slow, strong steering (correcting heading)
  7. approach      forward-phase default (driving toward the slot)

PERCEPTION IS LOAD-BEARING THROUGH THE FACTORS. Each decision cites the perceived
objects that actually constrain the maneuver (see schema.ROLE_*):
  flank_left/right  cars parked in the bays either side of the target slot — the
                    physical walls of the gap being reversed into
  front             a car ahead that caps how far the forward swing can go — this is
                    *why* the ego runs out of room and shifts to reverse
  rear              a car behind that limits how far it can back up
  swept             an object near the arc the ego is steering through
  in_path           an object in the immediate travel corridor (stop/creep)
Only the 1-3 objects that genuinely constrain are cited; listing every detected car
would dilute the causal signal into a different flavour of reasoning theatre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .scene_record import DetectedObject, SceneRecord
from .schema import (
    FactRecord,
    ROLE_FLANK_LEFT,
    ROLE_FLANK_RIGHT,
    ROLE_FRONT,
    ROLE_IN_PATH,
    ROLE_REAR,
    ROLE_SWEPT,
    metric_ref,
    obstacle_ref,
    slot_ref,
)

VEHICLE_CLASSES = frozenset({"car", "truck"})


@dataclass(frozen=True)
class Thresholds:
    """Decision-rule + perception thresholds (metres, m/s, radians, normalized steer).

    Calibrated on parking_infos_train.pkl (N=61,659 frames). Key percentiles:
      ego_speed: p10=0.05, p50=0.98, p90=1.52 m/s   (~10% of frames are < 0.05)
      |steer|:   p50=0.36, p90=1.00                 (steer is normalized [-1,1])
      dist_to_slot: p10=1.88, p50=4.41 m
      |dheading_to_slot|: p50=0.45 rad (~26 deg)
    Perception thresholds measured on the 371-frame epoch4 set: bays are 3.0 m wide;
    a flanking car sits ~3.1 m from the slot centre (median). Both flanks are occupied
    in only ~7% of frames, one flank in ~59% — so flank factors must handle 0/1/2.
    """

    speed_stop: float = 0.10       # ~p12: below this the ego is treated as stopped
    creep_speed: float = 0.50      # crawling band upper bound (below ~p30 of moving)
    align_speed: float = 0.60      # "slow" ceiling for the align (heading-correction) rule
    align_steer: float = 0.50      # strong steer (above the p50 of 0.36)
    near_slot_dist: float = 1.50   # "at the slot" radius for complete_park
    align_err: float = 0.20        # ~11 deg heading tolerance for complete_park
    corridor_half_width: float = 1.20   # half a vehicle width; keeps lane-adjacent parked
                                        #   cars out of the in-path test
    corridor_len_stop: float = 5.00     # how far ahead/behind counts as "in the path"
    creep_clearance: float = 4.00       # in-path object within this -> creep while moving
    slot_occupied_radius: float = 1.20  # a detection this close to the slot -> occupied

    # ── perception factors (Task 1) ──────────────────────────────────────────
    bay_width: float = 3.00        # bay pitch: neighbour bay centre is this far across
    flank_radius: float = 2.00     # a vehicle within this of a neighbour-bay centre = flank
    front_len: float = 8.00        # a vehicle this far ahead caps the forward swing
    front_half_width: float = 2.50 # swing envelope is wider than the stop corridor
    rear_len: float = 6.00         # a vehicle this far behind limits backing up
    rear_half_width: float = 2.00
    swept_radius: float = 2.50     # object within this of an upcoming waypoint = in the arc


DEFAULT_THRESHOLDS = Thresholds()


# ── perceived-object detectors (all read ONLY current-frame detections) ───────
def nearest_in_path(scene: SceneRecord, th: Thresholds = DEFAULT_THRESHOLDS
                    ) -> Optional[DetectedObject]:
    """Nearest detected object inside the immediate motion corridor (ahead when
    driving forward, behind when reversing). None if the path is clear."""
    reverse = scene.command.reverse
    best, best_gap = None, float("inf")
    for obj in scene.objects:
        along = -obj.forward if reverse else obj.forward
        if along <= 0.0 or along > th.corridor_len_stop:
            continue
        if abs(obj.right) > th.corridor_half_width:
            continue
        if along < best_gap:
            best_gap, best = along, obj
    return best


def bay_flanks(scene: SceneRecord, th: Thresholds = DEFAULT_THRESHOLDS
               ) -> Tuple[Optional[DetectedObject], Optional[DetectedObject]]:
    """Vehicles parked in the bays immediately left/right of the TARGET slot.

    These are the physical walls of the gap the ego reverses into. Neighbour-bay
    centres are the slot centre offset by +/- bay_width along the slot's across-axis.
    In ego (right, forward) coords the slot's right-unit is (cos dh, sin dh), where dh
    is the slot heading relative to ego forward (verified against a global-frame
    computation to 1e-6 m). Returns (left, right); either may be None — both flanks
    are occupied in only ~7% of frames.
    """
    if scene.command.slot_local is None:
        return None, None
    sr, sf, dh = scene.command.slot_local
    centre = np.array([sr, sf])
    right_unit = np.array([np.cos(dh), np.sin(dh)])
    targets = {
        ROLE_FLANK_RIGHT: centre + th.bay_width * right_unit,
        ROLE_FLANK_LEFT: centre - th.bay_width * right_unit,
    }
    found = {}
    for role, t in targets.items():
        best, best_d = None, th.flank_radius
        for obj in scene.objects:
            if obj.cls not in VEHICLE_CLASSES:
                continue
            d = float(np.hypot(obj.right - t[0], obj.forward - t[1]))
            if d <= best_d:
                best_d, best = d, obj
        found[role] = best
    return found[ROLE_FLANK_LEFT], found[ROLE_FLANK_RIGHT]


def front_blocker(scene: SceneRecord, th: Thresholds = DEFAULT_THRESHOLDS
                  ) -> Optional[DetectedObject]:
    """Nearest vehicle AHEAD inside the forward-swing envelope — the thing that caps
    how far the ego can pull forward, and hence why it must shift to reverse."""
    best, best_gap = None, float("inf")
    for obj in scene.objects:
        if obj.forward <= 0.0 or obj.forward > th.front_len:
            continue
        if abs(obj.right) > th.front_half_width:
            continue
        if obj.forward < best_gap:
            best_gap, best = obj.forward, obj
    return best


def rear_blocker(scene: SceneRecord, th: Thresholds = DEFAULT_THRESHOLDS
                 ) -> Optional[DetectedObject]:
    """Nearest object BEHIND inside the reversing envelope — limits how far back."""
    best, best_gap = None, float("inf")
    for obj in scene.objects:
        back = -obj.forward
        if back <= 0.0 or back > th.rear_len:
            continue
        if abs(obj.right) > th.rear_half_width:
            continue
        if back < best_gap:
            best_gap, best = back, obj
    return best


def swept_path_obstacle(scene: SceneRecord, planned_path, th: Thresholds = DEFAULT_THRESHOLDS
                        ) -> Optional[DetectedObject]:
    """Nearest perceived object lying close to the arc the ego is about to sweep.

    CAUSAL LOCALITY NOTE: `planned_path` (the upcoming waypoints) is used ONLY to
    *select* which currently-perceived object to cite — the cited entity is always a
    detection in the CURRENT scene, so nothing unperceivable ever enters the trace and
    `validators.causal_locality` still passes. The trace is an output *target* (like
    the trajectory itself), so future-derived selection is supervision, not an input
    leak. Pass planned_path=None to disable this factor entirely.
    """
    if planned_path is None:
        return None
    pts = np.asarray(planned_path, dtype=np.float64).reshape(-1, 3)[:, :2]
    best, best_d = None, th.swept_radius
    for obj in scene.objects:
        p = np.array([obj.right, obj.forward])
        d = float(np.min(np.linalg.norm(pts - p, axis=1)))
        if d <= best_d:
            best_d, best = d, obj
    return best


def _slot_occupied(scene: SceneRecord, th: Thresholds) -> bool:
    if scene.command.slot_local is None:
        return False
    sr, sf, _ = scene.command.slot_local
    for obj in scene.objects:
        if np.hypot(obj.right - sr, obj.forward - sf) <= th.slot_occupied_radius:
            return True
    return False


def _dedup(refs: Sequence) -> List:
    """Drop duplicate obstacle refs (the same object can be e.g. both front and swept);
    the first-cited role wins, so ordering encodes priority."""
    seen, out = set(), []
    for ef in refs:
        key = (ef.kind, ef.name, None if ef.r is None else round(ef.r, 1),
               None if ef.f is None else round(ef.f, 1))
        if ef.kind == "obstacle" and key in seen:
            continue
        seen.add(key)
        out.append(ef)
    return out


# ── main extractor ────────────────────────────────────────────────────────────
def extract_fact(scene: SceneRecord, th: Thresholds = DEFAULT_THRESHOLDS,
                 planned_path=None) -> FactRecord:
    """Deterministic SceneRecord -> FactRecord.

    `planned_path` (optional, 6x3 upcoming waypoints) is used ONLY to select which
    currently-perceived object sits in the ego's swept arc — see swept_path_obstacle.
    """
    speed = scene.ego.speed
    steer = abs(scene.ego.steer)
    reverse = scene.command.reverse
    dist = scene.dist_to_slot()
    aerr = scene.align_err()
    in_path = nearest_in_path(scene, th)
    occupied = _slot_occupied(scene, th)

    meta = {
        "token": scene.token, "reverse": reverse, "speed": speed,
        "steer": scene.ego.steer, "dist_to_slot": dist, "align_err": aerr,
        "side": scene.command.side, "source": scene.source,
    }

    slot_factors: List = []
    if scene.command.slot_local is not None:
        sr, sf, _ = scene.command.slot_local
        slot_factors.append(slot_ref(occupied, right=sr, forward=sf))
        slot_factors.append(metric_ref("dist_to_slot", dist))

    def flank_refs() -> List:
        left, right = bay_flanks(scene, th)
        out = []
        if left is not None:
            out.append(obstacle_ref(left.cls, left.right, left.forward, ROLE_FLANK_LEFT))
        if right is not None:
            out.append(obstacle_ref(right.cls, right.right, right.forward, ROLE_FLANK_RIGHT))
        return out

    def front_ref() -> List:
        o = front_blocker(scene, th)
        return [obstacle_ref(o.cls, o.right, o.forward, ROLE_FRONT)] if o else []

    def rear_ref() -> List:
        o = rear_blocker(scene, th)
        return [obstacle_ref(o.cls, o.right, o.forward, ROLE_REAR)] if o else []

    def swept_ref() -> List:
        o = swept_path_obstacle(scene, planned_path, th)
        return [obstacle_ref(o.cls, o.right, o.forward, ROLE_SWEPT)] if o else []

    # 1. gear change — cite the car ahead that ran the forward swing out of room.
    if scene.prev_reverse is not None and scene.prev_reverse != reverse:
        return FactRecord("shift_gear",
                          _dedup(front_ref() + slot_factors), meta)

    # 2/3. stopped states
    if speed < th.speed_stop:
        if in_path is not None:
            return FactRecord(
                "stop_yield",
                [obstacle_ref(in_path.cls, in_path.right, in_path.forward, ROLE_IN_PATH),
                 metric_ref("clearance", in_path.dist)],
                meta)
        if dist is not None and aerr is not None and dist < th.near_slot_dist and aerr < th.align_err:
            return FactRecord(
                "complete_park",
                _dedup(flank_refs() + slot_factors + [metric_ref("align_err", aerr)]),
                meta)

    # 4. reversing — the bay's flanking cars and whatever limits backing up.
    if reverse:
        factors = flank_refs() + rear_ref() + slot_factors
        if aerr is not None:
            factors.append(metric_ref("align_err", aerr))
        return FactRecord("reverse", _dedup(factors), meta)

    # forward phase (reverse == False)
    # 5. creep: crawling with a nearby in-path object
    if in_path is not None and in_path.dist <= th.creep_clearance and speed < th.creep_speed:
        return FactRecord(
            "creep",
            _dedup([obstacle_ref(in_path.cls, in_path.right, in_path.forward, ROLE_IN_PATH),
                    metric_ref("clearance", in_path.dist)] + swept_ref()),
            meta)

    # 6. align: slow + strong steer — cite what is forcing the steer.
    if steer > th.align_steer and speed < th.align_speed:
        factors = swept_ref() + front_ref() + slot_factors
        if aerr is not None:
            factors.append(metric_ref("align_err", aerr))
        return FactRecord("align", _dedup(factors), meta)

    # 7. approach: forward-phase default — cite what caps the approach.
    return FactRecord("approach",
                      _dedup(front_ref() + swept_ref() + slot_factors), meta)
