"""Counterfactual stop-injection.

Given a factual frame, synthesize the matched counterfactual: an obstacle appears
on the ego's path, so the ego must decelerate to a stop and its decision flips to
stop_yield. The pair shares an id (factual <-> counterfactual) so Step 8's swap /
disable causality gates can pair them, and it carries a synthetic-perception-token
spec so Step 2/3 can inject a matching detection/track token for the fake obstacle.

Everything here is pure kinematics + fact swapping (no LLM, no network). The
trajectory stays in the (right, forward, dheading) format over the same 6-step
horizon; only the source-side perception token spec is emitted for later injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .scene_record import DetectedObject, Frame
from .schema import FactRecord, metric_ref, obstacle_ref

# Kinematics / placement constants.
_MIN_PATH_LEN = 0.8      # net ego displacement (m) required to build a stop injection
_EPS = 1e-6
_OBSTACLE_CLS = "pedestrian"                 # a dynamic obstacle is the natural yield cause
_OBSTACLE_SIZE = (0.6, 0.6, 1.7)             # (w, l, h) metres, pedestrian
_INJECTED_TRACK_ID = -1                       # negative id marks a synthetic track


@dataclass
class CounterfactualPair:
    """A factual/counterfactual pair with a shared id and an injection spec."""

    pair_id: str
    factual_fact: FactRecord
    factual_traj: np.ndarray            # (6, 3)
    counterfactual_fact: FactRecord
    counterfactual_traj: np.ndarray     # (6, 3)
    synthetic_token_spec: Dict[str, Any]
    injection: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "factual": {"fact": self.factual_fact.to_dict(),
                        "trajectory": self.factual_traj.tolist()},
            "counterfactual": {"fact": self.counterfactual_fact.to_dict(),
                               "trajectory": self.counterfactual_traj.tolist()},
            "synthetic_token_spec": self.synthetic_token_spec,
            "injection": self.injection,
        }


def _decelerate_to_stop(traj: np.ndarray, ds: int) -> np.ndarray:
    """Real prefix traj[:ds], then a linear decelerate-to-stop from waypoint ds.

    Heading is frozen at the injection moment (the ego stops, it does not keep
    rotating). Motion continues along the ego's direction of travel entering the
    decel, with per-step displacement ramping linearly to zero by the horizon end.
    """
    P = traj[:, :2].astype(np.float64)
    prev = P[ds - 1] if ds >= 1 else np.zeros(2)     # position entering the decel
    prev2 = P[ds - 2] if ds >= 2 else np.zeros(2)    # position one step earlier
    step_vec = prev - prev2
    v0 = float(np.linalg.norm(step_vec))
    u = step_vec / v0 if v0 > _EPS else np.zeros(2)
    heading = float(traj[ds - 1, 2]) if ds >= 1 else 0.0

    out = [traj[k].astype(np.float64).copy() for k in range(ds)]
    n = 6 - ds  # steps remaining to ramp speed to zero
    pos = prev.copy()
    for i in range(1, n + 1):
        frac = max(0.0, 1.0 - i / float(n))          # 1 -> 0 across the tail
        pos = pos + u * (v0 * frac)
        out.append(np.array([pos[0], pos[1], heading], dtype=np.float64))
    return np.asarray(out, dtype=np.float64).reshape(6, 3)


def make_stop_injection(frame: Frame, obstacle_cls: str = _OBSTACLE_CLS
                        ) -> Optional[CounterfactualPair]:
    """Build the stop-injection counterfactual for `frame`, or None if the ego is
    not moving enough for a stop to be meaningful (e.g. already parked/stopped)."""
    traj = np.asarray(frame.trajectory, dtype=np.float64).reshape(6, 3)
    P = traj[:, :2]
    net = float(np.linalg.norm(P[-1]))
    if net < _MIN_PATH_LEN:
        return None

    # Decel starts partway along the path (leave >=2 steps to come to rest), and the
    # obstacle sits just beyond the stop point, where the ego would have driven.
    ds = 2 if len(P) >= 4 else 1
    cf_traj = _decelerate_to_stop(traj, ds)

    stop_pt = cf_traj[-1, :2]
    obst_idx = min(ds + 1, 5)
    obst_pt = P[obst_idx]
    gap = float(np.linalg.norm(obst_pt - stop_pt))

    synthetic = DetectedObject(
        cls=obstacle_cls,
        right=float(obst_pt[0]), forward=float(obst_pt[1]),
        yaw=0.0, size=_OBSTACLE_SIZE, vel=(0.0, 0.0),
        track_id=_INJECTED_TRACK_ID,
    )
    token_spec = {**synthetic.to_dict(), "injected": True, "source": "counterfactual"}

    cf_fact = FactRecord(
        decision="stop_yield",
        causal_factors=[
            obstacle_ref(synthetic.cls, synthetic.right, synthetic.forward),
            metric_ref("clearance", gap),
        ],
        meta={**dict(frame.fact.meta), "counterfactual": True,
              "of_decision": frame.fact.decision},
    )

    pair_id = f"{frame.token}::cf_stop"
    return CounterfactualPair(
        pair_id=pair_id,
        factual_fact=frame.fact,
        factual_traj=traj,
        counterfactual_fact=cf_fact,
        counterfactual_traj=cf_traj,
        synthetic_token_spec=token_spec,
        injection={"index": obst_idx, "decel_start": ds,
                   "right": float(obst_pt[0]), "forward": float(obst_pt[1]),
                   "cls": obstacle_cls},
    )
