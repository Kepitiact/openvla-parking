"""Validators — pure functions, reused verbatim as Step-8 causality/quality gates.

Signatures are written to be callable at eval time on (trace, fact, trajectory,
scene), not only at generation time:

  entity_fidelity(trace, fact)     mentioned entities subset of fact factors  [0..1]
  action_fidelity(decision, traj)  decision consistent with trajectory kinematics [0/1]
  causal_locality(fact, frame)     every factor grounded in the current scene   [0/1]
  schema_conformance(record)       decision in set, factors typed, tokens balanced [0/1]

Each returns a float in [0, 1]; 1.0 == pass. A record "passes" a gate when its
score >= PASS[<name>] (all 1.0 except action_fidelity, which is already 0/1).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .schema import (
    DECISIONS,
    EntityRef,
    FactRecord,
    REASON_START,
    REASON_END,
    TRAJ_START,
    TRAJ_END,
    extract_entity_mentions,
)

PASS = {
    "entity_fidelity": 1.0,
    "action_fidelity": 1.0,
    "causal_locality": 1.0,
    "schema_conformance": 1.0,
}

_GROUND_TOL = 0.7   # metres; an obstacle factor must match a detection within this


# ── entity fidelity ───────────────────────────────────────────────────────────
def entity_fidelity(trace: str, fact: FactRecord) -> float:
    """Coverage of the trace's named entities by the fact's factors (subset test).
    1.0 when every entity the trace mentions is present in the fact (no
    hallucination); 0 mentions -> vacuously 1.0."""
    mentions = extract_entity_mentions(trace)
    if not mentions:
        return 1.0
    allowed = fact.mention_keys()
    covered = sum(1 for m in mentions if m in allowed)
    return covered / len(mentions)


# ── action fidelity ───────────────────────────────────────────────────────────
def action_fidelity(decision: str, traj) -> float:
    """Does the trajectory *contradict* the decision? traj is 6x3
    (right, forward, dheading); the current pose is the origin (0,0,0).

    This is a **contradiction gate**, not a motion-confirmation test: it fails only
    when the trajectory clearly does the opposite of the decision, and passes on
    stationary or ambiguous frames (a paused ego does not contradict its gear). That
    is the semantics Step 8 needs — catch "reverse reasoning + a clearly-forward
    trajectory", not punish a legitimately slow/ambiguous frame.

    Two data facts drive the design: parking here is heavily *multi-point* (~5.6 gear
    changes/episode), and at 2 Hz the 3-s (6-step) horizon is longer than one gear
    phase — so direction decisions are judged on the *immediate* motion (waypoint 2,
    ~1 s) and only when the ego is actually moving (end displacement >= STATIONARY).
    """
    STATIONARY = 0.4   # metres of net motion below which the ego is "paused" (no contradiction)
    CLEAR = 0.2        # metres; forward/back motion beyond this counts as a clear direction

    t = np.asarray(traj, dtype=np.float64).reshape(-1, 3)
    P = t[:, :2]
    fwd = t[:, 1]
    early_fwd = float(fwd[1])                                 # forward pos after ~1 s
    last_step = float(np.linalg.norm(P[-1] - P[-2])) if len(P) >= 2 else 0.0
    end_disp = float(np.linalg.norm(P[-1]))
    dh_end = abs(float(t[-1, 2]))
    moving = end_disp >= STATIONARY

    if decision == "stop_yield":
        return 1.0 if last_step < 0.15 else 0.0
    if decision == "complete_park":
        return 1.0 if (end_disp < 0.6 and last_step < 0.15) else 0.0
    if decision == "reverse":
        return 0.0 if (moving and early_fwd > CLEAR) else 1.0        # clearly driving forward
    if decision == "approach":
        return 0.0 if (moving and early_fwd < -CLEAR) else 1.0       # clearly driving backward
    if decision == "creep":
        return 0.0 if (early_fwd < -CLEAR or early_fwd > 1.2) else 1.0  # reversing, or not crawling
    if decision == "align":
        return 0.0 if (end_disp > 3.0 and dh_end < 0.05) else 1.0    # long straight drive = not aligning
    if decision == "shift_gear":
        # A gear-change frame is a mode switch; its trajectory can look like either
        # gear, so it carries no direction constraint.
        return 1.0
    if decision == "abort":
        return 1.0
    return 0.0


# ── causal locality ───────────────────────────────────────────────────────────
def causal_locality(fact: FactRecord, frame) -> float:
    """Every causal factor must be resolvable from the CURRENT perceivable scene —
    not from the future. Obstacle factors must match a detection in the scene;
    slot/metric factors are current-frame ego measurements (always local).

    `frame` may be a SceneRecord or a Frame (anything exposing `.objects`, directly
    or via `.scene`)."""
    scene = getattr(frame, "scene", frame)
    objects = getattr(scene, "objects", [])
    for ef in fact.causal_factors:
        if ef.kind != "obstacle":
            continue
        grounded = any(
            np.hypot(o.right - ef.r, o.forward - ef.f) <= _GROUND_TOL
            for o in objects
        )
        if not grounded:
            return 0.0
    return 1.0


# ── schema conformance ────────────────────────────────────────────────────────
def _tokens_balanced(assistant_turn: str) -> bool:
    idxs = []
    for tok in (REASON_START, REASON_END, TRAJ_START, TRAJ_END):
        if assistant_turn.count(tok) != 1:
            return False
        idxs.append(assistant_turn.index(tok))
    return idxs == sorted(idxs)


def schema_conformance(record: Dict[str, Any]) -> float:
    """`record` is a serialized trace record (dict). Checks decision membership,
    factor typing, trajectory shape, and balanced/ordered special tokens."""
    if record.get("decision") not in DECISIONS:
        return 0.0

    for f in record.get("causal_factors", []):
        try:
            EntityRef.from_dict(f).canonical()
        except Exception:
            return 0.0

    traj = record.get("trajectory")
    if traj is not None:
        arr = np.asarray(traj, dtype=np.float64)
        if arr.shape != (6, 3):
            return 0.0

    at = record.get("assistant_turn")
    if at is not None and not _tokens_balanced(at):
        return 0.0

    return 1.0
