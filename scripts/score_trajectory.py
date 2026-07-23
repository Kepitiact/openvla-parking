"""Trajectory metrics for the parking task, including final-pose (position AND heading).

eval_drivevla -> planning_evaluation reports nuScenes-style L2/collision. For CARLA parking
we also want FINAL-POSE error: how far off, and how mis-aligned, is the car at the end of
the maneuver. ADE/FDE alone miss "parked crooked" -- a car at the right spot with the wrong
heading has small FDE but is not parked. Heading is the third trajectory component (x,y,h),
which retrieve_traj drops, so this parses the (x,y,h) triples itself.

GT comes from the teacher traces (their `trajectory` field is the GT 6x(x,y,h), keyed by
token). Convention: x=right, y=forward, h=heading in radians.

Usage:
  python scripts/score_trajectory.py \
      --student <infer_out>/planning_conversations_val.json \
      --traces  <staging>/reasoning/v1_uniad_epoch6/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re

_TRIPLE = re.compile(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)")


def parse_xyh(text: str):
    """(x,y,h) waypoints from a trajectory span. Empty list if unparseable."""
    return [(float(a), float(b), float(c)) for a, b, c in _TRIPLE.findall(text)]


def _wrap(a: float) -> float:
    """Angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _load_student(path: str):
    text = open(path).read().strip()
    rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines()]
    out = {}
    for r in rows:
        # inference ids carry a "_trajectory" suffix; traces are keyed by the bare token.
        tok = (r.get("id") or r.get("sample_id") or "").removesuffix("_trajectory")
        ans = r.get("answer")
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        out[tok] = parse_xyh(ans or "")
    return out


def _load_gt(traces_path: str):
    gt = {}
    for line in open(traces_path):
        r = json.loads(line)
        gt[r["token"]] = [tuple(w) for w in r["trajectory"]]
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--traces", required=True)
    args = ap.parse_args()

    pred, gt = _load_student(args.student), _load_gt(args.traces)
    common = [t for t in pred if t in gt and len(pred[t]) == len(gt[t]) == 6]
    dropped = sum(1 for t in pred if t in gt and len(pred[t]) != 6)
    if not common:
        raise SystemExit("no frames with a parseable 6-waypoint prediction")

    ade, fde, fpos, fhead = [], [], [], []
    for t in common:
        p, g = pred[t], gt[t]
        disp = [math.hypot(pi[0] - gi[0], pi[1] - gi[1]) for pi, gi in zip(p, g)]
        ade.append(sum(disp) / len(disp))
        fde.append(disp[-1])
        fpos.append(disp[-1])                                  # final position error
        fhead.append(abs(_wrap(p[-1][2] - g[-1][2])))          # final heading error (rad)

    n = len(common)
    mean = lambda xs: sum(xs) / len(xs)
    print(f"scored {n} frames ({dropped} predictions dropped: not 6 waypoints)\n")
    print(f"  ADE (avg displacement):     {mean(ade):.3f} m")
    print(f"  FDE (final displacement):   {mean(fde):.3f} m")
    print(f"  final position error:       {mean(fpos):.3f} m")
    print(f"  final heading error:        {mean(fhead):.3f} rad  ({math.degrees(mean(fhead)):.1f} deg)")
    # parked-well rate: within 0.5 m and ~10 deg of the GT end pose
    ok = sum(1 for i in range(n) if fpos[i] < 0.5 and fhead[i] < math.radians(10))
    print(f"  'parked well' (<0.5m, <10deg): {ok}/{n} = {ok/n:.2f}")


if __name__ == "__main__":
    main()
