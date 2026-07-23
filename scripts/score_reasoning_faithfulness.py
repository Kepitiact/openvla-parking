"""Score the STUDENT's reasoning: is it faithful, and is it diverse?

eval_drivevla already scores the trajectory (ADE/FDE via planning_evaluation). This scores
the other half -- the reasoning the student emits -- which nothing else looks at. Two things:

  FAITHFULNESS  run the same validators we gated the teacher with, now on the STUDENT's
                reasoning, against the ground-truth fact for that frame (the fact the
                teacher trace was built from). Catches a student that hallucinates objects
                or claims a different decision than the one its trajectory takes.
  DIVERSITY     distinct-n over the student's traces. A collapse to one template is the
                failure mode we watched the teacher for; the student can regress to it too.

The ground-truth fact per frame comes from the teacher traces (they carry decision +
causal_factors). So this needs: the student's inference output (with the `reasoning` field
that inference_drivevla now keeps) and the teacher traces.jsonl.

Usage:
  python scripts/score_reasoning_faithfulness.py \
      --student  <infer_out>/planning_conversations_val.json \
      --traces   <staging>/reasoning/v1_uniad_epoch6/traces.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from reasoning_data_gen import validators as V              # noqa: E402
from reasoning_data_gen.schema import FactRecord             # noqa: E402
from reasoning_data_gen.verbalizer import find_hallucinations  # noqa: E402


def _load_student(path: str) -> dict:
    """token -> student reasoning text. inference stores answer/reasoning as lists (one per
    returned sequence); take the first."""
    out = {}
    with open(path) as f:
        # inference writes either a JSON list or JSONL; handle both.
        text = f.read().strip()
        rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines()]
    for r in rows:
        # inference ids carry a "_trajectory" suffix; traces are keyed by the bare token.
        tok = (r.get("id") or r.get("sample_id") or "").removesuffix("_trajectory")
        reasoning = r.get("reasoning")
        if isinstance(reasoning, list):
            reasoning = reasoning[0] if reasoning else ""
        out[tok] = reasoning or ""
    return out


def _load_facts(traces_path: str) -> dict:
    """token -> FactRecord (the ground-truth fact the teacher verbalized)."""
    facts = {}
    with open(traces_path) as f:
        for line in f:
            r = json.loads(line)
            facts[r["token"]] = FactRecord.from_dict(r)   # r has decision + causal_factors
    return facts


def _distinct_n(traces, n: int) -> float:
    grams = [tuple(t.split()[i:i + n]) for t in traces
             for i in range(max(0, len(t.split()) - n + 1))]
    return len(set(grams)) / max(1, len(grams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="inference output with a `reasoning` field")
    ap.add_argument("--traces", required=True, help="teacher traces.jsonl (the GT facts)")
    args = ap.parse_args()

    student = _load_student(args.student)
    facts = _load_facts(args.traces)

    common = [t for t in student if t in facts and student[t].strip()]
    if not common:
        sys.exit("no overlapping tokens with non-empty reasoning; check the inputs")

    ent, cau = [], []
    hallucinating = 0
    per_decision_clean = collections.Counter()
    per_decision_total = collections.Counter()
    for tok in common:
        trace, fact = student[tok], facts[tok]
        ent.append(V.entity_fidelity(trace, fact))
        cau.append(1.0 if not find_hallucinations(trace, fact) else 0.0)  # clean == no hallucination
        if find_hallucinations(trace, fact):
            hallucinating += 1
        per_decision_total[fact.decision] += 1
        if not find_hallucinations(trace, fact):
            per_decision_clean[fact.decision] += 1

    traces = [student[t] for t in common]
    print(f"scored {len(common)} frames with student reasoning\n")
    print("FAITHFULNESS (student reasoning vs GT fact):")
    print(f"  entity_fidelity (mean):   {sum(ent)/len(ent):.3f}")
    print(f"  hallucination-free rate:  {1 - hallucinating/len(common):.3f}  "
          f"({hallucinating} frames name something not in the fact)")
    print("\n  clean rate by decision:")
    for dec in sorted(per_decision_total):
        c, t = per_decision_clean[dec], per_decision_total[dec]
        print(f"    {dec:>13}: {c}/{t} = {c/t:.2f}")

    print("\nDIVERSITY (student traces):")
    print(f"  unique {len(set(traces))}/{len(traces)} ({len(set(traces))/len(traces):.0%})"
          f"  distinct-2 {_distinct_n(traces,2):.2f}  distinct-3 {_distinct_n(traces,3):.2f}")

    # The headline single number: is the reasoning honest AND varied?
    hall_free = 1 - hallucinating / len(common)
    if hall_free < 0.9:
        print("\n!! low faithfulness: the student is hallucinating -- inspect before trusting traces.")
    if _distinct_n(traces, 2) < 0.1:
        print("!! low diversity: traces may have collapsed to a template.")


if __name__ == "__main__":
    main()
