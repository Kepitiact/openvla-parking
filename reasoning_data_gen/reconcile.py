"""Reconcile a perception-derived fact against the GT-derived fact.

At inference the SceneRecord comes from UniAD (imperfect). Reconciliation flags
frames where perception and GT disagree so the trainer can drop them from
reasoning-consistency training:

  ok                 perception and GT agree (same decision, same named entities)
  perception_limited GT sees causal entities perception missed (perception saw LESS)
                     -> the trace is still safe to learn from GT, but conservative
  contradiction      perception asserts an entity/decision GT does not support, or
                     the two decisions conflict in a safety-relevant way
                     -> unsafe to train the reasoning-consistency loss on

In Step 0 the perception source IS the GT source, so reconcile mostly returns 'ok';
the divergent branches are unit-tested with synthetic mismatches.
"""

from __future__ import annotations

from .schema import FactRecord

OK = "ok"
PERCEPTION_LIMITED = "perception_limited"
CONTRADICTION = "contradiction"

# Decisions where a mismatch is safety-relevant (yielding vs not yielding).
_SAFETY_DECISIONS = frozenset({"stop_yield"})


def reconcile(fact_from_perception: FactRecord, fact_from_gt: FactRecord) -> str:
    perc_keys = fact_from_perception.mention_keys()
    gt_keys = fact_from_gt.mention_keys()

    perc_only = perc_keys - gt_keys   # perception claims entities GT lacks
    gt_only = gt_keys - perc_keys     # GT has entities perception missed

    dp, dg = fact_from_perception.decision, fact_from_gt.decision
    safety_divergence = (dp in _SAFETY_DECISIONS) != (dg in _SAFETY_DECISIONS)

    # Perception hallucinating an entity, or a safety-relevant decision flip, is a
    # contradiction: the reasoning target would be grounded in something false.
    if perc_only or safety_divergence:
        return CONTRADICTION

    # Perception saw strictly less than GT (missed obstacles / occupancy).
    if gt_only:
        return PERCEPTION_LIMITED

    # Same named entities; a non-safety decision difference is still a soft limit.
    if dp != dg:
        return PERCEPTION_LIMITED

    return OK
