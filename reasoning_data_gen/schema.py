"""Schema: decisions, tokens, entity references, fact records, render/format helpers.

Everything downstream (fact extractor, verbalizer, counterfactuals, validators,
Step 5's conversation builder, Step 8's eval gates) speaks this vocabulary.

Forward-compatibility contract:
  * render_assistant_turn() emits EXACTLY the assistant-turn string Step 5 will
    produce: {REASON_START}{trace}{REASON_END}{TRAJ_START}{traj}{TRAJ_END}.
  * format_trajectory() reproduces build_llava_conversation's traj string byte for
    byte, so a reasoning record is a drop-in replacement for the plain-traj record.
  * EntityRef is typed + canonical + string-serializable so entity_fidelity is a
    pure set comparison, and the same refs serialize into JSON records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set

# ── Special tokens ────────────────────────────────────────────────────────────
# Reasoning-block delimiters. Registered in the tokenizer in a later step (Step 6);
# here they are only string constants.
REASON_START = "<reason_start>"
REASON_END = "<reason_end>"

# Trajectory-block delimiters. These MUST match OpenDriveVLA/llava/constants.py so
# render_assistant_turn() is byte-identical to what training emits.
try:  # pragma: no cover - llava is not on the path during Step-0 dev/CI
    from llava.constants import (  # type: ignore
        DEFAULT_TRAJ_START_TOKEN as TRAJ_START,
        DEFAULT_TRAJ_END_TOKEN as TRAJ_END,
    )
except Exception:
    # TODO(integration): unify with llava.constants when reasoning_data_gen runs
    # inside the training env. Kept identical to constants.py as of this commit.
    TRAJ_START = "<traj_start>"
    TRAJ_END = "<traj_end>"

# ── Decision set ──────────────────────────────────────────────────────────────
DECISION_SET: List[str] = [
    "approach",       # forward-phase driving toward the slot
    "align",          # correcting heading (strong steer, low speed) before/into reverse
    "creep",          # crawling forward with small clearance
    "reverse",        # actively backing into the slot
    "stop_yield",     # halted for an in-path obstacle
    "shift_gear",     # gear just changed (forward<->reverse transition frame)
    "complete_park",  # stopped, aligned, in the slot
    "abort",          # reserved (unused in v1 data)
]
DECISIONS: FrozenSet[str] = frozenset(DECISION_SET)

# Surface phrases -> canonical decision, used by the hallucination guard and by
# action_fidelity to recover which decision a free-form trace claims.
DECISION_LEXICON: Dict[str, str] = {
    "approach": "approach", "approaching": "approach", "drive forward": "approach",
    "align": "align", "aligning": "align", "line up": "align", "lining up": "align",
    "creep": "creep", "creeping": "creep", "crawl": "creep", "crawling": "creep",
    "reverse": "reverse", "reversing": "reverse", "back": "reverse", "backing": "reverse",
    "stop": "stop_yield", "yield": "stop_yield", "halt": "stop_yield", "yielding": "stop_yield",
    "shift": "shift_gear", "gear": "shift_gear", "change gear": "shift_gear",
    "parked": "complete_park", "finish": "complete_park", "complete": "complete_park",
    "abort": "abort", "aborting": "abort",
}

# ── Entities ──────────────────────────────────────────────────────────────────
# Obstacle classes the perceiver can report (matches gt_names in the infos).
OBSTACLE_CLASSES: FrozenSet[str] = frozenset({"car", "truck", "pedestrian"})
# Surface words -> obstacle-class mention key, used by entity_fidelity to map a
# free-form trace to a set of referenced entities.
OBSTACLE_LEXICON: Dict[str, str] = {
    "car": "car", "vehicle": "car", "sedan": "car",
    "truck": "truck", "van": "truck", "lorry": "truck",
    "pedestrian": "pedestrian", "person": "pedestrian", "walker": "pedestrian",
}
SLOT_WORDS: FrozenSet[str] = frozenset({"slot", "bay", "space", "spot", "parking space"})


@dataclass(frozen=True)
class EntityRef:
    """A typed, canonical, string-serializable reference to a perceivable thing.

    kind:
      'obstacle' -> name in OBSTACLE_CLASSES, at ego-local (r, f)
      'slot'     -> name in {'free', 'occupied'}
      'metric'   -> a scalar justification, name in {'dist_to_slot','align_err',...}, value=v
    All coordinates are ego frame: r=right (+right), f=forward (+forward), metres.
    """

    kind: str
    name: str = ""
    r: Optional[float] = None
    f: Optional[float] = None
    value: Optional[float] = None
    role: Optional[str] = None   # obstacles only: WHY this object constrains the maneuver

    def canonical(self) -> str:
        if self.kind == "obstacle":
            tag = f"{self.name}|{self.role}" if self.role else self.name
            return f"obstacle[{tag}]@({self.r:.1f},{self.f:.1f})"
        if self.kind == "slot":
            if self.r is not None and self.f is not None:
                return f"slot({self.name})@({self.r:.1f},{self.f:.1f})"
            return f"slot({self.name})"
        if self.kind == "metric":
            return f"{self.name}={self.value:.1f}"
        raise ValueError(f"unknown EntityRef.kind: {self.kind!r}")

    def bearing(self) -> Optional[str]:
        """Qualitative ego-relative bearing from (r, f), or None if unpositioned.
        Never numeric — obeys the 'keep headings qualitative' rule."""
        if self.r is None or self.f is None:
            return None
        return bearing(self.r, self.f)

    def mention_keys(self) -> FrozenSet[str]:
        """Coarse identity keys used by entity_fidelity. Metrics are not perceivable
        *named* entities (they are ego-relative measurements), so they carry none —
        only obstacles and the slot can be hallucinated."""
        if self.kind == "obstacle":
            return frozenset({"obstacle", f"obstacle:{self.name}"})
        if self.kind == "slot":
            return frozenset({"slot"})
        return frozenset()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "name": self.name,
            "r": self.r, "f": self.f, "value": self.value, "role": self.role,
            "canonical": self.canonical(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EntityRef":
        return EntityRef(kind=d["kind"], name=d.get("name", ""),
                         r=d.get("r"), f=d.get("f"), value=d.get("value"),
                         role=d.get("role"))


def bearing(right: float, forward: float, near: float = 0.7) -> str:
    """Ego-relative qualitative bearing for (right, forward) in metres. Purely
    lexical direction words — used by verbalizers so a trace can say 'behind on my
    left' without ever emitting degrees."""
    lon = "ahead" if forward > near else ("behind" if forward < -near else None)
    lat = "on my right" if right > near else ("on my left" if right < -near else None)
    if lon and lat:
        return f"{lon} {lat}"
    if lon:
        return lon
    if lat:
        return lat
    return "right here"


# Why a perceived object constrains the maneuver. These make perception LOAD-BEARING:
# the object is cited as the cause, not decoration.
ROLE_FLANK_LEFT = "flank_left"     # parked in the bay left of the target slot
ROLE_FLANK_RIGHT = "flank_right"   # parked in the bay right of the target slot
ROLE_FRONT = "front"               # ahead, caps how far the forward swing can go
ROLE_REAR = "rear"                 # behind, limits backing up
ROLE_IN_PATH = "in_path"           # sits in the immediate travel corridor
ROLE_SWEPT = "swept"               # near the arc the ego is steering through
OBSTACLE_ROLES = frozenset({ROLE_FLANK_LEFT, ROLE_FLANK_RIGHT, ROLE_FRONT,
                            ROLE_REAR, ROLE_IN_PATH, ROLE_SWEPT})


def obstacle_ref(cls: str, right: float, forward: float,
                 role: Optional[str] = None) -> EntityRef:
    return EntityRef("obstacle", cls, r=float(right), f=float(forward), role=role)


def slot_ref(occupied: bool, right: Optional[float] = None,
             forward: Optional[float] = None) -> EntityRef:
    """Slot reference. Carrying (right, forward) lets the verbalizer state a
    qualitative bearing ('the slot is behind me on the left'); the occupancy name
    is still the entity identity for fidelity checks."""
    return EntityRef("slot", "occupied" if occupied else "free",
                     r=None if right is None else float(right),
                     f=None if forward is None else float(forward))


def metric_ref(name: str, value: float) -> EntityRef:
    return EntityRef("metric", name, value=float(value))


# ── Fact record ───────────────────────────────────────────────────────────────
@dataclass
class FactRecord:
    """The deterministic, pre-computed structure the teacher verbalizes.

    A generated sentence may name *only* the entities/decision in here; anything
    else is a hallucination and is rejected (see verbalizer + validators).
    """

    decision: str
    causal_factors: List[EntityRef] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def mention_keys(self) -> FrozenSet[str]:
        keys: set = set()
        for ef in self.causal_factors:
            keys |= ef.mention_keys()
        return frozenset(keys)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "causal_factors": [ef.to_dict() for ef in self.causal_factors],
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FactRecord":
        return FactRecord(
            decision=d["decision"],
            causal_factors=[EntityRef.from_dict(x) for x in d.get("causal_factors", [])],
            meta=dict(d.get("meta", {})),
        )


# ── Render / format helpers ───────────────────────────────────────────────────
def format_trajectory(traj: Sequence[Sequence[float]]) -> str:
    """Format 6 waypoints (right, forward, dheading) EXACTLY as
    build_llava_conversation.generate_user_message does:
        "[(x,y,h),(x,y,h),...]" with 2-decimal fields.
    """
    pts = list(traj)
    if len(pts) != 6:
        raise ValueError(f"trajectory must have 6 waypoints, got {len(pts)}")
    return "[" + ",".join(f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})" for p in pts) + "]"


def render_assistant_turn(fact: FactRecord, trace: str, traj_str: str) -> str:
    """The exact assistant-turn string. Step 5 emits this verbatim.

    `traj_str` is the full bracketed trajectory string from format_trajectory()
    (i.e. what today's conversations[1] wraps in TRAJ_START/END). `fact` is accepted
    for signature stability / future schema checks; the string content is trace+traj.
    """
    return f"{REASON_START}{trace}{REASON_END}{TRAJ_START}{traj_str}{TRAJ_END}"


# ── Text -> mention extraction (shared by validators + verbalizer guard) ───────
_WORD_RE = re.compile(r"[a-z_]+")


def extract_entity_mentions(text: str) -> Set[str]:
    """Coarse entity mention keys referenced by a free-form trace, matching the
    keys produced by EntityRef.mention_keys(). Used by entity_fidelity and by the
    verbalizer's hallucination guard. Purely lexical (no LLM)."""
    words = set(_WORD_RE.findall(text.lower()))
    mentions: Set[str] = set()
    for w in words:
        if w in OBSTACLE_LEXICON:
            mentions.add("obstacle")
            mentions.add(f"obstacle:{OBSTACLE_LEXICON[w]}")
    if words & {w for phrase in SLOT_WORDS for w in phrase.split()}:
        mentions.add("slot")
    return mentions


def extract_decision_mentions(text: str) -> Set[str]:
    """Canonical decisions a free-form trace claims (via DECISION_LEXICON)."""
    low = text.lower()
    found: Set[str] = set()
    for phrase, decision in DECISION_LEXICON.items():
        if phrase in low:
            found.add(decision)
    return found
