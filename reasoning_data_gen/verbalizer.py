"""Verbalizers: turn a FactRecord into a 1-2 sentence natural-language trace.

The teacher only *verbalizes* the pre-computed fact — it may name only the
entities/decision in fact.causal_factors. Any sentence that references something
outside the fact is a hallucination and is rejected/regenerated.

  MockVerbalizer  — deterministic template fill, no network. Default for tests,
                    CI, and the dry run. Passes entity_fidelity by construction:
                    it only ever names entities present in the fact.
  QwenVerbalizer  — Qwen2.5-32B-Instruct behind a flag, designed to run as a HAL
                    batch job (reads model path / endpoint / batch size from CLI or
                    env, so it drops into an sbatch wrapper with no code change).
                    Not exercised in Step 0 (needs a GPU).
"""

from __future__ import annotations

import os
from typing import List, Optional

from .schema import (
    EntityRef,
    FactRecord,
    bearing,
    extract_decision_mentions,
    extract_entity_mentions,
)


# ── decision / factor -> qualitative phrasing (no numeric headings) ───────────
_DECISION_PHRASE = {
    "approach": "I am approaching my goal",
    "align": "I am aligning my heading",
    "creep": "I am creeping forward slowly",
    "reverse": "I am reversing",
    "stop_yield": "I am stopping to yield",
    "shift_gear": "I am shifting gears",
    "complete_park": "I have completed the park",
    "abort": "I am aborting the maneuver",
}


def _obstacle_phrase(ef: EntityRef) -> str:
    """Role-aware phrasing: say WHY the object constrains the maneuver, not just that
    it exists. The role comes from the fact, so this stays grounded."""
    dist = abs(ef.f) if ef.f is not None else 0.0
    euclid = float((ef.r or 0.0) ** 2 + (ef.f or 0.0) ** 2) ** 0.5
    # NB: avoid the word "parked" here — DECISION_LEXICON maps it to complete_park, so
    # it would read as a decision claim in a `reverse` trace and trip the guard.
    if ef.role == "flank_left":
        return f"a {ef.name} in the bay to my left"
    if ef.role == "flank_right":
        return f"a {ef.name} in the bay to my right"
    if ef.role == "front":
        return (f"a {ef.name} ahead about {dist:.1f} m away, limiting how far "
                "I can pull forward")
    if ef.role == "rear":
        return f"a {ef.name} behind me about {dist:.1f} m away, limiting how far I can back up"
    if ef.role == "swept":
        return f"a {ef.name} about {euclid:.1f} m from the path I am steering through"
    return f"a {ef.name} {bearing(ef.r or 0.0, ef.f or 0.0)} about {dist:.1f} m away"


def _factor_phrase(ef: EntityRef) -> Optional[str]:
    if ef.kind == "obstacle":
        return _obstacle_phrase(ef)
    if ef.kind == "slot":
        state = "clear" if ef.name == "free" else "occupied"
        brg = ef.bearing()
        return (f"the target slot is {state}, {brg}" if brg
                else f"the target slot is {state}")
    if ef.kind == "metric":
        if ef.name == "dist_to_slot":
            return f"the slot is about {ef.value:.1f} m away"
        if ef.name == "clearance":
            return f"there is about {ef.value:.1f} m of clearance"
        if ef.name == "align_err":
            return ("my heading is nearly on target" if (ef.value or 0.0) < 0.20
                    else "my heading still needs correcting")
    return None


class Verbalizer:
    """Interface: verbalize(fact) -> str (1-2 sentences)."""

    def verbalize(self, fact: FactRecord) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class MockVerbalizer(Verbalizer):
    """Deterministic template — same fact always yields the same trace."""

    def verbalize(self, fact: FactRecord) -> str:
        head = _DECISION_PHRASE.get(fact.decision, f"I am {fact.decision}")
        phrases = [p for p in (_factor_phrase(ef) for ef in fact.causal_factors) if p]
        if not phrases:
            return head + "."
        if len(phrases) == 1:
            reason = phrases[0]
        else:
            reason = ", ".join(phrases[:-1]) + ", and " + phrases[-1]
        return f"{head} because {reason}."


def find_hallucinations(trace: str, fact: FactRecord) -> List[str]:
    """Entities/decisions a trace names that are NOT in the fact. Empty => clean.
    Shared by QwenVerbalizer's guard and useful as a standalone check."""
    problems: List[str] = []
    allowed_entities = fact.mention_keys()
    for m in extract_entity_mentions(trace):
        if m not in allowed_entities:
            problems.append(f"entity:{m}")
    for d in extract_decision_mentions(trace):
        if d != fact.decision:
            problems.append(f"decision:{d}")
    return problems


# Master (system) prompt — CONSTANT across every frame. Encodes the teacher's role
# and the hard rules; the per-frame fact goes in the user prompt below.
QWEN_SYSTEM_PROMPT = (
    "You are the inner monologue of a self-parking car. You are given a maneuver that "
    "has ALREADY been decided and the grounded facts that justify it. Write the car's "
    "reasoning in the FIRST PERSON, 1-2 short natural sentences.\n"
    "Rules:\n"
    "1. State the decision and justify it using ONLY the grounding lines provided.\n"
    "2. Never mention any object, place, direction, or measurement that is not in the "
    "grounding. If it is not listed, it does not exist.\n"
    "3. Keep every direction qualitative (ahead / behind / left / right). NEVER output "
    "degrees, angles, or raw coordinates.\n"
    "4. Write it the way a careful driver actually thinks — do not just read the numbers "
    "back robotically.\n"
    "5. Output only the reasoning sentence(s). No preamble, no bullet points, no quotes."
)


class QwenVerbalizer(Verbalizer):
    """Qwen2.5-32B-Instruct teacher (HAL A100 batch job).

    Two-level prompt: a constant master/system prompt (QWEN_SYSTEM_PROMPT, the rules)
    plus a per-frame user prompt (the FactRecord's decision + grounding). The teacher
    only rewords the fact; every candidate passes find_hallucinations and is
    regenerated on any violation. Never invoked in Step 0 (no GPU in CI).

    Config from args/env so it drops into scripts/hal_reason_qwen.sbatch unchanged:
      QWEN_MODEL_PATH   local weights dir (transformers backend), or
      QWEN_ENDPOINT     OpenAI-compatible chat endpoint (e.g. a vLLM server on HAL)
      QWEN_MODEL_NAME   model name to send to the endpoint
      QWEN_BATCH_SIZE   batch size for the transformers backend
      QWEN_4BIT=1       load 4-bit (fits 32B on one 40 GB A100; else device_map shards)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        endpoint: Optional[str] = None,
        batch_size: Optional[int] = None,
        n_paraphrases: int = 3,
        temperature: float = 0.3,
        max_retries: int = 4,
        load_4bit: Optional[bool] = None,
    ):
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH")
        self.endpoint = endpoint or os.environ.get("QWEN_ENDPOINT")
        self.batch_size = int(batch_size or os.environ.get("QWEN_BATCH_SIZE", 8))
        self.n_paraphrases = n_paraphrases
        self.temperature = temperature
        self.max_retries = max_retries
        if load_4bit is None:
            load_4bit = os.environ.get("QWEN_4BIT", "").lower() in ("1", "true", "yes")
        self.load_4bit = load_4bit
        if not (self.model_path or self.endpoint):
            raise ValueError(
                "QwenVerbalizer needs QWEN_MODEL_PATH (transformers) or QWEN_ENDPOINT "
                "(OpenAI-compatible server). Run it as a HAL batch job; keep --teacher "
                "mock for local/CI."
            )
        self._tok = None
        self._model = None

    # -- prompt construction (pure) -------------------------------------------
    def build_messages(self, fact: FactRecord) -> List[dict]:
        """Master (system) + per-frame (user) chat messages. The grounding reuses the
        same qualitative phrasing the mock uses, so the teacher has faithful raw
        material and cannot drift off the fact."""
        grounding = [p for p in (_factor_phrase(ef) for ef in fact.causal_factors) if p]
        grounding_block = "\n".join(f"- {g}" for g in grounding) or "- (no extra grounding)"
        allowed = ", ".join(sorted(fact.mention_keys())) or "(no named entities)"
        user = (
            f"Decision: {fact.decision}\n"
            f"Grounding (use only these):\n{grounding_block}\n"
            f"Entities you may name: {allowed}\n"
            "Speak strictly in the first person (I, my). Never call yourself 'the car' or "
            "'the vehicle' — the guard reads those words as obstacle mentions.\n"
            "Write the reasoning:"
        )
        return [{"role": "system", "content": QWEN_SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    def verbalize(self, fact: FactRecord) -> str:
        messages = self.build_messages(fact)
        rejected = []   # (candidate, problems) — surfaced on failure, or a batch job on
                        # HAL dies with an opaque error and no way to see WHY
        for _ in range(self.max_retries):
            for cand in self._generate(messages, self.n_paraphrases):
                cand = cand.strip().strip('"')
                if not cand:
                    continue
                problems = find_hallucinations(cand, fact)
                if not problems:
                    return cand
                rejected.append((cand, problems))
        detail = "\n".join(f"  {p} <- {c!r}" for c, p in rejected[-4:])
        raise RuntimeError(
            f"QwenVerbalizer: no hallucination-free trace for decision="
            f"{fact.decision!r} after {self.max_retries} rounds. Last rejects:\n{detail}"
        )

    # -- backends (lazy; not run in Step 0) -----------------------------------
    def _generate(self, messages: List[dict], n: int) -> List[str]:
        if self.endpoint:
            return self._generate_http(messages, n)
        return self._generate_transformers(messages, n)

    def _generate_http(self, messages: List[dict], n: int) -> List[str]:  # pragma: no cover
        import json
        import urllib.request

        body = json.dumps({
            "model": os.environ.get("QWEN_MODEL_NAME", "Qwen2.5-32B-Instruct"),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 80,
            "n": n,
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return [c["message"]["content"] for c in data["choices"]]

    def _generate_transformers(self, messages: List[dict], n: int) -> List[str]:  # pragma: no cover
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_path)
            # fp16 only where it exists: CPU matmul has no Half kernels
            # ("addmm_impl_cpu_ not implemented for 'Half'"), so a CPU run (validation,
            # CI) needs fp32 while GPU runs keep fp16.
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            kwargs = dict(device_map="auto", torch_dtype=dtype)
            if self.load_4bit:  # fits Qwen2.5-32B on a single 40 GB A100
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok([prompt] * n, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs, do_sample=True, temperature=self.temperature,
            top_p=0.9, max_new_tokens=80)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self._tok.batch_decode(gen, skip_special_tokens=True)


def get_verbalizer(name: str, **kwargs) -> Verbalizer:
    if name == "mock":
        return MockVerbalizer()
    if name == "qwen":
        return QwenVerbalizer(**kwargs)
    raise ValueError(f"unknown teacher {name!r}; expected 'mock' or 'qwen'")
