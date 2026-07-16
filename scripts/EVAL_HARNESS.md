# Eval harness (Step F)

Four measurements. Two are built and self-tested now; two need the trained model and are
specified here so they can be implemented + tested in one pass once it lands.

## Built + tested (run post-hoc on the inference output)

**1. Trajectory quality — `score_trajectory.py`**
ADE/FDE + FINAL-POSE (position *and* heading) + a "parked well" rate (<0.5 m, <10 deg).
Heading is what ADE/FDE miss: right spot, wrong heading = not parked. GT = the teacher
traces' `trajectory` field, keyed by token.

**2. Reasoning faithfulness — `score_reasoning_faithfulness.py`**
The teacher's validators, now on the STUDENT's reasoning vs the GT fact. Catches
hallucinated objects / a decision the trajectory does not take, plus distinct-n diversity
(template collapse).

Both read `inference_drivevla`'s output, which now keeps a `reasoning` field (split from the
trajectory by `split_reason_traj`).

## Pending (need the trained model) — the causal tests

These are the headline: do the reasoning WORDS drive the trajectory? Both are **text-level
interventions**, not hidden-state hooks. Hidden-state ablation on a gated model is
tautological (we blocked the direct path, so removing the reasoning positions removes the
only path). Intervening on the reasoning TEXT tests whether the *content* matters.

Both need one new inference mode: **teacher-force a chosen reasoning, then generate only the
trajectory.** Construct `... <reason_start>{chosen}<reason_end><traj_start>` and let the model
generate from there. The chosen reasoning is:

**3. Ablation** — `chosen = ""` (empty reasoning).
Run inference normally AND with empty reasoning; compare trajectories with
`score_trajectory.py`.
- gated model: trajectory should DEGRADE hard (reasoning was the channel).
- baseline (if run): trajectory barely moves (it read perception directly).
The gap is the load-bearing signal.

**4. Trace-corruption** — `chosen = another frame's reasoning` (a plausible but WRONG trace).
Does the trajectory follow the wrong reasoning? If yes, the words causally drive the action
-- the strongest evidence the reasoning is not decoration. If the trajectory ignores the
swapped text, the words are cosmetic (and RL-for-consistency, FUTURE_WORK 3e, is the fix).

### Implementation note (one focused change to inference_drivevla)
Add `--reasoning-mode {normal,ablate,corrupt}`. For ablate/corrupt, build the input with the
chosen reasoning already in place and set `max_new_tokens` to just the trajectory span
(generation starts after `<traj_start>`). The reasoning-selection logic (empty / mismatched
token) is pure and unit-testable; only the generate call is model-dependent, so implement and
test it together when the model exists.
