# Future work — deliberate deferrals

Things we consciously chose NOT to do for v1, with the reason and the trigger for revisiting.
This is not a wishlist: each item is a decision we made, so it should be re-decided with
evidence rather than rediscovered.

---

## Priority roadmap (read this first)

The order to act, keyed to what the v1 eval shows. When training + eval finish, start here.

**A. The moment v1 eval lands — required + decision-driven:**
1. **§3b inference-time gate** — build BEFORE publishing any causality number. NOT optional
   (train/infer mismatch corrupts the ablation result). Do regardless of the numbers.
2. **The gate A/B (COMMITTED):** train a second arm **gate-off** (all tokens + reasoning,
   the field-standard config) alongside v1's `gate=track`, both from the same neutral align.
   Test BOTH with ablation + **wrong-reasoning injection** (trace-corruption). The gate-off
   corruption result is the real "is reasoning *naturally* load-bearing?" measurement — on
   the gated model, ablation is partly tautological. Optionally add the strong end
   (`track,map,scene`) for the full spectrum.
   - gate-off breaks under corruption → reasoning load-bearing naturally (strong, field-standard).
   - gate-off unchanged → reasoning was decorative → Step A5 (RL) is justified.
3. if trajectory **L2 is poor** → **§6 numeric/regression trajectory head** (Alpamayo's
   approach). The single most likely architectural upgrade — text digits are a known-bad
   numeric code.
4. if traces are **memorised templates** → **§5 perception-QA auxiliary task**.
5. **CONDITIONAL on A2** — if gate-off proved decorative (reasoning not naturally used) →
   **§3e RL for reasoning-action consistency**, the way to make "all tokens + reasoning"
   load-bearing. Largest-effort item; gate it on the A2 evidence, do not pre-commit. This is
   the main v2 architecture direction (Alpamayo path).

**B. v2 — the biggest lever (larger effort; plan after A):**
6. **§1 richer data** (pedestrians / denser lots / moving vehicles) — TOP v2 priority; it is
   what makes the causality real (today `stop_yield`/`creep` are 0% of frames).
7. **§2 counterfactuals** — likely made obsolete by §1; only if we still need paired data.
8. **§3c weather** — pending the CARLA owner's answer on whether it affects the GT.
9. **§4 colour** — conditional on §1/§5.
10. **§8 DAgger rebuild** — when DAgger data arrives; route through the main pipeline, do not
    resurrect the 0.5B DAgger scripts.

**C. Infra / cleanup (post-v1, no dependency — do when convenient):**
11. **§3d data-layout consolidation** — one data root, kill the dual config path.
12. **§9 base-VLM swap** — a physical-AI-pretrained VLM (Alpamayo uses Cosmos-Reason); a
    research consideration, not urgent.
13. **§7 general-data mixing** — only if the model degrades on general prompts.

---

## 1. Richer data (the biggest lever)

**Pedestrian-populated episodes + a yielding controller.**
Today `stop_yield` and `creep` are **0%** of frames — the ego never stops for anything. It
routes around obstacles via A*, so no example in the entire dataset shows *perception
changing the trajectory while the ego state stays fixed*. That is why the causality gate is
data-limited — the workaround (§2 counterfactuals) is synthetic and inferior to real
yielding data.

Collect episodes where a pedestrian crosses and a controller **stops and waits**. Then:
- `stop_yield` becomes real, with a real trajectory (the car actually stops)
- the pedestrian is really there, so UniAD really detects it → **no synthetic perception
  token injection is needed at all** (this is strictly better than counterfactuals)
- the reasoning writes itself: *"a pedestrian is close, I will wait until they pass"*

Also worth adding: **denser lots** (both bays flanked — currently only ~7% of frames have a
car on BOTH sides), and **moving vehicles** inside the lot.

**Bonus:** more pedestrian data also fixes UniAD's weakest class (epoch_6 pedestrian recall
0.42, precision **0.59** — by far the worst of the three).

**Trigger:** this is the top priority for v2.

---

## 1b. A* planner: fix the LATERAL alignment (fold into the v2 collection)

**Measured** over all 1308 training episodes (end-of-episode ego pose vs target slot, in the
slot frame; slot 5.5 x 3.0 m, car ~4.5 x 1.8 => slack ~0.5 m along, ~0.6 m across):

| axis | result |
|---|---|
| heading | **perfect** — p95 1.4 deg, max 8 deg |
| longitudinal (depth) | **perfect** — p99 0.28 m, ZERO episodes exceed the 0.5 m slack |
| **lateral (sideways)** | **the failure** — p90 0.90 m; **24.5%** exceed 0.6 m slack, **9.9%** clearly outside the slot |

So the planner nails depth and angle and fails sideways — ~10% of episodes end up straddling
a neighbouring bay. Mechanically sensible: depth is "back up until deep enough", lateral is
set by where the reverse begins and the steering arc. **It is a targeted fix (approach /
turn-in geometry, or lateral tracking in the controller), not a planner rewrite.**

**Why not regenerate now:** v1 trains on this data already, and the v1 thesis (reasoning is
load-bearing) does not depend on perfect GT. Regenerating costs the full pipeline (collect ->
~18 h extract -> ~8 h teacher). **Fold the lateral fix into the v2 collection (§1)** so we
regenerate once, not twice.

**CRITICAL for reading v1's eval — the model cannot beat its teacher.** The GT itself scores
only **68.9%** "parked well" (<0.5 m, <10 deg). A model at ~65% is nearly MATCHING its
demonstrations, i.e. a success. Always report model quality **relative to GT quality**, never
against perfection, or a good model reads as a bad one.

---

## 2. Counterfactual training (deferred, and possibly obsolete)

`reasoning_data_gen/counterfactual.py` generates stop-injection pairs and a
`synthetic_token_spec`. They are NOT fed to training, because **nothing injects the
synthetic obstacle into the UniAD perception tokens**. Train on them as-is and the model
sees *identical perception* paired with a *stop* trajectory → it learns to brake at random.

**If §1 lands, this may never be needed** — real yielding data is strictly better than
synthetic injection. Only build the token-injection path if we still need paired
factual/counterfactual data for the Step-8 swap/disable gates.

---

## 3. Gate the map (and scene) streams

`REASON_GATE` currently defaults to `track` — only the decoded objects are forced through
the reasoning.

**Why map is not gated yet — now measured (seg loss curve, stage1 training logs):**
| stream | loss start -> end | verdict |
|---|---|---|
| detection (loss_cls) | 0.82 -> 0.027 | strong (matches 0.80 recall) |
| map / things-seg (lot geometry) | 1.82 -> 0.58 | LEARNED, but weak -- large residual vs detection |
| map / stuff-seg | 0.0343, flat | DEAD -- the CARLA "stuff" mask is hardcoded to zeros (`nuscenes_e2e_dataset.py:503`) |

So the map head is not dead: things-seg (lot geometry via `CarlaVectorMap`) dropped ~68%, so
it learned *something*. But (a) its residual is far higher than detection, so the geometry is
coarse, and (b) the reasoning does not verbalize map content, so gating map would force a
weak, un-relayed stream through the reasoning -- deleting info with nothing to carry it. Keep
map ungated for v1.

**Still open (needs a GPU slot):** the loss curve says "learned moderately" but not "how
good." Render predicted lane/lot geometry from the seg DECODER against
`lot_map_gt_Town04_Opt.json` (IoU) -- the saved features are query embeddings, not polygons,
so this needs a UniAD forward pass, not just the .pth files. **Then** decide on
`REASON_GATE=track,map`.

**Note on scene (important, and a correction to an earlier claim):** gating `scene` does NOT
destroy the camera information. The reasoning TOKENS keep full attention to perception; only
the TRAJECTORY tokens are blocked. So the path is
`perception -> reasoning tokens' hidden states -> trajectory`, and the trajectory loss
backprops through those states, actively pressuring them to carry visual information — even
if the emitted TEXT never mentions colours or curbs. ~40 reasoning tokens x 2048 dims is not
a tight bottleneck for ~110 perception tokens.

**So `REASON_GATE=track,map,scene` is the strong version of the experiment**: with all
perception gated, ablating the reasoning block *necessarily* changes the trajectory, and the
reasoning is provably the channel. Run it as an A/B against `track` and compare trajectory
L2 + ablation sensitivity.

---

## 3b. Reason gate at inference time (known limitation)

The gate is a TRAINING-time mechanism. At `generate()` time,
`prepare_inputs_labels_for_multimodal_uniad_vlm` runs once on the prompt — before any
`<traj_start>` has been generated — so `is_traj` is all-False and the mask degenerates to
plain causal: **decoding is not gated**. Newly generated trajectory tokens can attend to
the cached perception keys.

Why this is acceptable for v1: the weights were trained with the direct
perception→trajectory path masked, so the model never *learned* to read perception from
trajectory positions — the information flow it learned routes through the reasoning
states. But it is a train/infer attention mismatch, and the strict version (masking
per-step during decode, via a custom attention processor or per-step 4D masks over the KV
cache) should be built before the Step-8 causality numbers are published.

**Trigger:** before running the final Step-8 ablation suite.

## 3c. Weather / road-surface as a causal factor (ASK THE CARLA AGENT)

CARLA writes a balanced `weather` preset per episode (10 classes) and it never enters the
model. We deliberately keep it OUT of the causal trace, because **we have no evidence the
GT trajectory depends on it** — the A* planner and controller appear blind to weather, so
wet-vs-dry episodes would carry identical trajectories, and a "wet road, so drive carefully"
clause would train the model that reasoning can be decoupled from action (Alpamayo's
"superficial reasoning").

**Open question for the CARLA-agent side:** does the ground-truth controller actually change
behaviour on wet/low-friction surfaces (slower approach, longer stopping, gentler steer)? If
YES, then wet-road IS causal and, exactly like Alpamayo, it belongs in the trace — add a
grounded `manner` fact (e.g. `low_traction`) from the weather preset. If NO, it stays out.

**Trigger:** confirm with the CARLA data owner whether weather affects the GT maneuver.
Most relevant when we move to real driving data, where surface friction is unquestionably
causal. Until proven, do not add it.

## 3d. Consolidate the data layout (post-v1 cleanup)

Data is split across two roots today: infos + conversations + cached_parking_info in the
REPO (/workspaces), features + reasoning traces on STAGING. Worse, TWO code paths resolve
the infos differently -- `carla_parking.py` uses `CARLA_DATA_ROOT`, `carla_parking_stage1.py`
uses a getcwd-based `_REPO`. That duplication caused the smoke's third crash (CARLA_DATA_ROOT
pointed at staging where infos don't live).

**Correct structure:** repo = code + config; ALL data under one root on staging, one env var.
- prep writes infos/conversations/cached to staging alongside the features.
- single `CARLA_DATA_ROOT` (staging), used by every config -- delete the `_REPO`/getcwd path.
- one source of truth for "where is the data".

**Why not now:** it is a refactor, not a fix (each file move needs its own verification), and
the working setup (CARLA_DATA_ROOT -> repo) is fine for v1. Doing it mid-flight, right before
the 8xA100 run, risks a new path bug at the worst time. The quota is not forcing it (~5.5 GB
of infos vs a ~100 GB workspace).

**Trigger:** after v1 trains, as a deliberate cleanup pass.

## 3e. RL for reasoning-action consistency (post-v1, Alpamayo path)

Our gate is a TRAINING-TIME ARCHITECTURAL mechanism: block the trajectory's direct view of
perception so it must route through reasoning. Alpamayo (2511.00088) reaches the same goal
differently -- a trajectory decoder CONDITIONED on the reasoning, plus an RL stage that
rewards reasoning-action CONSISTENCY (punishes a trajectory that disagrees with its stated
reasoning). SFT elicits the reasoning; RL enforces that it is actually used.

**Why it matters for us:** the gate proves reasoning is a *channel* (ablate it -> trajectory
breaks), but it does not directly pressure the reasoning to be *semantically correct* -- the
gated positions could carry useful activations while the surface text drifts. An RL
consistency reward targets exactly that gap, and it survives an architecture change (it is a
training objective, not a mask). If v2 moves toward the Alpamayo architecture, the gate goes
away but this reward is the thing that replaces its guarantee.

**Trigger:** after the first model's eval. Decide based on whether the gated model's traces
are faithful (validators on student output) AND causally used (ablation). If faithful+used,
the gate sufficed for v1; if the text drifts from the trajectory, RL consistency is the
next lever. This is the main v2 architecture direction.

## 4. Colour / appearance in the reasoning

The model DOES see colour: `<SCENE>` is the 6-camera RGB backbone features
(`llava_arch.py:330`, `img_feat_2D [1,6,256,15,25]` → 90 tokens). But
`reasoning_data_gen` only reads the **decoded boxes** (class + geometry), so the fact record
has no colour and the validators cannot check a colour claim.

**Never put colour in a trace unless it is in the FACT record** — otherwise it is
hallucination by construction, regardless of what the model can see.

To do it properly:
1. pull colour from CARLA GT (`actors.json` → blueprint) **into the fact record**
2. train
3. **ablate the scene tokens**: if colour claims degrade, the model genuinely read them; if
   they do not, it is confabulating from priors

Open question: the binding problem. Each camera is pooled to **15 tokens**
(`adaptive_max_pool2d(3,5)`), and colour must be bound to a *specific track* across two
separate token streams. Plausible for a near, large car; dubious for a small/far one.

---

## 5. Perception-QA auxiliary task

A second task ("is anything to my left?", "how many vehicles are within 20 m?") answered from
perception. Forces the model to actually READ the perception tokens, and diversifies the
language (attacking template memorisation). The plumbing already exists (`qwen_qa` template,
`<question_start>/<answer_start>`, and a QA branch in `build_llava_conversation`).

**Restrict to perceivable attributes only** (position, class, count, occupancy). No colour
(until §4), no brand, no intent.

**Trigger:** only if the trained model's traces turn out to be memorised templates. Measure
first — trace diversity in the trained model's output — then decide.

---

## 6. Numeric trajectory output

Waypoints are currently emitted as TEXT digits, learned by next-token cross-entropy. Text is
a poor numeric code: `0.17` and `0.18` are unrelated tokens, so there is **no metric
gradient**, and one mis-sampled digit throws a waypoint metres off.

**Trigger:** only if the first eval shows bad trajectory L2 or frequent parse failures. Then
switch to a numeric/regression head (the Alpamayo approach), keeping the reasoning as text.
Do not pre-emptively rebuild the head.

**Not the half-measure (weighting the trajectory tokens higher in the CE):** that makes the
model care *more* about exact digits but does not fix the root cause — CE still has no
"closer is better" gradient (`0.2` vs `0.3` costs the same as `0.2` vs `0.9`). If L2 is bad
enough to act, go straight to the regression head, which is the actual fix. The gate couples
reasoning ↔ trajectory, so do not over-weight the trajectory to the point the reasoning
collapses into unreadable code — that breaks the mechanism the project exists to show.

---

## 7. General-data mixing

Mix ~35% general instruction data to prevent the output distribution collapsing onto
parking-speak.

**Downgraded from "necessary" to "a seatbelt"**: the base 3B is frozen, only LoRA moves, and
the embedding matrix is now restricted to the 22 NEW rows (a gradient hook zeroes the
151,665 pretrained rows). There is very little left that *can* forget.

**Trigger:** if the model degrades on general prompts, or if its reasoning collapses into a
handful of memorised strings.

---

## 8. DAgger data path — rebuild against the reasoning-VLA (do NOT fix the old scripts)

`dagger/` is a self-quarantined mini-pipeline built for the OLD 0.5B trajectory-only model
(`ingest → extract → manifest → train`). It is broken on two levels: (1) the documented 0.5B
breakage (extraction contract, moved paths, 0.5B train path), and (2) the bigger reasoning-VLA
gap — it has **no teacher/reasoning step at all** (produces trajectory-only conversations), it
emits conversations **without the `split` tags** the trainer now requires, and it trains via
the retired `train_carla_parking.sh`.

**The fix is NOT to repair those scripts.** The main pipeline is now general — it turns *any*
CARLA episodes into training data. So "bake in DAgger data" = run DAgger episodes through the
main pipeline (`extract_uniad_features` → `run_generate` teacher traces →
`build_carla_conversations --split train` → `train_drivevla finetune`, warm-started from v1),
then blend. The CARLA agent must produce the same infos schema (see the closed-loop handoff)
and keep `result_track.detections` in the feature `.pth`.

**The one real DAgger-specific decision (recipe, not plumbing):** whether failure frames get
extra weight / a dedicated round vs. plain blending — that is where the DAgger value (focus on
failures) actually lives.

**Trigger:** when DAgger data is collected (post-closed-loop testing).

---

## 9. Base VLM choice (physical-AI-pretrained backbone)

We use Qwen2.5-3B-Instruct — a **general text** model. Alpamayo uses **Cosmos-Reason**, a VLM
pretrained for Physical AI (spatial/embodied reasoning). A physical-AI-pretrained backbone
could reason about geometry/occupancy better out of the box, which is most of what the
parking traces are about.

**Why deferred:** it is a backbone swap (rebuild `init`, re-run align/finetune) for an
uncertain gain, and it is orthogonal to the v1 thesis (reasoning is load-bearing) — that
result holds regardless of backbone. Revisit only if the trained model's *spatial* reasoning
is weak (traces get bearings/geometry wrong) despite faithful grounding.

**Trigger:** v1 eval shows spatially-wrong reasoning, or at a scale-up where a stronger
backbone is warranted. Check whether an open physical-AI VLM fits our 3B compute budget.
