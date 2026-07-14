# Future work — deliberate deferrals

Things we consciously chose NOT to do for v1, with the reason and the trigger for revisiting.
This is not a wishlist: each item is a decision we made, so it should be re-decided with
evidence rather than rediscovered.

---

## 1. Richer data (the biggest lever)

**Pedestrian-populated episodes + a yielding controller.**
Today `stop_yield` and `creep` are **0%** of frames — the ego never stops for anything. It
routes around obstacles via A*, so no example in the entire dataset shows *perception
changing the trajectory while the ego state stays fixed*. That is why the causality gate is
data-limited (see §4).

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

**Why map is not gated yet:** the map head is **completely unverified**.
`scripts/uniad_stage1_metrics.py` scores **detection only** (the 0.802 recall is
cars/trucks/pedestrians). The CARLA "stuff" mask is hardcoded to zeros
(`nuscenes_e2e_dataset.py:503`), though real lot geometry *is* loaded as "things" via
`CarlaVectorMap`. **Map/seg quality has never been measured, not once.** Gating a stream we
cannot show is meaningful is not a real experiment.

**Do first:** measure the map head (check the seg loss curve; render predicted lane/lot
geometry against `lot_map_gt_Town04_Opt.json`). **Then** turn on `REASON_GATE=track,map`.

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

---

## 7. General-data mixing

Mix ~35% general instruction data to prevent the output distribution collapsing onto
parking-speak.

**Downgraded from "necessary" to "a seatbelt"**: the base 3B is frozen, only LoRA moves, and
the embedding matrix is now restricted to the 22 NEW rows (a gradient hook zeroes the
151,665 pretrained rows). There is very little left that *can* forget.

**Trigger:** if the model degrades on general prompts, or if its reasoning collapses into a
handful of memorised strings.
