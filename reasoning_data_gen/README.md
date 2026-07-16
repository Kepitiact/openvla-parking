# reasoning_data_gen — reasoning + counterfactual data generator (Step 0)

Generates a natural-language **reasoning trace** that the VLA emits *before* its
trajectory, plus matched **counterfactual pairs**, grounded only in what the
perceiver can see. This is Step 0 of the reasoning-VLA arc: it runs today against
ground-truth scene data as a stand-in for UniAD, and is built so the input source
swaps GT→UniAD later by changing **one adapter** (`SceneRecord.from_uniad`).

## Design invariants (do not break)

- **Perception grounding, not GT grounding.** The reasoner sees only a
  `SceneRecord` (detections + ego + command + past-2s history). No A* path, no
  privileged obstacle list, no future — the same object a real perceiver would get.
- **Causal locality.** Only current frame + past 2 s + ego state + command. The
  future trajectory is a separate *output label*; it lives on `Frame`, never on
  `SceneRecord`, and is read only by the counterfactual synth and `action_fidelity`.
- **No hallucination.** Fact extraction is pure rules (deterministic). The teacher
  only *verbalizes* a `FactRecord`; any sentence naming an entity/decision outside
  the fact is rejected (`verbalizer.find_hallucinations`).
- **Versioned, never overwritten.** `run_generate` refuses a non-empty `--out`.

## Layout

| file | role |
|------|------|
| `schema.py` | tokens, `DECISION_SET`, `EntityRef`, `FactRecord`, `render_assistant_turn`, `format_trajectory`, mention extractors |
| `scene_record.py` | `SceneRecord` (perceivable scene), `DetectedObject`, `Frame`; `from_gt` + `from_uniad` (both real, share `_assemble`) |
| `fact_extractor.py` | deterministic rules + `Thresholds` → `FactRecord` |
| `verbalizer.py` | `MockVerbalizer` (default), `QwenVerbalizer` (HAL, behind a flag) |
| `reconcile.py` | perception-vs-GT status: `ok` / `perception_limited` / `contradiction` |
| `counterfactual.py` | `make_stop_injection` → decel-to-stop pair + synthetic-token spec |
| `validators.py` | `entity_fidelity`, `action_fidelity`, `causal_locality`, `schema_conformance` (reused as Step-8 gates) |
| `run_generate.py` | CLI + `generate()` → versioned `traces.jsonl` + `counterfactuals.jsonl` + `manifest.json` |
| `tests/` | `test_dry_run.py` + `fixtures/mini_infos.pkl` (20 frames of episode_0003) |

## Run the dry test

```bash
# 20-frame GT-placeholder end-to-end, CPU only, no network, no GPU:
python -m reasoning_data_gen.run_generate \
    --infos data_carla/processed/parking_infos_train.pkl \
    --source gt --teacher mock --limit 20 \
    --out data_carla/processed/reasoning/v0_dryrun/

pytest reasoning_data_gen/tests/ -q
```

## UniAD source (real perception)

`--source uniad` builds the scene from decoded UniAD detections instead of GT — the
*only* thing that changes when moving off the placeholder. `SceneRecord.from_uniad`
consumes the per-frame `.pth` that `drivevla/extract_uniad_features.py` writes
(`result_track.detections = boxes[N,9] (ego x,y,z,w,l,h,yaw,vx,vy) + scores + labels`,
same ego frame as `gt_boxes`); ego state / mission command / slot still come from the
info (the car always knows those — only *objects* are perceived). Low-confidence
boxes (`--uniad-score-thr`, default 0.3) are dropped. For every frame, `reconcile`
compares the UniAD-derived fact against the GT-derived fact and tags it
`ok` / `perception_limited` / `contradiction` so the trainer can drop the bad ones;
the manifest reports the histogram + `uniad.missing_pth_frames`.

Two-step HAL flow (see the sbatch headers):

```bash
# 1. GPU: decode UniAD -> per-frame .pth (existing script, with the UniAD weights, e.g. epoch4)
#    drivevla/extract_uniad_features.py  ->  <staging>/.../uniad_features/<token>.pth
# 2. CPU: sanity-check the conversion on ~10 episodes with the mock teacher
sbatch scripts/sbatch/hal_reason_uniad_convert.sbatch \
  --infos <staging>/.../parking_infos_val.pkl \
  --uniad-features-dir <staging>/.../uniad_features \
  --out <staging>/.../reasoning/v0_uniad_convert --limit 400
# 3. GPU: the real UniAD-grounded Qwen pass
sbatch scripts/sbatch/hal_reason_qwen.sbatch --source uniad \
  --uniad-features-dir <staging>/.../uniad_features --infos ... --out ...
```

The decoded-detection format is identical for epoch4 and the final model, so step 2
validates the whole path before any Qwen GPU time is spent.

## Forward-compatibility hooks (for the later steps)

- **Step 5** reads `assistant_turn` from each record. `render_assistant_turn`
  emits exactly `{REASON_START}{trace}{REASON_END}{TRAJ_START}{traj}{TRAJ_END}`,
  and `format_trajectory` reproduces `build_llava_conversation`'s trajectory string
  byte-for-byte, so a reasoning record is a drop-in for the plain-traj record.
- **Step 8** reuses `validators.py` verbatim: the four functions take
  `(trace, fact)`, `(decision, traj)`, `(fact, frame)`, `(record)` — callable at
  eval time, not just at gen time.
- **Step 3** is `SceneRecord.from_uniad` (now implemented + tested against the
  `extract_uniad_features` schema; validate on HAL with real epoch4 outputs). The
  counterfactual `synthetic_token_spec` is the descriptor Step 2/3 injects as a
  matching track token, and `pair_id` links factual↔counterfactual for the
  swap/disable gates.

## Teacher (Qwen2.5-32B) — the GT-grounded pass on HAL

`MockVerbalizer` is the default (deterministic, local, tests/CI). The real
`QwenVerbalizer` runs Qwen2.5-32B-Instruct as a HAL A100 batch job. **The wording
changes; the content does not** — Qwen only rewords the pre-computed `FactRecord`
(same decision, same causal factors), and every candidate passes the hallucination
guard (`find_hallucinations`) with regeneration on any violation. So anything you
want the language to say must live in the fact first (that is why the slot factor
carries a qualitative bearing — see below).

**Two-level prompt** (`QwenVerbalizer.build_messages`):
- **master / system prompt** (`QWEN_SYSTEM_PROMPT`, constant): the role + hard rules
  (first person, 1–2 sentences, justify only with the grounding, qualitative
  directions never degrees, name nothing outside the fact).
- **mini / user prompt** (per frame): the fact — `Decision:` + grounding lines
  (reusing the same qualitative phrasing the mock uses) + the allowed-entity
  whitelist.

Config comes from CLI/env so it drops into an sbatch wrapper unchanged:
`QWEN_MODEL_PATH` (transformers backend, `device_map=auto` shards the 32B across
GPUs) or `QWEN_ENDPOINT` (OpenAI-compatible server), `QWEN_4BIT=1` (fits the 32B on
one 40 GB A100), `QWEN_BATCH_SIZE`. Run it with:

```bash
# fit-check on ztest (2 GPUs, 45 min); outputs land on /staging (NOT the workspace quota)
sbatch scripts/sbatch/hal_reason_qwen.sbatch \
  --infos <staging>/data_carla/processed/parking_infos_train.pkl \
  --out   <staging>/data_carla/processed/reasoning/v1_qwen_gt \
  --limit 50
```

The SIF (`openvla.sif`, transformers 4.40.0.dev0) loads `Qwen2ForCausalLM` and runs
Qwen2.5 out of the box; the only real constraint is memory (32B fp16 ≈ 64 GB > 40 GB
→ multi-GPU or 4-bit). Swapping `--source uniad` later reuses this exact teacher.

## Perception factors — making perception causally load-bearing

Before this, the reasoning was effectively `f(ego_state, command)`: perception's only
footprint was slot occupancy (2/371 frames), so fixing a badly-broken perception model
changed the reasoning in ~0 frames. A VLA trained on that correctly learns to **ignore
the perception tokens**, and the Step-8 causality gates would fail by construction.

The fact extractor now cites the perceived objects that **actually constrain the
maneuver** (`schema.ROLE_*`), capped at the 1–3 that matter — listing every detected
car would be a subtler form of reasoning theatre:

| role | fires on | why it's load-bearing |
|------|----------|------------------------|
| `flank_left` / `flank_right` | `reverse`, `complete_park` | the cars bounding the bay are the physical walls of the gap being reversed into |
| `front` | `approach`, `shift_gear`, `align` | a car ahead caps the forward swing — **this is why the ego runs out of room and shifts to reverse** |
| `rear` | `reverse` | what limits backing up |
| `swept` | `align`, `creep` | an object near the arc being steered through |
| `in_path` | `stop_yield`, `creep` | an object in the immediate corridor |

**Perception does not gate the decision.** A detection can never flip the maneuver
label (a perception error must not turn `reverse` into `stop_yield`); it only supplies
the causal factors. The one exception is pre-existing and by design: `stop_yield` and
`creep` are *defined* by an in-path object.

Measured on the 371-frame epoch4 set (real UniAD perception):

| | before | after |
|---|---|---|
| frames citing ≥1 perceived object | **0%** | **56%** |
| trace changes when perception is ablated | **0%** | **56%** |
| trace changes when perception source changes (GT↔UniAD) | 0% | 12% |
| decision histogram | unchanged | unchanged *(by design)* |
| entity / causal / schema validators | 1.0 | 1.0 |

The 44% of frames citing no object genuinely have nothing constraining them (no
flanking car, nothing ahead) — that is honest, not a gap. `rear`/`swept`/`in_path`
fire 0× in this subset: the ego reverses into an *empty* bay (nothing behind), `swept`
de-dupes into `front`, and the ego never stops/crawls for an obstacle. They should be
re-checked on the full train set.

**Bay geometry.** Neighbour-bay centres are the slot centre ± `bay_width` along the
slot's across-axis; in ego `(right, forward)` coords the slot's right-unit is
`(cos dh, sin dh)` — verified to 1e-6 m against a global-frame computation. Both bays
are occupied in only ~7% of frames and one in ~59%, so flank factors handle 0/1/2.

**Causal-locality note (`planned_path`).** `swept_path_obstacle` receives the upcoming
waypoints, used **only to select which currently-perceived object to cite**. The cited
entity is always a current-frame detection, so nothing unperceivable enters the trace
and `causal_locality` still holds. The trace is an output *target* (like the trajectory
itself), so future-derived *selection* is supervision, not an input leak. Pass
`planned_path=None` to disable the factor entirely.

### Slot bearing (why the fact carries direction)

The slot factor is `slot(free|occupied)@(r,f)` — it carries the slot's ego-local
position so the trace can state a **qualitative bearing** ("the target slot is
clear, behind on my left") without ever emitting degrees. `bearing(right, forward)`
(schema.py) maps a position to ahead/behind + left/right; obstacles use the same
helper. Bearing words are spatial adverbs, not named entities, so they do not affect
`entity_fidelity` and are hallucination-safe.

---

## Calibration report

### Coordinate frame / units (assumptions made about `gt_boxes`)

`build_infos_pkl.py` documents `gt_boxes[N,7] = (x, y, z, w, l, h, yaw)` in the
**LIDAR == ego frame: x=forward, y=left, z=up** (nuScenes convention). The
trajectory/slot convention used everywhere else is **x=right, y=forward**. So the
adapter converts:

```
right   = -gt_boxes[:, 1]     # right = -left
forward =  gt_boxes[:, 0]
v_right = -gt_velocity[:, 1]
v_fwd   =  gt_velocity[:, 0]
yaw (object heading rel. to ego forward) kept as-is (gt_boxes[:, 6])
```

This matches `generate_cached_nuscenes_info.global_to_local_xy`, which returns
`[-left, forward]`. Verified against real data: forward-gear frames yield a future
trajectory with **forward increasing**; reverse-gear frames yield **forward
decreasing with the heading rotating** — the expected signs.

The ego future trajectory (the output label) is recomputed from the infos pose
chain (`ego2global_{translation,rotation}`) rather than the nuScenes DB, so Step 0
needs no DB and no network; it pads by repeating the last point at a scene boundary,
matching `collect_future_local`. Sample rate is **2 Hz**, so past-2s history = 4
frames and the 6-step horizon is 3 s.

### Decision thresholds and how they were calibrated

Calibrated against `parking_infos_train.pkl` (**N = 61,659** frames). Percentiles:

| quantity | p10 | p50 | p90 |
|----------|-----|-----|-----|
| `ego_speed` (m/s) | 0.05 | 0.98 | 1.52 |
| `|steer|` (normalized) | 0.00 | 0.36 | 1.00 |
| `dist_to_slot` (m) | 1.88 | 4.41 | 7.76 |
| `|dheading_to_slot|` (rad) | 0.00 | 0.45 | 1.58 |

`reverse` is set on 65.5% of frames; ~10% of frames have `ego_speed < 0.05`.

| threshold | value | rationale |
|-----------|-------|-----------|
| `speed_stop` | 0.10 m/s | ~p12 of speed; "stopped" for stop_yield/complete_park |
| `creep_speed` | 0.50 m/s | crawling band ceiling (below the moving p30) |
| `align_speed` | 0.60 m/s | "slow" ceiling for the heading-correction rule |
| `align_steer` | 0.50 | strong steer, above the p50 of 0.36 |
| `near_slot_dist` | 1.50 m | "at the slot" radius for complete_park |
| `align_err` | 0.20 rad (~11°) | heading tolerance for complete_park |
| `corridor_half_width` | 1.20 m | half a vehicle width — keeps the lot's many parked cars (48–52/frame) out of the in-path test |
| `corridor_len_stop` | 5.00 m | how far ahead/behind counts as "in the path" |
| `creep_clearance` | 4.00 m | in-path object within this while moving → creep |
| `slot_occupied_radius` | 1.20 m | a detection this close to the slot → occupied |

**Decision precedence** (first match wins): `shift_gear` (gear flip vs prev frame)
→ `stop_yield` (stopped + in-path object) → `complete_park` (stopped + near slot +
aligned) → `reverse` (reverse gear) → `creep` (forward, crawling, near in-path
object) → `align` (forward, slow, strong steer) → `approach` (forward default).

Resulting histogram on the full train set:
`reverse 56.2% · approach 25.7% · shift_gear 11.9% · align 3.8% · complete_park
2.3% · stop_yield 0.0% · creep 0.0%`.

`stop_yield`/`creep` are ~0% in GT because the GT ego never actually crashes or
crawls behind a blocker — those states are exercised through **counterfactual
injection** (and, later, through imperfect UniAD perception), which is the point.

### Validator pass rates — at scale (mock teacher)

Sampled across the full sets (train: 7,708 of 61,659 frames, stride 8; val: 2,364
of 9,454, stride 4), stable train↔val:

| validator | train | val |
|-----------|-------|-----|
| `entity_fidelity` | 1.0000 | 1.0000 |
| `causal_locality` | 1.0000 | 1.0000 |
| `schema_conformance` | 1.0000 | 1.0000 |
| `action_fidelity` | 0.946 | 0.940 |

Also at scale: **0** empty causal-factor frames, **0** missing slot poses, **0**
mock-teacher hallucinations, **72%** of frames are counterfactual-eligible.

`action_fidelity` is a **contradiction gate** (fails only when the trajectory
clearly does the opposite of the decision; stationary/ambiguous frames pass — a
paused ego does not contradict its gear). The residual ~5–6% are almost entirely
**phase-boundary frames of multi-point maneuvers**: the *last* frame of a reverse
run (gear = reverse, but the immediate future is the next forward run), verified by
`next_gear == 0` on the failures. At 2 Hz the 3-s (6-step) horizon is longer than a
single gear phase, so this tension is intrinsic to the data — exactly the frames
`reconcile`/soft-labeling should down-weight and Step 8 should treat carefully, not
a defect.

### Data shape that matters for later steps

Parking here is heavily **multi-point**: mean **~5.6 gear changes per episode**
(max 21), with only 87 isolated single-frame gear-flip toggles across all 1,308
episodes (negligible flag noise). This is why `shift_gear` is ~12% of frames and why
`action_fidelity` has an intrinsic phase-boundary residual. Episodes span 14–147
frames (median 36).

## Data-version log

| version | source | teacher | notes |
|---------|--------|---------|-------|
| `v0_dryrun` | gt | mock | acceptance dry run: 20 frames, CPU/no-network |

(Append a row per generated version. `run_generate` refuses to overwrite an
existing version dir.)
