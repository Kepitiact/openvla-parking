# Command Reference

Run everything from the repo root unless noted:
```bash
cd ~/projects/openvla_nuscenes && source .venv/bin/activate
```

Pipeline at a glance:
```
raw episodes → pkl → NuScenes DB → ego cache → conversations
            → UniAD feature cache → LoRA train → merge → inference → L2 / plots
```

---

## 1. Prepare a dataset (after collecting/adding episodes)

```bash
# 1. Build the infos pkl from raw episodes  (lives in the data-gen repo)
cd ~/projects/parking_data_gen && source venv/bin/activate
python scripts/build_infos_pkl.py --raw_dir ~/projects/openvla_nuscenes/data_carla/raw \
  --out ~/projects/openvla_nuscenes/data_carla/processed/parking_infos_temporal.pkl

cd ~/projects/openvla_nuscenes && source .venv/bin/activate
# 2. NuScenes DB tables (also rewrites camera paths to ABSOLUTE so they never break)
python scripts/build_carla_nusc_tables.py
# 3. Ego-state cache (speeds, history, 4-element command incl. reverse)
python scripts/generate_cached_nuscenes_info.py
# 4. Conversations (per-frame token references; content filled at train time)
python scripts/build_carla_conversations.py
```
Steps 2–4 default all paths via `scripts/paths.py` — no arguments needed.

---

## 2. Fine-tune (extract → train → merge)

```bash
cd ~/projects/openvla_nuscenes/OpenDriveVLA

# A. Pre-compute UniAD features once (~12–20h; slim ~1.2MB/frame, ~55GB total).
#    Also writes each feature path back into carla_conversations.json.
bash scripts/extract_carla_features.sh

# B. LoRA fine-tune (UniAD + Qwen2 base frozen). 3 epochs ≈ ~20h on the 8GB GPU.
bash scripts/train_carla_parking.sh ../checkpoints/OpenDriveVLA-0.5B 3

# C. Merge LoRA → standalone model (CPU, ~1–2 min). Required before inference.
python drivevla/merge_lora.py \
  --base ../checkpoints/OpenDriveVLA-0.5B \
  --lora ../checkpoints/OpenDriveVLA-0.5B-carla/epoch_003 \
  --out  ../checkpoints/OpenDriveVLA-0.5B-carla/merged
```

**What trains:** LoRA on Qwen2 attention (q,k,v,o, rank 64) + `mm_projector_*` →
**~11.75M trainable / 605M total**. UniAD and the Qwen2 base stay frozen.
**VRAM:** ~5GB peak (the train script sets `expandable_segments` + unbuffered logging;
trainable params kept fp32 so the GradScaler works). Saves a checkpoint per epoch.
**Sequential GPU only:** extraction (~3.7GB) and training (~2.5GB) can't run together on 8GB.

---

## 3. Inference + evaluation

```bash
cd ~/projects/openvla_nuscenes/OpenDriveVLA

# Full inference with the fine-tuned model (~20h for all 49,620 frames).
nohup bash scripts/eval_carla_parking.sh ../checkpoints/OpenDriveVLA-0.5B-carla/merged \
  > /tmp/eval_full.log 2>&1 &
tail -f /tmp/eval_full.log            # progress

# Scenario-split L2 (reverse vs forward), after it finishes:
cd ~/projects/openvla_nuscenes
python scripts/eval_carla_predictions.py \
  --plan-conv OpenDriveVLA/output/merged/<TIMESTAMP>/results/plan_conv.json
```

**Important:**
- The output dir is named after the **checkpoint** you run: merged model → `output/merged/...`; base model → `output/OpenDriveVLA-0.5B/...`.
- A **completed** run writes `plan_conv.json`; a **stopped** run only has `plan_conv_rank0.json`.
- **Always evaluate with the full** `carla_conversations.json` (the script's default). Passing a *subset* file does NOT limit inference — the DB drives iteration and frames outside the subset get a wrong default prompt → garbage. To truly limit frames you must subset the `v1.0-carla` DB.

---

## 4. Visualize

```bash
cd ~/projects/openvla_nuscenes
PRED=OpenDriveVLA/output/merged/<TIMESTAMP>/results/plan_conv.json   # or plan_conv_rank0.json

# One frame (red=pred, green=GT, blue=history) → outputs/sample_inspection/<token>/
python scripts/inspect_opendrivevla_sample.py --predictions "$PRED" --sample-token episode_0000_f0008
# add --print-prompt to see exactly what the model was told

# A whole episode → outputs/episode_views/<episode>/
#   episode_overview.png  (actual path + per-step predictions, top-down)
#   episode_filmstrip.png (front camera + GT/pred plot per frame)
python scripts/visualize_episode.py --episode episode_0000 --predictions "$PRED"
```

---

## 5. Handy ops

```bash
# latest inference run:
ls -lt OpenDriveVLA/output/*/*/results/ 2>/dev/null | head

# free the GPU safely (kills the GPU compute process by PID; avoids pkill -f self-matching your own shell):
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# reclaim space from old bloated feature files (keeps only what training reads):
python scripts/slim_uniad_features.py
```

---

## File map

| Path | What |
|---|---|
| `data_carla/raw/episode_XXXX/` | raw episode: 6 images + `poses.json` + `meta.json` |
| `data_carla/processed/parking_infos_temporal.pkl` | main dataset (1 record/frame) |
| `data_carla/processed/cached_parking_info.pkl` | ego-state cache (speeds, history, command) |
| `data_carla/processed/carla_conversations.json` | per-frame token references (+ `uniad_pth`) |
| `data_carla/processed/uniad_features/<token>.pth` | slim cached UniAD features (training) |
| `data/nuscenes/v1.0-carla/` | NuScenes-format DB tables |
| `checkpoints/OpenDriveVLA-0.5B` | base model |
| `checkpoints/OpenDriveVLA-0.5B-carla/{epoch_NNN,merged}` | LoRA adapters + merged model |
| `scripts/paths.py` | single source of truth for all paths |
| `scripts/*.py` | data prep, eval, visualization (run from repo root) |
| `OpenDriveVLA/drivevla/{extract_uniad_features,train_drivevla,merge_lora}.py` | model entry points |
| `OpenDriveVLA/scripts/*.sh` | launchers |

---

## Architecture: "brain" vs pipeline

`OpenDriveVLA/` is the model (upstream code). The few scripts inside it
(`drivevla/extract_uniad_features.py`, `train_drivevla.py`, `merge_lora.py`) must
live there because they import its internal `llava` / `projects` packages — they
**self-bootstrap** those paths + an nvcc shim, so they run standalone.

Everything else — data prep, caching, conversations, evaluation, visualization —
lives in `scripts/` and imports `scripts/paths.py`.

**While a run is in progress:** don't modify the pkl, cache, `carla_conversations.json`,
the `v1.0-carla` DB, or the active `output/.../` dir. Editing the Python/shell
sources is fine — changes only take effect on the next run.

---

## Mission-goal (command) scheme

The "mission goal" the model sees is **generated in code**, not stored in any doc. The
**current scheme is maneuver-level** (a per-episode goal, not a per-frame primitive):

```
Mission goal: reverse-perpendicular park, right side, into slot at (1.19,-4.53,0.62)
```

- `maneuver_type` + `side` come from the episode (carried in the pkl as `maneuver_type`,
  `side`, `target_slot`), constant across the episode.
- The triple `(right, forward, dheading)` is `slot_local` — the target slot in the
  **current** ego frame (x=right, y=forward), recomputed per frame by `slot_to_local`
  in `scripts/generate_cached_nuscenes_info.py`. `build_llava_conversation` renders it.
- **Legacy fallback:** frames with no `maneuver_type` fall back to the old per-frame
  `[right, left, forward, reverse]` vocabulary (`infer_future_command`).

**Why this scheme:** the old per-frame vocabulary couldn't tell a low-speed parking
*approach* from highway *cruising* — both became "keep forward" — so the model fell back
on its cruise prior (wide, slow turns). Telling it explicitly that it is *parking* into a
specific slot fixed that "forward problem." Measured on the training set, @3s L2 dropped
from **forward 0.62 / reverse 0.31 m** (per-frame) to **forward 0.48 / reverse 0.15 m**
(maneuver+slot).

**Inference-time contract (closed-loop harness):** any live runner MUST rebuild this exact
string — `maneuver_type`+`side` from the scenario, `slot_local` recomputed every step via
the same `slot_to_local` transform. Feeding the old "keep forward"/"reverse" prompt is
out-of-distribution and degrades the model. Source of truth: `slot_to_local` +
`build_llava_conversation.generate_user_message`.

---

## Data-collection rate

Current CARLA data is **2 Hz** (every 15th frame at 30 Hz sim) → 0.5 s/frame,
6 waypoints = 3 s horizon. For **real-world recording**, target **10 Hz** (record
fine, downsample later — you can't upsample). See the data-collection spec (in Teams).
