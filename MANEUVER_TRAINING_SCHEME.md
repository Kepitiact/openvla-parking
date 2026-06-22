# Maneuver-Level Training Scheme

**Status (2026-06-22): IMPLEMENTED + trained for `reverse_perpendicular`.** Sections 3–4
are done; the model is retrained on the maneuver+slot command. Training-set L2 @3s improved
from **forward 0.62 / reverse 0.31 m** (per-frame command) to **forward 0.48 / reverse 0.15 m**
(maneuver+slot). Still to do: fill Section 1 (which other methods to generate) and add them.

Move the model's command from **per-frame primitives** (keep forward / reverse /
turn-left/right) to a **maneuver-level mission goal + target slot** — matching how
production parking systems condition (slot + maneuver type; the forward / reverse /
gear-change phases *emerge in the trajectory output*, they are not per-frame labels).

This fixes the "forward problem": the model is told it is *parking* (not cruising),
so it stops falling back on the highway-cruise prior.

---

## 1. Parking methods to support  — **[FILL IN]**

List the maneuvers you want to generate and train on. (Current dataset =
`reverse_perpendicular` only, Town04_Opt.)

| Method id               | Description                              | Entry    | Sides | Generate? |
|-------------------------|------------------------------------------|----------|-------|-----------|
| `reverse_perpendicular` | back into a 90° bay                      | reverse  | L / R | ✅ have   |
| `forward_perpendicular` | drive forward into a 90° bay             | forward  | L / R | ?         |
| `parallel`              | parallel park at a curb                  | reverse  | L / R | ?         |
| `angled`                | 45–60° bay                               | fwd/rev  | L / R | ?         |
| `multi_point` (K-turn)  | tight bay needing >1 gear change         | mixed    | L / R | ?         |
| _(add your own)_        |                                          |          |       |           |

→ Fill the **Generate?** column; that drives what the data generator produces.

---

## 2. New data fields the GENERATOR must record (per episode)

In **`parking_data_gen`** (collection) → written into
`parking_infos_temporal.pkl` / episode meta:

- `maneuver_type` — one of the method ids in Section 1
- `side` — `left` | `right`
- `target_slot` — 4-corner polygon **and** target pose `(x, y, heading)`
- *(optional)* `gear_change_count`

These replace reliance on the per-frame `reverse` flag for building the command.

---

## 3. Command derivation change (this repo)

`scripts/generate_cached_nuscenes_info.py` → `infer_future_command`:
- **OLD:** per-frame `[right, left, forward, reverse]` from `|lateral| ≥ 2 m` + the reverse flag.
- **NEW:** emit the episode's `maneuver_type` + `side` + `target_slot` (constant across the episode).

---

## 4. Prompt format change (this repo / OpenDriveVLA)

`OpenDriveVLA/drivevla/data_utils/build_llava_conversation.py`:
- **OLD:** `Mission goal: keep forward`
- **NEW:** `Mission goal: reverse-perpendicular park, left side, into slot at (x, y, heading)`

---

## 5. Output is UNCHANGED

The training target stays the expert trajectory (`gt_ego_fut_trajs`). Forward,
reverse, and gear changes are **implicit in the trajectory's direction** — not
separate labels. A multi-point (K-turn) maneuver is just a trajectory that reverses
direction one or more times (needs a long-enough horizon, e.g. 6 s, to capture it).

---

## 6. How to apply

1. Decide Section 1 **before** large-scale data collection.
2. Add the fields (Section 2) in the data generator (`parking_data_gen`).
3. Update `infer_future_command` (Section 3) + `build_llava_conversation` (Section 4).
4. Regenerate the ego cache + conversations, then **retrain**.
   - **No re-extraction needed** — UniAD features are unchanged; only the command/prompt changes.

> Note: changing the command is a *prompt/label* change. It does not alter the model
> architecture or require abandoning OpenDriveVLA — the perception + LLM are reused.

---

## 7. Inference-time prompt contract (closed-loop harness)

The **closed-loop harness must rebuild the exact same mission-goal string live**, every
control step — not the old per-frame command. Getting this wrong (e.g. feeding "keep
forward") is out-of-distribution and silently degrades the model.

Exact format (from `build_llava_conversation.generate_user_message`):
```
Mission goal: reverse-perpendicular park, right side, into slot at (R,F,H)
```
- `maneuver_type` (→ `maneuver_type.replace('_','-') + " park"`) and `side` are
  **episode-constant**, known when the scenario starts.
- `(R,F,H)` = `slot_local` = the target slot expressed in the **current** ego frame,
  **recomputed every step**:
  - `R,F = global_to_local_xy(slot_xy, ego_xyz, ego_rot)` — x=right, y=forward (same
    convention as the trajectory output and the controller).
  - `H = normalize(slot_yaw − ego_yaw)`.
  - Source of truth: `slot_to_local()` in `scripts/generate_cached_nuscenes_info.py`.
- The harness also rebuilds the rest of the user message live (UniAD perception tokens
  from the 6 cameras, ego-states, last-2s history) — the mission goal is the only part
  that changed vs the pre-maneuver plan, but it's the part most easily gotten wrong.
