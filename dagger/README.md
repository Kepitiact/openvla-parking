# DAgger — data-generation loop (QUARANTINED, currently BROKEN)

The DAgger round loop: run the trained VLA in CARLA, ingest the frames where it went
wrong, merge them into the training set, retrain. It is the bridge between this repo
and the CARLA data-generation repo.

**Nothing on the current critical path uses these.** They are parked here so they stop
cluttering the root and `scripts/`, and so the breakage below is written down instead of
being rediscovered later.

## Contents

| file | role |
|------|------|
| `run_pipeline.sh` | one-shot: extract features → train → merge LoRA → eval |
| `run_dagger_round.sh` | one DAgger round: ingest → extract → manifest → train → merge |
| `ingest_dagger.py` | pull a round's collected frames into the infos pkl |
| `merge_dagger.py` | merge a round's data into the training set |
| `build_conversation_manifest.py` | rebuild `carla_conversations.json` for the merged set |
| `REPORT.md` | dated "HAL readiness" change report (superseded by `HAL_RUNBOOK.md`) |

`build_dataset.sh` deliberately stayed at the repo root: it is NOT DAgger-specific
(it only chains `build_carla_nusc_tables.py` → `generate_cached_nuscenes_info.py` →
`build_carla_conversations.py`) and is still the stage-0 data-prep entry point.

## ⚠️ What is broken, and why

**1. The extraction contract changed.** `OpenDriveVLA/scripts/extract_carla_features.sh`
was refactored so that UniAD is built directly from `(config, checkpoint)` — no LLaVA
wrapper, no OpenDriveVLA-0.5B. It **ignores `$1`** and now **hard-exits** unless
`UNIAD_CKPT` is set, because the filename `uniad_base_track_map.pth` used to name two
different models and picking the wrong one silently reverted the detector to nuScenes
weights.

Both of these therefore fail immediately today:
```
run_pipeline.sh:37      bash scripts/extract_carla_features.sh "${BASE}"      # $1 ignored
run_dagger_round.sh:43  bash scripts/extract_carla_features.sh "${WARM_START}" # $1 ignored
```
**Fix:** drop the positional arg and pass the *trained UniAD* checkpoint explicitly:
```bash
UNIAD_CKPT=/abs/path/to/checkpoints/stage1_carla_full/epoch_6.pth \
  bash OpenDriveVLA/scripts/extract_carla_features.sh
```
Note `$BASE`/`$WARM_START` was the **VLA** (0.5B) checkpoint — a different thing from the
UniAD checkpoint. That conflation is part of why the bug happened.

**2. Paths moved.** These scripts were written to run from the repo root and refer to
`scripts/ingest_dagger.py`, `scripts/merge_dagger.py`,
`scripts/build_conversation_manifest.py`, and `bash build_dataset.sh`. Those are now
`dagger/*.py` and `../build_dataset.sh`.

**3. The VLA backbone is changing.** `run_dagger_round.sh` trains OpenDriveVLA-0.5B via
`scripts/train_carla_parking.sh` + `drivevla/merge_lora.py`. The VLA is moving to
Qwen2.5-3B-Instruct with new projectors, so the train/merge half will need rewriting
against the new model regardless.

## Before reusing these

Fix (1) and (2) at minimum, and re-point (3) at the new backbone. Also note the reasoning
layer now expects `result_track.detections` in the feature `.pth` — do not reintroduce any
step that strips it (the old `slim_uniad_features.py` did exactly that and was deleted).
