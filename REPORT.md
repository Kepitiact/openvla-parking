# HAL Readiness — Change Report

Scope delivered: correctness landmines, distributed + resumable VLA training,
DAgger pipeline hardening, cleanup, and orchestration. **Out of scope (untouched):**
live-UniAD-in-VLA training path, Dockerfile/Apptainer, UniAD perception training/data.

The existing single-GPU LoRA path is preserved as the default; everything new is
additive/opt-in.

## A. Correctness landmines (fixed)

| ID | Fix | File |
|---|---|---|
| A1 | Removed the LRZ `rsync` fallback; missing feature `.pth` now raises a clear `FileNotFoundError` telling the user to re-run extraction. | `OpenDriveVLA/drivevla/data_utils/nuscenes_llava_dataset.py` |
| A2 | Hardcoded `/home/s0002438` paths replaced with `CARLA_DATA_ROOT` / `NUSC_DATA_ROOT` env vars (repo-root fallback), matching the clean scripts. | `OpenDriveVLA/projects/configs/stage1_track_map/carla_parking.py` |
| A3 | Bare `except:` → `except KeyError:` in the reorder fallback. | `nuscenes_llava_dataset.py` |
| A4 | Pre-train check now verifies each `uniad_pth` **file exists**, not just the field. | `train_drivevla.py` |
| A5 | `model.train()` at the start of each epoch. | `train_drivevla.py` |
| A6 | `merge_lora.py` errors clearly when `--base` mismatches the adapter's `base_model_name_or_path`. | `merge_lora.py` |

## B. HAL training readiness

- **B1 Distributed** — `train_drivevla.py` runs multi-GPU under `torchrun` via HuggingFace
  Accelerate (`--distributed`, auto-enabled when `WORLD_SIZE>1`). Single-GPU hand-rolled
  path stays the default.
- **B2 Resume** — full state (model, optimizer, scheduler, GradScaler, RNG, epoch,
  global_step, batch-in-epoch) saved per epoch and every `--save-steps` steps, behind
  `--resume`. **Mid-epoch resume** replays the same seeded order and skips processed
  batches. Writes are **atomic** (temp-then-rename) so a kill during a save can't corrupt
  the last good checkpoint.
- **B3 Determinism** — manual torch/numpy/python seeding, seeded dataloader
  (`worker_init_fn` + generator), explicit per-epoch reshuffle seed.
- **B4 Config** — `--num-workers`, `--batch-size`, `--grad-accum`, `--save-steps` exposed;
  wrapper env knobs (`NUM_WORKERS` default 4, etc.). NOTE: the UniAD collator asserts
  micro-batch == 1, so effective batch scales via grad-accum + multi-GPU, not micro-batch.
- **B5 Logging** — leveled logger writing to `checkpoints/<run>/train.log` + stdout;
  print-based training logs replaced.
- **B6 bf16** — precision handed to Accelerate (`mixed_precision=bf16/fp16`) on the
  distributed path; hand-rolled fp16+GradScaler retained only for the single-GPU default.
- **B7 Trainable groups** — declarative `--trainable-groups` (projectors/uniad/heads/llm)
  replaces the hardcoded `mm_projector` substring unfreeze. Default `projectors`
  reproduces prior behaviour.
- **B8 Reproducibility** — `requirements.lock.txt` (frozen `pip freeze`) + `ENV_SETUP.md`
  build recipe for the painful stack (mmcv-full 1.7.2, mmdet3d 1.0.0rc6, nvcc shim). The
  nvcc shim is now **conditional** (only when a real nvcc is absent) via `_bootstrap.py`.

## C. DAgger pipeline hardening

- **C1** `merge_dagger.py` merges are now **set differences** (idempotent; survive fresh
  and cumulative round-N packages without duplicating/clobbering).
- **C2** New idempotent `scripts/ingest_dagger.py`: merges cached+infos, registers DAgger
  tokens in the nuScenes DB (rebuild), emits a dagger-only infos file, and **normalizes
  the `v1.0-carla-dagger` → `v1.0-carla`** version string that broke extraction before.
- **C3** New `scripts/build_conversation_manifest.py` + `CONV` env in the wrappers wire the
  dataset's native yaml manifest so each round **adds** `dagger_r0N_conversations.json`
  instead of rewriting the base.
- **C4** `ingest_dagger.py` writes **round-aware backups** (`dagger_rounds/round_0N/backup/`)
  and a **per-round token manifest** (`round_0N_tokens.json`).

## D. Cleanup

- **D1** Deleted (with your approval) the `*.bak_predagger/preego/premaneuver` backups,
  `carla_conversations_revcheck.json`, and the `v1.0-carla.bak_predagger` DB snapshot
  (~640 MB). Kept the DAgger source package (`dagger_*.pkl/json`).
- **D2** `.gitignore` generalized: track only LoRA adapter files for any run
  (`…-carla`, `…-carla-dagger1`, …); exclude full weights, merged models, `train_state.pt`,
  `last_state/`.
- **D3** Duplicated nvcc-shim/sys.path bootstrap factored into
  `OpenDriveVLA/drivevla/_bootstrap.py`, imported by train/extract/merge_lora.
- **D4** `frames_upbound=32` annotated as an unused nuScenes multi-frame leftover for
  single-frame CARLA (train + extract).

## E. Orchestration

- `build_dataset.sh` — infos → DB → ego cache → conversations (stage-0 raw→infos entry
  point documented; it lives in the `parking_data_gen` harness).
- `run_pipeline.sh` — build → features → train → merge.
- `run_dagger_round.sh` — ingest → dagger-only extraction → per-round conversations →
  manifest → warm-start train → merge.

## Verified

- `grep -ri "ge86wob2\|lrz.de" OpenDriveVLA` → nothing.
- No hardcoded `/home/s0002438` in active configs/scripts.
- Single-GPU training runs and logs to `checkpoints/<run>/train.log`; loss decreases.
- **Kill mid-epoch + `--resume`** continues from the saved step (global_step 45, skip 362
  batches), not from scratch — verified on both the single-GPU and Accelerate paths.
- `merge_dagger.py` / `ingest_dagger.py` re-runs are no-ops (idempotent); version
  normalization observed (`v1.0-carla-dagger` → `v1.0-carla`).
- `merge_lora.py` base-mismatch guard errors clearly before loading.
- All shell scripts pass `bash -n`; all edited Python compiles with no errors.

## Not verified / remaining

- **True multi-GPU** (`torchrun --nproc_per_node=2`): this host has **1 GPU**, so only the
  single-process Accelerate path was exercised. The code is torchrun-ready
  (`NPROC>1` → `torchrun … --distributed`); validate on a multi-GPU HAL node.
- The `requirements.lock.txt` mmdet3d line records a machine-specific editable path —
  reinstall `-e OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6` (noted in `ENV_SETUP.md`).

## For the container agent

- Bake the painful stack per `ENV_SETUP.md` (torch 2.1.2+cu121, mmcv-full 1.7.2,
  mmdet3d 1.0.0rc6 editable). With a real CUDA toolkit in the image the nvcc shim is a
  no-op (already conditional).
- Mount persistent storage for `checkpoints/<run>/` (train.log + `last_state/` /
  `train_state.pt`) so `--resume` survives preemption.
- Slurm launch: `NPROC=<gpus> RESUME=1 bash OpenDriveVLA/scripts/train_carla_parking.sh`.
