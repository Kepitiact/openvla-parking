# Prompt for the HAL container / SIF agent

> Copy everything below the line into a fresh agent session (a Claude Code session
> is ideal, opened in the `openvla_nuscenes` repo). It is written so the agent
> proposes commands and **you run them** and paste back output — you do not need to
> give the agent SSH access. Fill in the two `<<...>>` placeholders first.

---

## Your role and goal

You are helping containerize the **`openvla_nuscenes`** repository (a UniAD /
OpenDriveVLA autonomous-driving model) so it runs on the **HAL Slurm cluster** at
Zenseact. The end goal is a training run, but **your job right now is only the
environment**: build a container image that reproduces this repo's exact Python/CUDA
stack, get it onto HAL, and prove it works by running a **1-step smoke test on a HAL
GPU**. No full training, no bulk data — a tiny 5-episode dataset is enough.

Do this now, in parallel with data collection happening elsewhere. If the env has
bugs (the mmcv / mmdet3d / deepspeed stack is where they hide), we want to find them
now, not when the full dataset is finally ready.

## How we work together (important)

- **You propose shell commands; I run them on my machine / HAL and paste the output
  back.** Keep commands small and individually verifiable. Do not assume you can SSH
  anywhere yourself.
- I can `ssh hal-login` (login node) and `ssh hal-data` (data node) from my machine —
  both already work passwordless. I may also open VS Code Remote-SSH into HAL, so I
  can run your commands in a terminal *on* HAL directly.
- **Before any slow or irreversible step** (building an image, transferring GBs,
  submitting a Slurm job), tell me the plan and expected cost first.
- I have limited cluster/SSH experience — explain what each non-obvious command does
  in one line.

## Placeholders to fill in

- `<<HAL_USER>>` = my HAL username (the `sXXXXXXX` in `ssh sXXXXXXX@hal-login01...`).
- `<<HAL_WORKSPACE>>` = `~/workspace` on HAL (a symlink to `/workspaces/<<HAL_USER>>`).

## Hard constraints

1. **The dependency stack is load-bearing and must be reproduced EXACTLY.** Other
   versions do not work. The authoritative recipe is in the repo:
   - `ENV_SETUP.md` — the build recipe for the painful stack.
   - `requirements.lock.txt` — a frozen `pip freeze`.
   The core pins: **Python 3.10, torch 2.1.2+cu121, mmcv-full 1.7.2, mmdet 2.28.x,
   mmsegmentation 0.30.x, mmdet3d 1.0.0rc6 (installed EDITABLE from
   `OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6`)**. Freezing these into the image
   is the entire point — the SIF must pin them so the env can never drift.
2. **Build approach**: a single Docker image built locally (Docker is available on my
   local machine), then converted to a Singularity/Apptainer **`.sif`** and moved to
   HAL. HAL uses Apptainer/Singularity, not Docker.
3. **No madame-web / run.py required** — I can submit my own Slurm jobs. (Verify this
   in Phase 0.)
4. **Acceptance test** = `scripts/uniad_stage1_smoke_test.py` runs 1 step on a HAL GPU
   inside the container, using the 5-episode dataset. This is the definitive proof the
   env works — the full BEVFormer forward pass OOMs on small GPUs, so it can only be
   validated on HAL.

## Known gotchas (bake these into the image / job)

- **deepspeed imports at startup and calls the CUDA compiler.** Locally this failed
  with `MissingCUDAException: CUDA_HOME does not exist`. In the image, install a real
  CUDA toolkit and set `CUDA_HOME` so this import succeeds.
- **nvcc shim**: the repo has a *conditional* nvcc shim (`OpenDriveVLA/drivevla/_bootstrap.py`)
  that only activates when a real `nvcc` is absent. With a real CUDA toolkit in the
  image it is a no-op — good, leave it.
- **Drop the local-only shims**: locally the repo was run with a `/tmp/fakecuda` nvcc
  stub and `PYTHONPATH=OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6`. In the image,
  do it properly: real CUDA toolkit + `pip install -e OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6`
  so the plugin resolves without the PYTHONPATH hack.
- **GPU passthrough**: run the container with `apptainer exec --nv <img>.sif ...`.
- **Repo paths / env vars** the config reads (see
  `OpenDriveVLA/projects/configs/stage1_track_map/carla_parking_stage1.py`):
  `NUSC_DATA_ROOT` (nuScenes-format DB root, default `<repo>/data/nuscenes`),
  `CARLA_INFOS_TRAIN` / `CARLA_INFOS_VAL`, `CARLA_LOT_GT`, and `UNIAD_WARMSTART`
  (default `ckpts/uniad_base_track_map.pth`).
- **Warm-start checkpoint**: the Stage-1 config sets `load_from = ckpts/uniad_base_track_map.pth`.
  For a pure *environment* smoke test you have two options — either transfer that
  checkpoint to HAL, or (simpler) confirm whether `uniad_stage1_smoke_test.py` already
  disables/relaxes `load_from`; if not, temporarily run randomly-initialized so the
  test validates the forward+backward pass without needing the (large) checkpoint.
  Read the smoke-test script first and follow what it expects.
- **Multi-GPU launcher** (for later, not the smoke test):
  `OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6/tools/dist_train.sh <config> <NUM_GPUS>`
  (torchrun). On Slurm this becomes `srun apptainer exec --nv img.sif bash dist_train.sh ...`.

## Datasets: a few small episodes for the smoke test, the full set for training

A full, validated **~1500-episode** dataset exists (the real training data). **Do NOT
ship the whole thing for the smoke test** — it's ~2.6 GB pkl + ~1.8 GB DB + tens of GB
of images, and loading that DB is slow. Use a few episodes for the smoke test.

**Smoke-test data flow (paths must be HAL-absolute, so rebuild on HAL — don't ship a
prebuilt pkl/DB; its image paths are baked to another machine):**
1. I scp ~3 small raw episodes into `data_carla/raw/` and the lot GT into
   `data_carla/processed/lot_map_gt_Town04_Opt.json` on HAL (each episode ~25 MB).
2. Rebuild in-repo, **from repo root**, so paths resolve on HAL:
   ```
   python scripts/build_infos_pkl.py --raw_dir data_carla/raw \
       --out data_carla/processed/parking_infos_temporal.pkl --absolute-paths --max-range 32
   python scripts/build_carla_nusc_tables.py          # writes data/nuscenes/v1.0-carla/
   python scripts/split_infos_train_val.py            # -> parking_infos_{train,val}.pkl
   ```
   All three builders are in `scripts/`. This yields the default layout the config
   expects (`data_carla/processed/*.pkl`, `data/nuscenes/v1.0-carla/`), so no env-var
   overrides are needed for the smoke test.

**Training → the full ~1500-episode set**, placed at the same `data_carla/` +
`data/nuscenes/` layout (scp raw + rebuild, or scp the prebuilt pkl/DB and rebuild).
Keep the image **data-free**; bind-mount the repo (which contains the data dirs) into
the container at runtime via `apptainer exec --bind ...`.

## Plan (phased — report back at the end of each phase before proceeding)

**Phase 0 — Probe HAL and confirm the workflow (no builds yet).**
Give me commands to run on HAL and report results:
- Container tooling: `which apptainer singularity; apptainer --version || singularity --version`.
- Can I build on HAL? Test `apptainer build --fakeroot` support, and whether there is a
  build node/queue — or whether I must build locally and transfer.
- GPUs & queues: `nvidia-smi` on a `ztest` node (GPU model + memory), `sinfo`,
  and confirm `ztest` (45-min debug queue) is the right place for the smoke test.
- Job submission: confirm a plain `sbatch` running `srun apptainer exec --nv ...` is
  allowed (no madame-web required).
- Storage: where the `.sif` and data should live (`<<HAL_WORKSPACE>>` vs a share vs
  `/staging/...`), and quotas.
- Is there an internal base image / container registry I'm expected to use, or should
  we start from a public base? **Flag this and ask me before committing** — I may need
  to check with my team.

Then propose the concrete build+transfer strategy based on what you found, and wait
for my OK.

**Phase 1 — Author and build the image locally.**
- Write a `Dockerfile` (and/or an Apptainer `.def`) that reproduces the stack from
  `ENV_SETUP.md` + `requirements.lock.txt` exactly, starting from a base that matches
  torch 2.1.2 + CUDA 12.1 (e.g. a `pytorch/pytorch:2.1.2-cuda12.1-cudnn8` or
  `nvidia/cuda:12.1.*-devel` base — pick per Phase 0), sets `CUDA_HOME`, and installs
  `mmcv-full==1.7.2`, `mmdet3d 1.0.0rc6` editable, etc.
- The repo itself is cloned from GitHub (I'll give you the clone URL/branch) either
  into the image or mounted at runtime — recommend **mounting the repo + data at
  runtime**, keeping the image to just the environment, so code changes don't require
  a rebuild.
- Build locally; validate inside the container that imports resolve:
  `python -c "import torch, mmcv, mmdet3d; from mmcv.ops import ...; print(torch.__version__, torch.version.cuda)"`
  and that deepspeed imports without the CUDA error.

**Phase 2 — Convert to SIF and transfer to HAL.**
- Convert the local Docker image to `.sif` (e.g. `apptainer build img.sif docker-daemon://<img>:<tag>`
  locally if Apptainer is available locally, or `docker save` + transfer + build on
  HAL, or push to a registry + `apptainer pull` on HAL — choose per Phase 0).
- Transfer the `.sif` to the HAL location chosen in Phase 0.

**Phase 3 — Stage code + the 5-episode data on HAL.**
- Clone/pull the repo into `<<HAL_WORKSPACE>>`; place the small dataset; set the env
  vars (`NUSC_DATA_ROOT`, `CARLA_INFOS_*`, `CARLA_LOT_GT`, and warm-start handling per
  the gotcha above).

**Phase 4 — Smoke test on a HAL GPU.**
- Write a minimal `sbatch` for the `ztest` queue that runs, inside the container with
  `--nv`, `python scripts/uniad_stage1_smoke_test.py` (1 step) against the 5-episode
  data. Bind-mount the repo + data into the container.
- Submit, monitor (`squeue --me`, `tail -f` the slurm log), and iterate on any failure
  — paste me errors and I'll run your fixes.

## Definition of done

- The `.sif` builds and lives on HAL.
- The repo + 5-episode data are staged on HAL with env vars set.
- `scripts/uniad_stage1_smoke_test.py` **completes 1 step on a HAL GPU inside the
  container** (forward + backward, no import/CUDA/shape errors).
- You hand me a short, reproducible runbook: exact build commands, transfer commands,
  the `sbatch` file, and the env vars — so I (and later a multi-GPU `dist_train.sh`
  run) can reproduce it.

## Reference files already in the repo

- `ENV_SETUP.md`, `requirements.lock.txt` — the authoritative env recipe.
- `REPORT.md` → section **"For the container agent"** — extra HAL notes (persistent
  mount for `checkpoints/<run>/` so `--resume` survives preemption; the conditional
  nvcc shim; Slurm launch shape).
- `scripts/uniad_stage1_smoke_test.py` — the acceptance test.
- `OpenDriveVLA/projects/configs/stage1_track_map/carla_parking_stage1.py` — the config
  (read the env-var header block at the top).
- `OpenDriveVLA/drivevla/_bootstrap.py` — the conditional nvcc shim.
