# HAL container runbook (UniAD Stage-1 env)

Reproducible steps to build the Singularity env for `openvla-parking` and run it on
the HAL cluster. The image is **env-only**; code + data + checkpoint are bind-mounted
at runtime, so code changes need only a `git pull` on HAL (no rebuild).

Placeholders: `s0002438` = HAL user, `/workspaces/s0002438` = HAL workspace.

## 1. Build the image (local machine with Docker)

```bash
cd <repo-root>
docker build -t openvla-parking:cu121 .          # see Dockerfile
# Clean single-manifest archive (avoids Singularity multi-manifest issues):
docker build --provenance=false -t openvla-parking:cu121 \
    -o type=docker,dest=/tmp/openvla.tar .
```
Image ≈ 11 GB archive. Stack pinned per `requirements.lock.txt` (torch 2.1.2+cu121,
mmcv-full 1.7.2, mmdet 2.26.0, mmseg 0.29.1, mmdet3d 1.0.0rc6 compiled for sm_80).

## 2. Transfer + build the SIF (on HAL)

```bash
scp -O /tmp/openvla.tar hal-login:/workspaces/s0002438/         # note: scp -O (legacy proto)

# On HAL — redirect Singularity scratch off small /tmp, then build (rootless):
export SINGULARITY_TMPDIR=/workspaces/s0002438/sing_tmp
export SINGULARITY_CACHEDIR=/workspaces/s0002438/sing_cache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
cd /workspaces/s0002438
singularity build openvla.sif docker-archive://openvla.tar     # -> openvla.sif ~9 GB
```

## 3. Stage code + data (on HAL)

```bash
cd /workspaces/s0002438
git clone https://github.com/Kepitiact/openvla-parking.git     # code only (public)
mkdir -p openvla-parking/OpenDriveVLA/ckpts \
         openvla-parking/data/nuscenes/maps/expansion
```
From local (repo root), scp the assets git ignores:
```bash
scp -O OpenDriveVLA/ckpts/uniad_base_track_map.pth  hal-login:/workspaces/s0002438/openvla-parking/OpenDriveVLA/ckpts/
scp -O -r data_carla/raw/episode_000{0,1,2,4,5}     hal-login:/workspaces/s0002438/openvla-parking/data_carla/raw/
scp -O data_carla/processed/lot_map_gt_Town04_Opt.json hal-login:/workspaces/s0002438/openvla-parking/data_carla/processed/
scp -O data/nuscenes/maps/*.png                     hal-login:/workspaces/s0002438/openvla-parking/data/nuscenes/maps/
scp -O data/nuscenes/expansion/*.json               hal-login:/workspaces/s0002438/openvla-parking/data/nuscenes/maps/expansion/
```
Rebuild the DB in-repo, inside the container, so paths are HAL-absolute:
```bash
cd /workspaces/s0002438/openvla-parking
RUN="singularity exec --bind /workspaces/s0002438 --pwd /workspaces/s0002438/openvla-parking /workspaces/s0002438/openvla.sif"
$RUN python scripts/build_infos_pkl.py --raw_dir data_carla/raw \
      --out data_carla/processed/parking_infos_temporal.pkl --absolute-paths --max-range 32
$RUN python scripts/build_carla_nusc_tables.py      # -> data/nuscenes/v1.0-carla/
$RUN python scripts/split_infos_train_val.py        # -> parking_infos_{train,val}.pkl
```
No env-var overrides needed: the config's `_REPO = parent(CWD)`, and running with
CWD=`OpenDriveVLA/` makes all default paths resolve (`data/nuscenes`, `data_carla`,
`ckpts/uniad_base_track_map.pth`).

## 4. Smoke test on a GPU (`scripts/sbatch/hal_smoke_test.sbatch`)

```bash
sbatch scripts/sbatch/hal_smoke_test.sbatch --tiny        # ztest A100-40GB, 45-min queue
squeue --me ; tail -f smoke_<jobid>.log
```
Key runtime flags baked into the sbatch:
- `--nv` GPU passthrough; `--bind /workspaces/s0002438`; `--pwd .../OpenDriveVLA`.
- `PYTHONPATH=.../OpenDriveVLA` — makes `projects.mmdet3d_plugin` / llava / drivevla importable (mounted, live).
- **`LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:/usr/local/cuda/lib64`**
  — container libs first so cv2's `libGLdispatch` resolves to the glibc-2.35 one, not
  the newer host lib `--nv` injects (else: `GLIBC_2.38 not found`).
- `TRITON_CACHE_DIR=/tmp/...` node-local (DeepSpeed/Triton NFS warning).
- `--tiny` (queue_length=1, 0.5x image) fits the 40 GB A100. **Full config
  (queue_length=5) OOMs at 40 GB** — real training needs 80 GB nodes, grad
  checkpointing, or smaller queue_length.

## Status: ENV VALIDATED + STAGE-1 TRAINING RESTORED ✅

`sbatch scripts/sbatch/hal_smoke_test.sbatch --tiny` completes a **full training step**
(forward → loss → backward → optimizer) on a HAL A100 inside the container:
`SMOKE TEST (full fwd/bwd) PASSED.` (total loss ~42, 67 `track.*`+`map.*` terms).
Proven: all imports + compiled `mmcv.ops` + `deepspeed` nvcc probe, mmdet3d CUDA ops,
full data pipeline, warm-start checkpoint key-match, forward through every head, and
track+map loss backward.

### Model-code fixes made to restore stage-1 training (were stubbed by the refactor)

The refactor had wired UniAD as a frozen feature-extractor for the VLM and stubbed the
training-loss path. Restored cleanly (no hard patches):

1. `uniad_e2e.py` `forward_train` — return the **loss dict** by default (mmdet
   contract); VLM features only on opt-in `return_vlm=True`.
2. `uniad_track_map.py` (VLM vision tower) — pass `return_vlm=True` (preserves VLM path).
3. `uniad_track.py:584` — un-stub: `losses = self.criterion.losses_dict` (was `{}`).
4. `panseg_head.py:1110` — un-stub: `losses_seg = self.loss(...)` (was `{}`).
5. `seg_assigner.py` — device fix: index the CPU `cost` tensor with CPU indices
   (scipy `linear_sum_assignment` runs on CPU; match indices were moved to GPU).
6. `nuscenes_e2e_dataset.py` — CARLA empty map_mask `[3,H,W]`→`[2,H,W]` so it yields
   exactly **1** stuff class (label 3), matching `num_stuff_classes=1` / `num_classes=4`
   and the nuScenes `obtain_map_info` layout. (Was emitting a spurious 2nd stuff class.)

**Commit these** — they're required for any UniAD stage-1 training, not just the smoke test.

### Training-scale notes (for the real run)

- **Full config (`queue_length=5`) OOMs at 40 GB.** The smoke test uses `--tiny`
  (queue_length=1, 0.5x image). For real training: 80 GB nodes (`zprod`), gradient
  checkpointing, or a smaller `queue_length`. Multi-GPU does not reduce per-GPU
  activation memory at batch=1.
- The 5-episode smoke split put all frames in `train`, `val` empty — expected at this
  size; the full ~1500-episode set splits normally.
- Multi-GPU launch (later): `srun singularity exec --nv ... bash
  OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6/tools/dist_train.sh <config> <N>`
  with the same `--bind` / `PYTHONPATH` / `LD_LIBRARY_PATH` env as the smoke sbatch.
