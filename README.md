# OpenDriveVLA — CARLA Parking Fine-Tuning

Fine-tunes [OpenDriveVLA](https://github.com/) (LLaVA = Qwen2-0.5B LLM + UniAD
BEV vision tower) to predict **parking trajectories** — including reverse
parking — from 6 surround cameras, using CARLA-collected data.

This repo contains **code only**. Data, the base model, the feature cache, and
inference outputs are **regenerated locally** (see *Setup*). The trained **LoRA
adapter** (the fine-tuned result) is included.

---

## What's in the repo vs. what you regenerate

| Tracked in git | Regenerated locally (gitignored, placeholder dirs) |
|---|---|
| `scripts/` (data prep, eval, visualization) | `data_carla/raw/` — raw CARLA episodes |
| `OpenDriveVLA/` (model code, de-nested) | `data_carla/processed/` — pkls, feature cache |
| `checkpoints/OpenDriveVLA-0.5B-carla/epoch_003/` (LoRA adapter, 45 MB) | `data/nuscenes/v1.0-carla/` — NuScenes-format DB |
| `COMMANDS.md`, `CLAUDE.md`, configs | `checkpoints/OpenDriveVLA-0.5B/` — base model (download) |
| | `checkpoints/.../merged/` — merged model (regenerate) |
| | `outputs/`, `OpenDriveVLA/output/` — runs/plots |

---

## Setup (make it functional after cloning)

**1. Environment** (Python 3.10):
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, mmcv/mmdet3d, peft, etc.
```

**2. Base model** — download the upstream `OpenDriveVLA-0.5B` checkpoint into
`checkpoints/OpenDriveVLA-0.5B/` (model weights + tokenizer). The fine-tuned LoRA
adapter in `checkpoints/OpenDriveVLA-0.5B-carla/epoch_003/` is applied on top of it.

**3. Data** — either collect CARLA episodes with the sibling
[`parking_data_gen`](../parking_data_gen) repo into `data_carla/raw/`, or drop an
existing `raw/` set there. Then build the dataset:
```bash
source .venv/bin/activate
python scripts/build_carla_nusc_tables.py        # raw → NuScenes DB (+ absolute image paths)
python scripts/generate_cached_nuscenes_info.py  # → cached_parking_info.pkl
python scripts/build_carla_conversations.py       # → carla_conversations.json
```
(The `parking_infos_temporal.pkl` is produced by `parking_data_gen/scripts/build_infos_pkl.py`.)

**4. Use the fine-tuned model** — merge the included LoRA adapter onto the base, then run:
```bash
cd OpenDriveVLA
python drivevla/merge_lora.py \
  --base ../checkpoints/OpenDriveVLA-0.5B \
  --lora ../checkpoints/OpenDriveVLA-0.5B-carla/epoch_003 \
  --out  ../checkpoints/OpenDriveVLA-0.5B-carla/merged
bash scripts/eval_carla_parking.sh ../checkpoints/OpenDriveVLA-0.5B-carla/merged
```

To re-train from scratch (extract features → LoRA train → merge → eval) and for
all command details, see **[COMMANDS.md](COMMANDS.md)**.

---

## Notes
- `scripts/paths.py` is the single source of truth for all paths.
- The model entry points under `OpenDriveVLA/drivevla/` self-bootstrap their
  import paths + an nvcc shim, so they run standalone.
- Hardware reference: developed on a single 8 GB GPU (LoRA rank 64). Sequential
  GPU use only — feature extraction and training can't run simultaneously.
- CARLA data collection lives in the separate `parking_data_gen` repo (Python 3.8 /
  CARLA 0.9.14).
