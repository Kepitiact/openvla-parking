# Reasoning-VLA for CARLA Parking

A vision-language-action model that **says why before it acts**: it reads 6 surround
cameras, writes a short natural-language reasoning trace, and *then* predicts a parking
trajectory (including reverse-perpendicular parking) — with the reasoning made
**causally load-bearing**, not decorative.

The design converges independently with NVIDIA's [Alpamayo-R1](https://arxiv.org/abs/2511.00088):
reasoning traces are grounded in a single explicit decision (**decision grounding**), cite
only what is observed (**causal locality**), and exclude superficial context (weather, road
type). Where Alpamayo couples reasoning to action with a conditioning decoder + RL, we use a
lighter **attention gate** that admits a clean controlled ablation. See
[FUTURE_WORK.md](FUTURE_WORK.md) §3e for the RL direction.

---

## The idea

Standard driving models map perception → trajectory directly, so any "reasoning" they emit
can be decoration. We block the direct path: the **trajectory tokens cannot attend to the
perception tokens** — object information can reach the waypoints *only* through the
reasoning's hidden states. If the reasoning is wrong, the trajectory is wrong. That is the
thesis, and it is testable (ablate the reasoning → does the trajectory break?).

## The cast

| role | model | job | trained? |
|---|---|---|---|
| **Perception** | UniAD (`epoch_6`) | 6 cameras → object / map / scene tokens | frozen |
| **Teacher** | Qwen2.5-**32B**-Instruct | writes the grounded reasoning *training data* | frozen |
| **Student** | Qwen2.5-**3B** (`DriveVLA-Qwen2.5-3B-init`) | learns reasoning + trajectory from perception | **trained** |

```
6 cameras → UniAD (frozen) → <SCENE><TRACK><MAP> tokens ─┐
                                                         ├─► Qwen2.5-3B (LoRA)
ego-state text (velocity, steer, ego-local slot pose) ──┘        │
                                                                 ▼
                        <reason_start> …reasoning… <reason_end>
                        <traj_start> (x,y,h)×6 <traj_end>
                        [REASON_GATE: trajectory can't read perception directly]
```

## Pipeline

1. **Extract** perception — `sbatch scripts/sbatch/hal_extract_features.sbatch` → per-frame
   UniAD feature `.pth` (object/map/scene streams).
2. **Reasoning data** — `reasoning_data_gen/` extracts a deterministic *fact* per frame
   (grounded in GT), the 32B teacher verbalizes it, and validators reject any hallucination.
   `sbatch scripts/sbatch/hal_reason_qwen_full.sbatch` (episode-sharded) →
   `reasoning/v1_uniad_epoch6/traces.jsonl`.
3. **Train** — `scripts/sbatch/hal_train_vla.sbatch`:
   - `align` — projectors + new-token embeddings learn to land perception in Qwen's space
     (LLM frozen).
   - `finetune` — LoRA on Qwen; the reason gate makes reasoning the causal channel.
4. **Eval** — see [scripts/EVAL_HARNESS.md](scripts/EVAL_HARNESS.md): trajectory metrics
   (ADE/FDE + final pose), reasoning faithfulness, and the causal ablation / trace-corruption
   tests.

## Layout

| path | what |
|---|---|
| `reasoning_data_gen/` | fact extraction, teacher verbalizer, validators, generation CLI |
| `OpenDriveVLA/` | model code — UniAD tower, Qwen VLA, the reason gate, train/infer |
| `scripts/` | Python tools (init backbone, probes, eval scorers) |
| `scripts/sbatch/` | all HAL Slurm jobs |
| `data_carla/processed/` | infos pkls, conversations index (gitignored) |
| `checkpoints/` | models — UniAD, the 3B backbone, trained adapters (gitignored) |

Data and models are regenerated / trained, not tracked in git. The heavy artifacts
(features, traces, the 32B) live on the cluster's staging store; infos + conversations live
in the repo. See [HAL_RUNBOOK.md](HAL_RUNBOOK.md) for the cluster workflow and
[ENV_SETUP.md](ENV_SETUP.md) for the environment.

## Status

Perception extracted (61,659 frames); teacher reasoning data generated and validated;
training path proven end-to-end (smoke); align/finetune + eval in progress. Research
directions and deliberate deferrals are tracked in [FUTURE_WORK.md](FUTURE_WORK.md).
