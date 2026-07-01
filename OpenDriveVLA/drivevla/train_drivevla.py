"""
Fine-tune OpenDriveVLA on CARLA parking data using LoRA.

LoRA keeps the 8GB GPU budget manageable:
  - Full model stays in fp16 (~1.2 GB) — UniAD and Qwen2 base frozen
  - LoRA adapters on Qwen2 attention layers (~5M trainable params)
  - mm_projectors fully trainable (small: ~2M params total)
  - Optimizer states for ~7M params ≈ 80 MB instead of 6 GB for full fine-tune

Requires pre-computed UniAD features from extract_uniad_features.py.

Usage (from OpenDriveVLA/ directory):
  bash scripts/train_carla_parking.sh
"""

import argparse
import json
import logging
import os
import pathlib
import random
import shutil
import sys
import time

import _bootstrap  # noqa: F401  (sys.path + conditional nvcc shim; must precede llava/mmdet3d)

import numpy as np
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from mmengine import Config
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.train.train import DataArguments

from data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="../checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--uniad-config", default="projects/configs/stage1_track_map/carla_parking.py")
    ap.add_argument("--conversations", default="../data_carla/processed/carla_conversations.json")
    ap.add_argument("--output-dir", default="../checkpoints/OpenDriveVLA-0.5B-carla")
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA adapters train faster than full fine-tune; 2e-4 is typical")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-mlp", action="store_true", default=False,
                    help="Also apply LoRA to the MLP layers (gate/up/down_proj), not just attention.")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="Also save resumable state every N optimizer steps (mid-epoch). "
                         "0 disables; only per-epoch state is saved.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Micro-batch per GPU. The UniAD collator supports only 1; scale "
                         "throughput via --num-workers, --grad-accum and multi-GPU, not this.")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", default=False)
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Resume optimizer/scheduler/scaler/RNG/epoch/step from the run's saved state.")
    ap.add_argument("--distributed", action="store_true", default=False,
                    help="Use HuggingFace Accelerate (DDP + framework-managed mixed precision). "
                         "Auto-enabled when launched under torchrun (WORLD_SIZE>1).")
    ap.add_argument("--trainable-groups", default="projectors",
                    help="Comma-separated non-LoRA groups to unfreeze: projectors,uniad,heads,llm. "
                         "Default 'projectors' reproduces the original behaviour.")
    return ap.parse_args()


# Declarative trainable-group policy (B7): name-substring predicates per group,
# replacing the single hardcoded mm_projector unfreeze. LoRA (on the LLM
# attention) is applied separately and is always the mechanism for the LLM.
TRAINABLE_GROUPS = {
    "projectors": ("mm_projector",),   # BEV -> LLM adapters (default)
    "uniad": ("uniad",),               # UniAD perception backbone
    "heads": ("lm_head", "embed_out"), # output heads
    "llm": ("model.layers",),          # full LLM fine-tune (in addition to LoRA)
}


def apply_trainable_groups(base, groups):
    patterns = []
    for g in groups:
        if g not in TRAINABLE_GROUPS:
            raise ValueError(
                f"Unknown trainable group '{g}'. Choose from {sorted(TRAINABLE_GROUPS)}")
        patterns += TRAINABLE_GROUPS[g]
    n = 0
    for name, param in base.named_parameters():
        if any(pat in name for pat in patterns):
            param.requires_grad = True
            n += 1
    return n


def setup_logger(out_dir, is_main):
    """Leveled logger -> stdout + checkpoints/<run>/train.log (B5)."""
    log = logging.getLogger("train_drivevla")
    log.setLevel(logging.INFO if is_main else logging.WARNING)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if is_main:
        fh = logging.FileHandler(out_dir / "train.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def set_seed(seed):
    """Manual seeding for torch/numpy/python (B3)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Deterministic per-worker seeding for the dataloader (B3)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _to_device(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=dtype)
    if isinstance(x, dict):
        return {k: _to_device(v, device, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device, dtype) for v in x]
    return x


def save_lora(out_dir, epoch, model, tokenizer, accelerator, is_main, log):
    """Save the per-epoch LoRA adapter (+ tokenizer)."""
    ckpt_path = out_dir / f"epoch_{epoch:03d}"
    if not is_main:
        return
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    unwrapped.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    log.info(f"  Saved LoRA adapter -> {ckpt_path}")


def save_full_state(out_dir, meta, model, optimizer, scheduler, scaler,
                    accelerator, is_main, log):
    """Save full resumable state (model + optimizer/scheduler/scaler/RNG + meta).

    `meta` carries {epoch, global_step, batch_in_epoch, epoch_done} so a run
    preempted mid-epoch resumes from the exact batch, not from scratch (B2).
    Writes are atomic (temp then rename) so a kill DURING a save can't corrupt
    the last good checkpoint.
    """
    if accelerator is not None:
        tmp_dir = out_dir / "last_state.tmp"
        final_dir = out_dir / "last_state"
        accelerator.wait_for_everyone()
        if is_main and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        accelerator.wait_for_everyone()
        # safe_serialization=False (torch .bin) preserves tied weights
        # (Qwen2 ties lm_head to embed_tokens); safetensors drops the shared
        # tensor and then strict load_state fails on the missing key.
        accelerator.save_state(str(tmp_dir), safe_serialization=False)
        accelerator.wait_for_everyone()
        if is_main:
            (tmp_dir / "meta.json").write_text(json.dumps(meta))
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(str(tmp_dir), str(final_dir))
    else:
        tmp = out_dir / "train_state.pt.tmp"
        torch.save({
            **meta,
            "model_trainable": {n: p.detach().cpu()
                                for n, p in model.named_parameters() if p.requires_grad},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }, tmp)
        os.replace(tmp, out_dir / "train_state.pt")
    if is_main:
        tag = "epoch-end" if meta["epoch_done"] else f"step {meta['batch_in_epoch']}"
        log.info(f"  Saved train state ({tag}, global_step {meta['global_step']})")


def load_training_state(out_dir, model, optimizer, scheduler, scaler, accelerator, log):
    """Restore full training state; returns (start_epoch, global_step, skip_batches)."""
    if accelerator is not None:
        state_dir = out_dir / "last_state"
        if not state_dir.exists():
            log.warning(f"--resume: no saved state at {state_dir}; starting fresh")
            return 1, 0, 0
        accelerator.load_state(str(state_dir))
        meta = json.loads((state_dir / "meta.json").read_text())
    else:
        state_path = out_dir / "train_state.pt"
        if not state_path.exists():
            log.warning(f"--resume: no saved state at {state_path}; starting fresh")
            return 1, 0, 0
        ck = torch.load(state_path, map_location="cpu")
        own = dict(model.named_parameters())
        for n, v in ck["model_trainable"].items():
            if n in own:
                own[n].data.copy_(v.to(own[n].dtype).to(own[n].device))
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["torch_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        meta = ck

    if meta["epoch_done"]:
        start_epoch, skip = meta["epoch"] + 1, 0
    else:
        start_epoch, skip = meta["epoch"], meta["batch_in_epoch"]
    log.info(f"Resumed from epoch {meta['epoch']} (global_step {meta['global_step']}, "
             f"skip {skip} batches)")
    return start_epoch, meta["global_step"], skip


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def main():
    args = parse_args()

    # --- Framework selection (B1/B6): opt-in Accelerate for multi-GPU / distributed.
    # Auto-enabled under torchrun (WORLD_SIZE>1); otherwise the plain single-GPU
    # path runs unchanged as the default.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_accelerate = args.distributed or world_size > 1
    accelerator = None
    if use_accelerate:
        from accelerate import Accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=args.grad_accum,
            mixed_precision="bf16" if args.bf16 else "fp16",
        )
        is_main = accelerator.is_main_process
        device = accelerator.device
    else:
        is_main = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = pathlib.Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    log = setup_logger(out_dir, is_main)

    # Determinism (B3): distinct per-rank seed so DDP replicas don't mirror RNG.
    set_seed(args.seed + (accelerator.process_index if accelerator else 0))

    # Fail fast if feature extraction hasn't been run yet. Verify each referenced
    # feature FILE exists, not just that the field is present (A4).
    with open(args.conversations) as f:
        convs_check = json.load(f)
    missing_field = sum(1 for c in convs_check if "uniad_pth" not in c)
    missing_file = [c["uniad_pth"] for c in convs_check
                    if "uniad_pth" in c and not os.path.exists(c["uniad_pth"])]
    if missing_field or missing_file:
        raise RuntimeError(
            f"{missing_field}/{len(convs_check)} conversations missing the 'uniad_pth' field; "
            f"{len(missing_file)} reference a feature file that does not exist"
            + (f" (e.g. {missing_file[0]})" if missing_file else "") + ".\n"
            "Run feature extraction first:\n"
            "  cd OpenDriveVLA && bash scripts/extract_carla_features.sh"
        )

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Load model (same as inference)
    log.info("Loading model...")
    disable_torch_init()
    overwrite_config = {
        "image_aspect_ratio": "pad",
        "vision_tower_test_mode": True,
    }
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        model_base=None,
        model_name="llava_qwen",
        device_map=str(device),
        multimodal=True,
        attn_implementation="eager",
        overwrite_config=overwrite_config,
    )
    model = model.to(dtype)

    # Freeze everything first (UniAD + LLM base weights)
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to Qwen2 attention projection layers only
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.lora_mlp:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Unfreeze the selected non-LoRA groups (B7). Default 'projectors' reproduces
    # the original mm_projector-only unfreeze. peft wraps the base at
    # model.base_model.model.
    groups = [g.strip() for g in args.trainable_groups.split(",") if g.strip()]
    n_unfroze = apply_trainable_groups(model.base_model.model, groups)
    log.info(f"Trainable groups {groups}: unfroze {n_unfroze} param tensors")

    # Keep trainable params in fp32 (frozen base stays fp16 to save memory).
    # GradScaler.unscale_ rejects fp16 gradients, so the LoRA adapters and
    # unfrozen groups — created in fp16 when the model was cast — must be fp32.
    # (Under Accelerate, the framework manages mixed precision, but fp32 master
    # weights for trainable params are still correct.)
    if dtype == torch.float16:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    # Enable gradient checkpointing to save activation memory
    model.gradient_checkpointing_enable()

    trainable, total = count_trainable(model)
    log.info(f"Parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.1f}M total "
             f"(LoRA rank={args.lora_rank} on {','.join(target_modules)})")

    # Dataset
    uniad_cfg = Config.fromfile(args.uniad_config)
    data_args = DataArguments(
        data_path=args.conversations,
        lazy_preprocess=True,
        # nuScenes multi-frame cap; unused for single-frame CARLA parking (D4).
        frames_upbound=32,
    )
    dataset = LLaVANuScenesDataset(
        tokenizer,
        data_args,
        uniad_cfg.data.test,
        llava_train_mode=True,
        use_uniad_pth=True,
    )
    log.info(f"Dataset: {len(dataset)} samples")

    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_train_mode=True)

    # Seeded, explicitly-shuffled loader (B3). The UniAD collator supports only a
    # micro-batch of 1; larger effective batches come from --grad-accum and
    # multi-GPU data parallelism, not --batch-size.
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=gen,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    # GradScaler for the hand-rolled single-GPU fp16 path. Under Accelerate the
    # framework owns the scaler, so disable ours there.
    scaler = torch.cuda.amp.GradScaler(
        enabled=(dtype == torch.float16 and accelerator is None))

    total_steps = args.num_epochs * len(loader)
    warmup_steps = min(100, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if accelerator is not None:
        accelerator.register_for_checkpointing(scheduler)
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    start_epoch, global_step, resume_skip = 1, 0, 0
    if args.resume:
        start_epoch, global_step, resume_skip = load_training_state(
            out_dir, model, optimizer, scheduler, scaler, accelerator, log)

    log.info(f"Starting training: epochs {start_epoch}..{args.num_epochs}, "
             f"lr={args.lr}, grad_accum={args.grad_accum}, "
             f"framework={'accelerate' if accelerator else 'single-gpu'}, "
             f"world_size={world_size}")

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()  # A5: ensure train mode each epoch (dropout on, etc.)
        # Deterministic, reproducible per-epoch shuffle so a mid-epoch resume
        # replays the same order and can skip already-processed batches (B3/B2).
        gen.manual_seed(args.seed + epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)  # per-epoch reshuffle for the distributed sampler

        skip = resume_skip if epoch == start_epoch else 0
        if skip:
            log.info(f"  resuming epoch {epoch}: skipping first {skip} batches")

        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(loader):
            if skip and step < skip:
                continue  # fast-forward to the resumed position within the epoch
            if "uniad_pth" not in batch:
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            uniad_pth = _to_device(batch["uniad_pth"], device, dtype)

            is_boundary = (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader)

            if accelerator is not None:
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        uniad_pth=uniad_pth,
                    )
                    accelerator.backward(outputs.loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                epoch_loss += outputs.loss.item()
            else:
                with torch.cuda.amp.autocast(dtype=dtype):
                    outputs = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        uniad_pth=uniad_pth,
                    )
                loss = outputs.loss / args.grad_accum
                scaler.scale(loss).backward()
                epoch_loss += outputs.loss.item()
                if is_boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            n_batches += 1

            if (step + 1) % 50 == 0 and is_main:
                avg = epoch_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                log.info(f"  epoch {epoch} step {step+1}/{len(loader)} "
                         f"loss={avg:.4f} lr={lr_now:.2e} t={time.time()-t0:.0f}s")

            # Mid-epoch resumable checkpoint (B2): survive Slurm preemption without
            # losing a whole epoch of work.
            if args.save_steps and global_step > 0 and (step + 1) % args.save_steps == 0:
                save_full_state(
                    out_dir,
                    {"epoch": epoch, "global_step": global_step,
                     "batch_in_epoch": step + 1, "epoch_done": False},
                    model, optimizer, scheduler, scaler, accelerator, is_main, log)

        avg_loss = epoch_loss / max(1, n_batches)
        log.info(f"Epoch {epoch}/{args.num_epochs} - avg_loss={avg_loss:.4f} "
                 f"({time.time()-t0:.0f}s)")

        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            save_lora(out_dir, epoch, model, tokenizer, accelerator, is_main, log)
            save_full_state(
                out_dir,
                {"epoch": epoch, "global_step": global_step,
                 "batch_in_epoch": 0, "epoch_done": True},
                model, optimizer, scheduler, scaler, accelerator, is_main, log)

    if is_main:
        log.info(f"Training complete. Final checkpoint: {out_dir}/epoch_{args.num_epochs:03d}")
        log.info("Merge LoRA before inference:\n"
                 f"  python drivevla/merge_lora.py --base {args.model_path} "
                 f"--lora {out_dir}/epoch_{args.num_epochs:03d} --out {out_dir}/merged\n"
                 f"  bash scripts/eval_carla_parking.sh {out_dir}/merged")


if __name__ == "__main__":
    main()
