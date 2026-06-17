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
import os
import pathlib
import sys
import time

# Bootstrap paths so this runs standalone (llava/projects live at the repo root,
# mmdet3d in third_party) and DeepSpeed's nvcc check passes (reuse the shared shim).
_ODV = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ODV / "third_party" / "mmdetection3d_1_0_0rc6"))
sys.path.insert(0, str(_ODV))
_SHIM = _ODV / ".cache" / "fake_cuda"
if not (_SHIM / "bin" / "nvcc").exists():
    (_SHIM / "bin").mkdir(parents=True, exist_ok=True)
    (_SHIM / "bin" / "nvcc").write_text(
        '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.1, V12.1.0"\n')
    os.chmod(_SHIM / "bin" / "nvcc", 0o755)
os.environ.setdefault("CUDA_HOME", str(_SHIM))
os.environ["PATH"] = f"{_SHIM / 'bin'}:{os.environ.get('PATH', '')}"

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
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", default=False)
    return ap.parse_args()


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def main():
    args = parse_args()

    # Fail fast if feature extraction hasn't been run yet
    with open(args.conversations) as f:
        convs_check = json.load(f)
    missing_pth = sum(1 for c in convs_check if "uniad_pth" not in c)
    if missing_pth > 0:
        raise RuntimeError(
            f"{missing_pth}/{len(convs_check)} conversations missing 'uniad_pth'. "
            "Run feature extraction first:\n"
            "  cd OpenDriveVLA && bash scripts/extract_carla_features.sh"
        )

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (same as inference)
    print("Loading model...")
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

    # Also unfreeze the mm_projectors (BEV→LLM adapters, small but important)
    # They sit on model.base_model.model.model since peft wraps the base
    base = model.base_model.model
    for name, param in base.named_parameters():
        if "mm_projector" in name:
            param.requires_grad = True

    # Keep trainable params in fp32 (frozen base stays fp16 to save memory).
    # GradScaler.unscale_ rejects fp16 gradients, so the LoRA adapters and
    # mm_projectors — created in fp16 when the model was cast — must be fp32.
    if dtype == torch.float16:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    # Enable gradient checkpointing to save activation memory
    model.gradient_checkpointing_enable()

    trainable, total = count_trainable(model)
    print(f"Parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.1f}M total")
    print(f"  (LoRA rank={args.lora_rank} on q,k,v,o + mm_projectors unfrozen)")

    # Dataset
    uniad_cfg = Config.fromfile(args.uniad_config)
    data_args = DataArguments(
        data_path=args.conversations,
        lazy_preprocess=True,
        frames_upbound=32,
    )
    dataset = LLaVANuScenesDataset(
        tokenizer,
        data_args,
        uniad_cfg.data.test,
        llava_train_mode=True,
        use_uniad_pth=True,
    )
    print(f"Dataset: {len(dataset)} samples")

    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_train_mode=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    # GradScaler prevents fp16 gradient underflow/overflow during backward
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    total_steps = args.num_epochs * len(loader)
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nStarting training: {args.num_epochs} epochs, "
          f"lr={args.lr}, grad_accum={args.grad_accum}")

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(loader):
            if "uniad_pth" not in batch:
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            uniad_pth = batch["uniad_pth"]

            def _to_device(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device, dtype=dtype)
                if isinstance(x, dict):
                    return {k: _to_device(v) for k, v in x.items()}
                if isinstance(x, list):
                    return [_to_device(v) for v in x]
                return x

            uniad_pth = _to_device(uniad_pth)

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
            n_batches += 1

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
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

            if (step + 1) % 50 == 0:
                avg = epoch_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch} step {step+1}/{len(loader)} "
                      f"loss={avg:.4f} lr={lr_now:.2e} t={time.time()-t0:.0f}s")

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch}/{args.num_epochs} — avg_loss={avg_loss:.4f} ({time.time()-t0:.0f}s)")

        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            ckpt_path = out_dir / f"epoch_{epoch:03d}"
            # save_pretrained on a peft model saves LoRA adapters + tokenizer
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            print(f"  Saved LoRA checkpoint → {ckpt_path}")

    print(f"\nTraining complete. Final checkpoint: {out_dir}/epoch_{args.num_epochs:03d}")
    print("\nTo run inference with the fine-tuned model, you need to merge LoRA weights first:")
    print(f"  python drivevla/merge_lora.py --base {args.model_path} "
          f"--lora {out_dir}/epoch_{args.num_epochs:03d} "
          f"--out {out_dir}/merged")
    print("Then:")
    print(f"  bash scripts/eval_carla_parking.sh {out_dir}/merged")


if __name__ == "__main__":
    main()
