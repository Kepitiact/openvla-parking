"""Build our own DriveVLA backbone: Qwen2.5-3B-Instruct + reasoning tokens + fresh projectors.

This replaces OpenDriveVLA-0.5B as the student. It produces a LLaVA-Qwen checkpoint that
llava.model.builder.load_pretrained_model can load, containing:

  * Qwen2.5-3B-Instruct weights (hidden_size 2048, vs the 0.5B's 896)
  * a tokenizer carrying all 19 project special tokens, INCLUDING the two new reasoning
    ones (<reason_start>, <reason_end>) — so the model can emit its reasoning before the
    trajectory
  * three FRESH mm_projectors (track / scene / map), Linear-stack 256 -> 2048. The 0.5B's
    projectors cannot be reused: they map into an 896-dim residual stream.

DELIBERATELY ABSENT: any `vision_model.*` (UniAD) weights.
  OpenDriveVLA-0.5B shipped with `mm_tunable_parts` including `mm_vision_tower`, so its
  safetensors baked 1742 nuScenes-UniAD weights. load_pretrained_model then restored them
  OVER the CARLA-trained checkpoint the vision tower had just loaded — silently running
  nuScenes UniAD on CARLA images (near-field recall 1.00 -> 0.04). We refuse to bake tower
  weights at all, so UniAD can only ever come from an explicit UNIAD_CKPT. That makes the
  bug structurally impossible in OUR model.

Usage (CPU is fine, ~10 GB RAM):
  python scripts/init_drivevla_qwen3b.py \
      --base /home/s0002438/models/Qwen2.5-3B-Instruct \
      --out  checkpoints/DriveVLA-Qwen2.5-3B-init
"""

import argparse
import json
import os
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "OpenDriveVLA" / "third_party" / "mmdetection3d_1_0_0rc6"))
sys.path.insert(0, str(_REPO / "OpenDriveVLA"))

import torch
from transformers import AutoTokenizer

from llava.constants import (
    DEFAULT_ANSWER_END,
    DEFAULT_ANSWER_START,
    DEFAULT_COMMAND_END_TOKEN,
    DEFAULT_COMMAND_START_TOKEN,
    DEFAULT_EGO_END_TOKEN,
    DEFAULT_EGO_START_TOKEN,
    DEFAULT_MAP_END_TOKEN,
    DEFAULT_MAP_START_TOKEN,
    DEFAULT_MAP_TOKEN,
    DEFAULT_QUESTION_END,
    DEFAULT_QUESTION_START,
    DEFAULT_REASON_END_TOKEN,
    DEFAULT_REASON_START_TOKEN,
    DEFAULT_SCENE_END_TOKEN,
    DEFAULT_SCENE_START_TOKEN,
    DEFAULT_SCENE_TOKEN,
    DEFAULT_TRACK_END_TOKEN,
    DEFAULT_TRACK_START_TOKEN,
    DEFAULT_TRACK_TOKEN,
    DEFAULT_TRAJ_END_TOKEN,
    DEFAULT_TRAJ_START_TOKEN,
    DEFAULT_TRAJ_TOKEN,
)

# Every special token the model must know. The <reason_*> pair is what Step 5 wraps the
# reasoning in; the rest mirror the 0.5B's tokenizer so the prompt format is unchanged.
SPECIAL_TOKENS = [
    DEFAULT_SCENE_TOKEN, DEFAULT_TRACK_TOKEN, DEFAULT_MAP_TOKEN, DEFAULT_TRAJ_TOKEN,
    DEFAULT_SCENE_START_TOKEN, DEFAULT_SCENE_END_TOKEN,
    DEFAULT_TRACK_START_TOKEN, DEFAULT_TRACK_END_TOKEN,
    DEFAULT_MAP_START_TOKEN, DEFAULT_MAP_END_TOKEN,
    DEFAULT_EGO_START_TOKEN, DEFAULT_EGO_END_TOKEN,
    DEFAULT_COMMAND_START_TOKEN, DEFAULT_COMMAND_END_TOKEN,
    DEFAULT_QUESTION_START, DEFAULT_QUESTION_END,
    DEFAULT_ANSWER_START, DEFAULT_ANSWER_END,
    DEFAULT_TRAJ_START_TOKEN, DEFAULT_TRAJ_END_TOKEN,
    DEFAULT_REASON_START_TOKEN, DEFAULT_REASON_END_TOKEN,   # <- the reasoning block
]

UNIAD_FEATURE_DIM = 256          # track_query_embeddings / seg queries are [N, 256]
MM_PROJECTOR_TYPE = "mlp2x_gelu"  # same family the 0.5B used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Qwen2.5-3B-Instruct weights dir")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    out = pathlib.Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")

    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
    from llava.model.multimodal_projector.builder import build_vision_projector

    print(f"[1/5] tokenizer <- {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    before = len(tok)
    added = tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    print(f"      vocab {before} -> {len(tok)}  (+{added} special tokens)")
    for t in (DEFAULT_REASON_START_TOKEN, DEFAULT_REASON_END_TOKEN):
        print(f"      {t} -> id {tok.convert_tokens_to_ids(t)}")

    print("[2/5] config: Qwen2.5-3B + multimodal fields")
    cfg = LlavaQwenConfig.from_pretrained(args.base)
    cfg.model_type = "llava_qwen"
    cfg.architectures = ["LlavaQwenForCausalLM"]
    cfg.vision_tower_pretrained = ""
    cfg.use_mm_proj = True
    cfg.mm_projector_type = MM_PROJECTOR_TYPE
    cfg.mm_hidden_size = UNIAD_FEATURE_DIM
    cfg.mm_vision_select_layer = -2
    cfg.mm_vision_select_feature = "patch"
    cfg.mm_patch_merge_type = "flat"
    # NOTE: mm_tunable_parts deliberately EXCLUDES mm_vision_tower — see module docstring.
    cfg.mm_tunable_parts = "mm_mlp_adapter"
    # `mm_vision_tower` is set AFTER construction on purpose: LlavaQwenModel.__init__
    # builds the vision tower (i.e. instantiates all of UniAD) the moment the config
    # carries that key. We do not need UniAD here — we are only assembling the LLM +
    # projectors, and we strip tower weights from the checkpoint anyway. Setting it
    # afterwards keeps init fast, CPU-only, and free of any UniAD dependency.
    print(f"      hidden_size={cfg.hidden_size}  mm_hidden_size={cfg.mm_hidden_size}")

    print(f"[3/5] loading Qwen2.5-3B weights (no vision tower is constructed)")
    model = LlavaQwenForCausalLM.from_pretrained(
        args.base, config=cfg, torch_dtype=dtype, attn_implementation="eager")

    print(f"[4/5] resize embeddings -> {len(tok)}; init new rows from the embedding mean")
    old_n = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok))
    n_new = len(tok) - old_n
    if n_new > 0:
        # Mean-init beats the default random init: the new tokens start in-distribution,
        # so they do not inject noise into a pretrained residual stream on step 0.
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean = emb[:old_n].mean(dim=0, keepdim=True)
            emb[-n_new:] = mean
            if not cfg.tie_word_embeddings:
                head = model.get_output_embeddings().weight
                head[-n_new:] = head[:old_n].mean(dim=0, keepdim=True)
        print(f"      {n_new} new embedding rows mean-initialised "
              f"(tied lm_head: {cfg.tie_word_embeddings})")

    # Fresh projectors: 256 -> 2048. The 0.5B's map into 896 and are unusable.
    mm = model.get_model()
    mm.mm_projector_track = build_vision_projector(cfg).to(dtype)
    mm.mm_projector_scene = build_vision_projector(cfg).to(dtype)
    mm.mm_projector_map = build_vision_projector(cfg).to(dtype)
    n_proj = sum(p.numel() for n, p in mm.named_parameters() if "mm_projector" in n)
    print(f"      3 fresh projectors built ({n_proj/1e6:.1f}M params, {MM_PROJECTOR_TYPE})")

    # Only now declare the vision tower, so the SAVED config tells the trainer to build
    # UniAD (from UNIAD_CKPT) while this init process never touched it.
    cfg.mm_vision_tower = "uniad_track_map"
    model.config.mm_vision_tower = "uniad_track_map"

    print(f"[5/5] saving -> {out}  (stripping any vision_model.* weights)")
    out.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    tower = [k for k in sd if "vision_model." in k or "vision_tower." in k]
    for k in tower:
        del sd[k]
    if tower:
        print(f"      dropped {len(tower)} vision-tower tensors (UniAD comes from UNIAD_CKPT)")
    model.save_pretrained(out, state_dict=sd, safe_serialization=True)
    tok.save_pretrained(out)

    # Hard guarantee: no baked tower weights, ever.
    from safetensors import safe_open
    leaked = []
    for f in out.glob("*.safetensors"):
        with safe_open(f, framework="pt") as fh:
            leaked += [k for k in fh.keys() if "vision_model." in k or "vision_tower." in k]
    if leaked:
        raise SystemExit(f"FAILED: {len(leaked)} vision-tower tensors leaked into the "
                         f"checkpoint (e.g. {leaked[0]}). This is the OpenDriveVLA-0.5B bug.")
    print(f"      verified: 0 vision-tower tensors in the checkpoint ✓")
    print(f"\nDone. Train with:  --model-path {out}  UNIAD_CKPT=<trained uniad>.pth")


if __name__ == "__main__":
    main()
