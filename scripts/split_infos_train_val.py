"""
Split the combined CARLA infos pkl into train/val by each episode's held-out
split (recorded in data_carla/raw/<episode>/meta.json -> collection_config.split).

The Stage-1 config points CARLA_INFOS_TRAIN / CARLA_INFOS_VAL at these outputs so
training and evaluation use disjoint slots (no frame/config leak).

Run from repo root:
  python scripts/split_infos_train_val.py
Outputs (next to the input pkl):
  parking_infos_train.pkl   parking_infos_val.pkl
"""

import argparse
import json
import pathlib
import pickle
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", default="data_carla/processed/parking_infos_temporal.pkl")
    ap.add_argument("--raw_dir", default="data_carla/raw")
    ap.add_argument("--val-split-name", default="test",
                    help="collection_config.split value that maps to the val pkl")
    args = ap.parse_args()

    infos_path = pathlib.Path(args.infos)
    raw_dir = pathlib.Path(args.raw_dir)

    # scene_token (episode dir name) -> split, from each episode's meta.json
    split_of = {}
    slots = {"train": set(), args.val_split_name: set()}
    for meta in sorted(raw_dir.glob("episode_*/meta.json")):
        ep = meta.parent.name
        try:
            cc = json.load(open(meta)).get("collection_config", {})
        except Exception:
            cc = {}
        sp = cc.get("split", "train")
        split_of[ep] = sp
        if sp in slots:
            slots[sp].add(cc.get("slot_idx"))

    with open(infos_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    meta_blob = data.get("metadata", {"version": "v1.0-carla"})

    train, val, unknown = [], [], Counter()
    for rec in infos:
        sp = split_of.get(rec["scene_token"], "train")
        (val if sp == args.val_split_name else train).append(rec)
        if rec["scene_token"] not in split_of:
            unknown[rec["scene_token"]] += 1

    out_train = infos_path.with_name("parking_infos_train.pkl")
    out_val = infos_path.with_name("parking_infos_val.pkl")
    with open(out_train, "wb") as f:
        pickle.dump({"infos": train, "metadata": meta_blob}, f)
    with open(out_val, "wb") as f:
        pickle.dump({"infos": val, "metadata": meta_blob}, f)

    n_ep_train = len({r["scene_token"] for r in train})
    n_ep_val = len({r["scene_token"] for r in val})
    overlap = slots["train"] & slots[args.val_split_name]
    print(f"train: {len(train):6d} frames / {n_ep_train} episodes -> {out_train}")
    print(f"val  : {len(val):6d} frames / {n_ep_val} episodes -> {out_val}")
    print(f"train slots: {sorted(s for s in slots['train'] if s is not None)}")
    print(f"val   slots: {sorted(s for s in slots[args.val_split_name] if s is not None)}")
    print(f"slot overlap (must be empty): {sorted(s for s in overlap if s is not None)}")
    if unknown:
        print(f"WARNING: {len(unknown)} episodes had no meta split -> defaulted to train")


if __name__ == "__main__":
    main()
