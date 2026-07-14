"""
Build carla_conversations.json for OpenDriveVLA training/inference.

One entry per frame: {qa_id, sample_id, empty conversation}. The actual prompt
and ground-truth answer are filled in at train/inference time by
build_llava_conversation from cached_parking_info.pkl, so this file only needs
the token references. (Feature extraction later adds the `uniad_pth` path.)

Run from the repo root:
  python scripts/build_carla_conversations.py
"""

import argparse
import json
import pickle

import paths


def main():
    ap = argparse.ArgumentParser(description="Build token-only carla_conversations.json.")
    ap.add_argument("--infos", default=str(paths.INFOS_PKL))
    ap.add_argument("--out", default=str(paths.CONVERSATIONS))
    # Stamped into every entry, and train_drivevla REFUSES to train on anything that is not
    # "train". The dataset does not filter -- with use_uniad_pth it takes the conversation
    # list verbatim (nuscenes_llava_dataset.py:185) -- so a combined train+val index would
    # be trained on end to end, silently, and every eval number afterwards would be a lie.
    ap.add_argument("--split", required=True, choices=["train", "val"],
                    help="which infos this index covers; guarded at train time")
    args = ap.parse_args()

    with open(args.infos, "rb") as f:
        obj = pickle.load(f)
    infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj

    convs = [
        {
            "qa_id": f"{info['token']}_trajectory",
            "sample_id": info["token"],
            "split": args.split,
            "conversations": [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": ""},
            ],
        }
        for info in infos
    ]

    with open(args.out, "w") as f:
        json.dump(convs, f)
    print(f"Wrote {len(convs)} {args.split} conversations to {args.out}")


if __name__ == "__main__":
    main()
