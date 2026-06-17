import argparse
import pickle
from nuscenes.nuscenes import NuScenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/infos/nuscenes_infos_temporal_val.pkl")
    parser.add_argument("--output", default="data/infos/nuscenes_infos_temporal_mini.pkl")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        src = pickle.load(f)

    if isinstance(src, dict) and "infos" in src:
        infos = src["infos"]
        metadata = src.get("metadata", {})
    else:
        infos = src
        metadata = {}

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    mini_tokens = {s["token"] for s in nusc.sample}

    mini_infos = [x for x in infos if x.get("token") in mini_tokens]
    metadata["version"] = args.version

    out = {"infos": mini_infos, "metadata": metadata}
    with open(args.output, "wb") as f:
        pickle.dump(out, f)

    print(f"Wrote {len(mini_infos)} entries to {args.output}")


if __name__ == "__main__":
    main()
