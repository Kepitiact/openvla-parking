"""Swap an absolute path prefix inside a CARLA infos pkl.

Use after moving the raw images to a new location (e.g. workspace -> staging): the
baked absolute image paths in the pkl are rewritten in place, so no full rebuild is
needed and the paths still point at the real data (no symlink).

  python scripts/rewrite_infos_paths.py --pkl data_carla/processed/parking_infos_val.pkl \
      --old /workspaces/s0002438/openvla-parking/data_carla/raw \
      --new /staging/short_range_perception/s0002438/openvla_parking/data_carla/raw
"""
import argparse
import pickle


def swap(obj, old, new, counter):
    if isinstance(obj, str):
        if old in obj:
            counter[0] += 1
            return obj.replace(old, new)
        return obj
    if isinstance(obj, dict):
        return {k: swap(v, old, new, counter) for k, v in obj.items()}
    if isinstance(obj, list):
        return [swap(x, old, new, counter) for x in obj]
    if isinstance(obj, tuple):
        return tuple(swap(x, old, new, counter) for x in obj)
    return obj  # numpy arrays / numbers untouched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    counter = [0]
    d = swap(d, args.old, args.new, counter)
    with open(args.pkl, "wb") as f:
        pickle.dump(d, f)
    print(f"{args.pkl}: rewrote {counter[0]} paths  {args.old} -> {args.new}")


if __name__ == "__main__":
    main()
