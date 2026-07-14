"""Do the perception streams actually carry information? A linear probe, per stream.

The reason gate forces a stream through the reasoning bottleneck. Gating a stream that
carries nothing is not an experiment, it is a coin flip -- so before turning on
REASON_GATE=track,map,scene we check each stream against a target only that stream could
possibly explain:

  weather   (10 CARLA presets)  -- appearance ONLY. No box, no map polygon, and no ego
                                   state encodes whether it is raining. If <SCENE> predicts
                                   it, <SCENE> is carrying real visual appearance, not just
                                   geometry re-derived from the cameras.
  occupancy (how full the lot is) -- geometry. <TRACK> should get this.

Both labels are free: CARLA wrote them into meta.json:collection_config, and neither has
ever entered the model. So this costs a couple of minutes of CPU and no training.

TWO THINGS THAT WOULD MAKE THE NUMBER A LIE, both guarded here:

1. Frame-level splits. Frames inside one episode are near-duplicate views of one lot under
   one weather preset, so a frame-level split leaks the answer and any probe scores ~1.0 by
   memorising the episode. We split by EPISODE (GroupKFold on scene_token).

2. Episode identity leaking through a stream that cannot see appearance. Hence the CONTROL:
   we probe <TRACK> (object queries -- geometry) for WEATHER too. Track has no legitimate
   path to weather, so it should sit at chance. If track predicts weather as well as scene
   does, the probe is reading lot layout, not rain, and the scene number means nothing.

Read the control before the result.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def stream_vectors(pth: dict) -> dict:
    """One fixed-length vector per stream, pooled the same way the projectors see them."""
    out = {}

    # <SCENE>: img_feat_2D [1,6,256,15,25] -> per-camera spatial mean -> 6*256.
    # Keep the cameras separate: "wet road" lives in the down-facing views, glare in the
    # front one -- averaging the 6 together would blur exactly the signal we are testing for.
    scene = pth["result_track"]["img_feat_2D"].float()      # [1,6,256,15,25]
    out["scene"] = scene.mean(dim=(-2, -1)).flatten().numpy()   # [1536]

    # <TRACK>: track query embeddings [N,256] -> mean. N varies per frame, and a frame
    # that tracked nothing carries None, not an empty tensor.
    trk = pth["result_track"]["track_query_embeddings"]
    out["track"] = (trk.float().mean(0).numpy() if trk is not None and trk.numel()
                    else np.zeros(256, dtype=np.float32))

    # <MAP>: seg queries.
    seg = pth["result_seg"]
    things = seg["chosen_output_query_things"].float()       # [64,256]
    out["map"] = things.mean(0).numpy()
    return out


def load(features_dir: str, raw_dir: str, stride: int = 1):
    files = sorted(glob.glob(os.path.join(features_dir, "*.pth")))
    if not files:
        raise SystemExit(f"no .pth under {features_dir}")
    # The probe is EPISODE-limited, not frame-limited: what it needs is many episodes per
    # weather class, and consecutive frames of one episode are near-duplicates that add
    # almost nothing. Striding keeps every episode while cutting the 78 GB read by `stride`.
    files = files[::stride]

    meta_cache: dict = {}
    X = collections.defaultdict(list)
    weather, occupancy, groups = [], [], []

    for f in files:
        ep = os.path.basename(f).split("_f")[0]
        if ep not in meta_cache:
            mpath = os.path.join(raw_dir, ep, "meta.json")
            if not os.path.exists(mpath):
                continue
            meta_cache[ep] = json.load(open(mpath))["collection_config"]
        cc = meta_cache[ep]

        for k, v in stream_vectors(torch.load(f, map_location="cpu")).items():
            X[k].append(v)
        weather.append(cc["weather"])
        occupancy.append(cc["occupancy"])
        groups.append(ep)

    return ({k: np.stack(v) for k, v in X.items()},
            np.array(weather), np.array(occupancy), np.array(groups))


def probe(X, y, groups, n_splits: int) -> tuple[float, float]:
    """Episode-grouped CV accuracy, and the majority-class baseline to beat."""
    n_groups = len(set(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        return float("nan"), float("nan")

    preds = np.empty(len(y), dtype=y.dtype)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[tr])) < 2:          # a fold with one class cannot train
            preds[te] = y[tr][0]
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict(sc.transform(X[te]))

    acc = float((preds == y).mean())
    majority = float(collections.Counter(y).most_common(1)[0][1] / len(y))
    return acc, majority


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data_carla/processed/uniad_features")
    ap.add_argument("--raw-dir", default="data_carla/raw")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth frame (episodes are all still covered)")
    args = ap.parse_args()

    X, weather, occupancy, groups = load(args.features_dir, args.raw_dir, args.stride)
    n_ep = len(set(groups))
    print(f"{len(groups)} frames / {n_ep} episodes / {len(set(weather))} weather classes\n")

    if n_ep < 20:
        print("!! WARNING: too few episodes for a trustworthy number. Episode-grouped CV\n"
              "!! with a handful of episodes per class is dominated by which episode landed\n"
              "!! in which fold. Treat this run as a MACHINERY check; rerun on the full\n"
              "!! extraction (1508 episodes) for the real verdict.\n")

    for target, y in (("WEATHER (appearance)", weather),
                      ("OCCUPANCY (geometry)", occupancy)):
        chance = 1.0 / len(set(y))
        print(f"--- {target}: {len(set(y))} classes, chance {chance:.2f}")
        for stream in ("scene", "track", "map"):
            acc, majority = probe(X[stream], y.astype(str), groups, args.n_splits)
            tag = ""
            if target.startswith("WEATHER"):
                tag = "  <- the result" if stream == "scene" else "  <- CONTROL: expect ~chance"
            print(f"   {stream:6s} acc {acc:.3f}   (majority baseline {majority:.3f}){tag}")
        print()


if __name__ == "__main__":
    main()
