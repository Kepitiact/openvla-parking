"""Score UniAD predictions against val GT — CARLA-native, no nuScenes toolkit.

Greedy match (per class, high-score first) of predicted boxes to GT by BEV center
distance, then report detection rate + localization error. Pure numpy — runs in the
plain venv (no mmdet3d), so scoring iterates freely off a dumped predictions pkl.

  python scripts/uniad_stage1_metrics.py \
      --preds  checkpoints/stage1_carla_full/preds_epoch2.pkl \
      --infos  data_carla/processed/parking_infos_val.pkl \
      [--dist-thr 2.0 --score-thr 0.3]
"""
import argparse
import pickle
from collections import defaultdict

import numpy as np


def _yaw_err(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--infos", required=True)
    ap.add_argument("--dist-thr", type=float, default=2.0,
                    help="a prediction matches a GT if BEV center distance <= this (m)")
    ap.add_argument("--score-thr", type=float, default=0.3,
                    help="ignore predictions below this confidence")
    args = ap.parse_args()

    P = pickle.load(open(args.preds, "rb"))
    class_names, preds = P["class_names"], P["preds"]
    d = pickle.load(open(args.infos, "rb"))
    infos = d["infos"] if isinstance(d, dict) else d
    by_token = {r["token"]: r for r in infos}

    TP, FP, FN = defaultdict(int), defaultdict(int), defaultdict(int)
    cerr, serr, yerr = defaultdict(list), defaultdict(list), defaultdict(list)
    frames = 0

    for fr in preds:
        info = by_token.get(fr["token"])
        if info is None:
            continue
        frames += 1
        keep = fr["scores"] >= args.score_thr
        pb, pl, ps = fr["boxes"][keep], fr["labels"][keep], fr["scores"][keep]
        order = np.argsort(-ps)            # high score first (greedy matching order)
        pb, pl = pb[order], pl[order]

        gb = np.asarray(info["gt_boxes"], dtype=np.float32)
        gn = np.asarray(info["gt_names"])
        vf = np.asarray(info.get("valid_flag", np.ones(len(gb), bool))).astype(bool)
        gb, gn = gb[vf], gn[vf]

        for ci, cname in enumerate(class_names):
            g = gb[gn == cname]            # (G,7) GT of this class
            p = pb[pl == ci]               # (Pc,7) preds of this class, score-sorted
            matched = np.zeros(len(g), dtype=bool)
            for k in range(len(p)):
                if len(g) == 0:
                    FP[cname] += 1
                    continue
                dist = np.linalg.norm(g[:, :2] - p[k, :2], axis=1)
                dist[matched] = 1e9
                j = int(np.argmin(dist))
                if dist[j] <= args.dist_thr:
                    matched[j] = True
                    TP[cname] += 1
                    cerr[cname].append(dist[j])
                    serr[cname].append(np.abs(g[j, 3:6] - p[k, 3:6]).mean())
                    yerr[cname].append(_yaw_err(g[j, 6], p[k, 6]))
                else:
                    FP[cname] += 1
            FN[cname] += int((~matched).sum())

    print(f"\n=== UniAD detection eval | {frames} frames | "
          f"dist_thr={args.dist_thr} m, score_thr={args.score_thr} ===")
    hdr = (f"{'class':<16}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
           f"{'recall':>8}{'prec':>8}{'cErr m':>8}{'sErr m':>8}{'yErr °':>8}")
    print(hdr)
    print("-" * len(hdr))
    tot = dict(gt=0, tp=0, fp=0, fn=0)
    for ci, cname in enumerate(class_names):
        gt = TP[cname] + FN[cname]
        if gt == 0 and (TP[cname] + FP[cname]) == 0:
            continue
        rec = TP[cname] / gt if gt else 0.0
        prec = TP[cname] / (TP[cname] + FP[cname]) if (TP[cname] + FP[cname]) else 0.0
        ce = np.mean(cerr[cname]) if cerr[cname] else float("nan")
        se = np.mean(serr[cname]) if serr[cname] else float("nan")
        ye = np.degrees(np.mean(yerr[cname])) if yerr[cname] else float("nan")
        print(f"{cname:<16}{gt:>6}{TP[cname]:>6}{FP[cname]:>6}{FN[cname]:>6}"
              f"{rec:>8.3f}{prec:>8.3f}{ce:>8.3f}{se:>8.3f}{ye:>8.1f}")
        tot["gt"] += gt; tot["tp"] += TP[cname]; tot["fp"] += FP[cname]; tot["fn"] += FN[cname]
    if tot["gt"]:
        rec = tot["tp"] / tot["gt"]
        prec = tot["tp"] / (tot["tp"] + tot["fp"]) if (tot["tp"] + tot["fp"]) else 0.0
        print("-" * len(hdr))
        print(f"{'ALL':<16}{tot['gt']:>6}{tot['tp']:>6}{tot['fp']:>6}{tot['fn']:>6}"
              f"{rec:>8.3f}{prec:>8.3f}")


if __name__ == "__main__":
    main()
