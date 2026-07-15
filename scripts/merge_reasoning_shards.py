"""Merge episode-sharded reasoning output into one versioned dir.

The full teacher run is split into N shard jobs (run_generate --num-shards N --shard-idx k),
each writing a self-contained dir. This concatenates their traces and re-aggregates the
manifest, and -- critically -- VERIFIES the shards partition the data cleanly: every episode
present, no episode produced by two shards (which episode-level sharding must guarantee).

Usage:
  python scripts/merge_reasoning_shards.py \
      --shard-dirs <base>/shard00 <base>/shard01 ... \
      --out        <base>/v1_uniad_epoch6
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def _read_jsonl(p: pathlib.Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_traces = []
    manifests = []
    episodes_by_shard = {}
    expected_shards = None

    for d in sorted(args.shard_dirs):
        d = pathlib.Path(d)
        man = json.loads((d / "manifest.json").read_text())
        manifests.append(man)
        traces = _read_jsonl(d / "traces.jsonl")
        all_traces.extend(traces)
        idx = man["shard"]["shard_idx"]
        episodes_by_shard[idx] = {t["scene_token"] for t in traces}
        expected_shards = expected_shards or man["shard"]["num_shards"]
        if man["shard"]["num_shards"] != expected_shards:
            sys.exit(f"shard {d} has num_shards={man['shard']['num_shards']}, "
                     f"expected {expected_shards}")

    # 1. all shards present
    missing = set(range(expected_shards)) - set(episodes_by_shard)
    if missing:
        sys.exit(f"missing shard(s) {sorted(missing)} of {expected_shards}; do not merge a "
                 "partial run -- re-run the missing shard first.")

    # 2. episodes partition cleanly (episode-sharding's whole promise)
    seen, overlap = set(), set()
    for eps in episodes_by_shard.values():
        overlap |= (seen & eps)
        seen |= eps
    if overlap:
        sys.exit(f"{len(overlap)} episode(s) appear in more than one shard, e.g. "
                 f"{sorted(overlap)[:3]} -- sharding is not disjoint, ABORT.")

    # aggregate manifest
    fallbacks = sum(m["counts"]["teacher_fallbacks"] for m in manifests)
    total = len(all_traces)
    agg = {
        "version": out.name,
        "merged_from": [m["shard"]["shard_idx"] for m in manifests],
        "source": manifests[0]["source"],
        "teacher": manifests[0]["teacher"],
        "counts": {"frames": total, "episodes": len(seen),
                   "teacher_fallbacks": fallbacks},
        "decision_histogram": dict(sum((collections.Counter(m["decision_histogram"])
                                        for m in manifests), collections.Counter())),
        "fallback_rate": round(fallbacks / total, 4) if total else 0.0,
    }
    (out / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in all_traces) + "\n")
    (out / "manifest.json").write_text(json.dumps(agg, indent=2))

    print(f"merged {total} traces / {len(seen)} episodes from {len(manifests)} shards")
    print(f"teacher_fallbacks: {fallbacks} ({agg['fallback_rate']:.2%})")
    print(f"-> {out}/traces.jsonl")
    if fallbacks:
        print("NOTE: fallbacks > 0 -- inspect before training (grounded but templated).")


if __name__ == "__main__":
    main()
