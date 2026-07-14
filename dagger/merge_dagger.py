"""
Merge the DAgger package into the canonical CARLA dataset (single source of truth
under data_carla/processed/). ADDITIVE only: every original token is kept, the
DAgger tokens are appended.

Three independent targets (run any subset via --targets):
  cached        cached_parking_info.pkl  += dagger_cached.pkl   (dict update)
  infos         parking_infos_temporal.pkl["infos"] += dagger_infos.pkl["infos"]  (list extend)
  conversations carla_conversations.json += dagger_conversations.json  (list concat)

Ordering matters: 'cached' and 'infos' must be merged BEFORE UniAD extraction so
the extractor's dataset (ann_file = parking_infos_temporal.pkl) can resolve the
DAgger tokens. 'conversations' must be merged AFTER extraction so every DAgger
entry already carries the uniad_pth the extractor stamps on it.

Each target is backed up once (<file>.bak_predagger) before the first write and
the merge is refused if the DAgger tokens are already present (idempotent guard).

Run from the repo root in the model venv:
  python scripts/merge_dagger.py --targets cached infos
  # ... extract + slim ...
  python scripts/merge_dagger.py --targets conversations
"""

import argparse
import json
import pickle
import shutil

import paths

PROC = paths.PROCESSED_DIR
BAK = ".bak_predagger"


def _backup(path):
    bak = path.with_name(path.name + BAK)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backed up {path.name} -> {bak.name}")
    else:
        print(f"  backup already exists: {bak.name} (left as-is)")


def merge_cached():
    orig_p = PROC / "cached_parking_info.pkl"
    dag_p = PROC / "dagger_cached.pkl"
    orig = pickle.load(open(orig_p, "rb"))
    dag = pickle.load(open(dag_p, "rb"))
    assert isinstance(orig, dict) and isinstance(dag, dict)

    # Set difference: add only the dagger keys not already present. Idempotent,
    # and survives a cumulative round-N package (which re-includes earlier rounds)
    # without duplicating or clobbering.
    new_keys = set(dag) - set(orig)
    if not new_keys:
        print(f"  SKIP cached: all {len(dag)} dagger keys already present. No-op.")
        return
    _backup(orig_p)
    orig.update({k: dag[k] for k in new_keys})
    pickle.dump(orig, open(orig_p, "wb"))
    print(f"  cached: wrote {len(orig)} tokens (+{len(new_keys)} new dagger)")


def merge_infos():
    orig_p = PROC / "parking_infos_temporal.pkl"
    dag_p = PROC / "dagger_infos.pkl"
    orig = pickle.load(open(orig_p, "rb"))
    dag = pickle.load(open(dag_p, "rb"))
    assert isinstance(orig, dict) and "infos" in orig
    dag_infos = dag["infos"] if isinstance(dag, dict) else dag

    orig_tokens = {e["token"] for e in orig["infos"]}
    new = [e for e in dag_infos if e["token"] not in orig_tokens]
    if not new:
        print(f"  SKIP infos: all {len(dag_infos)} dagger tokens already present. No-op.")
        return
    _backup(orig_p)
    orig["infos"].extend(new)
    pickle.dump(orig, open(orig_p, "wb"))
    print(f"  infos: wrote {len(orig['infos'])} entries (+{len(new)} new dagger)")


def merge_conversations():
    orig_p = PROC / "carla_conversations.json"
    dag_p = PROC / "dagger_conversations.json"
    orig = json.load(open(orig_p))
    dag = json.load(open(dag_p))
    assert isinstance(orig, list) and isinstance(dag, list)

    orig_ids = {e["sample_id"] for e in orig}
    new = [e for e in dag if e["sample_id"] not in orig_ids]
    if not new:
        print(f"  SKIP conversations: all {len(dag)} dagger ids already present. No-op.")
        return

    # Every NEW dagger entry must already carry uniad_pth (stamped by the
    # extractor), otherwise training would reference features that don't exist.
    missing = [e["sample_id"] for e in new if not e.get("uniad_pth")]
    if missing:
        raise SystemExit(
            f"  ABORT conversations: {len(missing)} new dagger entries lack 'uniad_pth' "
            f"(e.g. {missing[0]}). Run UniAD extraction first.")

    _backup(orig_p)
    merged = orig + new
    json.dump(merged, open(orig_p, "w"))
    print(f"  conversations: wrote {len(merged)} entries (+{len(new)} new dagger)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+",
                    choices=["cached", "infos", "conversations"],
                    default=["cached", "infos", "conversations"])
    args = ap.parse_args()

    fns = {"cached": merge_cached, "infos": merge_infos,
           "conversations": merge_conversations}
    for t in args.targets:
        print(f"[{t}]")
        fns[t]()


if __name__ == "__main__":
    main()
