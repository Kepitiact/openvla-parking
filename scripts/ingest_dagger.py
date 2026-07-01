"""Idempotent DAgger ingest for one round (single source of truth under
data_carla/processed/).

Does everything that must happen BEFORE UniAD feature extraction, in one
re-runnable command:

  1. Round-aware backup of the canonical files (C4).
  2. Merge dagger_cached.pkl  -> cached_parking_info.pkl        (set difference)
  3. Merge dagger_infos.pkl   -> parking_infos_temporal.pkl     (set difference)
  4. Emit a dagger-ONLY infos file (patched + canonical version) so extraction can
     process just the new tokens via extract_uniad_features.py --ann-file.
  5. Register the dagger tokens in the nuScenes-format DB (rebuild v1.0-carla from
     the merged infos) so the extractor/loader can resolve them.
  6. NORMALIZE the version string to the canonical 'v1.0-carla' (a non-canonical
     'v1.0-carla-dagger' broke extraction before).
  7. Write a per-round token manifest (C4).

Re-running the same round is a no-op: the merges are set differences, the DB
rebuild is deterministic, and backups/manifests are written once.

Conversations are merged AFTER extraction (they must carry uniad_pth), so that
step stays in merge_dagger.py:
  python scripts/ingest_dagger.py --round 1
  cd OpenDriveVLA && bash scripts/extract_carla_features.sh && cd ..
  python scripts/merge_dagger.py --targets conversations

Run from the repo root in the model venv.
"""

import argparse
import json
import pickle
import shutil

import paths
import merge_dagger
from build_carla_nusc_tables import build_tables, patch_pkl

PROC = paths.PROCESSED_DIR
CANON_VERSION = paths.NUSC_VERSION  # "v1.0-carla"
ROUNDS_DIR = PROC / "dagger_rounds"

_CANON_FILES = (
    "cached_parking_info.pkl",
    "parking_infos_temporal.pkl",
    "carla_conversations.json",
)


def _round_backup(round_id):
    """Round-aware backup of the canonical files (written once per round)."""
    bdir = ROUNDS_DIR / f"round_{round_id:02d}" / "backup"
    bdir.mkdir(parents=True, exist_ok=True)
    for name in _CANON_FILES:
        src = PROC / name
        dst = bdir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  round backup: {name} -> {dst}")


def _normalize_version(infos_dict):
    md = infos_dict.setdefault("metadata", {})
    old = md.get("version")
    if old != CANON_VERSION:
        md["version"] = CANON_VERSION
        print(f"  normalized version '{old}' -> '{CANON_VERSION}'")
    return infos_dict


def emit_dagger_only_infos(round_id):
    """Write a patched, canonical-version, dagger-only infos file for extraction."""
    dag = pickle.load(open(PROC / "dagger_infos.pkl", "rb"))
    if not (isinstance(dag, dict) and "infos" in dag):
        dag = {"infos": dag}
    _normalize_version(dag)
    out = PROC / f"dagger_r{round_id:02d}_infos.pkl"
    pickle.dump(dag, open(out, "wb"))
    # Patch cam paths (absolute) + lidar stubs + canonical version, matching the
    # base build so the extractor's dataset can load these tokens.
    patch_pkl(out, paths.RAW_DIR)
    tokens = [e["token"] for e in dag["infos"]]
    print(f"  emitted dagger-only infos -> {out.name} ({len(tokens)} tokens)")
    return out, tokens


def write_token_manifest(round_id, tokens):
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    mpath = ROUNDS_DIR / f"round_{round_id:02d}_tokens.json"
    json.dump({"round": round_id, "count": len(tokens), "tokens": tokens},
              open(mpath, "w"))
    print(f"  token manifest -> {mpath.name} ({len(tokens)} tokens)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True,
                    help="DAgger round number (used for backups + token manifest).")
    ap.add_argument("--no-rebuild-db", action="store_true", default=False,
                    help="Skip rebuilding the v1.0-carla nuScenes DB from the merged infos.")
    args = ap.parse_args()

    print(f"=== Ingesting DAgger round {args.round} ===")
    _round_backup(args.round)

    print("[cached]")
    merge_dagger.merge_cached()
    print("[infos]")
    merge_dagger.merge_infos()

    print("[dagger-only infos]")
    _, tokens = emit_dagger_only_infos(args.round)

    if not args.no_rebuild_db:
        print("[register tokens in nuScenes DB]")
        # Rebuild v1.0-carla from the merged infos (now includes dagger tokens),
        # then re-patch the merged infos (idempotent: absolute cam paths, lidar
        # stubs, canonical version).
        build_tables(paths.INFOS_PKL, paths.NUSC_DB)
        patch_pkl(paths.INFOS_PKL, paths.RAW_DIR)

    print("[token manifest]")
    write_token_manifest(args.round, tokens)

    print("\nDone. Next:")
    print("  cd OpenDriveVLA && bash scripts/extract_carla_features.sh && cd ..")
    print(f"  # (or extract only this round: "
          f"--ann-file data_carla/processed/dagger_r{args.round:02d}_infos.pkl)")
    print("  python scripts/merge_dagger.py --targets conversations")


if __name__ == "__main__":
    main()
