"""CLI: infos pkl + source + teacher + version -> versioned reasoning traces + manifest.

  python -m reasoning_data_gen.run_generate \
      --infos data_carla/processed/parking_infos_train.pkl \
      --source gt --teacher mock --limit 20 \
      --out data_carla/processed/reasoning/v0_dryrun/

Writes, under the (versioned, never-overwritten) --out dir:
  traces.jsonl          one factual trace record per frame
  counterfactuals.jsonl one stop-injection pair per eligible frame (shared pair_id)
  manifest.json         counts, version, source, teacher, thresholds, histograms, rates

The GT infos are the Step-0 stand-in for perception. Swapping --source uniad calls
SceneRecord.from_uniad(), which raises until Step 3 — proving the source adapter is
the only thing that changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import pickle
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

# scripts/paths.py is the single source of truth for dataset paths.
_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import paths  # noqa: E402

from . import reconcile as reconcile_mod
from . import validators as V
from .counterfactual import make_stop_injection
from .fact_extractor import DEFAULT_THRESHOLDS, Thresholds, extract_fact
from .scene_record import DetectedObject, Frame, SceneRecord, _global_to_ego_rf, _norm_angle
from .schema import FactRecord, format_trajectory, render_assistant_turn
from .verbalizer import MockVerbalizer, Verbalizer, get_verbalizer

COORD_CONVENTION = "ego frame: x=right(+right), y=forward(+forward); dheading rad in [-pi,pi]"


# ── infos indexing + causal-local geometry ────────────────────────────────────
def _load_infos(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj


def _index(infos: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {i["token"]: i for i in infos}


def _shard_by_episode(infos: List[Dict[str, Any]], num_shards: int, shard_idx: int
                      ) -> List[Dict[str, Any]]:
    """Keep only the WHOLE episodes assigned to this shard. Sharding by episode (not by
    frame index) is what makes parallel generation safe: prev_reverse -- the gear-change
    signal -- is looked up within an episode, so splitting an episode would corrupt
    shift_gear detection at the boundary. Deterministic: episodes are sorted, then split
    into num_shards contiguous groups, so shard k is the same set on every run."""
    if not (0 <= shard_idx < num_shards):
        raise SystemExit(f"--shard-idx {shard_idx} out of range for --num-shards {num_shards}")
    episodes = sorted({i["scene_token"] for i in infos})
    keep = set(np.array_split(np.array(episodes, dtype=object), num_shards)[shard_idx])
    return [i for i in infos if i["scene_token"] in keep]


def _load_uniad_pth(features_dir: str, token: str):
    """Load one frame's decoded-UniAD .pth (from extract_uniad_features.py). Returns
    None when the file is absent — treated as 'perception saw nothing this frame',
    which reconcile then flags against GT. torch is imported lazily so the gt/mock
    path stays torch-free."""
    p = pathlib.Path(features_dir) / f"{token}.pth"
    if not p.exists():
        return None
    import torch  # lazy: only the uniad source needs it
    return torch.load(p, map_location="cpu", weights_only=False)


def _same_scene_next(info, by_token):
    nt = info.get("next")
    if not nt:
        return None
    nb = by_token.get(nt)
    if nb is None or nb.get("scene_token") != info.get("scene_token"):
        return None
    return nb


def _same_scene_prev(info, by_token):
    pt = info.get("prev")
    if not pt:
        return None
    pb = by_token.get(pt)
    if pb is None or pb.get("scene_token") != info.get("scene_token"):
        return None
    return pb


def _future_traj(info, by_token, steps: int = 6) -> np.ndarray:
    """Ego future trajectory (steps x 3) in the current ego frame, from the infos
    chain (no nuScenes DB). Pads by repeating the last point at scene end — matching
    collect_future_local. This is an OUTPUT label, never fed to the reasoner."""
    origin = np.asarray(info["ego2global_translation"], dtype=np.float64)
    rot = Quaternion(info["ego2global_rotation"])
    cur_yaw = rot.yaw_pitch_roll[0]
    pts: List[np.ndarray] = []
    cur = info
    for _ in range(steps):
        nxt = _same_scene_next(cur, by_token)
        if nxt is None:
            pts.append(pts[-1].copy() if pts else np.zeros(3))
            continue
        r, f = _global_to_ego_rf(nxt["ego2global_translation"][:2], origin, rot)
        dh = _norm_angle(Quaternion(nxt["ego2global_rotation"]).yaw_pitch_roll[0] - cur_yaw)
        pts.append(np.array([r, f, dh], dtype=np.float64))
        cur = nxt
    return np.asarray(pts, dtype=np.float64).reshape(steps, 3)


def _history(info, by_token, steps: int = 4) -> List[Tuple[float, float]]:
    """Past-2s ego positions (right, forward) in the current ego frame, oldest-first.
    Causal-local: only past frames."""
    origin = np.asarray(info["ego2global_translation"], dtype=np.float64)
    rot = Quaternion(info["ego2global_rotation"])
    past: List[Tuple[float, float]] = []
    cur = info
    for _ in range(steps):
        prev = _same_scene_prev(cur, by_token)
        if prev is None:
            break
        past.append(_global_to_ego_rf(prev["ego2global_translation"][:2], origin, rot))
        cur = prev
    return past[::-1]


# ── record assembly ───────────────────────────────────────────────────────────
def _make_record(token, role, pair_id, source, teacher, fact: FactRecord,
                 trace: str, traj: np.ndarray, scene: SceneRecord,
                 reconcile_status: str, synthetic_tokens: List[Dict[str, Any]]
                 ) -> Dict[str, Any]:
    traj_str = format_trajectory(traj.tolist())
    assistant_turn = render_assistant_turn(fact, trace, traj_str)
    record = {
        "token": token,
        "scene_token": scene.scene_token,
        "frame_idx": scene.frame_idx,
        "pair_id": pair_id,
        "role": role,
        "source": source,
        "teacher": teacher,
        "decision": fact.decision,
        "causal_factors": [ef.to_dict() for ef in fact.causal_factors],
        "trace": trace,
        "trajectory": traj.tolist(),
        "traj_str": traj_str,
        "assistant_turn": assistant_turn,
        "synthetic_tokens": synthetic_tokens,
        "reconcile": reconcile_status,
        "meta": fact.meta,
    }
    record["validators"] = {
        "entity_fidelity": V.entity_fidelity(trace, fact),
        "action_fidelity": V.action_fidelity(fact.decision, traj),
        "causal_locality": V.causal_locality(fact, scene),
        "schema_conformance": V.schema_conformance(record),
    }
    return record


def _scene_with_injected(scene: SceneRecord, spec: Dict[str, Any]) -> SceneRecord:
    """Copy of `scene` with the synthetic obstacle added, so a counterfactual
    record's causal_locality resolves the injected entity (Step 3 adds its token)."""
    obj = DetectedObject(
        cls=spec["cls"], right=spec["right"], forward=spec["forward"],
        yaw=spec.get("yaw", 0.0), size=tuple(spec.get("size", (0.6, 0.6, 1.7))),
        vel=tuple(spec.get("vel", (0.0, 0.0))), track_id=spec.get("track_id"),
    )
    return dataclasses.replace(scene, objects=list(scene.objects) + [obj])


# ── main generation ───────────────────────────────────────────────────────────
def generate(
    infos_path: str,
    source: str,
    teacher: str,
    out_dir: pathlib.Path,
    limit: Optional[int] = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    verbalizer: Optional[Verbalizer] = None,
    uniad_features_dir: Optional[str] = None,
    uniad_score_thr: float = 0.3,
    counterfactuals: bool = False,
    num_shards: int = 1,
    shard_idx: int = 0,
    **teacher_kwargs,
) -> Dict[str, Any]:
    out_dir = pathlib.Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite existing version dir {out_dir} (non-empty). "
            "Pick a new --out version.")
    version = out_dir.name  # dir is created only just before writing (below), so a
                            # failed run (e.g. --source uniad) leaves nothing behind.

    infos = _load_infos(infos_path)
    by_token = _index(infos)              # FULL index: history/prev lookups must resolve
                                          # even when this job only generates one shard.
    if num_shards > 1:
        infos = _shard_by_episode(infos, num_shards, shard_idx)
    if limit is not None:
        infos = infos[:limit]

    if source == "uniad" and not uniad_features_dir:
        raise SystemExit(
            "--source uniad requires --uniad-features-dir (a dir of <token>.pth files "
            "produced by drivevla/extract_uniad_features.py). Run that first on HAL.")

    verb = verbalizer or get_verbalizer(teacher, **teacher_kwargs)
    fallback_verb = MockVerbalizer()

    def verbalize_or_fall_back(f) -> tuple:
        """Never let ONE frame kill the run. Three separate 32B jobs have now died at a
        single unverbalizable frame, and on the 62k-frame run that is hours of A100 thrown
        away at frame 40,000 with nothing written. The mock is grounded by construction
        (it only ever names what is in the fact), so a fallback is safe -- but it is a
        template, so it must never be silent: every fallback is tagged in the record and
        counted in the manifest. If that count is not ~0, the teacher has a real problem
        and we look at it BEFORE training on the output."""
        try:
            return verb.verbalize(f), teacher
        except RuntimeError as e:
            fallback_reasons.append(str(e).split("\n")[0])
            return fallback_verb.verbalize(f), f"{teacher}_fallback_mock"

    trace_records: List[Dict[str, Any]] = []
    cf_pairs: List[Dict[str, Any]] = []
    decision_hist: Counter = Counter()
    reconcile_hist: Counter = Counter()
    fallback_reasons: List[str] = []
    missing_uniad = 0

    for info in infos:
        history = _history(info, by_token)
        prev = _same_scene_prev(info, by_token)
        prev_reverse = bool(prev["reverse"]) if prev is not None else None

        # The upcoming waypoints. This is the OUTPUT label; it is also handed to
        # extract_fact as `planned_path`, used ONLY to select which currently-perceived
        # object lies in the ego's swept arc (see fact_extractor.swept_path_obstacle).
        # The cited entity is always a current-frame detection, so causal_locality holds.
        traj = _future_traj(info, by_token)

        # Source adapter — the single swap point. GT is the Step-0 stand-in; UniAD is
        # real perception (objects from decoded detections, ego/command still from info).
        if source == "gt":
            scene = SceneRecord.from_gt(info, history=history, prev_reverse=prev_reverse)
            fact = extract_fact(scene, thresholds, planned_path=traj)
            # source IS gt, so the gt-side fact == the perception fact.
            status = reconcile_mod.reconcile(fact, fact)
        elif source == "uniad":
            decoded = _load_uniad_pth(uniad_features_dir, info["token"])
            if decoded is None:
                missing_uniad += 1
            scene = SceneRecord.from_uniad(decoded, info, history=history,
                                           prev_reverse=prev_reverse,
                                           score_thr=uniad_score_thr)
            fact = extract_fact(scene, thresholds, planned_path=traj)
            # Reconcile perception vs GT so limited/contradiction frames can be dropped.
            gt_scene = SceneRecord.from_gt(info, history=history, prev_reverse=prev_reverse)
            gt_fact = extract_fact(gt_scene, thresholds, planned_path=traj)
            status = reconcile_mod.reconcile(fact, gt_fact)
        else:
            raise ValueError(f"unknown source {source!r}; expected 'gt' or 'uniad'")

        trace, used_teacher = verbalize_or_fall_back(fact)

        record = _make_record(info["token"], "factual", info["token"], source, used_teacher,
                              fact, trace, traj, scene, status, synthetic_tokens=[])
        trace_records.append(record)
        decision_hist[fact.decision] += 1
        reconcile_hist[status] += 1

        # Counterfactual stop-injection (skipped when the ego is barely moving).
        # OFF by default: counterfactuals are deferred to v2 (real pedestrian episodes beat
        # synthetic injection, and nothing injects the synthetic obstacle into the UniAD
        # tokens anyway -- see FUTURE_WORK 2). Generating them is not free: the teacher must
        # verbalize a stop_yield it has no grounding for, and one unverbalizable frame kills
        # the whole run. Do not pay that for data we do not train on.
        if not counterfactuals:
            continue
        frame = Frame(token=info["token"], scene=scene, fact=fact, trajectory=traj)
        pair = make_stop_injection(frame)
        if pair is not None:
            cf_scene = _scene_with_injected(scene, pair.synthetic_token_spec)
            fac_rec = _make_record(info["token"], "factual", pair.pair_id, source, teacher,
                                   pair.factual_fact, trace, pair.factual_traj, scene,
                                   status, synthetic_tokens=[])
            cf_trace, _ = verbalize_or_fall_back(pair.counterfactual_fact)
            cf_rec = _make_record(info["token"], "counterfactual", pair.pair_id, source,
                                  teacher, pair.counterfactual_fact, cf_trace,
                                  pair.counterfactual_traj, cf_scene, status,
                                  synthetic_tokens=[pair.synthetic_token_spec])
            cf_pairs.append({
                "pair_id": pair.pair_id,
                "token": info["token"],
                "factual": fac_rec,
                "counterfactual": cf_rec,
                "synthetic_token_spec": pair.synthetic_token_spec,
                "injection": pair.injection,
            })

    # Validator pass rates over factual frames.
    n = len(trace_records)
    pass_rates = {}
    for name, thr in V.PASS.items():
        if n == 0:
            pass_rates[name] = 1.0
        else:
            pass_rates[name] = sum(1 for r in trace_records
                                   if r["validators"][name] >= thr) / n

    manifest = {
        "version": version,
        "source": source,
        "teacher": teacher,
        "infos": str(infos_path),
        "limit": limit,
        "coordinate_convention": COORD_CONVENTION,
        "shard": {"num_shards": num_shards, "shard_idx": shard_idx,
                  "episodes": len({r["scene_token"] for r in trace_records})},
        "counts": {"frames": n, "counterfactual_pairs": len(cf_pairs),
                   # Frames the teacher could not verbalize cleanly, which fell back to the
                   # (template) mock. Should be ~0. If it is not, READ fallback_examples
                   # before training on this data -- the traces are grounded but templated.
                   "teacher_fallbacks": len(fallback_reasons)},
        "fallback_examples": fallback_reasons[:5],
        "thresholds": dataclasses.asdict(thresholds),
        "decision_histogram": dict(decision_hist),
        "reconcile_histogram": dict(reconcile_hist),
        "validator_pass_rates": pass_rates,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if source == "uniad":
        manifest["uniad"] = {
            "features_dir": uniad_features_dir,
            "score_thr": uniad_score_thr,
            "missing_pth_frames": missing_uniad,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "traces.jsonl", trace_records)
    _write_jsonl(out_dir / "counterfactuals.jsonl", cf_pairs)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _write_jsonl(path: pathlib.Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infos", default=str(paths.PROCESSED_DIR / "parking_infos_train.pkl"))
    ap.add_argument("--source", choices=["gt", "uniad"], default="gt")
    ap.add_argument("--teacher", choices=["mock", "qwen"], default="mock")
    ap.add_argument("--out", required=True, help="versioned output dir (never overwritten)")
    ap.add_argument("--limit", type=int, default=None)
    # UniAD source (real perception): dir of <token>.pth from extract_uniad_features.py.
    ap.add_argument("--uniad-features-dir", default=None)
    ap.add_argument("--uniad-score-thr", type=float, default=0.3,
                    help="drop UniAD detections below this confidence")
    ap.add_argument("--counterfactuals", action="store_true",
                    help="also generate stop-injection pairs. OFF by default: deferred to "
                         "v2, not fed to training, and one unverbalizable frame kills the run")
    # Episode-level sharding for the full 62k run: launch N jobs, one per shard, into
    # sibling out dirs, then scripts/merge_reasoning_shards.py. Whole episodes only.
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    # QwenVerbalizer config (unused for --teacher mock); drops into an sbatch wrapper.
    ap.add_argument("--qwen-model-path", default=None)
    ap.add_argument("--qwen-endpoint", default=None)
    ap.add_argument("--qwen-batch-size", type=int, default=None)
    args = ap.parse_args(argv)

    teacher_kwargs = {}
    if args.teacher == "qwen":
        teacher_kwargs = {
            "model_path": args.qwen_model_path,
            "endpoint": args.qwen_endpoint,
            "batch_size": args.qwen_batch_size,
        }

    manifest = generate(
        infos_path=args.infos,
        source=args.source,
        teacher=args.teacher,
        out_dir=pathlib.Path(args.out),
        limit=args.limit,
        uniad_features_dir=args.uniad_features_dir,
        uniad_score_thr=args.uniad_score_thr,
        counterfactuals=args.counterfactuals,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
        **teacher_kwargs,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
