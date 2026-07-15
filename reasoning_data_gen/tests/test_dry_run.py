"""Step-0 dry run: 20 GT-placeholder frames end-to-end with the MockVerbalizer.

Covers the acceptance criteria:
  * 20 frames -> 20 fact records + traces + >=1 counterfactual pair
  * 100% schema_conformance / entity_fidelity / causal_locality (by construction)
  * deterministic fact extraction (same input twice -> identical FactRecord)
  * from_uniad is a documented stub (raises)
  * render_assistant_turn is the exact Step-5 assistant-turn string
  * manifest.json carries version + reconcile histogram + validator rates
Runs on CPU, no network, no GPU, no nuScenes DB.
"""

import json
import pickle
from pathlib import Path

import pytest

from reasoning_data_gen import reconcile as R
from reasoning_data_gen import validators as V
from reasoning_data_gen.counterfactual import make_stop_injection
from reasoning_data_gen.fact_extractor import extract_fact
from reasoning_data_gen.run_generate import _future_traj, _index, _load_infos, generate
from reasoning_data_gen.scene_record import Frame, SceneRecord
from reasoning_data_gen.schema import (
    DECISIONS,
    MANNER_NAMES,
    REASON_END,
    REASON_START,
    TRAJ_END,
    TRAJ_START,
    render_assistant_turn,
)
from reasoning_data_gen.verbalizer import MockVerbalizer, find_hallucinations

FIXTURE = Path(__file__).parent / "fixtures" / "mini_infos.pkl"


def _infos():
    return _load_infos(str(FIXTURE))


def test_dry_run_end_to_end(tmp_path):
    out = tmp_path / "v0_test"
    manifest = generate(str(FIXTURE), source="gt", teacher="mock", out_dir=out)

    assert manifest["counts"]["frames"] == 20
    # Counterfactuals are OFF by default (deferred to v2, never fed to training). Asserting
    # 0 here is the point: an unverbalizable stop_yield killed the whole 32B teacher run,
    # and we do not pay that cost for data we do not train on.
    assert manifest["counts"]["counterfactual_pairs"] == 0

    traces = [json.loads(l) for l in (out / "traces.jsonl").read_text().splitlines()]
    assert len(traces) == 20

    # 100% gates guaranteed by construction with the mock teacher.
    for name in ("schema_conformance", "entity_fidelity", "causal_locality"):
        assert manifest["validator_pass_rates"][name] == 1.0, name

    for r in traces:
        assert r["decision"] in DECISIONS
        for v in ("schema_conformance", "entity_fidelity", "causal_locality"):
            assert r["validators"][v] == 1.0

    # manifest completeness
    assert manifest["version"] == "v0_test"
    assert manifest["source"] == "gt" and manifest["teacher"] == "mock"
    assert "reconcile_histogram" in manifest and "decision_histogram" in manifest
    assert "thresholds" in manifest
    assert (out / "counterfactuals.jsonl").exists()


def test_grounded_manner_facts_present_and_clean(tmp_path):
    """Q2/Q3 enrichment: grounded qualitative manner facts appear, and adding them does
    NOT break entity/causal/schema fidelity (manner carries no mention key, no coords)."""
    out = tmp_path / "v0_manner"
    manifest = generate(str(FIXTURE), source="gt", teacher="mock", out_dir=out)
    traces = [json.loads(l) for l in (out / "traces.jsonl").read_text().splitlines()]
    manner = [e for r in traces for e in r["causal_factors"] if e["kind"] == "manner"]
    assert manner, "no grounded manner facts were produced"
    assert {m["name"] for m in manner} <= MANNER_NAMES
    for name in ("entity_fidelity", "causal_locality", "schema_conformance"):
        assert manifest["validator_pass_rates"][name] == 1.0, name


def test_counterfactuals_are_opt_in(tmp_path):
    """The machinery still works when explicitly asked for -- it is deferred, not deleted."""
    out = tmp_path / "v0_cf"
    manifest = generate(str(FIXTURE), source="gt", teacher="mock", out_dir=out,
                        counterfactuals=True)
    assert manifest["counts"]["counterfactual_pairs"] >= 1


def test_refuses_overwrite(tmp_path):
    out = tmp_path / "v0_test"
    generate(str(FIXTURE), source="gt", teacher="mock", out_dir=out)
    with pytest.raises(SystemExit):
        generate(str(FIXTURE), source="gt", teacher="mock", out_dir=out)


def test_fact_extraction_deterministic():
    infos = _infos()
    by_token = _index(infos)
    scenes = [SceneRecord.from_gt(i) for i in infos]
    facts_a = [extract_fact(s).to_dict() for s in scenes]
    facts_b = [extract_fact(SceneRecord.from_gt(i)).to_dict() for i in infos]
    assert facts_a == facts_b


def _write_uniad_pth(infos, out_dir):
    """Synthesize per-frame decoded-UniAD .pth files (extract_uniad_features format)
    from the fixture's GT boxes, so the full --source uniad path is testable offline.
    boxes[N,9] = gt_boxes[N,7] (ego x,y,z,w,l,h,yaw) + gt_velocity[N,2] (vx,vy)."""
    import numpy as np
    import torch

    from reasoning_data_gen.scene_record import UNIAD_CLASS_NAMES

    out_dir.mkdir(parents=True, exist_ok=True)
    for info in infos:
        gb = np.asarray(info["gt_boxes"], dtype=float).reshape(-1, 7)
        gv = np.asarray(info["gt_velocity"], dtype=float).reshape(-1, 2)
        boxes = np.concatenate([gb, gv], axis=1) if len(gb) else np.zeros((0, 9))
        labels = [UNIAD_CLASS_NAMES.index(str(n)) if str(n) in UNIAD_CLASS_NAMES else 0
                  for n in info.get("gt_names", [])]
        det = {"boxes": torch.tensor(boxes),
               "scores": torch.ones(len(boxes)),
               "labels": torch.tensor(labels, dtype=torch.long)}
        torch.save({"result_track": {"detections": det}},
                   out_dir / f"{info['token']}.pth")


def test_from_uniad_conversion_matches_schema():
    import numpy as np

    info = _infos()[0]
    decoded = {"result_track": {"detections": {
        "boxes": np.array([
            [3.0, -1.0, 0.0, 1.8, 4.5, 1.6, 0.1, 0.0, 0.0],   # car, y=-1(left) -> right=+1
            [-2.0, 2.0, 0.0, 0.6, 0.6, 1.7, 0.0, 0.0, 0.0],   # pedestrian, behind-left
            [5.0, 0.0, 0.0, 1.8, 4.5, 1.6, 0.0, 0.0, 0.0],    # low score -> dropped
        ], dtype=float),
        "scores": np.array([0.9, 0.8, 0.1]),
        "labels": np.array([0, 8, 0]),   # car, pedestrian, car
    }}}
    scene = SceneRecord.from_uniad(decoded, info, score_thr=0.3)
    assert scene.source == "uniad"
    assert len(scene.objects) == 2                       # 3rd dropped by score
    assert scene.objects[0].cls == "car"
    assert abs(scene.objects[0].right - 1.0) < 1e-6      # right = -y = +1
    assert abs(scene.objects[0].forward - 3.0) < 1e-6    # forward = x = 3
    assert scene.objects[1].cls == "pedestrian"
    assert scene.command.maneuver_type == info["maneuver_type"]  # command from info
    assert extract_fact(scene).decision in DECISIONS


def test_from_uniad_empty_perception():
    scene = SceneRecord.from_uniad(None, _infos()[0])
    assert scene.objects == [] and scene.source == "uniad"


def test_uniad_source_requires_features_dir(tmp_path):
    with pytest.raises(SystemExit):
        generate(str(FIXTURE), source="uniad", teacher="mock", out_dir=tmp_path / "x")


def test_uniad_source_end_to_end(tmp_path):
    feats = tmp_path / "feats"
    _write_uniad_pth(_infos(), feats)
    out = tmp_path / "v_uniad"
    m = generate(str(FIXTURE), source="uniad", teacher="mock", out_dir=out,
                 uniad_features_dir=str(feats))
    assert m["counts"]["frames"] == 20
    assert m["source"] == "uniad"
    assert m["uniad"]["missing_pth_frames"] == 0
    # the synthetic .pth mirrors GT, so perception agrees with GT everywhere
    assert set(m["reconcile_histogram"]) == {"ok"}
    recs = [json.loads(l) for l in (out / "traces.jsonl").read_text().splitlines()]
    assert all(r["source"] == "uniad" for r in recs)
    assert all(r["validators"]["schema_conformance"] == 1.0 for r in recs)


def test_render_assistant_turn_is_step5_string():
    infos = _infos()
    by_token = _index(infos)
    info = infos[0]
    scene = SceneRecord.from_gt(info)
    fact = extract_fact(scene)
    trace = MockVerbalizer().verbalize(fact)
    traj = _future_traj(info, by_token)
    from reasoning_data_gen.schema import format_trajectory

    traj_str = format_trajectory(traj.tolist())
    turn = render_assistant_turn(fact, trace, traj_str)
    # exact structure Step 5 will emit: reason block then trajectory block
    assert turn == f"{REASON_START}{trace}{REASON_END}{TRAJ_START}{traj_str}{TRAJ_END}"
    assert turn.index(REASON_START) < turn.index(REASON_END) < turn.index(TRAJ_START) < turn.index(TRAJ_END)
    # trajectory block matches the plain build_llava_conversation format exactly
    assert turn.split(TRAJ_START)[1].split(TRAJ_END)[0] == traj_str


def test_mock_traces_have_no_hallucination():
    for info in _infos():
        fact = extract_fact(SceneRecord.from_gt(info))
        trace = MockVerbalizer().verbalize(fact)
        assert find_hallucinations(trace, fact) == []


def test_counterfactual_pair_shares_id_and_flips_to_stop():
    infos = _infos()
    by_token = _index(infos)
    made = 0
    for info in infos:
        scene = SceneRecord.from_gt(info)
        fact = extract_fact(scene)
        traj = _future_traj(info, by_token)
        pair = make_stop_injection(Frame(info["token"], scene, fact, traj))
        if pair is None:
            continue
        made += 1
        assert pair.pair_id == f"{info['token']}::cf_stop"
        assert pair.counterfactual_fact.decision == "stop_yield"
        # counterfactual trajectory comes to rest
        assert V.action_fidelity("stop_yield", pair.counterfactual_traj) == 1.0
        # synthetic-perception-token spec is present for Step-2/3 injection
        spec = pair.synthetic_token_spec
        assert spec["injected"] is True and "cls" in spec and "right" in spec
    assert made >= 1


def test_slot_bearing_is_grounded_in_fact():
    # The slot factor must carry its ego-local position so the trace can state a
    # qualitative bearing, and the bearing word must not count as a hallucination.
    infos = _infos()
    for info in infos:
        fact = extract_fact(SceneRecord.from_gt(info))
        slot = next((e for e in fact.causal_factors if e.kind == "slot"), None)
        if slot is None:
            continue
        assert slot.r is not None and slot.f is not None
        assert slot.bearing() is not None
        trace = MockVerbalizer().verbalize(fact)
        # bearing words are spatial adverbs, not named entities -> still no hallucination
        assert find_hallucinations(trace, fact) == []


def test_qwen_prompt_is_two_level_and_grounded():
    from reasoning_data_gen.verbalizer import QWEN_SYSTEM_PROMPT, QwenVerbalizer

    info = _infos()[8]
    fact = extract_fact(SceneRecord.from_gt(info))
    qv = QwenVerbalizer(endpoint="http://dummy/v1/chat/completions")  # no network in build
    msgs = qv.build_messages(fact)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == QWEN_SYSTEM_PROMPT           # constant master prompt
    assert f"Decision: {fact.decision}" in msgs[1]["content"]  # per-frame fact
    # the user prompt must not leak an entity outside the fact's allowed set
    assert "Entities you may name:" in msgs[1]["content"]


def test_render_parse_roundtrip_and_malformed_output():
    """render_assistant_turn and parse_assistant_turn are THE write/read pair — training,
    inference and eval all use them, so they must never drift. A malformed model output
    must fail loudly (so a format-failure RATE can be reported), never silently score as
    a trajectory of zeros."""
    from reasoning_data_gen.schema import format_trajectory, parse_assistant_turn

    infos = _infos()
    by = _index(infos)
    for info in infos[:5]:
        fact = extract_fact(SceneRecord.from_gt(info))
        trace = MockVerbalizer().verbalize(fact)
        traj = _future_traj(info, by)
        turn = render_assistant_turn(fact, trace, format_trajectory(traj.tolist()))

        p = parse_assistant_turn(turn)
        assert p.ok and p.error is None
        assert p.trace == trace
        assert len(p.trajectory) == 6
        for got, want in zip(p.trajectory, traj.tolist()):
            for a, b in zip(got, want):
                assert abs(a - round(b, 2)) < 1e-9      # 2-dp render is the contract

    # malformed output must never yield a usable trajectory silently
    for bad in ["<reason_start>x<reason_end>",                                   # no traj
                "<reason_start>x<reason_end><traj_start>[(1,2,3),(1,2,3)]<traj_end>",  # short
                "just some words"]:
        p = parse_assistant_turn(bad)
        assert not p.ok and p.trajectory is None and p.error


def test_bay_flank_detection_geometry():
    """A vehicle placed in the bay adjacent to the target slot must be found as a
    flank, on the correct side (the ego-frame across-axis was verified to 1e-6 m
    against a global-frame computation)."""
    import numpy as np

    from reasoning_data_gen.fact_extractor import DEFAULT_THRESHOLDS, bay_flanks
    from reasoning_data_gen.scene_record import DetectedObject

    info = _infos()[0]
    scene = SceneRecord.from_gt(info)
    sr, sf, dh = scene.command.slot_local
    right_unit = np.array([np.cos(dh), np.sin(dh)])
    bay = DEFAULT_THRESHOLDS.bay_width

    def car_at(p):
        return DetectedObject(cls="car", right=float(p[0]), forward=float(p[1]),
                              yaw=0.0, size=(1.8, 4.5, 1.6), vel=(0.0, 0.0))

    centre = np.array([sr, sf])
    scene.objects = [car_at(centre + bay * right_unit)]      # occupy the RIGHT bay
    left, right = bay_flanks(scene, DEFAULT_THRESHOLDS)
    assert right is not None and left is None

    scene.objects = [car_at(centre - bay * right_unit)]      # occupy the LEFT bay
    left, right = bay_flanks(scene, DEFAULT_THRESHOLDS)
    assert left is not None and right is None

    scene.objects = []                                        # empty bays -> no flanks
    assert bay_flanks(scene, DEFAULT_THRESHOLDS) == (None, None)


def test_front_blocker_is_cited_and_gates_no_decision():
    """A car ahead must be CITED (it caps the forward swing) without changing the
    DECISION — the new perception factors supply causes, they never gate the maneuver
    label. (stop_yield/creep remain perception-gated by original design: they are
    *defined* by an in-path object. So place the car outside the stop corridor
    (|right| > corridor_half_width) and use a moving frame, to isolate `front`.)"""
    from reasoning_data_gen.fact_extractor import DEFAULT_THRESHOLDS, extract_fact
    from reasoning_data_gen.scene_record import DetectedObject

    th = DEFAULT_THRESHOLDS
    info = next(i for i in _infos()
                if not i["reverse"] and abs(i["ego_speed"]) > th.creep_speed)

    clear = SceneRecord.from_gt(info)
    clear.objects = []
    fact_clear = extract_fact(clear)
    assert not [e for e in fact_clear.causal_factors if e.kind == "obstacle"]

    blocked = SceneRecord.from_gt(info)
    # inside the front-swing envelope, OUTSIDE the in-path stop corridor
    blocked.objects = [DetectedObject(cls="car", right=2.0, forward=4.0, yaw=0.0,
                                      size=(1.8, 4.5, 1.6), vel=(0.0, 0.0))]
    fact_blocked = extract_fact(blocked)

    assert fact_blocked.decision == fact_clear.decision           # decision UNCHANGED
    roles = {e.role for e in fact_blocked.causal_factors if e.kind == "obstacle"}
    assert "front" in roles                                       # but the car IS cited


def test_perception_moves_the_trace():
    """The whole point of Task 1: removing the perceived objects must change the
    trace. If this passes trivially, perception is not load-bearing."""
    from reasoning_data_gen.fact_extractor import extract_fact

    infos = _infos()
    by = _index(infos)
    changed = 0
    for info in infos:
        scene = SceneRecord.from_gt(info)
        traj = _future_traj(info, by)
        with_p = MockVerbalizer().verbalize(extract_fact(scene, planned_path=traj))
        scene.objects = []                                  # ablate perception
        without_p = MockVerbalizer().verbalize(extract_fact(scene, planned_path=traj))
        if with_p != without_p:
            changed += 1
    assert changed > 0, "ablating perception changed no trace -> perception is decorative"


def test_reconcile_divergence_cases():
    # Perfect agreement -> ok.
    infos = _infos()
    f = extract_fact(SceneRecord.from_gt(infos[0]))
    assert R.reconcile(f, f) == R.OK

    from reasoning_data_gen.schema import FactRecord, metric_ref, obstacle_ref, slot_ref

    # GT sees an obstacle perception missed -> perception_limited.
    perc = FactRecord("approach", [slot_ref(False), metric_ref("dist_to_slot", 4.0)])
    gt = FactRecord("approach", [slot_ref(False), metric_ref("dist_to_slot", 4.0),
                                 obstacle_ref("car", 0.5, 3.0)])
    assert R.reconcile(perc, gt) == R.PERCEPTION_LIMITED

    # Perception hallucinates an obstacle GT does not have -> contradiction.
    perc2 = FactRecord("stop_yield", [obstacle_ref("pedestrian", 0.0, 3.0)])
    gt2 = FactRecord("approach", [slot_ref(False)])
    assert R.reconcile(perc2, gt2) == R.CONTRADICTION
