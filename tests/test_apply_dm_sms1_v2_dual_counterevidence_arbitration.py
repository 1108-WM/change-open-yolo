import copy
import json

from tools.apply_dm_sms1_v2_dual_counterevidence_arbitration import (
    FI1_FOUNDATION,
    RULE_ID,
    decide_row,
    run,
)


def _manifest():
    return {
        "task_id": "task-1",
        "scene_name": "scene0000_00",
        "geometry_key": "scene0000_00:geometry:1",
        "geometry_hash": "geometry-hash",
        "canonical_frozen_class_index": 0,
        "candidate_hypotheses": [
            {"class_index": 0, "class_name": "incumbent"},
            {"class_index": 1, "class_name": "alternative"},
        ],
        "candidate_order_ab": ["incumbent", "alternative"],
        "candidate_order_ba": ["alternative", "incumbent"],
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }


def _item(class_index, supported, strong_counterevidence, confidence=0.8):
    return {
        "class_index": class_index,
        "supported": supported,
        "strong_counterevidence": strong_counterevidence,
        "support_evidence": f"support-{class_index}",
        "counterevidence": f"counter-{class_index}",
        "confidence": confidence,
    }


def _evidence():
    return {
        "task_id": "task-1",
        "scene_name": "scene0000_00",
        "geometry_hash": "geometry-hash",
        "valid": True,
        "order_ab": {"candidate_results": [
            _item(0, False, True),
            _item(1, True, False),
        ]},
        "order_ba": {"candidate_results": [
            _item(1, True, False),
            _item(0, False, True),
        ]},
        "ground_truth_read": False,
        "ap_computed": False,
    }


def test_all_six_gates_are_required_for_a_change():
    changed = decide_row(_manifest(), _evidence())
    assert changed["arbitrated_class_index"] == 1
    assert changed["class_changed"] is True
    assert changed["all_change_gates_passed"] is True
    assert changed["change_reason"] == RULE_ID
    assert changed["fi1_foundation"] == FI1_FOUNDATION

    mutations = (
        ("order_ab", 1, "supported", False, "alternative_missing_dual_support_keep_frozen_control"),
        ("order_ba", 0, "supported", False, "alternative_missing_dual_support_keep_frozen_control"),
        ("order_ab", 1, "strong_counterevidence", True, "alternative_has_strong_counterevidence_keep_frozen_control"),
        ("order_ba", 0, "strong_counterevidence", True, "alternative_has_strong_counterevidence_keep_frozen_control"),
        ("order_ab", 0, "strong_counterevidence", False, "incumbent_missing_dual_strong_counterevidence_keep_frozen_control"),
        ("order_ba", 1, "strong_counterevidence", False, "incumbent_missing_dual_strong_counterevidence_keep_frozen_control"),
    )
    for order, item_index, field, value, reason in mutations:
        evidence = _evidence()
        evidence[order]["candidate_results"][item_index][field] = value
        decision = decide_row(_manifest(), evidence)
        assert decision["arbitrated_class_index"] == 0
        assert decision["class_changed"] is False
        assert decision["change_reason"] == reason


def test_confidence_does_not_gate_the_v2_decision():
    low = _evidence()
    high = _evidence()
    for order in ("order_ab", "order_ba"):
        for item in low[order]["candidate_results"]:
            item["confidence"] = 0.0
        for item in high[order]["candidate_results"]:
            item["confidence"] = 1.0
    assert decide_row(_manifest(), low)["arbitrated_class_index"] == 1
    assert decide_row(_manifest(), high)["arbitrated_class_index"] == 1


def test_invalid_or_misordered_evidence_safely_keeps_the_incumbent():
    invalid = _evidence()
    invalid["valid"] = False
    invalid["error"] = "synthetic invalid record"
    decision = decide_row(_manifest(), invalid)
    assert decision["arbitrated_class_index"] == 0
    assert decision["model_evidence_valid"] is False
    assert decision["change_reason"] == "invalid_evidence_keep_frozen_control"

    misordered = _evidence()
    misordered["order_ba"]["candidate_results"].reverse()
    decision = decide_row(_manifest(), misordered)
    assert decision["arbitrated_class_index"] == 0
    assert decision["model_evidence_valid"] is False


def test_run_writes_only_a_new_v2_ledger(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    candidate_path.write_text(json.dumps(_manifest()) + "\n")
    evidence_path.write_text(json.dumps(_evidence()) + "\n")
    output_root = tmp_path / "dm_sms1_v2_synthetic_ledger"
    summary = run(candidate_path, evidence_path, output_root)
    assert summary["class_change_count"] == 1
    assert summary["fi1_foundation"] == "FI1-Legacy"
    saved = json.loads((output_root / "v2_safe_decisions.jsonl").read_text())
    assert saved["evidence_snapshot"]["order_ab"]["incumbent"]["strong_counterevidence"] is True
    assert not (tmp_path / "safe_decisions.jsonl").exists()


def test_manifest_input_is_not_mutated():
    manifest = _manifest()
    evidence = _evidence()
    original_manifest = copy.deepcopy(manifest)
    original_evidence = copy.deepcopy(evidence)
    decide_row(manifest, evidence)
    assert manifest == original_manifest
    assert evidence == original_evidence

