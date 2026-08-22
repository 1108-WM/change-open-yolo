import json

from tools.apply_dm_sms1_v2_dual_counterevidence_arbitration import run
from tools.audit_dm_sms1_v2_dual_counterevidence_decisions import audit


def _inputs(tmp_path):
    candidate = {
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
    incumbent = {
        "class_index": 0,
        "supported": False,
        "strong_counterevidence": True,
        "support_evidence": "none",
        "counterevidence": "direct target evidence",
        "confidence": 0.7,
    }
    alternative = {
        "class_index": 1,
        "supported": True,
        "strong_counterevidence": False,
        "support_evidence": "direct target evidence",
        "counterevidence": "none",
        "confidence": 0.8,
    }
    evidence = {
        "task_id": "task-1",
        "scene_name": "scene0000_00",
        "geometry_hash": "geometry-hash",
        "valid": True,
        "order_ab": {"candidate_results": [incumbent, alternative]},
        "order_ba": {"candidate_results": [alternative, incumbent]},
        "ground_truth_read": False,
        "ap_computed": False,
    }
    candidate_path = tmp_path / "candidates.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    candidate_path.write_text(json.dumps(candidate) + "\n")
    evidence_path.write_text(json.dumps(evidence) + "\n")
    return candidate_path, evidence_path


def test_independent_audit_accepts_an_untampered_v2_ledger(tmp_path):
    candidate_path, evidence_path = _inputs(tmp_path)
    decision_root = tmp_path / "dm_sms1_v2_synthetic_audit_ok"
    run(candidate_path, evidence_path, decision_root)
    result = audit(decision_root, candidate_path, evidence_path)
    assert result["audit_valid"] is True
    assert result["error_count"] == 0
    assert result["ground_truth_read"] is False
    assert result["ap_computed"] is False


def test_independent_audit_detects_a_tampered_gate_and_selected_class(tmp_path):
    candidate_path, evidence_path = _inputs(tmp_path)
    decision_root = tmp_path / "dm_sms1_v2_synthetic_audit_tampered"
    run(candidate_path, evidence_path, decision_root)
    decision_path = decision_root / "v2_safe_decisions.jsonl"
    row = json.loads(decision_path.read_text())
    row["decision_gates"]["incumbent_strong_counterevidence_ab"] = False
    row["arbitrated_class_index"] = 0
    decision_path.write_text(json.dumps(row) + "\n")
    result = audit(decision_root, candidate_path, evidence_path)
    assert result["audit_valid"] is False
    assert result["error_count"] >= 2

