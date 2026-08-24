from tools.build_dm_sms1_safe_decision_ledger import fallback_decision


def test_invalid_evidence_falls_back_without_mutating_frozen_fields():
    row = {
        "scene_name": "scene", "geometry_key": "scene:g", "geometry_hash": "g",
        "canonical_frozen_class_index": 4,
        "candidate_hypotheses": [{"class_index": 4}],
    }
    decision = fallback_decision(row, "task", "bad JSON")
    assert decision["arbitrated_class_index"] == 4
    assert decision["class_changed"] is False
    assert decision["model_evidence_valid"] is False
    assert decision["geometry_mutation"] is False
