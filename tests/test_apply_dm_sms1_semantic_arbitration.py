import pytest

from tools.apply_dm_sms1_semantic_arbitration import decide_row


def _manifest():
    return {
        "scene_name": "scene", "plan_key": "plan", "geometry_key": "plan",
        "visual_geometry_key": "scene:g", "geometry_hash": "g",
        "task_id": "task",
        "candidate_source": "native", "challenger_score": 0.5, "append_only": False,
        "canonical_frozen_class_index": 0,
        "candidate_hypotheses": [{"class_index": 0}, {"class_index": 1}],
    }


def _evidence(supported_ab=True, supported_ba=True):
    def item(class_index, supported):
        return {"class_index": class_index, "supported": supported, "strong_counterevidence": False,
                "support_evidence": "evidence", "counterevidence": "none", "confidence": 0.8}
    return {
        "task_id": "task",
        "order_ab": {"candidate_results": [item(0, False), item(1, supported_ab)]},
        "order_ba": {"candidate_results": [item(1, supported_ba), item(0, False)]},
    }


def test_both_orders_are_required_for_a_class_change():
    result = decide_row(_manifest(), _evidence(True, True))
    assert result["arbitrated_class_index"] == 1
    assert result["class_changed"] is True
    assert result["geometry_mutation"] is False
    assert decide_row(_manifest(), _evidence(True, False))["arbitrated_class_index"] == 0


def test_support_true_requires_nonempty_support_evidence():
    evidence = _evidence(True, True)
    evidence["order_ba"]["candidate_results"][0]["support_evidence"] = ""
    with pytest.raises(ValueError, match="support is true"):
        decide_row(_manifest(), evidence)


def test_strong_counterevidence_true_requires_nonempty_counterevidence():
    evidence = _evidence(True, True)
    evidence["order_ab"]["candidate_results"][0]["strong_counterevidence"] = True
    evidence["order_ab"]["candidate_results"][0]["counterevidence"] = ""
    with pytest.raises(ValueError, match="strong counterevidence is true"):
        decide_row(_manifest(), evidence)
