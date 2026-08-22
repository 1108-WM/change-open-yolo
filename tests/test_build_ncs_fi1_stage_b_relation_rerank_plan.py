import pytest

from tools.build_ncs_fi1_stage_b_relation_rerank_plan import (
    classify_relation,
    relation_decay_action,
)


def _classify(**updates):
    values = {
        "same_hash": False,
        "pair_union_parent": False,
        "union_new_fraction": 0.0,
        "iou": 0.2,
        "coverage_a_in_b": 0.3,
        "coverage_b_in_a": 0.4,
        "cross_source": True,
        "direct_evidence": False,
        "same_fraction": 0.0,
        "different_fraction": 0.0,
    }
    values.update(updates)
    return classify_relation(**values)


def test_relation_taxonomy_uses_frozen_priority_order():
    assert _classify(same_hash=True, pair_union_parent=True) == "exact_duplicate"
    assert _classify(pair_union_parent=True, union_new_fraction=0.2) == "complementary"
    assert _classify(coverage_a_in_b=0.95, coverage_b_in_a=0.92) == "near_duplicate"
    assert _classify(coverage_a_in_b=0.95, coverage_b_in_a=0.5) == "containment"
    assert _classify(iou=0.01) == "conflict"
    assert _classify(cross_source=False, iou=0.2) == "uncertain"


def test_near_duplicate_decay_never_selects_native_lower_candidate():
    relation = {"relation_type": "near_duplicate", "minimum_bidirectional_coverage": 0.9}
    native = {"geometry_key": "n", "candidate_source": "native", "candidate_id": 1, "stage_a_quality": 0.2}
    track = {"geometry_key": "t", "candidate_source": "track", "candidate_id": 2, "stage_a_quality": 0.8}
    assert relation_decay_action(relation, native, track) is None


def test_containment_and_conflict_apply_continuous_non_native_decay():
    containment = {
        "relation_type": "containment",
        "coverage_a_in_b": 0.95,
        "coverage_b_in_a": 0.4,
        "maximum_bidirectional_coverage": 0.95,
    }
    track = {"geometry_key": "t", "candidate_source": "track", "candidate_id": 2, "stage_a_quality": 0.3}
    union = {"geometry_key": "u", "candidate_source": "pair_union", "candidate_id": 3, "stage_a_quality": 0.7}
    action = relation_decay_action(containment, track, union)
    assert action["geometry_key"] == "t"
    assert action["factor"] == pytest.approx(0.525)

    conflict = {"relation_type": "conflict"}
    native = {"geometry_key": "n", "candidate_source": "native", "candidate_id": 1, "stage_a_quality": 0.4}
    union = {"geometry_key": "u", "candidate_source": "pair_union", "candidate_id": 3, "stage_a_quality": 0.8}
    action = relation_decay_action(conflict, native, union)
    assert action == {"geometry_key": "u", "factor": 0.5, "reason": "conflict_native_cap"}
