import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_same_class_competition_audit.py"
    spec = importlib.util.spec_from_file_location("same_class_competition_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_containment_state_keeps_exact_set_fact_without_iou_threshold():
    module = _module()
    contained = module.containment_state({"intersection_point_count": 4}, 4, 7)
    partial = module.containment_state({"intersection_point_count": 3}, 4, 7)
    assert contained["state"] == "left_strictly_contained_by_right"
    assert contained["left_exclusive_point_count"] == 0
    assert contained["right_exclusive_point_count"] == 3
    assert partial["state"] == "partial_geometric_overlap"
    assert not partial["has_strict_set_containment"]


def test_audit_requires_pareto_and_containment_for_strongest_audit_category():
    module = _module()
    left = {"candidate_id": 1, "source_track_id": 11, "class_id": 2, "point_count": 4, "gvc_score": .8, "semantic_vote_margin": .7, "semantic_normalized_entropy": .2, "independent_from_native_ratio": .6}
    right = {"candidate_id": 2, "source_track_id": 12, "class_id": 2, "point_count": 7, "gvc_score": .7, "semantic_vote_margin": .6, "semantic_normalized_entropy": .3, "independent_from_native_ratio": .5}
    relation = {"left_candidate_id": 1, "right_candidate_id": 2, "same_class": True, "semantic_js_divergence": .1, "geometry": {"intersection_point_count": 4, "iou": 4 / 7, "left_coverage": 1., "right_coverage": 4 / 7}, "cross_view_and_granularity_evidence": {}}
    records, excluded = module.build_scene_audit([left, right], [relation], [])
    assert excluded == 0
    assert records[0]["pareto_dominance_direction"] == "left_dominates_right"
    assert records[0]["competition_audit_category"] == "pareto_dominance_with_strict_containment"
    assert records[0]["decision_state"].startswith("只读审计")
