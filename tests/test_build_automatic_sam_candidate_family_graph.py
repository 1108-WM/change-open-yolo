import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_candidate_family_graph.py"
    spec = importlib.util.spec_from_file_location("candidate_family_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geometry_relation_keeps_continuous_overlap_values():
    module = _module()
    relation = module.geometry_relation(4, 6, 3)
    assert relation == {"intersection_point_count": 3, "iou": 3 / 7, "left_coverage": .75, "right_coverage": .5}


def test_relation_tags_do_not_make_a_competition_decision():
    module = _module()
    geometry = module.geometry_relation(4, 4, 0)
    evidence = {
        "cross_view_edge_count": 2,
        "mean_bidirectional_reprojection_support": .4,
        "mean_jointly_visible_shared_superpoint_count": 1.,
        "same_frame_granularity_relation_count": 1,
        "max_same_frame_bbox_nesting": .9,
        "hierarchy_evidence_limit": "approximate",
    }
    assert module._relation_tags(False, geometry, evidence) == [
        "cross_class_relation", "cross_view_evidence", "same_frame_granularity_evidence",
        "cross_view_evidence_without_geometric_overlap",
    ]
