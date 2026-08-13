import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_candidate_complementarity_ledger.py"
    spec = importlib.util.spec_from_file_location("candidate_complementarity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complementarity_requires_both_increment_and_cross_view_evidence_for_hypothesis():
    module = _module()
    relation = {
        "containment": {"state": "partial_geometric_overlap", "has_strict_set_containment": False},
        "cross_view_and_granularity_evidence": {"cross_view_edge_count": 1, "same_frame_granularity_relation_count": 0},
    }
    left = {"multiview_increment": {"candidate_independent_observation_count": 2}}
    right = {"multiview_increment": {"candidate_independent_observation_count": 3}}
    states = module.complementarity_states(relation, left, right)
    assert "both_candidates_have_multiview_native_increment" in states
    assert "two_sided_exclusive_geometry_observed" in states
    assert "multiview_complementarity_hypothesis" in states
    assert "containment_boundary_competition_hypothesis" not in states
