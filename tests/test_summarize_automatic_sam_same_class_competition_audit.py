import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "summarize_automatic_sam_same_class_competition_audit.py"
    spec = importlib.util.spec_from_file_location("same_class_competition_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_keeps_continuous_geometry_by_audit_category():
    module = _module()
    records = [{
        "competition_audit_category": "pareto_dominance_without_strict_containment",
        "geometry": {"iou": .5, "left_coverage": .6, "right_coverage": .8},
        "semantic_js_divergence": .2,
    }]
    summary = module.summarize(records)
    result = summary["pareto_dominance_without_strict_containment"]
    assert result["iou"]["mean"] == .5
    assert result["min_coverage"]["p50"] == .6
    assert result["max_coverage"]["p90"] == .8


def test_orientation_distinguishes_a_dominant_container_from_dominant_part():
    module = _module()
    container = {
        "pareto_dominance_direction": "right_dominates_left",
        "containment": {"state": "left_strictly_contained_by_right"},
    }
    part = {
        "pareto_dominance_direction": "left_dominates_right",
        "containment": {"state": "left_strictly_contained_by_right"},
    }
    assert module.dominance_containment_orientation(container) == "dominant_candidate_is_geometric_container"
    assert module.dominance_containment_orientation(part) == "dominant_candidate_is_geometrically_contained"
