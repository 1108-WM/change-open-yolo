import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_aggregation_boundary_variant_plan.py"
    spec = importlib.util.spec_from_file_location("aggregation_boundary_variant_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relation(states):
    return {"pair_evidence_states": states}


def test_multiframe_non_nested_exact_evidence_only_plans_a_reversible_aggregation_variant():
    module = _module()
    exact = {"distinct_frame_count": 2, "max_left_coverage": .7, "max_right_coverage": .8}
    state, exact_state = module.plan_state(_relation(["multiview_complementarity_hypothesis"]), exact)
    assert exact_state == "multiframe_exact_2d_evidence"
    assert state == "aggregation_variant_hypothesis_keep_originals"


def test_near_containment_never_plans_an_aggregation_union():
    module = _module()
    exact = {"distinct_frame_count": 3, "max_left_coverage": .99, "max_right_coverage": .5}
    state, exact_state = module.plan_state(_relation(["multiview_complementarity_hypothesis"]), exact)
    assert exact_state == "near_containment_exact_2d_evidence"
    assert state == "boundary_competition_variant_hypothesis_keep_originals"


def test_tiny_multiframe_boundary_touch_does_not_plan_an_aggregation_variant():
    module = _module()
    exact = {"distinct_frame_count": 2, "max_left_coverage": .001, "max_right_coverage": .03}
    state, exact_state = module.plan_state(_relation(["multiview_complementarity_hypothesis"]), exact)
    assert exact_state == "weak_multiframe_exact_2d_boundary_touch_evidence"
    assert state == "keep_originals_only_insufficient_exact_2d_support"
