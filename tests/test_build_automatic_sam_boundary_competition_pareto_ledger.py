import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_boundary_competition_pareto_ledger.py"
    spec = importlib.util.spec_from_file_location("boundary_competition_pareto", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pareto_direction_requires_all_four_single_candidate_objectives():
    module = _module()
    left = {"gvc_score": .8, "semantic_vote_margin": .7, "semantic_normalized_entropy": .2, "weighted_independent_ratio": .6}
    right = {"gvc_score": .7, "semantic_vote_margin": .6, "semantic_normalized_entropy": .3, "weighted_independent_ratio": .5}
    assert module.direction(left, right) == "left_pareto_dominates_right"
    right["weighted_independent_ratio"] = .9
    assert module.direction(left, right) == "pareto_incomparable"
