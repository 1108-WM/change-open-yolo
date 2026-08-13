import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "summarize_automatic_sam_candidate_family_graph.py"
    spec = importlib.util.spec_from_file_location("candidate_family_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pareto_dominance_requires_no_worse_all_objectives():
    module = _module()
    strong = {"gvc_score": .8, "semantic_vote_margin": .7, "semantic_normalized_entropy": .2, "independent_from_native_ratio": .6}
    weak = {"gvc_score": .7, "semantic_vote_margin": .6, "semantic_normalized_entropy": .3, "independent_from_native_ratio": .5}
    mixed = {"gvc_score": .9, "semantic_vote_margin": .4, "semantic_normalized_entropy": .1, "independent_from_native_ratio": .7}
    assert module.dominates(strong, weak)
    assert not module.dominates(mixed, weak)
