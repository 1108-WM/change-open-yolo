import math

from tools.diagnose_official100_component_reranking_oracle_gt import (
    component_permutation_scores,
    maximum_matching_selection,
    shared_gt_quality_scores,
    threshold_oracle_scores,
)


def test_maximum_matching_selection_finds_augmenting_path():
    edges = {
        ("native", 0): [(0, 0.9), (1, 0.8)],
        ("track", 0): [(0, 0.7)],
    }
    selected = maximum_matching_selection(edges)
    assert len(selected) == 2
    assert {value[0] for value in selected.values()} == {0, 1}


def test_component_joint_permutation_preserves_score_multiset():
    base = {("native", 0): 0.9, ("native", 1): 0.3, ("track", 0): 0.6}
    quality = {("native", 0): 0.2, ("native", 1): 0.8, ("track", 0): 0.5}
    components = [{"native": [("native", 0), ("native", 1)], "track": [("track", 0)]}]
    result = component_permutation_scores(base, quality, components, "component_joint")
    assert sorted(result.values()) == sorted(base.values())
    assert result[("native", 1)] == 0.9
    assert result[("native", 0)] == 0.3


def test_source_separate_does_not_exchange_levels_across_sources():
    base = {("native", 0): 0.9, ("native", 1): 0.8, ("track", 0): 0.2, ("track", 1): 0.1}
    quality = {("native", 0): 0.1, ("native", 1): 0.9, ("track", 0): 0.1, ("track", 1): 0.9}
    components = [{
        "native": [("native", 0), ("native", 1)],
        "track": [("track", 0), ("track", 1)],
    }]
    result = component_permutation_scores(base, quality, components, "component_source_separate")
    assert {result[("native", 0)], result[("native", 1)]} == {0.8, 0.9}
    assert {result[("track", 0)], result[("track", 1)]} == {0.1, 0.2}


def test_joint_threshold_oracle_keeps_candidates_but_separates_selected_scores():
    base = {("native", 0): 0.9, ("track", 0): 0.6, ("track", 1): 0.4}
    quality = {("native", 0): 0.8, ("track", 0): 0.7, ("track", 1): 0.1}
    edges = {
        ("native", 0): [(0, 0.8)],
        ("track", 0): [(0, 0.7)],
        ("track", 1): [],
    }
    scores, selected = threshold_oracle_scores(
        base,
        quality,
        edges,
        {("native", 0)},
        {("track", 0), ("track", 1)},
        "component_joint",
    )
    assert set(scores) == set(base)
    assert len(selected) == 1
    assert all(scores[key] > 1.0 for key in selected)
    assert all(scores[key] < 0.0 for key in set(base) - selected)
    assert all(math.isfinite(value) for value in scores.values())


def test_shared_joint_oracle_promotes_one_candidate_per_target_across_sources():
    base = {("native", 0): 0.9, ("track", 0): 0.6, ("track", 1): 0.4}
    quality = {("native", 0): 0.8, ("track", 0): 0.7, ("track", 1): 0.9}
    target_ids = {("native", 0): 4, ("track", 0): 4, ("track", 1): 8}
    scores, selected = shared_gt_quality_scores(
        base,
        quality,
        target_ids,
        {("native", 0)},
        {("track", 0), ("track", 1)},
        "component_joint",
    )
    assert selected == {("native", 0), ("track", 1)}
    assert scores[("track", 0)] < 0.0
    assert scores[("native", 0)] > 1.0
