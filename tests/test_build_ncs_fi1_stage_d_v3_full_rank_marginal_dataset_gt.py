import numpy as np

from tools.build_ncs_fi1_stage_d_v3_full_rank_marginal_dataset_gt import (
    append_precedes, full_rank_prefix,
)


def _append(key, score):
    return {"candidate_key": key, "reference_score": score}


def test_full_rank_prefix_includes_baseline_ties_and_higher_append_candidates():
    baseline = [
        {"reference_score": 0.8, "node": {"geometry_key": "n1"}},
        {"reference_score": 0.5, "node": {"geometry_key": "n2"}},
        {"reference_score": 0.2, "node": {"geometry_key": "n3"}},
    ]
    current = _append("scene:union:0002:original", 0.5)
    earlier_tie = _append("scene:union:0001:refined", 0.5)
    later_tie = _append("scene:union:0003:original", 0.5)
    higher = _append("scene:union:0004:original", 0.7)
    selected = full_rank_prefix(baseline, [current, earlier_tie, later_tie, higher], current)
    assert selected == [baseline[0], baseline[1], earlier_tie, higher]
    assert current not in selected


def test_append_tie_order_is_stable_candidate_key_order():
    left = _append("a", np.float64(0.5))
    right = _append("b", 0.5)
    assert append_precedes(left, right)
    assert not append_precedes(right, left)
