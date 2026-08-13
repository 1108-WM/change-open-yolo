import math

from tools.train_candidate_component_union_residual_expert_head_oof import (
    fixed_geometric_keep,
)


def test_fixed_geometric_keep_is_symmetric_and_not_scanned():
    assert math.isclose(fixed_geometric_keep(0.25, 1.0), 0.5)
    assert fixed_geometric_keep(0.2, 0.8) == fixed_geometric_keep(0.8, 0.2)
