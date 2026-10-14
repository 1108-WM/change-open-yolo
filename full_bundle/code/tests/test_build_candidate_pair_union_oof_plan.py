import pytest

from tools.build_candidate_pair_union_oof_plan import append_score


def test_pair_union_append_score_uses_fixed_cubic_focal_shape():
    assert append_score(0.8, 0.0) == pytest.approx(0.0)
    assert append_score(0.8, 1.0) == pytest.approx(0.8)
    assert append_score(0.8, 0.5) == pytest.approx(0.7)
