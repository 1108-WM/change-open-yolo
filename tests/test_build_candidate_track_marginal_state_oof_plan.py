import pytest

from tools.build_candidate_track_marginal_state_oof_plan import suppressed_track_score


def test_suppressed_track_score_is_monotone_and_never_increases_original():
    assert suppressed_track_score(0.7, 0.0) == pytest.approx(0.0)
    assert suppressed_track_score(0.7, 1.0) == pytest.approx(0.7)
    assert 0.0 < suppressed_track_score(0.7, 0.5) < 0.7
    with pytest.raises(ValueError):
        suppressed_track_score(0.7, 1.1)
