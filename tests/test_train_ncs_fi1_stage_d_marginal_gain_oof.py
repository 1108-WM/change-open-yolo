import numpy as np

from tools.train_ncs_fi1_stage_d_marginal_gain_oof import conservative_append_score


def test_stage_d_score_uses_only_positive_lower_bound_and_is_quality_bounded():
    conservative, score = conservative_append_score(
        np.asarray([0.10, 0.03, 0.80]),
        np.asarray([0.04, 0.04, 0.10]),
        np.asarray([0.50, 0.90, 0.25]),
    )
    assert np.allclose(conservative, [0.06, 0.0, 0.70])
    assert np.allclose(score, [0.03, 0.0, 0.175])
    assert np.all(score <= np.asarray([0.50, 0.90, 0.25]))
