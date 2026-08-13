import pytest

from tools.diagnose_candidate_track_marginal_harm_oof import (
    attach_marginal_labels,
    joint_harm_probability,
)


def test_joint_harm_probability_uses_all_three_channels_monotonically():
    base = joint_harm_probability(0.2, 0.0, 0.0)
    assert base == pytest.approx(0.2)
    assert joint_harm_probability(0.3, 0.0, 0.0) > base
    assert joint_harm_probability(0.2, 0.5, 0.0) > base
    assert joint_harm_probability(0.2, 0.0, 0.5) > base


def test_attach_marginal_labels_keeps_only_tracks_and_requires_exact_coverage():
    rows = [
        {
            "scene_name": "s", "relation_component_id": 2,
            "candidate_source": "native_mask3d_yoloworld", "candidate_id": 1,
        },
        {
            "scene_name": "s", "relation_component_id": 2,
            "candidate_source": "d2b_track", "candidate_id": 7,
        },
    ]
    label = {
        "label_demote_harms_ap": 1,
        "label_keep_utility": 0.1,
        "label_positive_harm_magnitude": 0.1,
        "label_delta_official_ap": -0.1,
        "label_decision": "suppress_harmful",
        "evaluator_candidate_present": True,
    }
    output = attach_marginal_labels(rows, {("s", 2, 7): label})
    assert len(output) == 1
    assert output[0]["candidate_id"] == 7
    assert output[0]["label_demote_harms_ap"] == 1
