from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE
from tools.build_frozen_track_harm_score_plan import (
    _component_candidates,
    fixed_track_score,
)


def test_fixed_track_score_uses_frozen_cubic_suppression():
    multiplier, planned = fixed_track_score(0.8, 0.5)
    assert multiplier == 0.875
    assert abs(planned - 0.7) < 1e-12


def test_fixed_track_score_never_increases_track():
    for keep in (0.0, 0.2, 0.8, 1.0):
        _multiplier, planned = fixed_track_score(0.73, keep)
        assert 0.0 <= planned <= 0.73


def test_component_candidates_selects_highest_score_native_representative():
    scene = "scene_x"
    components = [{
        "relation_component_id": 0,
        "native_exact_geometry_group_ids": [f"{scene}:native_geometry:0000"],
        "track_ids": [7],
    }]
    relation_rows = [{
        "relation_component_id": 0,
        "native_exact_geometry_group_id": f"{scene}:native_geometry:0000",
        "native_member_candidate_ids": [0, 2],
        "track_id": 7,
    }]
    feature_rows = [
        {"scene_name": scene, "candidate_source": NATIVE_SOURCE, "candidate_id": 0,
         "original_source_score": 0.8},
        {"scene_name": scene, "candidate_source": NATIVE_SOURCE, "candidate_id": 2,
         "original_source_score": 1.0},
        {"scene_name": scene, "candidate_source": TRACK_SOURCE, "candidate_id": 7,
         "original_source_score": 0.7},
    ]
    result = _component_candidates(scene, components, relation_rows, feature_rows)
    assert [(row["candidate_source"], row["candidate_id"]) for row in result] == [
        (NATIVE_SOURCE, 2), (TRACK_SOURCE, 7),
    ]
