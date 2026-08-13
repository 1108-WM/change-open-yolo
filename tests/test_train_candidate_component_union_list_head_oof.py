from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE
from tools.train_candidate_component_union_list_head_oof import (
    augment_union_features,
    collapse_multitask_auxiliary_to_candidate_win_residual,
    official_relevance_target,
    strip_multitask_auxiliary_features,
)
from tools.train_candidate_component_list_calibration_head_oof import (
    build_candidate_feature_rows,
)


def test_augment_union_features_keeps_schema_for_native_and_track_rows():
    rows = [
        {"scene_name": "s", "relation_component_id": 0, "candidate_source": NATIVE_SOURCE,
         "candidate_id": 3, "model_features": {"base": 1.0}},
        {"scene_name": "s", "relation_component_id": 0, "candidate_source": TRACK_SOURCE,
         "candidate_id": 7, "model_features": {"base": 2.0}},
    ]
    lookup = {("s", 0, 7): {"coverage": 0.75}}
    output = augment_union_features(rows, lookup, ["coverage"])
    assert output[0]["model_features"]["union__coverage"] == 0.0
    assert output[0]["model_features"]["union__not_applicable"] == 1.0
    assert output[1]["model_features"]["union__coverage"] == 0.75
    assert output[1]["model_features"]["union__not_applicable"] == 0.0


def test_official_relevance_target_is_continuous_and_zero_for_nonwinner():
    assert official_relevance_target({
        "label_official_relevance": 4.0 / 9.0,
        "label_component_unique_winner": 1,
    }) == 4.0 / 9.0
    assert official_relevance_target({
        "label_official_relevance": 0.0,
        "label_component_unique_winner": 0,
    }) == 0.0


def test_multitask_relation_scores_are_appended_without_replacing_incumbent_scores():
    candidate = {
        "scene_name": "s",
        "relation_component_id": 0,
        "candidate_source": TRACK_SOURCE,
        "candidate_id": 7,
        "original_source_score": 0.8,
    }
    raw = {
        "track_id": 7,
        "native_member_candidate_ids": [3],
        "native_exact_geometry_group_id": "g",
        "features": {
            name: 0.1 for name in (
                "point_iou", "native_inside_track_ratio", "track_inside_native_ratio",
                "aabb_iou", "centroid_distance_normalized",
                "native_shared_superpoint_fraction", "track_shared_superpoint_fraction",
                "mean_rgb_distance", "mean_normal_difference",
                "exclusive_boundary_contact_ratio_mean", "public_projected_box_iou_mean",
                "public_same_matched_observation_fraction",
                "public_different_matched_observation_fraction",
                "public_track_minus_native_gvc",
            )
        },
    }
    stacked = {
        "nested_track_q": 0.7,
        "nested_native_q_median": 0.6,
        "nested_track_valid25": 0.8,
        "nested_native_valid25_median": 0.7,
        "nested_track_valid50": 0.6,
        "nested_native_valid50_median": 0.5,
        "same_target_score": 0.9,
        "different_target_score": 0.1,
        "track_better_score": 0.4,
        "baseline_better_score": 0.6,
        "track_win_relation_score": 0.36,
        "baseline_win_relation_score": 0.54,
        "coexist_relation_score": 0.1,
        "multitask_track_better_score": 0.7,
        "multitask_baseline_better_score": 0.3,
        "multitask_track_win_relation_score": 0.63,
        "multitask_baseline_win_relation_score": 0.27,
    }
    row = build_candidate_feature_rows(
        [candidate], {("s", 0): [raw]}, {("s", 0): [stacked]}
    )[0]
    features = row["model_features"]
    assert features["relation__candidate_win_relation_score__mean"] == 0.36
    assert features["relation__multitask_candidate_win_relation_score__mean"] == 0.63
    assert features["relation__multitask_track_better_score__mean"] == 0.7
    baseline = strip_multitask_auxiliary_features([row])[0]["model_features"]
    residual = collapse_multitask_auxiliary_to_candidate_win_residual(
        [row]
    )[0]["model_features"]
    assert not any(name.startswith("relation__multitask_") for name in baseline)
    assert residual["relation__candidate_win_relation_score__mean"] == 0.36
    assert residual["relation__multitask_candidate_win_residual__mean"] == 0.27
    assert len(residual) == len(baseline) + 1
