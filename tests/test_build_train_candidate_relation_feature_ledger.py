import json

import numpy as np

from tools.build_train_candidate_relation_feature_ledger import (
    _candidate_stats,
    _geometry_features,
    _exclusive_point_features,
    _exclusive_public_view_features,
    _load_fixed_scores,
    _public_view_features,
    _relation_components,
    build_inference_relation_rows,
)


def test_fixed_score_loader_does_not_admit_relation_identity(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({
        "scene_name": "scene0001_00",
        "track_id": 7,
        "native_exact_geometry_group_id": "group0",
        "label_pair_preference": "prefer_track",
        "delta_raw_original_score": 0.2,
        "track_D_plus_gvc_q": 0.8,
        "native_median_D_plus_gvc_q": 0.6,
        "delta_D_plus_gvc_q": 0.2,
    }) + "\n")
    values = next(iter(_load_fixed_scores(path).values()))
    assert "track_id" not in values
    assert "label_pair_preference" not in values
    assert values["delta_D_plus_gvc_q"] == 0.2


def test_geometry_features_record_shared_superpoints_and_boundary_continuity():
    processed = np.zeros((6, 12), dtype=np.float32)
    processed[:, :3] = np.asarray([
        [0, 0, 0], [0.01, 0, 0], [0.02, 0, 0],
        [0.04, 0, 0], [0.05, 0, 0], [0.06, 0, 0],
    ])
    processed[:, 3:6] = np.asarray([[255, 0, 0]] * 3 + [[250, 5, 0]] * 3)
    processed[:, 6:9] = np.asarray([[0, 0, 1]] * 6)
    processed[:, 9] = np.asarray([0, 0, 1, 1, 2, 2])
    edge01 = {
        "neighbor_superpoint_id": 1,
        "boundary_contact_count": 4,
        "boundary_contact_ratio": 0.5,
        "mean_boundary_distance": 0.02,
        "mean_normal_difference": 0.0,
        "mean_color_difference": 0.02,
    }
    edge10 = {**edge01, "neighbor_superpoint_id": 0}
    edge12 = {**edge01, "neighbor_superpoint_id": 2}
    edge21 = {**edge01, "neighbor_superpoint_id": 1}
    context = {
        "raw_ids": np.asarray([0, 1, 2]),
        "sizes": np.asarray([2, 2, 2]),
        "neighbors": {0: [edge01], 1: [edge10, edge12], 2: [edge21]},
    }
    track = _candidate_stats(np.asarray([0, 1, 2, 3]), processed, context)
    native = _candidate_stats(np.asarray([2, 3, 4, 5]), processed, context)
    features = _geometry_features(track, native, context)
    assert features["shared_superpoint_count"] == 1
    assert features["track_superpoint_component_count"] == 1
    assert features["exclusive_boundary_superpoint_edge_count"] == 0
    assert features["all_cross_boundary_superpoint_edge_count"] == 2
    assert features["mean_normal_difference"] == 0.0


def test_public_views_capture_projected_overlap_same_observation_and_range():
    def row(frame, box, observation):
        return {
            "frame_index": frame,
            "visible_point_count": 100,
            "visible_point_fraction": 0.5,
            "projected_box": np.asarray(box, dtype=np.float64),
            "matched_observation_id": observation,
            "depth_consistent": True,
            "gvc_frame_score": 0.8,
        }

    track = {0: row(0, [0, 0, 10, 10], 5), 1: row(1, [0, 0, 10, 10], 6)}
    native = {0: row(0, [0, 0, 10, 10], 5), 1: row(1, [20, 20, 30, 30], 7)}
    features = _public_view_features(
        track, native, np.asarray([1.0, 0.0, 0.0]), np.asarray([1.1, 0.0, 0.0]),
        {0: np.zeros(3), 1: np.zeros(3)},
    )
    assert features["public_common_selected_view_count"] == 2
    assert features["public_same_matched_observation_count"] == 1
    assert features["public_different_matched_observation_count"] == 1
    assert features["public_projected_box_iou_mean"] == 0.5
    assert features["public_centroid_camera_range_absolute_delta_mean"] > 0.0


def test_exclusive_pair_features_preserve_direction_and_view_conflict():
    point_features, track_only, native_only = _exclusive_point_features(
        np.asarray([0, 1, 2, 3]), np.asarray([2, 3, 4]),
    )
    assert track_only.tolist() == [0, 1]
    assert native_only.tolist() == [4]
    assert point_features["track_exclusive_fraction_of_pair_exclusive"] == 2.0 / 3.0

    def row(frame, box, observation, gvc):
        return {
            "frame_index": frame,
            "visible_point_count": 100,
            "visible_point_fraction": 0.5,
            "projected_box": np.asarray(box, dtype=np.float64),
            "matched_observation_id": observation,
            "mask_point_support": gvc,
            "gvc_frame_score": gvc,
        }

    track = {
        0: row(0, [0, 0, 10, 10], 5, 0.8),
        1: row(1, [0, 0, 10, 10], 6, 0.2),
    }
    native = {
        0: row(0, [0, 0, 10, 10], 5, 0.3),
        1: row(1, [20, 20, 30, 30], 7, 0.7),
    }
    features = _exclusive_public_view_features(track, native)
    assert features["exclusive_public_same_matched_observation_fraction"] == 0.5
    assert features["exclusive_public_different_matched_observation_fraction"] == 0.5
    assert features["exclusive_public_track_gvc_win_fraction"] == 0.5
    assert features["exclusive_public_native_gvc_win_fraction"] == 0.5


def test_relation_components_conserve_pairs_and_native_members():
    rows = [
        {"track_id": 1, "native_exact_geometry_group_id": "a", "native_exact_geometry_group_size": 3},
        {"track_id": 1, "native_exact_geometry_group_id": "b", "native_exact_geometry_group_size": 2},
        {"track_id": 2, "native_exact_geometry_group_id": "b", "native_exact_geometry_group_size": 2},
    ]
    lookup, components = _relation_components(rows)
    assert len(lookup) == 3
    assert len(components) == 1
    assert components[0]["track_count"] == 2
    assert components[0]["native_geometry_group_count"] == 2
    assert components[0]["native_member_candidate_count"] == 5


def test_inference_relation_builder_requires_neither_labels_nor_learned_scores():
    processed = np.zeros((4, 10), dtype=np.float32)
    processed[:, :3] = np.asarray([
        [0.0, 0.0, 0.0], [0.01, 0.0, 0.0],
        [0.02, 0.0, 0.0], [0.03, 0.0, 0.0],
    ])
    processed[:, 6:9] = np.asarray([[0, 0, 1]] * 4)
    processed[:, 9] = 0
    context = {
        "raw_ids": np.asarray([0]),
        "sizes": np.asarray([4]),
        "neighbors": {0: []},
    }
    pair = {
        "track_id": 7,
        "native_exact_geometry_group_id": "g0",
        "native_member_candidate_ids": [0, 1],
        "native_exact_geometry_group_size": 2,
        "point_iou": 0.5,
        "track_inside_native_ratio": 1.0,
        "native_inside_track_ratio": 0.5,
        "mutual_duplicate_strict_099": False,
        "track_original_score": 0.8,
        "native_original_score_median": 0.7,
    }
    rows, components, summary = build_inference_relation_rows(
        "scene0000_00", [pair], processed, context,
        {7: np.asarray([0, 1])}, {"g0": np.asarray([0, 1, 2, 3])},
        {7: {}}, {"g0": {}}, {},
    )
    assert len(rows) == len(components) == 1
    assert "labels" not in rows[0]
    assert rows[0]["contracts"]["feature_ground_truth_usage"] == "none"
    assert rows[0]["contracts"]["label_ground_truth_usage"] == "none"
    assert summary["learned_score_fields_attached"] is False
