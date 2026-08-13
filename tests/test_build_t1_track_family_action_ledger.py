import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_t1_track_family_action_ledger.py"
    spec = importlib.util.spec_from_file_location("t1_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_track_family_actions_keep_frozen_tracks_and_never_materialize():
    module = _module()
    nodes = {
        1: {"frame_index": 0, "lifted_superpoints": [1, 2]},
        2: {"frame_index": 1, "lifted_superpoints": [1, 2]},
        3: {"frame_index": 2, "lifted_superpoints": [1]},
        4: {"frame_index": 3, "lifted_superpoints": [2]},
    }
    tracks = {
        10: {"track_id": 10, "observation_ids": (1, 2)},
        20: {"track_id": 20, "observation_ids": (4,)},
    }
    owner = {1: 10, 2: 10, 4: 20}
    features = {
        (2, 3): {"recall_reasons": ["shared_lifted_superpoint"], "common_visible_superpoint_siou": .8,
                 "relative_depth_weighted_iou": .7, "left_to_right_reprojection_support_ratio": .5,
                 "right_to_left_reprojection_support_ratio": .5, "centroid_distance": .1,
                 "rgb_mean_l2_difference": .1, "texture_rgb_std_l2_difference": .1,
                 "normal_difference": .1, "dino_vits14_state": "deferred_cuda_unavailable"},
        (2, 4): {"recall_reasons": ["centroid_knn"], "common_visible_superpoint_siou": .3,
                 "relative_depth_weighted_iou": .2, "left_to_right_reprojection_support_ratio": .2,
                 "right_to_left_reprojection_support_ratio": .2, "centroid_distance": .2,
                 "rgb_mean_l2_difference": .2, "texture_rgb_std_l2_difference": .2,
                 "normal_difference": .2, "dino_vits14_state": "deferred_cuda_unavailable"},
    }
    relations, actions = module.build_scene_ledger("scene_unit", nodes, tracks, owner, features, {}, {})
    assert any(row["action_type"] == "attach" for row in actions)
    assert any(row["action_type"] == "merge" for row in actions)
    assert any(row["action_type"] == "reassign" for row in actions)
    keeps = [row for row in actions if row["action_type"] == "keep"]
    assert {row["source_track_id"] for row in keeps} == {10, 20}
    assert all(row["proposal_materialization_applied"] is False for row in actions)
    assert all(row["ground_truth_usage"] == "none" for row in relations)
