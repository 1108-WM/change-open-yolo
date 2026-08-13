import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_gvc_class_agnostic_ap.py"
    spec = importlib.util.spec_from_file_location("gvc_class_agnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_class_agnostic_mapping_preserves_instances_and_discards_void_labels():
    module = _module()
    valid_class = next(iter(module.VALID_GT_CLASSES))
    ids = np.asarray([0, valid_class * 1000 + 1, valid_class * 1000 + 1, 999 * 1000 + 1, valid_class * 1000 + 2])
    mapped = module._class_agnostic_gt_ids(ids)
    assert mapped.tolist() == [0, 2001, 2001, 0, 2002]


def test_append_tracks_preserves_native_predictions(tmp_path):
    module = _module()
    points_path = tmp_path / "track_points.npz"
    np.savez_compressed(points_path, point_indices=np.asarray([1, 3], dtype=np.int64))
    native_masks = np.asarray([[1], [0], [1], [0]], dtype=bool)
    native_scores = np.asarray([0.8], dtype=np.float32)
    combined = module.append_track_predictions(
        {
            "pred_masks": native_masks,
            "pred_scores": native_scores,
            "pred_classes": np.asarray([5], dtype=np.int64),
        },
        [{"points_path": str(points_path), "mean_node_quality": 0.6}],
    )
    assert np.array_equal(combined["pred_masks"][:, :1], native_masks)
    assert np.array_equal(combined["pred_scores"][:1], native_scores)
    assert combined["pred_masks"][:, 1].tolist() == [False, True, False, True]
    assert np.isclose(combined["pred_scores"][1], 0.6)


def test_merge_scan_matches_concatenates_predictions_and_checks_gt_identity():
    module = _module()
    native_gt = {
        "chair": [{
            "instance_id": 2001, "label_id": 2, "vert_count": 10,
            "med_dist": 0.0, "dist_conf": 1.0, "matched_pred": [{"uuid": "n"}],
        }]
    }
    track_gt = {
        "chair": [{
            "instance_id": 2001, "label_id": 2, "vert_count": 10,
            "med_dist": 0.0, "dist_conf": 1.0, "matched_pred": [{"uuid": "t"}],
        }]
    }
    native_pred = {"chair": [{"uuid": "n"}]}
    track_pred = {"chair": [{"uuid": "t"}]}
    merged_gt, merged_pred = module._merge_scan_matches(
        native_gt, native_pred, track_gt, track_pred
    )
    assert [row["uuid"] for row in merged_gt["chair"][0]["matched_pred"]] == ["n", "t"]
    assert [row["uuid"] for row in merged_pred["chair"]] == ["n", "t"]

    track_gt["chair"][0]["instance_id"] = 2002
    try:
        module._merge_scan_matches(native_gt, native_pred, track_gt, track_pred)
    except ValueError as error:
        assert "identity differs" in str(error)
    else:
        raise AssertionError("mismatched GT identity was accepted")
