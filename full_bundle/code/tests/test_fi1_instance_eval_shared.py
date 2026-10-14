import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "fi1_instance_eval_shared.py"
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
