import numpy as np

from tools.train_candidate_relative_margin_oof import (
    IOU_MARGIN,
    margin_training_indexes,
    nested_margin_fit_predict,
    scene_block_conformal_correction,
)


def _row(scene, track_id, state, margin=None):
    reliable = state != "unknown"
    target = "same_target" if state != "coexist" and reliable else (
        "different_target_coexist" if state == "coexist" else "unknown"
    )
    return {
        "scene_name": scene,
        "track_id": track_id,
        "labels": {
            "reliable_pair": reliable,
            "target_state": target,
            "relative_quality_state": state,
            "iou_margin_track_minus_native": margin,
        },
    }


def test_margin_contract_includes_equivalent_and_excludes_unknown_and_coexist():
    rows = [
        _row("a", 0, "prefer_track", 0.2),
        _row("a", 1, "prefer_native", -0.2),
        _row("b", 2, "equivalent_abstain", 0.01),
        _row("b", 3, "coexist", 0.3),
        _row("c", 4, "unknown", None),
    ]
    assert margin_training_indexes(rows).tolist() == [0, 1, 2]


def test_scene_block_conformal_uses_maximum_residual_per_scene():
    rows = [
        _row("a", 0, "prefer_track", 0.2),
        _row("a", 1, "prefer_native", -0.2),
        _row("b", 2, "equivalent_abstain", 0.0),
        _row("b", 3, "prefer_track", 0.3),
    ]
    margins = np.asarray([0.2, -0.2, 0.0, 0.3])
    lower = np.asarray([0.1, 0.1, -0.1, 0.2])
    correction, diagnostics = scene_block_conformal_correction(
        rows, np.arange(4), margins, lower, alpha=0.5
    )
    # Scene a's maximum residual is 0.3; scene b's is -0.1.
    assert np.isclose(correction, 0.3)
    assert diagnostics["calibration_scene_count"] == 2


def test_nested_margin_outputs_probability_and_conservative_lower_bound():
    rows, margins, matrix, scene_to_fold = [], [], [], {}
    for scene_index in range(10):
        scene = f"scene{scene_index}"
        scene_to_fold[scene] = scene_index // 2
        for relation_index, margin in enumerate((-0.3, -0.1, 0.0, 0.15)):
            state = "prefer_track" if margin > IOU_MARGIN else (
                "prefer_native" if margin < -IOU_MARGIN else "equivalent_abstain"
            )
            rows.append(_row(scene, relation_index, state, margin))
            margins.append(margin)
            matrix.append([margin + 0.02 * scene_index, relation_index])
    margins = np.asarray(margins, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    train = np.asarray([
        index for index, row in enumerate(rows) if scene_to_fold[row["scene_name"]] != 0
    ])
    validation = np.asarray([
        index for index, row in enumerate(rows) if scene_to_fold[row["scene_name"]] == 0
    ])
    values, diagnostics = nested_margin_fit_predict(
        rows, matrix, margins, train, validation, scene_to_fold, outer_fold_index=0
    )
    assert set(values) == {
        "center_margin", "probability_margin_gt_005",
        "lower_quantile_raw", "lower_conformal",
    }
    assert all(len(value) == len(validation) for value in values.values())
    assert np.all((values["probability_margin_gt_005"] >= 0) &
                  (values["probability_margin_gt_005"] <= 1))
    assert diagnostics["conformal"]["calibration_scene_count"] == 8
