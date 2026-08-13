from tools.train_candidate_component_list_calibration_head_oof import (
    assign_component_labels,
    calibrated_score,
    focal_risk_target,
)


def test_assign_component_labels_keeps_one_candidate_per_target_jointly():
    rows = [
        {
            "scene_name": "s",
            "relation_component_id": 0,
            "candidate_source": "native_mask3d_yoloworld",
            "candidate_id": 0,
            "label_best_gt_instance_id": 4,
            "label_best_gt_iou": 0.8,
            "original_source_score": 1.0,
        },
        {
            "scene_name": "s",
            "relation_component_id": 0,
            "candidate_source": "d2b_track",
            "candidate_id": 0,
            "label_best_gt_instance_id": 4,
            "label_best_gt_iou": 0.7,
            "original_source_score": 0.9,
        },
        {
            "scene_name": "s",
            "relation_component_id": 0,
            "candidate_source": "d2b_track",
            "candidate_id": 1,
            "label_best_gt_instance_id": 8,
            "label_best_gt_iou": 0.6,
            "original_source_score": 0.8,
        },
    ]
    assign_component_labels(rows)
    assert [row["label_component_unique_winner"] for row in rows] == [1, 0, 1]
    assert rows[1]["label_calibrated_quality"] == 0.0
    assert rows[1]["label_harm_kind"] == "duplicate_valid"


def test_calibrated_score_exactly_uses_logit_residual_contract():
    new_score, delta = calibrated_score(0.99, 0.2)
    assert abs(new_score - 0.2) < 1e-12
    assert delta < 0.0


def test_focal_risk_target_preserves_confident_native_and_moves_tracks_both_ways():
    assert focal_risk_target(1.0, 0.95) > 0.999
    assert focal_risk_target(0.7, 0.95) > 0.95
    assert focal_risk_target(0.7, 0.05) < 0.2
