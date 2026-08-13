from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE
from tools.train_candidate_union_ap25_protected_head_oof import (
    assign_track_protection_labels,
    protected_preserve_probability,
    suppressed_track_score,
)


def test_track_protection_labels_preserve_unique_and_better_tracks():
    rows = [
        {"scene_name": "s", "relation_component_id": 0, "candidate_source": NATIVE_SOURCE,
         "candidate_id": 0, "label_best_gt_instance_id": 4, "label_best_gt_iou": 0.6},
        {"scene_name": "s", "relation_component_id": 0, "candidate_source": TRACK_SOURCE,
         "candidate_id": 1, "label_best_gt_instance_id": 4, "label_best_gt_iou": 0.5},
        {"scene_name": "s", "relation_component_id": 0, "candidate_source": TRACK_SOURCE,
         "candidate_id": 2, "label_best_gt_instance_id": 8, "label_best_gt_iou": 0.4},
        {"scene_name": "s", "relation_component_id": 0, "candidate_source": TRACK_SOURCE,
         "candidate_id": 3, "label_best_gt_instance_id": 4, "label_best_gt_iou": 0.7},
    ]
    assign_track_protection_labels(rows)
    tracks = {row["candidate_id"]: row for row in rows if row["candidate_source"] == TRACK_SOURCE}
    assert tracks[1]["label_native_dominated"] == 1
    assert tracks[1]["label_safe_keep25"] == 0
    assert tracks[2]["label_unique_valid25"] == 1
    assert tracks[2]["label_safe_keep25"] == 1
    assert tracks[3]["label_native_dominated"] == 0
    assert tracks[3]["label_safe_keep25"] == 1


def test_protected_probability_uses_unique_value_to_veto_domination_risk():
    low_unique = protected_preserve_probability(0.9, 0.0, 0.8)
    high_unique = protected_preserve_probability(0.9, 1.0, 0.8)
    assert high_unique == 0.9
    assert high_unique > low_unique


def test_suppressed_track_score_is_continuous_and_nonincreasing():
    multiplier, score = suppressed_track_score(0.8, 0.5)
    assert multiplier == 0.875
    assert abs(score - 0.7) < 1e-12
