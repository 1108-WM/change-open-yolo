import numpy as np

from utils.backprojection_fusion import append_backprojection_proposals


def test_gvc_append_only_keeps_native_and_deduplicates_only_same_class(tmp_path):
    def candidate(candidate_id, class_id, points):
        path = tmp_path / f"{candidate_id}.npz"
        np.savez_compressed(path, point_indices=np.asarray(points, dtype=np.int64))
        return {
            "candidate_id": candidate_id,
            "source_kind": "gvc_append_only",
            "class_id": class_id,
            "class_name": str(class_id),
            "score": 0.8,
            "fusion_score": 0.8,
            "proposal_priority": 0.8,
            "seed_points_path": str(path),
            "num_seed_points": len(points),
            "best_existing_iou": 1.0,
            "seed_in_existing_mask_ratio": 1.0,
        }

    masks = np.asarray([[1], [1], [0], [0]], dtype=bool)
    classes = np.asarray([0], dtype=np.int64)
    scores = np.asarray([0.9], dtype=np.float32)
    candidates = {"scene0000_00": [candidate(1, 2, [0, 1]), candidate(2, 2, [0, 1]), candidate(3, 3, [0, 1])]}
    output_masks, output_classes, output_scores, report = append_backprojection_proposals(
        "scene0000_00", masks, classes, scores, candidates,
        min_score=0.0, min_seed_points=1, max_existing_iou=0.0, max_seed_in_existing_mask_ratio=0.0,
        append_only_source_kinds="gvc_append_only", append_only_same_class_dedup_iou=0.5,
    )
    assert output_masks.shape[1] == 3
    assert output_classes.tolist() == [0, 2, 3]
    assert output_scores[0] == scores[0]
    assert report["summary"]["applied_count"] == 2
    assert any(item["reason"] == "duplicate_append_only_same_class_proposal" for item in report["skipped"])
