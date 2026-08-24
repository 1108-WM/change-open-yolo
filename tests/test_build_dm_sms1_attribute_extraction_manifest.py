from tools.build_dm_sms1_attribute_extraction_manifest import (
    ATTRIBUTE_PROMPT,
    build_attribute_row,
)


def test_attribute_row_hides_candidate_labels_and_keeps_only_input_evidence():
    row = {
        "scene_name": "scene0001_00",
        "geometry_key": "scene0001_00:geometry:x",
        "geometry_hash": "x",
        "point_count": 10,
        "finite_class_hypotheses": [{"class_index": 1}, {"class_index": 2}],
        "selected_views": [{
            "selection_rank": 0, "frame_id": "0", "frame_index": 0,
            "rgb_path": "/tmp/rgb.jpg", "depth_path": "/tmp/depth.png",
            "pose_path": "/tmp/pose.txt", "intrinsics_path": "/tmp/intrinsics.txt",
            "sam_box_prompt_xyxy": [0, 0, 1, 1], "sam_mask_sha256": "m",
            "visible_ratio": 0.5, "visible_point_count": 5,
        }],
    }
    result = build_attribute_row(row)
    assert result["candidate_labels_hidden"] is True
    assert "finite_class_hypotheses" not in result
    assert result["candidate_hypothesis_count"] == 2
    assert result["attribute_prompt"] == ATTRIBUTE_PROMPT
    assert result["class_decision_made"] is False
