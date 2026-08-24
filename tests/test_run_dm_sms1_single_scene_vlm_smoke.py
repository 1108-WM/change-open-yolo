from PIL import Image

import pytest

from tools.run_dm_sms1_single_scene_vlm_smoke import (
    _target_images,
    _validate_candidate_output,
)


def test_target_images_build_three_equal_width_composites(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (80, 60), (120, 130, 140)).save(path)
    row = {
        "view_inputs": [
            {"rgb_path": str(path), "sam_box_prompt_xyxy": [20, 15, 40, 35]}
            for _ in range(3)
        ]
    }
    images = _target_images(row)
    assert len(images) == 3
    assert all(image.size == (160, 60) for image in images)


def _candidate_item(supported=False, strong=False, support="", counter=""):
    return {
        "class_index": 0,
        "supported": supported,
        "strong_counterevidence": strong,
        "support_evidence": support,
        "counterevidence": counter,
        "confidence": 0.8,
    }


def test_candidate_output_rejects_empty_asserted_support_text():
    output = {"candidate_results": [_candidate_item(supported=True)]}
    with pytest.raises(ValueError, match="support true"):
        _validate_candidate_output(output, [0])


def test_candidate_output_rejects_empty_asserted_counterevidence_text():
    output = {"candidate_results": [_candidate_item(strong=True)]}
    with pytest.raises(ValueError, match="strong counterevidence true"):
        _validate_candidate_output(output, [0])
