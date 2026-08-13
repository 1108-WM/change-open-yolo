import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_anchor_guided_track_expansions.py"
    spec = importlib.util.spec_from_file_location("anchor_guided_expansions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selects_complete_high_quality_mask_for_anchor():
    module = _module()
    observations = [
        {"observation_id": 1, "frame_id": "0", "frame_index": 0, "predicted_iou": 0.99, "stability_score": 0.99,
         "points": np.asarray([0, 1, 7])},
        {"observation_id": 2, "frame_id": "0", "frame_index": 0, "predicted_iou": 0.90, "stability_score": 0.95,
         "points": np.asarray([0, 1, 2, 3, 4, 8])},
    ]
    result = module.select_mask_for_anchor(np.asarray([0, 1, 2, 3, 4]), observations, 0.60, 3)
    assert result["observation_id"] == 2
    assert result["shared_anchor_point_count"] == 5
