import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_yoloe_independent_coverage_gt.py"
    spec = importlib.util.spec_from_file_location("diagnose_yoloe_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_coverage_uses_only_the_requested_class():
    module = _module()
    prediction = {
        "boxes_xyxy": np.asarray([[0, 0, 2, 2], [3, 3, 5, 5]], dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "scores": np.asarray([0.8, 0.9], dtype=np.float32),
    }
    coords = np.asarray([[0, 0], [1, 1], [4, 4]], dtype=np.int64)
    coverage, score, count = module._frame_coverage(coords, prediction, 1, (1.0, 1.0))
    assert coverage == 1 / 3
    assert score == np.float32(0.9)
    assert count == 1


def test_comparison_label_marks_yoloe_only_evidence():
    module = _module()
    assert module._comparison_label({"reliable": False}, {"reliable": True}) == "仅 YOLOE 有可靠二维证据"
