import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_counterevidence_reliability_ledger.py"
    spec = importlib.util.spec_from_file_location("reliability_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_sampling_marks_uncovered_pixels_next_to_boundary_as_uncertain():
    module = _module()
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    coverage, uncertain, interior = module._mask_sample_features(mask, np.asarray([[1, 4], [4, 4]]))
    assert coverage == 0.5
    assert uncertain == 0.5
    assert interior == 0.5


def test_view_independence_requires_two_distinct_camera_centers():
    module = _module()
    baseline, angle = module._pairwise_view_features([np.asarray([0.0, 0.0, 0.0])], np.asarray([0.0, 0.0, 2.0]))
    assert baseline == 0.0
    assert angle == 0.0
