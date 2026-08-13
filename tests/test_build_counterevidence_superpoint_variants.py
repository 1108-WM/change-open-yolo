import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_counterevidence_superpoint_variants.py"
    spec = importlib.util.spec_from_file_location("counterevidence_variants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kept_superpoints_only_removes_visible_negative_margin_evidence():
    module = _module()
    kept = module._kept_superpoints(
        superpoint_ids=np.asarray([10, 20, 30]),
        visible_view_counts=np.asarray([2, 0, 3]),
        evidence_margins=np.asarray([-0.2, -1.0, 0.0]),
    )
    assert kept.tolist() == [20, 30]


def test_variant_points_preserves_points_from_unknown_superpoints():
    module = _module()
    variant = module._variant_points(
        points=np.asarray([0, 1, 2, 3]),
        point_superpoints=np.asarray([10, 10, 20, 30]),
        kept_superpoints=np.asarray([20, 30]),
    )
    assert variant.tolist() == [2, 3]
