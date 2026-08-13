import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_anchor_guided_expansion_alphaclip_semantics.py"
    spec = importlib.util.spec_from_file_location("expansion_alphaclip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_views_restricts_to_supported_frames():
    module = _module()
    points = np.asarray([0, 1, 2, 3])
    projections = np.asarray([[[2, 2], [4, 4], [6, 6], [8, 8]], [[2, 2], [4, 4], [6, 6], [8, 8]]])
    visibility = np.asarray([[True, True, False, False], [True, True, True, True]])
    views = module._select_views(points, ["b"], {"a": 0, "b": 1}, projections, visibility, (1.0, 1.0), (20, 20), 3, 2)
    assert len(views) == 1
    assert views[0]["frame_id"] == "b"
