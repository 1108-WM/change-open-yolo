import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_automatic_track_alphaclip_semantics.py"
    spec = importlib.util.spec_from_file_location("automatic_track_alphaclip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_track_views_uses_only_track_frames_and_visibility():
    module = _module()
    points = np.asarray([0, 1, 2, 3])
    projections = np.asarray([[[2, 2], [4, 4], [6, 6], [8, 8]], [[2, 2], [4, 4], [6, 6], [8, 8]]])
    visibility = np.asarray([[True, True, False, False], [True, True, True, True]])
    selected = module._select_track_views(
        points, ["a", "b"], {"a": 0, "b": 1}, projections, visibility, (1.0, 1.0), (20, 20), 1, 2
    )
    assert len(selected) == 1
    assert selected[0]["frame_id"] == "b"
    assert selected[0]["visible_points"] == 4
