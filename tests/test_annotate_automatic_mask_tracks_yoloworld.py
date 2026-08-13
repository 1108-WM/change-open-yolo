import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "annotate_automatic_mask_tracks_yoloworld.py"
    spec = importlib.util.spec_from_file_location("automatic_track_semantics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_class_votes_keeps_strongest_box_per_class():
    module = _module()
    votes = module.frame_class_votes(
        np.asarray([5.0, 6.0, 20.0]),
        np.asarray([5.0, 6.0, 20.0]),
        np.asarray([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [15.0, 15.0, 25.0, 25.0]]),
        np.asarray([2, 2, 3]),
        np.asarray([0.4, 0.8, 0.9]),
    )
    assert votes[2] == 0.8 * (2 / 3)
    assert votes[3] == 0.9 * (1 / 3)
