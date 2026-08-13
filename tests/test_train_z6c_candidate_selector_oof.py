import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "train_z6c_candidate_selector_oof.py"
    spec = importlib.util.spec_from_file_location("train_z6c_selector", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_select_option_prefers_current_on_exact_tie():
    module = _module()
    rows = [
        {"option_is_current": False, "option_class_index": 1},
        {"option_is_current": True, "option_class_index": 2},
    ]
    chosen, margin = module._select_option(rows, np.asarray([0, 1]), np.asarray([0.5, 0.5]))
    assert chosen == 1
    assert margin == 0.0


def test_prediction_groups_preserve_first_seen_order():
    module = _module()
    rows = [
        {"scene_name": "b", "prediction_index": 2},
        {"scene_name": "b", "prediction_index": 2},
        {"scene_name": "a", "prediction_index": 0},
    ]
    groups = module._prediction_groups(rows)
    assert [group.tolist() for group in groups] == [[0, 1], [2]]
