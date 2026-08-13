import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "train_z6c_nested_abstain_gate_oof.py"
    spec = importlib.util.spec_from_file_location("z6c_nested_gate", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_choose_returns_selected_and_current_rows():
    module = _module()
    rows = [
        {"option_is_current": True, "option_class_index": 1},
        {"option_is_current": False, "option_class_index": 2},
    ]
    selected, current, margin = module._choose(rows, np.asarray([0, 1]), np.asarray([0.2, 0.8]))
    assert (selected, current) == (1, 0)
    assert np.isclose(margin, 0.6)


def test_gate_feature_has_fixed_class_id_free_shape():
    module = _module()
    features = np.zeros((2, 30), dtype=np.float32)
    value = module._gate_feature(features, np.asarray([0.2, 0.8]), 1, 0, 0.6)
    assert value.shape == (32,)


def test_gate_feature_batch_is_predictable():
    module = _module()
    features = np.zeros((2, 30), dtype=np.float32)
    batch = np.asarray([
        module._gate_feature(features, np.asarray([0.2, 0.8]), 1, 0, 0.6),
        module._gate_feature(features, np.asarray([0.3, 0.7]), 1, None, 0.4),
    ], dtype=np.float32)
    assert batch.shape == (2, 32)
