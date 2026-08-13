import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "train_z6d_nested_improvement_gate_oof.py"
    spec = importlib.util.spec_from_file_location("z6d_nested_gate", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_binary_gate_predicts_improvement_probability():
    module = _module()
    gate = module._make_gate(7)
    features = np.concatenate([
        np.zeros((100, 32), dtype=np.float32),
        np.ones((100, 32), dtype=np.float32),
    ])
    labels = np.concatenate([np.zeros(100, dtype=np.int8), np.ones(100, dtype=np.int8)])
    gate.fit(features, labels)
    probabilities = gate.predict_proba(features)[:, 1]
    assert probabilities[:100].mean() < 0.5
    assert probabilities[100:].mean() > 0.5
