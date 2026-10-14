import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "fi1_superpoint_geometry.py"
    spec = importlib.util.spec_from_file_location("automatic_sam_track_growth_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_superpoint_context_keeps_original_ids_and_contact_features():
    ledger = _module()
    processed = np.zeros((6, 10), dtype=np.float32)
    processed[:, :3] = np.asarray([
        [0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0],
        [0.03, 0.0, 0.0], [0.04, 0.0, 0.0], [0.05, 0.0, 0.0],
    ])
    processed[:, 3:6] = 128
    processed[:, 6:9] = [0.0, 0.0, 1.0]
    processed[:, 9] = [10, 10, 10, 30, 30, 30]
    context = ledger._raw_superpoint_context(processed, 2, 0.03, 1, 0.0)
    assert context["raw_ids"].tolist() == [10, 30]
    assert context["neighbors"][10][0]["neighbor_superpoint_id"] == 30
    assert context["neighbors"][10][0]["boundary_contact_count"] > 0


