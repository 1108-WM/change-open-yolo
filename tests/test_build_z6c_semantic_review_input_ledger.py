import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_z6c_semantic_review_input_ledger.py"
    spec = importlib.util.spec_from_file_location("z6c_review_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_rows_build_deterministic_model_union():
    module = _module()
    yolo = np.asarray([0.8, 0.0, 0.2, 0.1], dtype=np.float32)
    alpha = np.asarray([0.0, 0.9, 0.3, 0.0], dtype=np.float32)
    rows, yolo_top, alpha_top = module._candidate_rows(
        yolo, alpha, ["a", "b", "c", "d"], 2
    )
    assert yolo_top == [0, 2]
    assert alpha_top == [1, 2]
    assert [row["class_index"] for row in rows] == [1, 0, 2]
    assert rows[2]["in_yolo_top5"] and rows[2]["in_alpha_top5"]


def test_margin_and_entropy_handle_empty_distribution():
    module = _module()
    values = np.zeros(198, dtype=np.float32)
    assert module._margin(values) == 0.0
    assert module._entropy(values) == 0.0


def test_pair_union_keeps_frozen_original_score():
    module = _module()
    row = {
        "candidate_source": "pair_union",
        "original_score": 0.125,
        "oof_predictions": {"C_joint_yolo_alpha": 0.9},
    }
    assert module._hypothesis_score(row) == 0.125
