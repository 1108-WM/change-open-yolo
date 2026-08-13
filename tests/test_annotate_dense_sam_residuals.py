import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "annotate_dense_sam_residuals.py"
SPEC = importlib.util.spec_from_file_location("annotate_dense_sam_residuals", MODULE_PATH)
ANNOTATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANNOTATOR)


def test_overlap_metrics_keep_any_and_same_class_residuals_separate():
    masks = np.zeros((8, 3), dtype=bool)
    masks[[0, 1, 2], 0] = True
    masks[[0, 3, 5], 1] = True
    masks[[4, 6], 2] = True
    classes = np.asarray([4, 7, 4], dtype=np.int64)

    metrics, residual_any, residual_same = ANNOTATOR._overlap_metrics(
        masks,
        classes,
        np.asarray([0, 1, 2, 5]),
        class_id=4,
    )

    assert metrics["best_any_candidate_id"] == 0
    assert metrics["best_same_class_candidate_id"] == 0
    assert np.isclose(metrics["best_same_class_seed_coverage"], 0.75)
    assert np.isclose(metrics["any_candidate_seed_coverage"], 1.0)
    assert residual_any.size == 0
    assert residual_same.tolist() == [5]


def test_overlap_metrics_handles_empty_prediction_set():
    metrics, residual_any, residual_same = ANNOTATOR._overlap_metrics(
        np.zeros((5, 0), dtype=bool),
        np.asarray([], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        class_id=2,
    )

    assert metrics["candidate_count"] == 0
    assert residual_any.tolist() == [1, 3]
    assert residual_same.tolist() == [1, 3]
