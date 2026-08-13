import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "evaluate_z6c_candidate_selector_oof_gt.py"
    spec = importlib.util.spec_from_file_location("eval_z6c_selector", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_apply_selections_keeps_scores_and_changes_registered_class_only():
    module = _module()
    prediction = {
        "pred_masks": np.ones((2, 2), dtype=bool),
        "pred_classes": np.asarray([1, 2]),
        "pred_scores": np.asarray([0.5, 0.2]),
    }
    selections = [{
        "prediction_index": 1, "current_class_index": 2,
        "selectors": {"semantic_plus_dino": {"selected_class_index": 3}},
    }]
    output = module._apply_selections(prediction, selections, "semantic_plus_dino")
    assert output["pred_classes"].tolist() == [1, 3]
    assert np.array_equal(output["pred_scores"], prediction["pred_scores"])
