import importlib.util
from pathlib import Path

import numpy as np
import torch


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_yoloe_f30_mask_observations.py"
    spec = importlib.util.spec_from_file_location("export_yoloe_masks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_to_visible_points_uses_projection_scaling():
    module = _module()
    mask = np.asarray([[True, False], [False, True]])
    projections = np.asarray([[[0, 0], [2, 2], [3, 0]]], dtype=np.float32)
    visibility = np.asarray([[True, True, True]])
    points = module._mask_to_visible_points(mask, 0, projections, visibility, (2.0, 2.0))
    assert np.array_equal(points, np.asarray([0, 1]))


def test_result_arrays_resizes_masks_and_keeps_detection_order():
    module = _module()

    class Boxes:
        cls = torch.tensor([4, 2])
        conf = torch.tensor([0.7, 0.8])
        xyxy = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=torch.float32)

        def __len__(self):
            return 2

    class Masks:
        data = torch.tensor([[[1]], [[0]]], dtype=torch.float32)

    class Result:
        boxes = Boxes()
        masks = Masks()

    masks, labels, scores, boxes = module._result_arrays(Result(), (2, 3))
    assert masks.shape == (2, 2, 3)
    assert masks[0].all() and not masks[1].any()
    assert np.array_equal(labels, np.asarray([4, 2]))
    assert np.allclose(scores, [0.7, 0.8])
    assert boxes.shape == (2, 4)
