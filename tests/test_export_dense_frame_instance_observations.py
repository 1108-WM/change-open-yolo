import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "export_dense_frame_instance_observations.py"
SPEC = importlib.util.spec_from_file_location("export_dense_frame_instance_observations", MODULE_PATH)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def _item(mask, priority, detection_id):
    return {"mask": mask, "area": int(mask.sum()), "priority": priority, "detection_id": detection_id}


def test_frame_selection_removes_same_mask_but_keeps_nested_small_object():
    large = np.zeros((8, 8), dtype=bool)
    large[:6, :6] = True
    duplicate = large.copy()
    small = np.zeros((8, 8), dtype=bool)
    small[2:4, 2:4] = True

    kept = EXPORTER.select_frame_observations(
        [_item(large, 0.8, 0), _item(duplicate, 0.7, 1), _item(small, 0.6, 2)]
    )

    assert len(kept) == 2
    assert {item["detection_id"] for item in kept} == {0, 2}


def test_frame_label_map_gives_overlap_to_smaller_mask():
    large = np.zeros((5, 5), dtype=bool)
    large[:4, :4] = True
    small = np.zeros((5, 5), dtype=bool)
    small[1:3, 1:3] = True

    labels, assigned = EXPORTER.build_frame_label_map([_item(large, 0.8, 0), _item(small, 0.7, 1)])

    by_detection = {item["detection_id"]: label_id for item, label_id, _ in assigned}
    assert labels[1, 1] == by_detection[1]
    assert labels[0, 0] == by_detection[0]
    assert labels[4, 4] == 0
