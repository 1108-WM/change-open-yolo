import importlib.util
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "build_ibsp_label_maps.py"
SPEC = importlib.util.spec_from_file_location("build_ibsp_label_maps", MODULE_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_label_map_removes_duplicate_masks_and_preserves_smaller_overlap(tmp_path):
    large = np.zeros((4, 4), dtype=np.uint8)
    large[0:3, 0:3] = 255
    duplicate = large.copy()
    small = np.zeros((4, 4), dtype=np.uint8)
    small[1:3, 1:3] = 255
    paths = []
    for name, mask in (("large", large), ("duplicate", duplicate), ("small", small)):
        path = tmp_path / f"{name}.png"
        imageio.imwrite(path, mask)
        paths.append(path)

    labels, stats = BUILDER.build_label_map(paths, min_area=1, duplicate_iou=0.90)

    assert stats["kept_masks"] == 2
    assert labels[1, 1] == 1
    assert labels[0, 0] == 2
    assert labels[3, 3] == 0
