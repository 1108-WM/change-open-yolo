import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_yoloe_f30_boxes.py"
    spec = importlib.util.spec_from_file_location("export_yoloe_f30_boxes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_sort_key_orders_numeric_ids_before_text_ids():
    module = _module()
    paths = [Path("10.jpg"), Path("2.jpg"), Path("preview.jpg")]
    assert [path.name for path in sorted(paths, key=module._frame_sort_key)] == ["2.jpg", "10.jpg", "preview.jpg"]


def test_sampling_stride_includes_the_configured_source_frequency(tmp_path):
    module = _module()
    color = tmp_path / "color"
    color.mkdir()
    for index in range(30):
        (color / f"{index}.jpg").touch()
    paths = module._sample_color_paths(tmp_path, frame_stride=1, source_frame_frequency=10, max_frames=3)
    assert [path.stem for path in paths] == ["0", "10", "20"]


def test_result_record_serializes_empty_boxes():
    module = _module()

    class EmptyBoxes:
        def __len__(self):
            return 0

    class Result:
        boxes = EmptyBoxes()

    record = module._result_record(Path("12.jpg"), Result())
    assert record == {"frame_id": "12", "boxes_xyxy": [], "labels": [], "scores": []}
