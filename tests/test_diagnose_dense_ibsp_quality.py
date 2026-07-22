import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "diagnose_dense_ibsp_quality.py"
SPEC = importlib.util.spec_from_file_location("diagnose_dense_ibsp_quality", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_review_scenes_unions_low_coverage_and_high_instance_cases():
    rows = [
        {"scene_name": "scene_a", "mean_label_coverage": 0.10, "output_instances": 2, "single_frame_instances": 0},
        {"scene_name": "scene_b", "mean_label_coverage": 0.40, "output_instances": 9, "single_frame_instances": 2},
        {"scene_name": "scene_c", "mean_label_coverage": 0.20, "output_instances": 3, "single_frame_instances": 0},
    ]

    selected = MODULE.select_review_scenes(rows, low_coverage_count=1, high_instance_count=1)

    by_scene = {item["row"]["scene_name"]: item["reasons"] for item in selected}
    assert by_scene == {"scene_a": ["low_coverage"], "scene_b": ["high_instance_count"]}


def test_overlay_labels_keeps_background_unchanged_and_colors_foreground():
    import numpy as np

    image = np.full((2, 2, 3), 100, dtype=np.uint8)
    labels = np.array([[0, 1], [2, 0]], dtype=np.uint16)

    output = MODULE._overlay_labels(image, labels, alpha=0.55)

    assert output[0, 0].tolist() == [100, 100, 100]
    assert output[0, 1].tolist() != [100, 100, 100]
