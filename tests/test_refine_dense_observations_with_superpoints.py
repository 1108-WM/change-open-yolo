import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "refine_dense_observations_with_superpoints.py"
SPEC = importlib.util.spec_from_file_location("refine_dense_observations_with_superpoints", MODULE_PATH)
REFINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFINER)


def _observation(tmp_path, observation_id, frame_index, point_indices, priority=3.0, class_name="chair"):
    path = tmp_path / f"observation{observation_id}.npz"
    np.savez_compressed(path, point_indices=np.asarray(point_indices, dtype=np.int64))
    return {
        "observation_id": observation_id,
        "frame_index": frame_index,
        "point_indices_path": str(path),
        "priority": priority,
        "class_name": class_name,
    }


def test_refiner_merges_cross_view_tracks_and_keeps_separate_instance(tmp_path):
    # Four contiguous synthetic superpoints, four points each.
    point_superpoints = np.repeat(np.arange(4, dtype=np.int64), 4)
    observations = [
        _observation(tmp_path, 0, 0, range(0, 8)),
        _observation(tmp_path, 1, 1, range(0, 8), priority=2.5),
        _observation(tmp_path, 2, 2, range(8, 12), class_name="table"),
    ]

    instances, diagnostics = REFINER.build_refined_instances(
        observations,
        point_superpoints,
        min_support_views=2,
        singleton_min_confidence=10.0,
    )

    assert diagnostics["raw_track_count"] == 2
    assert diagnostics["output_instance_count"] == 1
    assert instances[0]["superpoint_ids"] == [0, 1]
    assert instances[0]["class_name"] == "chair"


def test_refiner_drops_ambiguous_superpoint_instead_of_assigning_it(tmp_path):
    point_superpoints = np.repeat(np.arange(2, dtype=np.int64), 4)
    observations = [
        _observation(tmp_path, 0, 0, range(0, 4), priority=3.0, class_name="chair"),
        _observation(tmp_path, 1, 1, range(0, 4), priority=3.0, class_name="table"),
    ]

    instances, diagnostics = REFINER.build_refined_instances(
        observations,
        point_superpoints,
        min_support_views=1,
        singleton_min_confidence=0.0,
        min_core_iou=1.1,
        min_core_containment=1.1,
        duplicate_iou=1.1,
        ambiguity_ratio=0.90,
    )

    assert diagnostics["ambiguous_superpoint_count"] == 1
    assert instances == []
