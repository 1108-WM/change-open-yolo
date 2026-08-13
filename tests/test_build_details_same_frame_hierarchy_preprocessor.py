import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_same_frame_hierarchy_preprocessor.py"
    )
    spec = importlib.util.spec_from_file_location("details_same_frame", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(observation_id, quality, area=100):
    return {
        "observation_id": observation_id,
        "scene_name": "scene0000_00",
        "frame_id": "0",
        "frame_index": 0,
        "area": area,
        "predicted_iou": quality,
        "stability_score": 1.0,
        "mask_rle": {"size": [10, 10], "counts": [0, area]},
        "point_indices_path": f"obs{observation_id}.npz",
    }


def _relation(left, right, kind):
    return {
        "left_observation_id": left,
        "right_observation_id": right,
        "relation_kind": kind,
    }


def test_duplicate_component_keeps_deterministic_best_representative():
    module = _module()
    observations = [
        _observation(7, 0.90),
        _observation(2, 0.95),
        _observation(4, 0.95),
    ]
    relations = [_relation(7, 2, "duplicate"), _relation(7, 4, "duplicate")]
    kept, actions = module.preprocess_hierarchy_safe(observations, relations)
    assert [row["observation_id"] for row in kept] == [2]
    assert {row["representative_observation_id"] for row in actions} == {2}
    assert sum(row["action"] == "suppressed_near_duplicate" for row in actions) == 2


def test_containment_and_partial_relations_never_suppress():
    module = _module()
    observations = [_observation(0, 0.8), _observation(1, 0.9), _observation(2, 0.7)]
    relations = [_relation(0, 1, "containment"), _relation(1, 2, "partial")]
    kept, actions = module.preprocess_hierarchy_safe(observations, relations)
    assert [row["observation_id"] for row in kept] == [0, 1, 2]
    assert all(row["action"] == "kept" for row in actions)


def test_hierarchy_safe_result_does_not_depend_on_input_order():
    module = _module()
    observations = [_observation(0, 0.8), _observation(1, 0.9), _observation(2, 0.7)]
    relations = [_relation(0, 1, "duplicate"), _relation(1, 2, "partial")]
    expected = module.preprocess_hierarchy_safe(observations, relations)
    actual = module.preprocess_hierarchy_safe(
        list(reversed(observations)), list(reversed(relations))
    )
    assert actual == expected


def test_kept_observation_contract_is_byte_equivalent_at_object_level():
    module = _module()
    observations = [_observation(0, 0.8), _observation(1, 0.9)]
    kept, actions = module.preprocess_hierarchy_safe(observations, [])
    assert kept == observations
    for source, output in zip(observations, kept):
        for key in (
            "observation_id",
            "frame_id",
            "frame_index",
            "mask_rle",
            "point_indices_path",
        ):
            assert output[key] == source[key]
    assert all(not row["mask_changed"] for row in actions)
    assert all(not row["point_indices_changed"] for row in actions)


def test_suppressed_observation_is_traceable_to_kept_representative():
    module = _module()
    observations = [_observation(0, 0.8), _observation(1, 0.9)]
    kept, actions = module.preprocess_hierarchy_safe(
        observations, [_relation(0, 1, "duplicate")]
    )
    kept_ids = {row["observation_id"] for row in kept}
    for action in actions:
        if action["action"] == "suppressed_near_duplicate":
            assert action["representative_observation_id"] in kept_ids
    module.validate_preprocessed_output("hierarchy_safe", observations, kept, actions)


def test_relation_classification_separates_hierarchy_states():
    module = _module()
    threshold = 0.95
    assert module.classify_relation(0.0, 0.0, 0.0, threshold) == "disjoint"
    assert module.classify_relation(0.92, 0.96, 0.96, threshold) == "duplicate"
    assert module.classify_relation(0.40, 0.99, 0.41, threshold) == "containment"
    assert module.classify_relation(0.20, 0.40, 0.30, threshold) == "partial"


def test_mask_bbox_uses_image_coordinates_for_odd_area():
    module = _module()
    mask = np.asarray(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]], dtype=bool
    )
    assert module.mask_bbox_xywh(mask) == [1, 1, 2, 2]


def test_details_exact_gives_smaller_mask_overlap_priority(tmp_path):
    module = _module()
    masks = [
        np.asarray([[1, 1, 0], [0, 0, 0]], dtype=bool),
        np.asarray([[1, 1, 1], [1, 0, 0]], dtype=bool),
    ]
    observations = []
    for observation_id, mask in enumerate(masks):
        point_path = tmp_path / f"source{observation_id}.npz"
        points = np.asarray([0, 1] if observation_id == 0 else [0, 1, 2, 3])
        np.savez_compressed(point_path, point_indices=points)
        observations.append(
            {
                "observation_id": observation_id,
                "scene_name": "scene0000_00",
                "frame_id": "0",
                "frame_index": 0,
                "area": int(mask.sum()),
                "bbox_xywh": module.mask_bbox_xywh(mask),
                "predicted_iou": 0.9,
                "stability_score": 0.95,
                "mask_rle": module.encode_binary_mask_rle(mask),
                "point_indices_path": str(point_path),
            }
        )
    output_points = tmp_path / "output_points"
    kept, actions = module.preprocess_details_exact(
        observations, output_points, Path("published/points")
    )
    assert [row["area"] for row in kept] == [2, 2]
    assert np.array_equal(module.decode_binary_mask_rle(kept[0]["mask_rle"]), masks[0])
    assert not np.any(
        np.logical_and(
            module.decode_binary_mask_rle(kept[0]["mask_rle"]),
            module.decode_binary_mask_rle(kept[1]["mask_rle"]),
        )
    )
    with np.load(output_points / "obs000001_points.npz") as payload:
        assert np.array_equal(payload["point_indices"], np.asarray([2, 3]))
    assert actions[1]["removed_pixel_count"] == 2
    assert actions[1]["removed_point_count"] == 2
