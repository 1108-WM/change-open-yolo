import importlib.util
import json
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "verify_sam_automatic_rle_reexport.py"
    spec = importlib.util.spec_from_file_location("verify_sam_automatic_rle_reexport", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_verify_scene_requires_identical_existing_observations_and_new_rle(tmp_path):
    module = _module()
    old_scene, new_scene = tmp_path / "old" / "scene0000_00", tmp_path / "new" / "scene0000_00"
    old_points, new_points = old_scene / "points.npz", new_scene / "points.npz"
    old_scene.mkdir(parents=True)
    new_scene.mkdir(parents=True)
    np.savez_compressed(old_points, point_indices=np.asarray([1, 3, 4], dtype=np.int64))
    np.savez_compressed(new_points, point_indices=np.asarray([1, 3, 4], dtype=np.int64))
    common = {"observation_id": 0, "scene_name": "scene0000_00", "frame_id": "0", "frame_index": 0,
              "area": 6, "bbox_xywh": [0, 0, 2, 3], "crop_box_xywh": [0, 0, 2, 3],
              "predicted_iou": 0.9, "stability_score": 0.95}
    _write_jsonl(old_scene / "automatic_observations.jsonl", [{**common, "point_indices_path": str(old_points)}])
    _write_jsonl(new_scene / "automatic_observations.jsonl", [{**common, "point_indices_path": str(new_points), "mask_rle": {"size": [2, 3], "counts": [0, 6]}}])
    _write_jsonl(new_scene / "same_frame_mask_relations.jsonl", [])
    result = module.verify_scene(tmp_path / "old", tmp_path / "new", "scene0000_00")
    assert result == {"scene_name": "scene0000_00", "observation_count": 1, "same_frame_relation_count": 0}
