import json
from pathlib import Path

import pytest

from tools.run_scannet200_train_scene_track_pipeline import _validate_inputs


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_validate_inputs_accepts_complete_uniform30_contract(tmp_path):
    scene = "scene0001_01"
    prepared = tmp_path / "prepared"
    records = tmp_path / "records" / scene
    _write(
        prepared / scene / "stream_prepare_manifest.json",
        {"scene_name": scene, "split": "official_scannet200_train"},
    )
    _write(
        records / "native_export_manifest.json",
        {
            "scene_name": scene,
            "cache_contract": "Mask3D + YOLO-World only",
            "ground_truth_usage": "none",
            "point_count": 12,
        },
    )
    sam = records / "sam_automatic_uniform30" / scene
    _write(
        sam / "summary.json",
        {
            "frame_count": 30,
            "observation_count": 1,
            "mask_rle_saved": True,
            "exact_same_frame_relations_saved": True,
        },
    )
    (sam / "automatic_observations.jsonl").write_text("{}\n")
    (sam / "points").mkdir()
    (sam / "points" / "obs000000_points.npz").write_bytes(b"test")
    assert _validate_inputs(scene, prepared, records) == (12, 1)


def test_validate_inputs_rejects_non_uniform_sam(tmp_path):
    scene = "scene0001_01"
    prepared = tmp_path / "prepared"
    records = tmp_path / "records" / scene
    _write(
        prepared / scene / "stream_prepare_manifest.json",
        {"scene_name": scene, "split": "official_scannet200_train"},
    )
    _write(
        records / "native_export_manifest.json",
        {
            "scene_name": scene,
            "cache_contract": "Mask3D + YOLO-World only",
            "ground_truth_usage": "none",
            "point_count": 12,
        },
    )
    _write(
        records / "sam_automatic_uniform30" / scene / "summary.json",
        {
            "frame_count": 3,
            "observation_count": 1,
            "mask_rle_saved": True,
            "exact_same_frame_relations_saved": True,
        },
    )
    with pytest.raises(ValueError, match="not uniform30"):
        _validate_inputs(scene, prepared, records)
