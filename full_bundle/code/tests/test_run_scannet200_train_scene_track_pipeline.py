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
    with pytest.raises(ValueError, match="not the uniform30"):
        _validate_inputs(scene, prepared, records)


def test_validate_inputs_accepts_all_available_when_scene_has_fewer_than_30_frames(tmp_path):
    scene = "scene0150_02"
    prepared = tmp_path / "prepared"
    records = tmp_path / "records" / scene
    _write(
        prepared / scene / "stream_prepare_manifest.json",
        {"scene_name": scene, "split": "official_scannet200_train"},
    )
    color = prepared / scene / "color"
    color.mkdir()
    for index in range(13):
        (color / f"{index}.jpg").write_bytes(b"x")
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
            "frame_count": 13,
            "observation_count": 1,
            "mask_rle_saved": True,
            "exact_same_frame_relations_saved": True,
        },
    )
    (sam / "automatic_observations.jsonl").write_text("{}\n")
    (sam / "points").mkdir()
    (sam / "points" / "obs000000_points.npz").write_bytes(b"test")
    assert _validate_inputs(scene, prepared, records) == (12, 1)


def test_validate_inputs_counts_configured_views_before_max30_sampling(tmp_path):
    scene = "scene0150_02"
    prepared = tmp_path / "prepared"
    records = tmp_path / "records" / scene
    config = tmp_path / "config.yaml"
    config.write_text("openyolo3d:\n  frequency: 10\n")
    _write(
        prepared / scene / "stream_prepare_manifest.json",
        {
            "scene_name": scene,
            "split": "official_scannet200_train",
            "sens": {"frame_count": 130},
        },
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
            "frame_count": 13,
            "observation_count": 1,
            "mask_rle_saved": True,
            "exact_same_frame_relations_saved": True,
        },
    )
    (sam / "automatic_observations.jsonl").write_text("{}\n")
    (sam / "points").mkdir()
    (sam / "points" / "obs000000_points.npz").write_bytes(b"test")
    assert _validate_inputs(
        scene, prepared, records, config_path=config
    ) == (12, 1)


def test_validate_inputs_accepts_ncs_train_shared_automatic_root(tmp_path):
    scene = "scene0001_00"
    prepared = tmp_path / "prepared"
    records = tmp_path / "records" / scene
    automatic = tmp_path / "pipeline" / "sam_automatic_uniform30"
    _write(
        prepared / scene / "stream_prepare_manifest.json",
        {
            "scene_name": scene,
            "ncs_split": "NCS-train",
            "gt_policy": "train_allowed",
            "operations": {"candidate_generation": False},
        },
    )
    _write(
        records / "native_export_manifest.json",
        {
            "scene_name": scene,
            "ncs_split": "NCS-train",
            "cache_contract": "frozen Mask3D + YOLO-World only",
            "ground_truth_usage": "none",
            "point_count": 12,
        },
    )
    sam = automatic / scene
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
    assert _validate_inputs(
        scene, prepared, records, "NCS-train", automatic
    ) == (12, 1)
