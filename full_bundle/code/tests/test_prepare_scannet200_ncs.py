import json
from pathlib import Path

from tools import prepare_scannet200_ncs as ncs


def test_read_scenes_rejects_duplicates_and_invalid_names(tmp_path: Path):
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("scene0001_00\nscene0001_00\n")
    try:
        ncs._read_scenes(duplicate)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("expected duplicate scene list to be rejected")

    invalid = tmp_path / "invalid.txt"
    invalid.write_text("scan0001_00\n")
    try:
        ncs._read_scenes(invalid)
    except ValueError as error:
        assert "invalid scene" in str(error)
    else:
        raise AssertionError("expected invalid scene list to be rejected")


def test_geometry_only_scene_requires_no_label_files(tmp_path: Path, monkeypatch):
    raw = tmp_path / "scene0001_00"
    out = tmp_path / "prepared"
    raw.mkdir()
    (raw / "scene0001_00.sens").write_bytes(b"sens")
    (raw / "scene0001_00_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": [0]})
    )
    (raw / "scene0001_00_vh_clean_2.ply").write_bytes(b"ply")
    assert all(
        path.name not in {"scene0001_00_vh_clean_2.labels.ply", "scene0001_00.aggregation.json"}
        for path in ncs._required_files(raw, with_gt=False)
    )
    assert any(
        path.name == "scene0001_00_vh_clean_2.labels.ply"
        for path in ncs._required_files(raw, with_gt=True)
    )


def test_existing_manifest_requires_matching_gt_policy(tmp_path: Path):
    scene_root = tmp_path / "scene0001_00"
    scene_root.mkdir()
    manifest = {
        "scene_name": "scene0001_00",
        "ncs_split": "NCS-validation",
        "gt_policy": "validation_forbidden",
        "operations": {"candidate_generation": False},
    }
    (scene_root / "stream_prepare_manifest.json").write_text(json.dumps(manifest))
    assert ncs._existing_is_compatible(scene_root, "scene0001_00", "NCS-validation", False)
    assert not ncs._existing_is_compatible(scene_root, "scene0001_00", "NCS-validation", True)
