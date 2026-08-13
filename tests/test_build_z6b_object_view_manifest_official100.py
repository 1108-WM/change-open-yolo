import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_z6b_object_view_manifest_official100.py"
    spec = importlib.util.spec_from_file_location("z6b_object_view_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asset_paths_follow_prepared_rgbd_contract(tmp_path):
    module = _module()
    paths = module._asset_paths(tmp_path, "scene0001_01", "120")
    assert paths["rgb_path"].endswith("scene0001_01/color/120.jpg")
    assert paths["depth_path"].endswith("scene0001_01/depth/120.png")
    assert paths["pose_path"].endswith("scene0001_01/poses/120.txt")
    assert paths["intrinsics_path"].endswith("scene0001_01/intrinsics.txt")


def test_view_role_uses_exact_z1_frame_metadata():
    module = _module()
    evidence = {
        "semantic_evidence_node_key": "scene:1",
        "distribution": {"views": [
            {"frame_id": "20", "frame_index": 2, "view_role": "track_support_view"},
        ]},
    }
    assert module._view_role(evidence, "20") == (2, "track_support_view")


def test_validate_top3_rejects_unsorted_visibility():
    module = _module()
    views = [
        {"view_rank": 1, "frame_id": "0", "visible_point_count": 10},
        {"view_rank": 2, "frame_id": "10", "visible_point_count": 20},
    ]
    with pytest.raises(ValueError, match="not sorted"):
        module._validate_top3(views, "scene:1")
