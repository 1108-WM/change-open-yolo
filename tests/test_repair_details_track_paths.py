import importlib.util
from pathlib import Path


def test_repair_tool_is_scoped_to_atomic_staging_marker():
    path = Path(__file__).parents[1] / "tools" / "repair_details_track_paths.py"
    spec = importlib.util.spec_from_file_location("repair_details_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert ".scene0000_00.writing" in "/tmp/.scene0000_00.writing/track_points/track0000_points.npz"
