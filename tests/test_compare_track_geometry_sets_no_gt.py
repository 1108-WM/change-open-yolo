import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "compare_track_geometry_sets_no_gt.py"
    spec = importlib.util.spec_from_file_location("compare_track_geometry_sets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track(track_id, superpoints):
    return {"track_id": track_id, "superpoint_ids": superpoints}


def test_compare_scene_tracks_preserves_duplicate_geometry_multiplicity():
    module = _module()
    source = [_track(0, [1, 2]), _track(1, [1, 2]), _track(2, [3])]
    candidate = [_track(4, [1, 2]), _track(5, [4])]
    summary, changed = module.compare_scene_tracks("scene0000_00", source, candidate)
    assert summary["exact_shared_track_count"] == 1
    assert summary["source_only_track_count"] == 2
    assert summary["candidate_only_track_count"] == 1
    assert summary["changed_geometry_count"] == 3
    by_geometry = {tuple(row["superpoint_ids"]): row for row in changed}
    assert by_geometry[(1, 2)]["source_only_count"] == 1
    assert by_geometry[(3,)]["source_only_count"] == 1
    assert by_geometry[(4,)]["candidate_only_count"] == 1


def test_geometry_signature_rejects_unsorted_or_duplicate_superpoints():
    module = _module()
    for values in ([2, 1], [1, 1], []):
        try:
            module.geometry_signature(_track(0, values))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid superpoints accepted: {values}")
