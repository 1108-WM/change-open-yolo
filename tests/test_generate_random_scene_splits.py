import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "generate_random_scene_splits.py"
SPEC = importlib.util.spec_from_file_location("generate_random_scene_splits", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_make_splits_is_reproducible_and_keeps_expected_relationships():
    scenes = [f"scene{i:04d}_00" for i in range(240)]

    first = MODULE._make_splits(scenes, seed=17)
    second = MODULE._make_splits(scenes, seed=17)

    assert first == second
    assert len(first["even48"]) == 48
    assert len(first["even96"]) == 96
    assert len(first["odd96"]) == 96
    assert set(first["even48"]).issubset(first["even96"])
    assert set(first["even96"]).isdisjoint(first["odd96"])
    assert first["even48"] == sorted(first["even48"])


def test_make_splits_rejects_too_few_scenes():
    scenes = [f"scene{i:04d}_00" for i in range(100)]

    try:
        MODULE._make_splits(scenes, seed=17)
    except ValueError as exc:
        assert "Need at least" in str(exc)
    else:
        raise AssertionError("Expected ValueError for too few scenes")
