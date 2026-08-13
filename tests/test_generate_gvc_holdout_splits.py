import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "generate_gvc_holdout_splits.py"
    spec = importlib.util.spec_from_file_location("generate_gvc_holdout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gvc_holdout_splits_are_disjoint_and_deterministic(tmp_path):
    generate = _module().generate
    scene_root = tmp_path / "scenes"
    scene_root.mkdir()
    all_scenes = [f"scene{index:04d}_00" for index in range(312)]
    for scene in all_scenes:
        (scene_root / scene).mkdir()
    even96 = tmp_path / "even96.txt"
    odd96 = tmp_path / "odd96.txt"
    even96.write_text("\n".join(all_scenes[:96]) + "\n")
    odd96.write_text("\n".join(all_scenes[96:192]) + "\n")
    manifest = generate(scene_root, even96, odd96, tmp_path / "out", seed=17)
    safety = set((tmp_path / "out" / "gvc_safety60.txt").read_text().splitlines())
    test = set((tmp_path / "out" / "gvc_test60.txt").read_text().splitlines())
    assert len(safety) == len(test) == 60
    assert safety.isdisjoint(test)
    assert safety.isdisjoint(all_scenes[:192])
    assert test.isdisjoint(all_scenes[:192])
    assert manifest["disjoint"]["safety_test"] is True
