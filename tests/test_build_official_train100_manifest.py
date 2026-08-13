from tools.build_official_train100_manifest import build_manifest


def test_manifest_preserves_current_scenes_and_builds_seeded_80_20_protocol():
    official = [f"scene{index:04d}_00" for index in range(120)]
    current = official[:20]
    scenes, manifest = build_manifest(official, current, {official[100]}, 100, 11, 22)
    assert scenes[:20] == current
    assert len(scenes) == 100
    assert official[100] not in scenes
    assert all((len(fold["train_scenes"]), len(fold["validation_scenes"])) == (80, 20) for fold in manifest["folds"])
    assert len({scene for fold in manifest["folds"] for scene in fold["validation_scenes"]}) == 100
