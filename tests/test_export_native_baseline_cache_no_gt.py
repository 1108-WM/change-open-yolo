import importlib.util
from pathlib import Path

import numpy as np
from types import SimpleNamespace


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_native_baseline_cache_no_gt.py"
    spec = importlib.util.spec_from_file_location("native_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_keeps_native_nonnegative_scores(tmp_path):
    module = _module()
    count = module._save("scene0000_00", (np.asarray([[1, 0], [0, 1]]), np.asarray([3, 4]), np.asarray([0.2, -0.1])), tmp_path)
    assert count == 1
    assert np.load(tmp_path / "scene0000_00_pred_classes.npy").tolist() == [3]


def test_mask3d_yoloworld_only_returns_unfused_prediction(tmp_path):
    module = _module()

    class FakeModel:
        world2cam = None

        def predict(self, **_):
            return {"scene0000_00": (np.asarray([[1], [0]]), np.asarray([4]), np.asarray([0.8]), "unused")}

    args = SimpleNamespace(
        processed_scene_root=tmp_path, dataset_root=tmp_path, mask_root=tmp_path,
        bboxes_2d_root=tmp_path, mask3d_yoloworld_only=True,
    )
    output = module._scene_prediction(FakeModel(), "scene0000_00", args, {}, 1000.0)
    assert len(output) == 3
    assert output[1].tolist() == [4]


def test_without_bpr_still_uses_fusion_path(tmp_path, monkeypatch):
    module = _module()
    calls = []

    class FakeWorld:
        mesh = "unused"

        @staticmethod
        def load_ply(_):
            return np.zeros((2, 3), dtype=np.float32), None

    class FakeModel:
        world2cam = FakeWorld()

        def predict(self, **_):
            return {"scene0000_00": (np.asarray([[1], [0]]), np.asarray([4]), np.asarray([0.8]), "unused")}

    def fake_fusion(*args, **kwargs):
        calls.append((args, kwargs))
        return args[1], args[2], args[3]

    monkeypatch.setattr(module, "append_backprojection_proposals", fake_fusion)
    processed_scene = tmp_path / "scene0000_00"
    processed_scene.mkdir()
    np.save(processed_scene / "0000_00.npy", np.zeros((2, 10), dtype=np.float32))
    args = SimpleNamespace(
        processed_scene_root=tmp_path, dataset_root=tmp_path, mask_root=tmp_path,
        bboxes_2d_root=tmp_path, mask3d_yoloworld_only=False, without_bpr=True,
    )
    output = module._scene_prediction(FakeModel(), "scene0000_00", args, {}, 1000.0)
    assert output[1].tolist() == [4]
    assert len(calls) == 1
