import numpy as np
import pytest

from tools.export_scannet200_train_scene_native_cache import _save_native


def test_save_native_uses_point_by_candidate_layout(tmp_path):
    masks = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=bool)
    classes = np.asarray([4, 8], dtype=np.int64)
    scores = np.asarray([0.6, 0.4], dtype=np.float32)
    summary = _save_native(tmp_path, "scene0001_01", (masks, classes, scores))
    saved = np.load(tmp_path / "scene0001_01_pred_masks.npy")
    assert saved.shape == (3, 2)
    assert summary == {"point_count": 3, "native_candidate_count": 2}


def test_save_native_rejects_candidate_dimension_mismatch(tmp_path):
    masks = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=bool)
    classes = np.asarray([4], dtype=np.int64)
    scores = np.asarray([0.6, 0.4], dtype=np.float32)
    with pytest.raises(ValueError, match="candidate dimensions disagree"):
        _save_native(tmp_path, "scene0001_01", (masks, classes, scores))
