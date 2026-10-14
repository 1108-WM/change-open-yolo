import torch
import pytest

from tools.annotate_automatic_mask_tracks_yoloworld import _load_predictions


def test_load_predictions_accepts_signed_ncs_cache(tmp_path):
    path = tmp_path / "scene.pt"
    predictions = {"0": {"bbox": torch.zeros((0, 4))}}
    torch.save({
        "metadata": {"scene_name": "scene0001_00"},
        "predictions": predictions,
    }, path)
    loaded, contract = _load_predictions(path, "scene0001_00")
    assert contract == "signed_metadata_predictions"
    assert set(loaded) == {"0"}


def test_load_predictions_rejects_signed_scene_mismatch(tmp_path):
    path = tmp_path / "scene.pt"
    torch.save({
        "metadata": {"scene_name": "scene9999_00"},
        "predictions": {},
    }, path)
    with pytest.raises(ValueError, match="metadata mismatch"):
        _load_predictions(path, "scene0001_00")
