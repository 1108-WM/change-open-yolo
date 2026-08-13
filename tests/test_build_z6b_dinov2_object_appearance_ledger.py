import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_z6b_dinov2_object_appearance_ledger.py"
    spec = importlib.util.spec_from_file_location("z6b_dino_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_embeddings_selects_central_medoid_and_reports_dispersion():
    module = _module()
    features = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    result = module._aggregate_embeddings(features)
    assert result["medoid_view_index"] == 1
    assert np.isclose(np.linalg.norm(result["mean_embedding"]), 1.0)
    assert np.isclose(np.linalg.norm(result["medoid_embedding"]), 1.0)
    assert result["pairwise_cosine_min"] < result["pairwise_cosine_mean"]
    assert np.isclose(
        result["dispersion_one_minus_pairwise_mean"],
        1.0 - result["pairwise_cosine_mean"],
    )


def test_single_view_has_zero_dispersion():
    module = _module()
    result = module._aggregate_embeddings(np.asarray([[3.0, 4.0]], dtype=np.float32))
    assert result["medoid_view_index"] == 0
    assert result["pairwise_cosine_mean"] == 1.0
    assert result["pairwise_cosine_min"] == 1.0
    assert result["pairwise_cosine_std"] == 0.0
    assert result["dispersion_one_minus_pairwise_mean"] == 0.0


def test_masked_crop_tensor_masks_background_and_has_expected_shape():
    module = _module()
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    coords = np.asarray([[3, 3], [4, 4]], dtype=np.int64)
    tensor = module._masked_crop_tensor(image, [2, 2, 6, 6], coords, 14, 0)
    assert tuple(tensor.shape) == (3, 14, 14)
    assert np.isfinite(tensor.numpy()).all()
