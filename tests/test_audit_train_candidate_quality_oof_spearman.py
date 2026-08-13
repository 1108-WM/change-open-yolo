import numpy as np

from tools.audit_train_candidate_quality_oof_spearman import audit


def _row(scene, source, candidate_id, label, score):
    return {"scene_name": scene, "candidate_source": source, "candidate_id": candidate_id, "label_best_gt_iou": label, "predictions": {"D": {"q": score}}}


def test_geometry_group_spearman_collapses_native_class_expansion(tmp_path):
    scene = "scene0001_00"
    cache = tmp_path / scene / "native_cache"
    cache.mkdir(parents=True)
    np.save(cache / f"{scene}_pred_masks.npy", np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool))
    rows = [
        _row(scene, "native_mask3d_yoloworld", 0, .8, .7),
        _row(scene, "native_mask3d_yoloworld", 1, .8, .9),
        _row(scene, "native_mask3d_yoloworld", 2, .2, .2),
        _row(scene, "d2b_track", 0, .9, .8),
        _row(scene, "d2b_track", 1, .1, .1),
    ]
    report = audit(rows, tmp_path)
    metrics = report["model_q_spearman"]["D"]
    assert metrics["native_geometry_group_count"] == 2
    assert metrics["track_raw_count"] == 2
    assert abs(metrics["native_geometry_group_spearman"] - 1.0) < 1e-12
