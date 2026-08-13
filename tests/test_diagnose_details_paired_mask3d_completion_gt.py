import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "diagnose_details_paired_mask3d_completion_gt.py"
    )
    spec = importlib.util.spec_from_file_location("paired_completion_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attribute_completion_separates_aligned_overexpansion_from_mismatch():
    module = _module()
    gt = np.asarray([1001, 1001, 1001, 1001, 2001, 2001, 0], dtype=np.int64)
    sizes = {1001: 4, 2001: 2}
    aligned = module.attribute_completion([0, 1, 2], [0, 1, 2, 4, 6], [0, 1], gt, sizes)
    assert aligned["failure_category"] == "aligned_overexpansion_original_iou_ge50"
    assert aligned["added_original_target_precision"] == 0.0
    mismatch = module.attribute_completion([0, 1, 2], [0, 1, 2, 3], [4, 5], gt, sizes)
    assert mismatch["failure_category"] == "core_native_target_mismatch"
    assert mismatch["core_native_target_aligned"] is False


def test_attribute_completion_records_aligned_improvement_and_target_metrics():
    module = _module()
    gt = np.asarray([1001, 1001, 1001, 1001, 0], dtype=np.int64)
    result = module.attribute_completion(
        [0, 1], [0, 1, 2, 3], [0, 1], gt, {1001: 4}
    )
    assert result["failure_category"] == "aligned_improvement"
    assert result["original_target_iou"] == 0.5
    assert result["refined_target_iou"] == 1.0
    assert result["added_original_target_precision"] == 1.0


def test_threshold_transitions_include_ap25_and_full_ap_range():
    module = _module()
    transitions = module.threshold_transitions(0.48, 0.56)
    assert transitions["iou25"] == {"up": False, "down": False}
    assert transitions["iou50"] == {"up": True, "down": False}
    assert transitions["iou55"] == {"up": True, "down": False}
    assert transitions["iou60"] == {"up": False, "down": False}
    assert "iou90" in transitions


def test_summary_counts_threshold_losses_and_high_quality_damage():
    module = _module()
    base = {
        "failure_category": "aligned_overexpansion_original_iou_ge50",
        "original_best_gt_instance_id": 1001,
        "core_native_target_aligned": True,
        "refined_best_target_switched": False,
        "refined_minus_original_target_iou": -0.3,
        "original_target_iou": 0.8,
        "refined_target_iou": 0.5,
        "added_original_target_precision": 0.1,
        "expansion_ratio": 0.5,
        "native_score": 1.0,
        "native_score_scene_percentile": 0.9,
    }
    for name, transition in module.threshold_transitions(0.8, 0.5).items():
        base[f"{name}_up"] = transition["up"]
        base[f"{name}_down"] = transition["down"]
    summary = module.summarize_rows([base])
    assert summary["target_iou_worsened_count"] == 1
    assert summary["worsened_original_iou_ge75_count"] == 1
    assert summary["threshold_transitions"]["iou75"]["downward_cross_count"] == 1
