import importlib.util
import json
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "diagnose_details_core_prompt_native_relation_gt.py"
    )
    spec = importlib.util.spec_from_file_location("prompt_native_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_metrics_compare_the_same_instance():
    module = _module()
    gt = np.asarray([1001, 1001, 1001, 0, 2001], dtype=np.int64)
    metrics = module.target_metrics([0, 1, 3], gt, 1001, 3)
    assert metrics["iou"] == 0.5
    assert metrics["precision"] == 2 / 3
    assert metrics["coverage"] == 2 / 3


def test_native_target_relation_reports_best_iou_and_highest_eligible_score():
    module = _module()
    gt = np.asarray([1001, 1001, 1001, 0], dtype=np.int64)
    masks = np.asarray([
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
    ], dtype=bool)
    relation = module.native_target_relation(
        masks, np.asarray([0.4, 0.8, 0.9]), gt, 1001, 3
    )
    assert relation["best_candidate_id"] == 1
    assert relation["best_iou"] == 0.75
    assert relation["max_score_at_iou50"] == 0.8


def test_summary_separates_unique_recovery_from_native_redundancy():
    module = _module()
    base = {
        "target_gt_instance_id": 1001,
        "prompt_minus_v0_target_iou": 0.2,
        "v0_target_iou": 0.2,
        "prompt_target_iou": 0.4,
        "prompt_top_native_iou": 0.1,
        "prompt_native_mutual_best": False,
        "track_score_above_native_iou25_max": None,
        "added_target_precision": 0.8,
    }
    unique = {**base, "native_target_best_iou": 0.1}
    redundant = {
        **base,
        "native_target_best_iou": 0.6,
        "prompt_top_native_iou": 0.6,
        "track_score_above_native_iou25_max": False,
    }
    summary = module.summarize_rows([unique, redundant])
    assert summary["upward_cross_ap25_count"] == 2
    assert summary["unique_vs_native_upward_cross_ap25_count"] == 1
    assert summary["improved_but_native_already_ap50_count"] == 1
    assert summary["prompt_mask_duplicate_iou50_count"] == 1


def test_gt_diagnostic_cache_contract_reads_explicit_mask3d_mode(tmp_path):
    module = _module()
    (tmp_path / "native_cache_no_gt_manifest.json").write_text(json.dumps({
        "candidate_inputs": {"mode": "mask3d_yoloworld_only"},
        "decision_state": "pure",
    }))
    contract = module._native_cache_contract(tmp_path, "mask3d_yoloworld_only")
    assert contract["mode"] == "mask3d_yoloworld_only"
