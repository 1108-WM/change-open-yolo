import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "diagnose_mv3dis_baseline_adapted_variant_gt.py"
    )
    spec = importlib.util.spec_from_file_location("mv3dis_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_gt_uses_iou_then_intersection_then_lowest_instance_id():
    module = _module()
    gt = np.asarray([2001, 2001, 2002, 2002, 0, 0])
    best = module.best_gt_instance(
        np.asarray([0, 2]), gt, [(2001, 2), (2002, 2)]
    )
    assert best["instance_id"] == 2001
    assert np.isclose(best["iou"], 1 / 3)
    assert module.best_gt_instance(np.asarray([4, 5]), gt, [(2001, 2)]) is None


def test_native_target_stats_reports_threshold_coverage_and_scores():
    module = _module()
    gt = np.asarray([2001, 2001, 2001, 0, 0])
    masks = np.asarray([
        [1, 0],
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ], dtype=bool)
    result = module.native_target_stats(
        masks, np.asarray([0.4, 0.9]), gt, 2001, 3
    )
    assert result["best_native_iou"] == 1.0
    assert result["native_iou50_candidate_count"] == 1
    assert np.isclose(result["max_native_score_at_iou50"], 0.4)
    assert result["native_iou25_candidate_count"] == 2
    assert np.isclose(result["max_native_score_at_iou25"], 0.9)


def test_summary_counts_crossings_and_native_dilution():
    module = _module()
    rows = [
        {
            "fixed_gt_available": True,
            "fixed_gt_iou_delta": 0.1,
            "best_gt_target_changed": False,
            "fixed_gt_iou25_state": "upcross",
            "fixed_gt_iou50_state": "both_fail",
            "native_already_covers_fixed_gt_at_iou25": True,
            "native_already_covers_fixed_gt_at_iou50": False,
            "native_higher_score_cover_at_iou25": True,
            "native_higher_score_cover_at_iou50": False,
        },
        {
            "fixed_gt_available": True,
            "fixed_gt_iou_delta": -0.1,
            "best_gt_target_changed": True,
            "fixed_gt_iou25_state": "downcross",
            "fixed_gt_iou50_state": "downcross",
            "native_already_covers_fixed_gt_at_iou25": False,
            "native_already_covers_fixed_gt_at_iou50": True,
            "native_higher_score_cover_at_iou25": False,
            "native_higher_score_cover_at_iou50": True,
        },
        {"fixed_gt_available": False},
    ]
    summary = module.summarize(rows)
    assert summary["changed_proposal_count"] == 3
    assert summary["fixed_gt_iou_improved_count"] == 1
    assert summary["fixed_gt_iou_declined_count"] == 1
    assert summary["fixed_gt_iou25_upcross_count"] == 1
    assert summary["fixed_gt_iou25_downcross_count"] == 1
    assert summary["downcross_with_native_cover_at_iou50_count"] == 1
