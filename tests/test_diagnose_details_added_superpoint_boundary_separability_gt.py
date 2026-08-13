import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "diagnose_details_added_superpoint_boundary_separability_gt.py"
    )
    spec = importlib.util.spec_from_file_location("boundary_sep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atom(superpoint_id, point_count, inside_ratio, owner=4, evidence=True):
    return {
        "scene_name": "scene0000_00",
        "track_id": 2,
        "added_superpoint_id": superpoint_id,
        "added_point_count": point_count,
        "core_best_native_candidate_id": owner,
        "added_inside_core_best_native_ratio": inside_ratio,
        "joint_visible_independent_frame_count": 2 if evidence else 0,
        "reliable_anchor_frame_count": 2 if evidence else 0,
        "positive_anchor_frame_count": 1 if evidence else 0,
        "exclusion_anchor_frame_count": 1 if evidence else 0,
        "delta_mean_best_siou": 0.2 if evidence else 0.0,
        "delta_support_frame_rate": 0.1 if evidence else 0.0,
        "direct_core_neighbor_count": 1,
        "max_direct_core_contact_density": 0.25,
        "contact_weighted_boundary_distance": 0.02,
        "contact_weighted_normal_difference": 0.3,
        "contact_weighted_color_difference": 0.4,
        "reachable_from_core_via_prompt_added": True,
        "graph_hops_from_core": 1,
        "prompt_support_frame_count": 2,
        "best_other_score_one_native_coverage_ratio": 0.5,
    }


def _gt(added_count=100):
    return {
        "scene_name": "scene0000_00",
        "track_id": "2",
        "native_candidate_id": "4",
        "failure_category": "aligned_improvement",
        "refined_minus_original_target_iou": "0.1",
        "common_core_native_iou": "0.4",
        "expansion_ratio": "0.2",
        "added_point_count": str(added_count),
    }


def test_candidate_aggregation_preserves_missing_evidence_and_point_weights():
    module = _module()
    atoms = [_atom(10, 100, 0.5, evidence=True), _atom(11, 50, 0.0, evidence=False)]
    row, prepared = module.aggregate_candidate(_gt(100), atoms)
    assert [item["actual_added_point_count"] for item in prepared] == [50, 50]
    assert row["independent_joint_visible_atom_ratio"] == 0.5
    assert row["independent_joint_visible_point_ratio"] == 0.5
    assert row["delta_mean_best_siou_point_weighted_all"] == 0.1
    assert row["delta_mean_best_siou_point_weighted_evidenced"] == 0.2
    assert row["positive_anchor_frame_rate"] == 0.5
    assert np.isclose(row["already_inside_native_point_ratio"], 1.0 / 3.0)


def test_fully_native_atoms_do_not_dilute_effective_atom_ratios():
    module = _module()
    atoms = [_atom(10, 100, 1.0, evidence=False), _atom(11, 50, 0.0, evidence=True)]
    row, _ = module.aggregate_candidate(_gt(50), atoms)
    assert row["declared_added_superpoint_count"] == 2
    assert row["added_superpoint_count"] == 1
    assert row["independent_joint_visible_atom_ratio"] == 1.0


def test_candidate_aggregation_rejects_wrong_native_owner():
    module = _module()
    with pytest.raises(ValueError, match="核心 owner"):
        module.aggregate_candidate(_gt(100), [_atom(10, 100, 0.0, owner=3)])


def test_candidate_aggregation_rejects_point_conservation_mismatch():
    module = _module()
    with pytest.raises(ValueError, match="固定候选"):
        module.aggregate_candidate(_gt(99), [_atom(10, 100, 0.0)])


def test_auc_and_summary_use_candidates_not_atomic_repetitions():
    module = _module()
    assert module._auc_high_value([3.0, 2.0], [1.0, 2.0]) == 0.875
    rows = []
    for outcome, value in (("improvement", 2.0), ("overexpansion", 1.0)):
        row = {feature: value for feature in module.FEATURES}
        row["outcome"] = outcome
        rows.append(row)
    summary = module.summarize_features(rows)
    assert summary["common_core_native_iou"]["high_value_toward_improvement_auc"] == 1.0
    assert summary["common_core_native_iou"]["improvement"]["count"] == 1


def test_fixed_scene_splits_report_candidate_level_direction():
    module = _module()
    rows = []
    for split in ("A", "B"):
        for outcome, value in (("improvement", 2.0), ("overexpansion", 1.0)):
            row = {feature: value for feature in module.FEATURES}
            row.update({"scene_split": split, "outcome": outcome})
            rows.append(row)
    summary = module.summarize_fixed_scene_splits(rows)
    assert summary["A"]["candidate_counts"] == {
        "improvement": 1,
        "overexpansion": 1,
    }
    assert summary["B"]["feature_distributions"]["expansion_ratio"][
        "high_value_toward_improvement_auc"
    ] == 1.0
