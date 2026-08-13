import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_mv3dis_boundary_action_feature_separability_gt.py"
    spec = importlib.util.spec_from_file_location("boundary_feature_separability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_join_labels_uses_only_oracle_as_post_hoc_label():
    module = _module()
    features = [{
        "scene_name": "scene_unit", "superpoint_id": 3,
        "candidate_owner_proposal_id": 20, "ground_truth_usage": "none",
        "direct_core_contact_ratio_max": .7,
    }]
    oracle = [{
        "scene_name": "scene_unit", "superpoint_id": 3,
        "action_results": [{
            "action_kind": "assign_owner", "target_owner_proposal_id": 20,
            "locally_feasible_no_empty_proposal": True,
            "fixed_gt_iou_improved_proposal_count": 1,
            "fixed_gt_iou_declined_proposal_count": 0,
            "fixed_gt_iou25_upcross_proposal_count": 0,
            "fixed_gt_iou25_downcross_proposal_count": 0,
            "fixed_gt_iou50_upcross_proposal_count": 0,
            "fixed_gt_iou50_downcross_proposal_count": 0,
            "proposal_outcomes": [
                {"fixed_gt_available": True, "fixed_gt_iou_delta": .2},
                {"fixed_gt_available": True, "fixed_gt_iou_delta": -.1},
            ],
        }],
    }]
    rows = module.join_feature_labels(features, oracle)
    assert rows[0]["gt_only_local_fixed_target_label"] == "beneficial"
    assert rows[0]["gt_only_local_fixed_target_iou_delta_sum"] == .1
    assert rows[0]["ground_truth_usage"] != "none"
    assert features[0]["ground_truth_usage"] == "none"
