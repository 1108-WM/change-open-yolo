import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_t1_track_family_action_oracle_gt.py"
    spec = importlib.util.spec_from_file_location("t1_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_t1_action_application_preserves_noop_and_empty_track_fallback():
    module = _module()
    association = {10: [1, 2], 20: [3]}
    attach, attach_diag = module.apply_t1_action(association, {
        "action_type": "attach", "source_track_id": None, "target_track_id": 20,
        "source_observation_ids": [4],
    })
    assert attach_diag["action_applied"]
    assert attach == {10: [1, 2], 20: [3, 4]}
    reassigned, reassign_diag = module.apply_t1_action(association, {
        "action_type": "reassign", "source_track_id": 20, "target_track_id": 10,
        "source_observation_ids": [3],
    })
    assert not reassign_diag["action_applied"]
    assert reassign_diag["fallback_reason"] == "empty_source_track_atomic_fallback"
    assert reassigned == association
    merged, merge_diag = module.apply_t1_action(association, {
        "action_type": "merge", "source_track_id": 10, "target_track_id": 20,
        "source_observation_ids": [1, 2],
    })
    assert merge_diag["action_applied"]
    assert merged == {20: [1, 2, 3]}


def test_t1_action_oracle_uses_fixed_baseline_target():
    module = _module()
    gt_ids = np.asarray([1001, 1001, 0, 0], dtype=np.int64)
    gt_sizes = {1001: 2}
    baseline = [{
        "proposal_id": 10, "lineage_proposal_ids": [10],
        "points": np.asarray([0, 2]), "best_gt_instance_id": 1001, "best_gt_iou": 1 / 3,
    }]
    changed = [{
        "proposal_id": 10, "lineage_proposal_ids": [10],
        "points": np.asarray([0, 1]), "best_gt_instance_id": 1001, "best_gt_iou": 1.0,
    }]
    rows = module.action_oracle_rows({}, baseline, changed, gt_ids, gt_sizes, [10], [10])
    assert len(rows) == 1
    assert rows[0]["fixed_gt_instance_id"] == 1001
    assert np.isclose(rows[0]["delta_iou"], 2 / 3)
    assert rows[0]["iou50_up"]
