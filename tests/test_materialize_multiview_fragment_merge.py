import importlib.util
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).parents[1]


def _module():
    path = (
        PROJECT_ROOT
        / "tools"
        / "materialize_multiview_fragment_merge.py"
    )
    spec = importlib.util.spec_from_file_location("fragment_merge_materializer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_resolves_repository_tools_package():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "materialize_multiview_fragment_merge.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--plan-root" in result.stdout


def _track(proposal_id, superpoints, observations, lineage=None):
    lineage = lineage or [proposal_id]
    return {
        "proposal_id": proposal_id,
        "track_id": proposal_id,
        "source_track_id": proposal_id,
        "source_track_ids": [proposal_id],
        "lineage_proposal_ids": lineage,
        "superpoint_ids": superpoints,
        "superpoint_count": len(superpoints),
        "point_count": len(superpoints) * 2,
        "observation_ids": observations,
        "node_ids": observations,
        "frame_ids": [str(item) for item in observations],
        "mean_node_quality": 0.8,
        "mean_predicted_iou": 0.9,
        "mean_stability_score": 0.95,
        "mean_edge_score": 0.7,
        "merge_action_count": 0,
        "points_path": "unused.npz",
        "unrelated_field": {"preserve": True},
    }


def _action(anchor=10, absorbed=20, index=0):
    return {
        "action_index": index,
        "anchor_proposal_id": anchor,
        "absorbed_proposal_id": absorbed,
        "bridge_frame_count": 2,
        "bridge_observation_count": 2,
    }


def test_prepare_restores_aggregation_state_and_preserves_lineage():
    module = _module()
    prepared = module.prepare_d2b_proposals(
        [_track(10, [1], [1, 2], [10, 11])], {1: 2}
    )[0]
    assert prepared["lineage_proposal_ids"] == [10, 11]
    assert prepared["_observation_weight"] == 2
    assert prepared["_quality_sum"] == 1.6
    assert prepared["unrelated_field"] == {"preserve": True}


def test_applied_pair_removes_one_proposal_and_conserves_lineage():
    module = _module()
    tracks = [_track(10, [1], [1, 2]), _track(20, [2], [3, 4]), _track(30, [3], [5, 6])]

    def merge(anchor, absorbed, observations, sizes, round_index):
        result = dict(anchor)
        result["lineage_proposal_ids"] = sorted(
            anchor["lineage_proposal_ids"] + absorbed["lineage_proposal_ids"]
        )
        result["superpoint_ids"] = sorted(
            anchor["superpoint_ids"] + absorbed["superpoint_ids"]
        )
        result["superpoint_count"] = 2
        result["point_count"] = 4
        return result, {"removed_by_refinement_count": 0}

    final, ledger, changed = module.materialize_fragment_merges(
        tracks, [_action()], {}, {1: 2, 2: 2, 3: 2}, merge_function=merge
    )
    assert [row["proposal_id"] for row in final] == [10, 30]
    assert final[0]["lineage_proposal_ids"] == [10, 20]
    assert ledger[0]["merge_action_applied"] is True
    assert ledger[0]["score_used_for_decision"] is False
    assert changed == {10}


def test_empty_refinement_atomically_falls_back_to_both_proposals():
    module = _module()
    tracks = [_track(10, [1], [1, 2]), _track(20, [2], [3, 4])]

    def empty(*args):
        raise ValueError("merge 10 <- 20 became empty after frozen consensus refinement")

    final, ledger, changed = module.materialize_fragment_merges(
        tracks, [_action()], {}, {1: 2, 2: 2}, merge_function=empty
    )
    assert [row["proposal_id"] for row in final] == [10, 20]
    assert ledger[0]["materialization_state"] == "atomic_fallback_empty_refinement"
    assert ledger[0]["merge_action_applied"] is False
    assert changed == set()


def test_overlapping_or_reversed_plan_actions_are_rejected():
    module = _module()
    tracks = [_track(10, [1], [1]), _track(20, [2], [2]), _track(30, [3], [3])]
    for actions in (
        [_action(20, 10)],
        [_action(10, 20, 0), _action(20, 30, 1)],
    ):
        try:
            module.materialize_fragment_merges(
                tracks, actions, {}, {1: 2, 2: 2, 3: 2}
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid action plans must be rejected")


def test_cli_has_no_gt_native_semantic_or_action_threshold_inputs():
    module = _module()
    options = {
        item
        for action in module.build_parser()._actions
        for item in action.option_strings
    }
    assert {
        "--scene-list", "--track-root", "--automatic-root", "--plan-root",
        "--processed-scene-root", "--dataset-root", "--config-path", "--output-root",
    } <= options
    assert not {
        "--gt-instance-dir", "--native-prediction-cache", "--semantic-root",
        "--score-field", "--merge-threshold",
    } & options
