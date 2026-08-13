import copy
import importlib.util
import json
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_iterative_proposal_merges.py"
    )
    spec = importlib.util.spec_from_file_location("details_iterative_merges", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track(track_id, superpoints, observation_ids, quality=0.8):
    return {
        "track_id": track_id,
        "source_track_id": track_id,
        "superpoint_ids": sorted(superpoints),
        "superpoint_count": len(superpoints),
        "point_count": len(superpoints),
        "observation_ids": observation_ids,
        "node_ids": observation_ids,
        "frame_ids": [str(item) for item in observation_ids],
        "support_view_count": len(observation_ids),
        "mean_node_quality": quality,
        "mean_predicted_iou": quality + 0.1,
        "mean_stability_score": quality + 0.05,
        "mean_edge_score": 0.7,
        "points_path": f"track{track_id}.npz",
    }


def _observations(tracks, all_superpoints):
    rows = {}
    for track in tracks:
        for observation_id in track["observation_ids"]:
            rows[observation_id] = {
                "frame_index": observation_id,
                "visible_counts": {item: 10 for item in all_superpoints},
                "inside_counts": {item: 9 for item in track["superpoint_ids"]},
            }
    return rows


def test_strict_upper_triangle_anchor_can_absorb_multiple_columns_per_round():
    module = _module()
    tracks = [
        _track(0, [1, 2, 3, 4], [0, 1]),
        _track(1, [1, 2, 3, 5], [2, 3]),
        _track(2, [1, 2, 4, 6], [4, 5]),
    ]
    sizes = {item: 1 for item in range(1, 7)}
    observations = _observations(tracks, sizes)
    final, actions, rounds, _ = module.iterative_merge_proposals(
        tracks, observations, sizes
    )
    assert len(final) == 1
    assert [row["round_index"] for row in actions] == [0, 0]
    assert [row["anchor_proposal_id"] for row in actions] == [0, 0]
    assert [row["absorbed_proposal_id"] for row in actions] == [1, 2]
    assert rounds[0]["merge_action_count"] == 2
    assert rounds[-1]["terminal"] is True


def test_relationships_are_recomputed_after_each_round():
    module = _module()
    tracks = [
        _track(0, [1, 2, 3], [0, 1]),
        _track(1, [1, 2, 4], [2, 3]),
        _track(2, [3, 4], [4, 5]),
    ]
    sizes = {item: 1 for item in range(1, 5)}
    observations = _observations(tracks, sizes)
    final, actions, rounds, relations = module.iterative_merge_proposals(
        tracks, observations, sizes
    )
    assert len(final) == 1
    assert [row["round_index"] for row in actions] == [0, 1]
    assert rounds[0]["eligible_pair_count"] == 1
    assert rounds[1]["eligible_pair_count"] == 1
    first_round_pair = next(
        row
        for row in relations
        if row["round_index"] == 0
        and row["left_proposal_id"] == 0
        and row["right_proposal_id"] == 2
    )
    second_round_pair = next(
        row
        for row in relations
        if row["round_index"] == 1
        and row["left_proposal_id"] == 0
        and row["right_proposal_id"] == 2
    )
    assert first_round_pair["point_iou"] == 0.25
    assert second_round_pair["point_iou"] == 0.5


def test_point_iou_equal_to_threshold_does_not_merge():
    module = _module()
    tracks = [
        _track(0, [1, 2, 3, 4, 5, 6], [0, 1]),
        _track(1, [1, 2, 3, 7, 8, 9, 10], [2, 3]),
    ]
    sizes = {item: 1 for item in range(1, 11)}
    observations = _observations(tracks, sizes)
    final, actions, rounds, relations = module.iterative_merge_proposals(
        tracks, observations, sizes
    )
    assert relations[0]["point_iou"] == 0.3
    assert relations[0]["details_merge_eligible_observed"] is False
    assert len(final) == 2
    assert actions == []
    assert len(rounds) == 1 and rounds[0]["terminal"] is True


def test_merge_refinement_only_removes_from_current_union():
    module = _module()
    tracks = [
        _track(0, [1, 2, 3], [0, 1]),
        _track(1, [1, 2, 4], [2, 3]),
    ]
    sizes = {item: 1 for item in range(1, 6)}
    observations = _observations(tracks, sizes)
    # Superpoint 5 has strong evidence in every observation but is absent from
    # both current proposals, so Algorithm 1 refinement must not reintroduce it.
    for row in observations.values():
        row["inside_counts"][5] = 9
    final, actions, _, _ = module.iterative_merge_proposals(
        tracks, observations, sizes
    )
    assert len(actions) == 1
    assert final[0]["superpoint_ids"] == [1, 2, 3, 4]
    assert 5 not in final[0]["superpoint_ids"]


def test_lineage_scores_and_source_inputs_are_conserved_without_mutation():
    module = _module()
    tracks = [
        _track(5, [1, 2, 3], [0, 1], quality=0.6),
        _track(9, [1, 2, 4], [2, 3, 4, 5], quality=0.9),
        _track(12, [8], [6, 7], quality=0.5),
    ]
    sizes = {item: 1 for item in range(1, 9)}
    observations = _observations(tracks, sizes)
    before_tracks = copy.deepcopy(tracks)
    before_observations = copy.deepcopy(observations)
    final, actions, rounds, relations = module.iterative_merge_proposals(
        tracks, observations, sizes
    )
    assert tracks == before_tracks
    assert observations == before_observations
    assert sorted(item for row in final for item in row["lineage_proposal_ids"]) == [5, 9, 12]
    assert len(final) + len(actions) == len(tracks)
    merged = next(row for row in final if row["proposal_id"] == 5)
    assert merged["mean_node_quality"] == (0.6 * 2 + 0.9 * 4) / 6
    module.validate_merge_result(tracks, final, actions, rounds)
    json.dumps(relations, sort_keys=True)


def test_cli_exposes_no_merge_or_consensus_tuning_parameters():
    source = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_iterative_proposal_merges.py"
    ).read_text()
    assert "--merge-iou" not in source
    assert "--min-consensus-rate" not in source
    assert "--semantic-root" not in source
    assert "--evaluation-root" not in source
