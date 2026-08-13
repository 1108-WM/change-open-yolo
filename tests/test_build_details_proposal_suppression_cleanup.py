import copy
import importlib.util
import json
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_proposal_suppression_cleanup.py"
    )
    spec = importlib.util.spec_from_file_location("details_suppression_cleanup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proposal(proposal_id, superpoints, lineage=None):
    return {
        "proposal_id": proposal_id,
        "track_id": proposal_id,
        "superpoint_ids": sorted(superpoints),
        "superpoint_count": len(superpoints),
        "point_count": len(superpoints),
        "lineage_proposal_ids": lineage or [proposal_id],
        "mean_node_quality": 0.8,
        "points_path": f"track{proposal_id}.npz",
    }


def test_directional_coverage_equal_to_point_99_is_not_suppressed():
    module = _module()
    proposals = [_proposal(0, [1]), _proposal(1, [1, 2])]
    sizes = {1: 99, 2: 1}
    final, suppressed, actions, relations = module.cleanup_proposals(proposals, sizes)
    assert relations[0]["left_point_coverage"] == 1.0
    assert relations[0]["right_point_coverage"] == 0.99
    # Left is fully in right and is removed; right's exact 0.99 is not itself
    # considered contained in left.
    assert [row["proposal_id"] for row in final] == [1]
    assert [row["proposal_id"] for row in suppressed] == [0]
    assert actions[0]["action"] == "suppressed_asymmetrically_included"


def test_exact_threshold_on_contained_side_is_strictly_excluded():
    module = _module()
    # Proposal 0 contains 99/100 of proposal 1, exactly the threshold.
    proposals = [_proposal(0, [1]), _proposal(1, [1, 2])]
    sizes = {1: 99, 2: 1}
    relation = module.pair_relations(proposals, sizes)[0]
    assert relation["right_point_coverage"] == 0.99
    assert relation["right_in_left_observed"] is False


def test_mutual_duplicate_component_keeps_deterministic_lowest_id():
    module = _module()
    proposals = [
        _proposal(9, [1, 2]),
        _proposal(3, [1, 2]),
        _proposal(7, [1, 2]),
    ]
    sizes = {1: 1, 2: 1}
    final, suppressed, actions, _ = module.cleanup_proposals(proposals, sizes)
    assert [row["proposal_id"] for row in final] == [3]
    assert [row["proposal_id"] for row in suppressed] == [7, 9]
    assert all(row["action"] == "suppressed_mutual_duplicate" for row in actions)
    assert {row["replacement_proposal_id"] for row in actions} == {3}


def test_asymmetric_inclusion_removes_small_proposal_once():
    module = _module()
    proposals = [
        _proposal(0, [1]),
        _proposal(1, [1, 2, 3, 4]),
        _proposal(2, [8]),
    ]
    sizes = {item: 1 for item in range(1, 9)}
    final, suppressed, actions, relations = module.cleanup_proposals(proposals, sizes)
    assert [row["proposal_id"] for row in final] == [1, 2]
    assert [row["proposal_id"] for row in suppressed] == [0]
    assert actions[0]["replacement_proposal_id"] == 1
    assert actions[0]["direct_container_proposal_ids"] == [1]
    module.validate_converged_d2b(relations)


def test_inclusion_chain_points_every_action_to_a_surviving_container():
    module = _module()
    proposals = [
        _proposal(0, [1]),
        _proposal(1, [1, 2, 3, 4]),
        _proposal(2, list(range(1, 15))),
    ]
    sizes = {item: 1 for item in range(1, 15)}
    final, suppressed, actions, relations = module.cleanup_proposals(proposals, sizes)
    assert [row["proposal_id"] for row in final] == [2]
    assert [row["proposal_id"] for row in suppressed] == [0, 1]
    assert {row["replacement_proposal_id"] for row in actions} == {2}
    assert len(actions) == 2
    module.validate_converged_d2b(relations)


def test_nonconverged_merge_input_is_rejected():
    module = _module()
    proposals = [_proposal(0, [1, 2, 3]), _proposal(1, [1, 2, 4])]
    sizes = {item: 1 for item in range(1, 5)}
    relations = module.pair_relations(proposals, sizes)
    try:
        module.validate_converged_d2b(relations)
    except ValueError as error:
        assert "not merge-converged" in str(error)
    else:
        raise AssertionError("D2c must reject a D2b input with IoU > 0.3")


def test_cleanup_does_not_mutate_source_and_conserves_lineage():
    module = _module()
    proposals = [
        _proposal(0, [1], lineage=[0, 5]),
        _proposal(1, [1, 2, 3, 4], lineage=[1, 8]),
        _proposal(2, [9], lineage=[2]),
    ]
    sizes = {item: 1 for item in range(1, 10)}
    before = copy.deepcopy(proposals)
    final, suppressed, actions, relations = module.cleanup_proposals(proposals, sizes)
    assert proposals == before
    module.validate_cleanup_result(proposals, final, suppressed, actions)
    json.dumps(relations, sort_keys=True)


def test_cli_has_no_threshold_semantic_or_evaluation_inputs():
    source = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_proposal_suppression_cleanup.py"
    ).read_text()
    assert "--inclusion-threshold" not in source
    assert "--duplicate-threshold" not in source
    assert "--semantic-root" not in source
    assert "--evaluation-root" not in source
