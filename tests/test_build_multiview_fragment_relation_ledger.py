import copy
import importlib.util
import json
from itertools import combinations
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_multiview_fragment_relation_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("fragment_relation_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nodes_and_geometry():
    nodes = [{"proposal_id": item} for item in (10, 20, 30)]
    geometry = []
    for left, right in combinations((10, 20, 30), 2):
        geometry.append({
            "left_proposal_id": left,
            "right_proposal_id": right,
            "has_spatial_contact": (left, right) == (10, 20),
            "point_iou": 0.0,
            "containment_direction": "none",
        })
    return nodes, geometry


def _link(proposal_id, observation_id, frame_id):
    return {
        "guide_proposal_id": proposal_id,
        "observation_id": observation_id,
        "frame_id": str(frame_id),
        "depth_weight_adapter": "mv3dis_relative_depth_rle_reference",
    }


def _hierarchy(left, right, frame_id, kind):
    return {
        "left_observation_id": left,
        "right_observation_id": right,
        "frame_id": str(frame_id),
        "relation_kind": kind,
    }


def _by_pair(rows):
    return {
        (row["left_proposal_id"], row["right_proposal_id"]): row
        for row in rows
    }


def test_same_mask_matching_two_guides_is_a_bridge():
    module = _module()
    nodes, geometry = _nodes_and_geometry()
    matches = [
        _link(10, 100, 0), _link(20, 100, 0),
        _link(10, 200, 20), _link(20, 200, 20),
    ]
    rows = module.build_fragment_pair_evidence(nodes, geometry, matches, [])
    bridge = _by_pair(rows)[(10, 20)]
    assert bridge["bridge_observation_ids"] == [100, 200]
    assert bridge["bridge_frame_ids"] == ["0", "20"]
    assert bridge["bridge_frame_count"] == 2
    assert bridge["separation_frame_count"] == 0
    assert bridge["multiview_evidence_state"] == (
        "bridge_without_separation_counterevidence"
    )


def test_distinct_disjoint_masks_are_separation_counterevidence():
    module = _module()
    nodes, geometry = _nodes_and_geometry()
    matches = [
        _link(10, 101, 0), _link(20, 102, 0),
        _link(10, 201, 20), _link(20, 202, 20),
    ]
    hierarchy = [
        _hierarchy(101, 102, 0, "disjoint"),
        _hierarchy(201, 202, 20, "partial"),
    ]
    separated = _by_pair(module.build_fragment_pair_evidence(
        nodes, geometry, matches, hierarchy
    ))[(10, 20)]
    assert separated["bridge_frame_count"] == 0
    assert separated["separation_frame_ids"] == ["0", "20"]
    assert separated["separation_observation_pair_count"] == 2
    assert separated["same_frame_exclusive_relation_counts"] == {
        "disjoint": 1, "partial": 1
    }
    assert separated["multiview_evidence_state"] == (
        "separation_counterevidence_only"
    )


def test_duplicate_and_containment_masks_are_not_separation_counterevidence():
    module = _module()
    nodes, geometry = _nodes_and_geometry()
    matches = [_link(10, 101, 0), _link(20, 102, 0)]
    for kind in ("duplicate", "containment"):
        row = _by_pair(module.build_fragment_pair_evidence(
            nodes, geometry, matches, [_hierarchy(101, 102, 0, kind)]
        ))[(10, 20)]
        assert row["separation_frame_count"] == 0
        assert row["same_frame_exclusive_relation_counts"] == {kind: 1}


def test_bridge_and_separation_are_both_preserved_without_action():
    module = _module()
    nodes, geometry = _nodes_and_geometry()
    matches = [
        _link(10, 100, 0), _link(20, 100, 0),
        _link(10, 101, 0), _link(20, 102, 0),
    ]
    row = _by_pair(module.build_fragment_pair_evidence(
        nodes, geometry, matches, [_hierarchy(101, 102, 0, "disjoint")]
    ))[(10, 20)]
    assert row["bridge_frame_count"] == 1
    assert row["separation_frame_count"] == 1
    assert row["multiview_evidence_state"] == (
        "bridge_with_separation_counterevidence"
    )
    assert row["fragment_action"] == "none_ledger_only"
    assert row["proposal_mutation_count"] == 0


def test_all_pairs_are_conserved_and_inputs_are_not_mutated():
    module = _module()
    nodes, geometry = _nodes_and_geometry()
    matches = [_link(10, 100, 0), _link(20, 100, 0)]
    before = copy.deepcopy((nodes, geometry, matches))
    rows = module.build_fragment_pair_evidence(nodes, geometry, matches, [])
    assert (nodes, geometry, matches) == before
    assert len(rows) == 3
    assert [
        (row["left_proposal_id"], row["right_proposal_id"])
        for row in rows
    ] == list(combinations((10, 20, 30), 2))
    module.validate_fragment_ledger(nodes, rows)
    json.dumps(rows, sort_keys=True)


def test_unknown_proposal_and_duplicate_links_are_rejected():
    module = _module()
    nodes, geometry = _nodes_and_geometry()
    for matches in (
        [_link(99, 100, 0)],
        [_link(10, 100, 0), _link(10, 100, 0)],
    ):
        try:
            module.build_fragment_pair_evidence(nodes, geometry, matches, [])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid guide links must be rejected")


def test_cli_has_no_gt_native_semantic_score_or_threshold_inputs():
    module = _module()
    actions = module.build_parser()._actions
    option_strings = {item for action in actions for item in action.option_strings}
    assert {
        "--scene-list", "--proposal-root", "--matching-root", "--automatic-root",
        "--processed-scene-root", "--output-root",
    } <= option_strings
    assert not {
        "--gt-instance-dir", "--native-prediction-cache", "--semantic-root",
        "--score-field", "--bridge-threshold", "--separation-threshold",
    } & option_strings
