import copy
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_consensus_proposal_relation_graph.py"
    )
    spec = importlib.util.spec_from_file_location("details_proposal_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    # Four raw superpoints with two points each. Proposal 20 is contained in 10,
    # proposal 30 only contacts 10, and proposal 40 is spatially disjoint.
    processed = np.zeros((8, 10), dtype=np.float32)
    processed[:, :3] = np.asarray(
        [
            [0.00, 0.00, 0.00], [0.01, 0.00, 0.00],
            [0.02, 0.00, 0.00], [0.03, 0.00, 0.00],
            [0.04, 0.00, 0.00], [0.05, 0.00, 0.00],
            [1.00, 0.00, 0.00], [1.01, 0.00, 0.00],
        ]
    )
    processed[:, 9] = [1, 1, 2, 2, 3, 3, 4, 4]
    scene_root = tmp_path / "scene0000_00"
    scene_root.mkdir()
    specs = [
        (10, [1, 2], [0, 1], ["0", "10"], 0.8),
        (20, [1], [1, 2], ["10", "20"], 0.9),
        (30, [3], [3], ["30"], 0.7),
        (40, [4], [4], ["40"], 0.6),
    ]
    tracks = []
    for track_id, superpoints, observations, frames, quality in specs:
        points = np.flatnonzero(np.isin(processed[:, 9], superpoints))
        point_path = scene_root / f"track{track_id}.npz"
        np.savez_compressed(point_path, point_indices=points)
        tracks.append(
            {
                "track_id": track_id,
                "source_track_id": track_id + 100,
                "superpoint_ids": superpoints,
                "point_count": len(points),
                "points_path": str(point_path),
                "observation_ids": observations,
                "frame_ids": frames,
                "support_view_count": len(frames),
                "mean_node_quality": quality,
                "unrelated_source_field": {"must": "survive"},
            }
        )
    contact_map = {
        (2, 3): {"boundary_contact_count": 7, "boundary_contact_ratio": 0.5}
    }
    return processed, scene_root, tracks, contact_map


def _relation_by_pair(relations):
    return {
        (row["left_proposal_id"], row["right_proposal_id"]): row
        for row in relations
    }


def test_geometry_metrics_are_exact_and_validate_intersection():
    module = _module()
    assert module.geometry_metrics(4, 6, 3) == {
        "intersection_count": 3,
        "iou": 3 / 7,
        "left_coverage": 3 / 4,
        "right_coverage": 1 / 2,
    }
    try:
        module.geometry_metrics(2, 3, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid intersections must be rejected")


def test_details_observation_thresholds_are_strictly_greater_than():
    module = _module()
    assert module.containment_direction(0.99, 0.0) == "none"
    assert module.containment_direction(0.990001, 0.0) == "left_in_right"
    assert 3 / 10 == module.DETAILS_MERGE_IOU
    assert not (3 / 10 > module.DETAILS_MERGE_IOU)


def test_scene_graph_keeps_exact_point_superpoint_and_temporal_metrics(tmp_path):
    module = _module()
    processed, scene_root, tracks, contact_map = _fixture(tmp_path)
    _, relations = module.build_scene_graph(
        "scene0000_00", tracks, processed, scene_root, contact_map
    )
    rows = _relation_by_pair(relations)

    contained = rows[(10, 20)]
    assert contained["point_intersection_count"] == 2
    assert contained["point_iou"] == 0.5
    assert contained["left_point_coverage"] == 0.5
    assert contained["right_point_coverage"] == 1.0
    assert contained["superpoint_intersection_count"] == 1
    assert contained["superpoint_iou"] == 0.5
    assert contained["left_superpoint_coverage"] == 0.5
    assert contained["right_superpoint_coverage"] == 1.0
    assert contained["shared_observation_count"] == 1
    assert contained["shared_frame_count"] == 1
    assert contained["frame_union_count"] == 3
    assert contained["containment_direction"] == "right_in_left"
    assert contained["details_merge_eligible_observed"] is True
    assert contained["details_inclusion_observed"] is True


def test_overlap_contact_only_and_disjoint_are_exclusive(tmp_path):
    module = _module()
    processed, scene_root, tracks, contact_map = _fixture(tmp_path)
    _, relations = module.build_scene_graph(
        "scene0000_00", tracks, processed, scene_root, contact_map
    )
    rows = _relation_by_pair(relations)
    assert rows[(10, 20)]["relation_kind"] == "overlap"
    assert rows[(10, 30)]["relation_kind"] == "contact_only"
    assert rows[(10, 30)]["adjacent_superpoint_pair_count"] == 1
    assert rows[(10, 30)]["boundary_contact_count"] == 7
    assert rows[(10, 40)]["relation_kind"] == "disjoint"
    assert sum(row["relation_kind"] == "overlap" for row in relations) == 1
    assert sum(row["relation_kind"] == "contact_only" for row in relations) == 1
    assert sum(row["relation_kind"] == "disjoint" for row in relations) == 4


def test_pair_order_is_deterministic_and_input_order_independent(tmp_path):
    module = _module()
    processed, scene_root, tracks, contact_map = _fixture(tmp_path)
    expected = module.build_scene_graph(
        "scene0000_00", tracks, processed, scene_root, contact_map
    )
    actual = module.build_scene_graph(
        "scene0000_00", list(reversed(tracks)), processed, scene_root, contact_map
    )
    assert actual == expected
    proposal_ids = [10, 20, 30, 40]
    assert [
        (row["left_proposal_id"], row["right_proposal_id"])
        for row in actual[1]
    ] == list(combinations(proposal_ids, 2))


def test_graph_conserves_nodes_and_relations_and_references_valid_nodes(tmp_path):
    module = _module()
    processed, scene_root, tracks, contact_map = _fixture(tmp_path)
    nodes, relations = module.build_scene_graph(
        "scene0000_00", tracks, processed, scene_root, contact_map
    )
    ids = {node["proposal_id"] for node in nodes}
    assert ids == {track["track_id"] for track in tracks}
    assert len(relations) == len(nodes) * (len(nodes) - 1) // 2
    assert all(row["left_proposal_id"] in ids for row in relations)
    assert all(row["right_proposal_id"] in ids for row in relations)
    module.validate_scene_graph(nodes, relations)


def test_graph_build_does_not_mutate_tracks_or_point_files(tmp_path):
    module = _module()
    processed, scene_root, tracks, contact_map = _fixture(tmp_path)
    before_tracks = copy.deepcopy(tracks)
    before_files = {
        path.name: path.read_bytes() for path in sorted(scene_root.glob("*.npz"))
    }
    module.build_scene_graph(
        "scene0000_00", tracks, processed, scene_root, contact_map
    )
    after_files = {
        path.name: path.read_bytes() for path in sorted(scene_root.glob("*.npz"))
    }
    assert tracks == before_tracks
    assert after_files == before_files


def test_raw_superpoint_contact_map_records_filtered_boundary_counts():
    module = _module()
    processed = np.zeros((4, 10), dtype=np.float32)
    processed[:, :3] = [
        [0.00, 0.00, 0.00], [0.01, 0.00, 0.00],
        [0.02, 0.00, 0.00], [0.03, 0.00, 0.00],
    ]
    processed[:, 9] = [7, 7, 9, 9]
    contacts = module.build_superpoint_contact_map(
        processed,
        adjacency_knn=3,
        adjacency_max_distance=0.05,
        min_contact_points=1,
        min_contact_ratio=0.0,
    )
    assert set(contacts) == {(7, 9)}
    assert contacts[(7, 9)]["boundary_contact_count"] > 0


def test_cli_contract_has_only_frozen_tracks_processed_scene_and_output_inputs():
    source = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_consensus_proposal_relation_graph.py"
    ).read_text()
    required_arguments = {
        "--scene-list", "--track-root", "--processed-scene-root", "--output-root"
    }
    assert all(argument in source for argument in required_arguments)
    forbidden_arguments = {
        "--prediction-cache", "--semantic-root", "--label-root", "--evaluation-root"
    }
    assert not any(argument in source for argument in forbidden_arguments)


def test_saved_relation_rows_are_json_serializable(tmp_path):
    module = _module()
    processed, scene_root, tracks, contact_map = _fixture(tmp_path)
    nodes, relations = module.build_scene_graph(
        "scene0000_00", tracks, processed, scene_root, contact_map
    )
    json.dumps(nodes, sort_keys=True)
    json.dumps(relations, sort_keys=True)
