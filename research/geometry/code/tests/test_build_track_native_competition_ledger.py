import importlib.util
import json
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_track_native_competition_ledger.py"
    spec = importlib.util.spec_from_file_location("competition_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relations_conserve_pairs_and_record_geometry_and_score_facts():
    module = _module()
    masks = np.asarray([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=bool)
    sizes = np.count_nonzero(masks, axis=0)
    rows, summary = module.track_native_relations(
        np.asarray([0, 1, 2]),
        masks,
        sizes,
        np.asarray([1.0, 0.4, 0.9]),
        proposal_id=7,
        track_score=0.8,
        geometry_variant="source",
    )
    assert len(rows) == 2
    assert summary["overlap_native_candidate_count"] == 2
    assert summary["disjoint_native_candidate_count"] == 1
    assert summary["best_point_iou"] == 1.0
    assert rows[0]["track_inside_native_strict_099"] is True
    assert rows[0]["mutual_duplicate_strict_099"] is True
    assert rows[0]["score_relation"] == "native_higher"
    assert rows[0]["native_contains_track_with_higher_or_equal_score_observed"] is True


def test_comparison_preserves_non_geometry_metadata_and_tracks_new_overlap():
    module = _module()
    source = {
        "proposal_id": 1, "track_id": 1, "lineage_proposal_ids": [1, 4],
        "mean_node_quality": 0.8, "superpoint_ids": [10], "point_count": 2,
        "points_path": "source.npz", "decision_state": "source",
    }
    grow = dict(source)
    grow.update({
        "superpoint_ids": [10, 11], "point_count": 3,
        "points_path": "grow.npz", "decision_state": "grow",
    })
    source_summary = {"best_point_iou": 0.2}
    grow_summary = {"best_point_iou": 0.3}
    row = module.compare_proposal(
        source, grow, np.asarray([0, 1]), np.asarray([0, 1, 2]),
        source_summary, grow_summary, {3}, {3, 5},
    )
    assert row["grow_geometry_changed"] is True
    assert row["grow_added_point_count"] == 1
    assert row["best_native_iou_change"] == "increased"
    assert row["added_overlap_native_candidate_ids"] == [5]
    assert row["lineage_proposal_ids"] == [1, 4]


def test_parser_has_no_gt_ap_semantic_or_action_inputs():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("gt", "ap", "class", "semantic", "suppress", "score-field")
    assert not any(any(token in option for token in forbidden) for option in options)


def test_native_contract_accepts_strict_single_scene_train_stream_manifest(tmp_path):
    module = _module()
    cache_root = tmp_path / "scene0001_01" / "native_cache"
    cache_root.mkdir(parents=True)
    manifest = {
        "scene_name": "scene0001_01",
        "split": "official_scannet200_train",
        "cache_contract": "Mask3D + YOLO-World only",
        "ground_truth_usage": "none",
        "sam_inference": False,
        "d2b_inference": False,
    }
    (cache_root.parent / "native_export_manifest.json").write_text(
        json.dumps(manifest)
    )

    contract = module._native_cache_contract(
        cache_root, 1, ["scene0001_01"]
    )

    assert contract["mode"] == "mask3d_yoloworld_only"
    assert contract["manifest_path"].endswith("native_export_manifest.json")


def test_native_contract_rejects_stream_manifest_scene_mismatch(tmp_path):
    module = _module()
    cache_root = tmp_path / "scene0001_01" / "native_cache"
    cache_root.mkdir(parents=True)
    manifest = {
        "scene_name": "scene9999_99",
        "split": "official_scannet200_train",
        "cache_contract": "Mask3D + YOLO-World only",
        "ground_truth_usage": "none",
        "sam_inference": False,
        "d2b_inference": False,
    }
    (cache_root.parent / "native_export_manifest.json").write_text(
        json.dumps(manifest)
    )

    try:
        module._native_cache_contract(cache_root, 1, ["scene0001_01"])
    except ValueError as error:
        assert "scene differs" in str(error)
    else:
        raise AssertionError("mismatched stream manifest was accepted")
