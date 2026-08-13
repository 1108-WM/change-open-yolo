import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_growth_variant_plan.py"
    spec = importlib.util.spec_from_file_location("automatic_sam_growth_variant_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track_record():
    return {
        "scene_name": "scene0000_00", "track_id": 3, "source_observation_ids": [4, 5], "source_view_count": 2,
        "internal_cross_view_edge_count": 1, "mean_internal_reprojection_support": 0.7,
        "supported_superpoints": [{"superpoint_id": 10}, {"superpoint_id": 20}],
        "growth_frontier_superpoints": [{
            "superpoint_id": 30, "adjacent_seed_superpoint_ids": [20], "adjacent_seed_superpoint_count": 1,
            "boundary_contact_count_sum": 4, "mean_normal_difference": 0.1, "mean_color_difference": 0.2,
            "mean_boundary_distance": 0.01, "cross_view_link_count": 2,
            "mean_cross_view_reprojection_support": 0.8, "linked_seed_node_count": 2,
        }],
    }


def test_variant_plan_keeps_seed_and_each_one_hop_action_separate():
    planner = _module()
    variants = planner.expand_track_variant_plan(_track_record())
    assert len(variants) == 2
    assert variants[0]["variant_type"] == "seed_superpoint_closure"
    assert variants[0]["added_superpoint_ids"] == []
    assert variants[1]["variant_type"] == "one_hop_positive_evidence_addition"
    assert variants[1]["added_superpoint_ids"] == [30]
    assert variants[1]["base_superpoint_ids"] == [10, 20]
    assert "accepted" not in variants[1]


def test_scene_build_writes_plan_without_prediction_masks(tmp_path):
    planner = _module()
    scene_name = "scene0000_00"
    source = tmp_path / "ledger" / scene_name
    source.mkdir(parents=True)
    (source / "automatic_sam_track_growth_ledger.json").write_text(json.dumps([_track_record()]))
    args = SimpleNamespace(growth_ledger_root=tmp_path / "ledger", output_root=tmp_path / "out")
    summary = planner._build_scene(scene_name, args)
    assert summary["variant_count"] == 2
    output = args.output_root / scene_name
    assert (output / "automatic_sam_growth_variant_plan.jsonl").is_file()
    assert not list(output.glob("*pred_mask*"))
