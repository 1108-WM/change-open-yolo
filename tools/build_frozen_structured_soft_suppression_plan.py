#!/usr/bin/env python3
"""Build a frozen structured soft-suppression plan without ground truth.

The deployable official100 full-data package first selects the single frozen
``structured_lower_relation_veto`` action for every relation component.  No
candidate is deleted.  Candidates that the hard action would remove receive
the fixed score factor

``1 - max(0, P(positive) - P(harmful))``.

The formula has no fitted coefficient or threshold.  This tool only writes a
no-GT plan and immutable-input audit; it never evaluates AP or changes a
candidate file.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.build_safety60_structured_action_plan import (  # noqa: E402
    POLICY,
    _resolve,
    _scene,
    _write_jsonl,
)
from tools.build_train_candidate_component_action_utility_ledger import _sha256  # noqa: E402
from tools.evaluate_official100_structured_soft_suppression_oof_ap import (  # noqa: E402
    suppression_strength,
)


VERSION = "frozen_structured_soft_suppression_plan_v1"
SCORE_RULE = "score *= 1 - max(0, P(positive)-P(harmful)) for candidates the hard action removes"


def soft_plan_row(action: dict, prediction: dict | None) -> dict:
    component_groups = {str(value) for value in action["component_native_exact_geometry_group_ids"]}
    component_tracks = {int(value) for value in action["component_track_ids"]}
    kept_groups = {str(value) for value in action["kept_native_exact_geometry_group_ids"]}
    kept_tracks = {int(value) for value in action["kept_track_ids"]}
    if not kept_groups <= component_groups or not kept_tracks <= component_tracks:
        raise ValueError("selected action keeps a candidate outside its relation component")
    if action["selected_action_kind"] == "coexist":
        if prediction is not None:
            raise ValueError("coexist action must not have a noncoexist prediction")
        strength = 0.0
        probability = None
    else:
        if prediction is None:
            raise ValueError("noncoexist action lacks its frozen state prediction")
        strength = suppression_strength(prediction)
        probability = prediction["state_probability"]
    factor = 1.0 - strength
    return {
        **action,
        "score_rule": SCORE_RULE,
        "selected_action_state_probability": probability,
        "suppression_strength": strength,
        "suppression_factor": factor,
        "suppressed_native_exact_geometry_group_ids": sorted(component_groups - kept_groups),
        "suppressed_track_ids": sorted(component_tracks - kept_tracks),
        "candidate_retained": True,
        "candidate_count_modified": False,
        "candidate_geometry_modified": False,
        "candidate_class_modified": False,
    }


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != args.expected_scene_count:
        raise ValueError(
            f"expected {args.expected_scene_count} evaluation scenes, got {len(scenes)}"
        )
    evaluation = set(scenes)
    overlap_counts = {}
    disjoint_digests = {}
    for path in args.disjoint_scene_list:
        reference = set(read_scene_list(path))
        overlap = sorted(evaluation & reference)
        if overlap:
            raise ValueError(f"evaluation scenes overlap {path}: {overlap}")
        overlap_counts[str(path)] = 0
        disjoint_digests[str(path)] = _sha256(path)

    package_path = args.model_root / "model_package.pkl"
    metadata_path = args.model_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if _sha256(package_path) != metadata["model_package_sha256"]:
        raise ValueError("frozen model package SHA-256 mismatch")
    with package_path.open("rb") as handle:
        package = pickle.load(handle)
    if package["metadata"]["version"] != metadata["version"]:
        raise ValueError("model package metadata version mismatch")
    if metadata.get("frozen_policy") != POLICY or metadata.get("threshold_scanning"):
        raise ValueError("model package is not the frozen single-policy contract")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    aggregate_rows = []
    scene_summaries = []
    try:
        for index, scene in enumerate(scenes, start=1):
            relation_rows, prediction_rows, action_rows, summary = _scene(scene, args, package)
            prediction_by_key = {
                (int(row["relation_component_id"]), str(row["action_name"])): row
                for row in prediction_rows
            }
            soft_rows = []
            for action in action_rows:
                key = (int(action["relation_component_id"]), str(action["selected_action_name"]))
                prediction = None if action["selected_action_kind"] == "coexist" else prediction_by_key.get(key)
                soft_rows.append(soft_plan_row(action, prediction))

            scene_root = staging / scene
            scene_root.mkdir()
            _write_jsonl(scene_root / "relation_features_no_gt.jsonl", relation_rows)
            _write_jsonl(scene_root / "action_predictions_no_gt.jsonl", prediction_rows)
            _write_jsonl(scene_root / "soft_suppression_plan.jsonl", soft_rows)
            summary = {
                **summary,
                "score_rule": SCORE_RULE,
                "soft_plan_component_count": len(soft_rows),
                "soft_target_native_group_count": sum(
                    len(row["suppressed_native_exact_geometry_group_ids"]) for row in soft_rows
                ),
                "soft_target_track_count": sum(len(row["suppressed_track_ids"]) for row in soft_rows),
                "score_changed_target_count": sum(
                    (len(row["suppressed_native_exact_geometry_group_ids"]) + len(row["suppressed_track_ids"]))
                    for row in soft_rows if row["suppression_factor"] < 1.0
                ),
            }
            (scene_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            aggregate_rows.extend(soft_rows)
            scene_summaries.append(summary)
            print(f"[frozen soft no-GT plan] {index}/{len(scenes)} {scene}", flush=True)

        _write_jsonl(staging / "soft_suppression_plan.jsonl", aggregate_rows)
        component_count = sum(int(row["relation_component_count"]) for row in scene_summaries)
        output = {
            "version": VERSION,
            "source_policy": POLICY,
            "score_rule": SCORE_RULE,
            "fitted_soft_suppression_parameter_count": 0,
            "scene_count": len(scenes),
            "relation_count": sum(int(row["relation_count"]) for row in scene_summaries),
            "relation_component_count": component_count,
            "selected_action_kind_counts": dict(sorted(Counter(
                row["selected_action_kind"] for row in aggregate_rows
            ).items())),
            "soft_target_native_group_count": sum(
                len(row["suppressed_native_exact_geometry_group_ids"]) for row in aggregate_rows
            ),
            "soft_target_track_count": sum(len(row["suppressed_track_ids"]) for row in aggregate_rows),
            "score_changed_target_count": sum(
                (len(row["suppressed_native_exact_geometry_group_ids"]) + len(row["suppressed_track_ids"]))
                for row in aggregate_rows if row["suppression_factor"] < 1.0
            ),
            "disjoint_scene_overlap_counts": overlap_counts,
            "feature_schema_verified": True,
            "plan_complete": len(aggregate_rows) == component_count,
            "ground_truth_usage": "none",
            "candidate_files_modified": False,
            "candidate_file_modification_count": 0,
            "candidate_count_modified": False,
            "candidate_geometry_modified": False,
            "candidate_class_modified": False,
            "ap_evaluation_run": False,
            "ap_evaluation_run_count": 0,
            "threshold_scanning": False,
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "disjoint_scene_list_sha256": disjoint_digests,
                "model_package_sha256": metadata["model_package_sha256"],
                "scene_candidate_input_sha256": {
                    row["scene_name"]: row["candidate_input_combined_sha256"]
                    for row in scene_summaries
                },
            },
        }
        if not output["plan_complete"]:
            raise AssertionError("soft plan does not cover every relation component")
        (staging / "summary.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--disjoint-scene-list", type=Path, action="append", default=[])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--yoloworld-sam-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "model_root", "native_cache", "track_root", "quality_ledger_root",
        "automatic_root", "yoloworld_sam_root", "dataset_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    args.disjoint_scene_list = [_resolve(path) for path in args.disjoint_scene_list]
    if any("ground_truth" in str(getattr(args, name)).lower() for name in (
        "native_cache", "track_root", "quality_ledger_root", "automatic_root",
        "yoloworld_sam_root", "dataset_root",
    )):
        raise SystemExit("no-GT planning refuses any input path containing ground_truth")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root exists and is non-empty: {args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
