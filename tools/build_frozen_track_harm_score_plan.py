#!/usr/bin/env python3
"""Build a no-GT frozen score plan for track-harm focal suppression.

The input relation ledger must contain only the raw no-GT inference features
previously materialized for the target scenes.  Candidate quality, relation,
and component-list probabilities are recomputed from the frozen full-data
model package.  The frozen coexist score is the exact native source score for
native candidates and the full-data quality-head ``q`` score for tracks.
Native scores are never transformed; only related track coexist scores receive
the fixed multiplier ``1 - (1 - P(keep))**3``.  This tool does not materialize
candidates or evaluate AP.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    read_jsonl,
    read_scene_list,
)
from tools.build_safety60_structured_action_plan import (  # noqa: E402
    _quality_predictions,
    _relation_predictions,
)
from tools.build_train_candidate_component_action_utility_ledger import _sha256  # noqa: E402
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    load_safety_scene,
)
from tools.train_candidate_component_list_calibration_head_oof import (  # noqa: E402
    _feature_matrix,
    build_candidate_feature_rows,
)


VERSION = "frozen_track_harm_score_plan_v2"
POLICY = "track_harm_focal_suppression"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def fixed_track_score(original_score: float, keep_probability: float) -> tuple[float, float]:
    """Return the fixed cubic multiplier and suppressed track score."""
    original = float(original_score)
    keep = float(keep_probability)
    if not 0.0 <= original <= 1.0 or not 0.0 <= keep <= 1.0:
        raise ValueError("score probabilities must lie in [0, 1]")
    multiplier = 1.0 - (1.0 - keep) ** 3
    planned = original * multiplier
    if planned > original or planned < 0.0:
        raise AssertionError("fixed track suppression violated score bounds")
    return multiplier, planned


def _components(relation_rows: list[dict]) -> list[dict]:
    by_component: dict[int, list[dict]] = defaultdict(list)
    for row in relation_rows:
        if row.get("contracts", {}).get("feature_ground_truth_usage") != "none":
            raise ValueError("relation ledger violates no-GT feature contract")
        by_component[int(row["relation_component_id"])].append(row)
    ids = sorted(by_component)
    if ids != list(range(len(ids))):
        raise ValueError("relation component ids are not contiguous")
    return [{
        "relation_component_id": component_id,
        "native_exact_geometry_group_ids": sorted({
            str(row["native_exact_geometry_group_id"])
            for row in by_component[component_id]
        }),
        "track_ids": sorted({
            int(row["track_id"]) for row in by_component[component_id]
        }),
    } for component_id in ids]


def _component_candidates(
    scene: str,
    components: list[dict],
    relation_rows: list[dict],
    feature_rows: list[dict],
) -> list[dict]:
    lookup = {
        (str(row["candidate_source"]), int(row["candidate_id"])): row
        for row in feature_rows
    }
    groups: dict[str, list[int]] = {}
    for row in relation_rows:
        group_id = str(row["native_exact_geometry_group_id"])
        members = sorted(map(int, row["native_member_candidate_ids"]))
        previous = groups.setdefault(group_id, members)
        if previous != members:
            raise ValueError(f"{scene}: inconsistent native geometry group {group_id}")
    result = []
    seen = set()
    for component in components:
        component_id = int(component["relation_component_id"])
        for group_id in component["native_exact_geometry_group_ids"]:
            members = groups[str(group_id)]
            representative = min(members, key=lambda candidate_id: (
                -float(lookup[(NATIVE_SOURCE, candidate_id)]["original_source_score"]),
                candidate_id,
            ))
            key = (NATIVE_SOURCE, representative)
            if key in seen:
                raise ValueError(f"{scene}: native representative repeated across components")
            seen.add(key)
            result.append({
                **lookup[key],
                "relation_component_id": component_id,
                "native_exact_geometry_group_id": str(group_id),
            })
        for track_id in component["track_ids"]:
            key = (TRACK_SOURCE, int(track_id))
            if key in seen:
                raise ValueError(f"{scene}: track repeated across components")
            seen.add(key)
            result.append({
                **lookup[key],
                "relation_component_id": component_id,
            })
    return result


def _stacked_relation_rows(
    relation_rows: list[dict], feature_rows: list[dict], quality: dict, package: dict,
) -> list[dict]:
    candidate_index = {
        (str(row["candidate_source"]), int(row["candidate_id"])): index
        for index, row in enumerate(feature_rows)
    }
    relation_predictions = _relation_predictions(package, relation_rows)
    result = []
    for row, relation in zip(relation_rows, relation_predictions):
        track_index = candidate_index[(TRACK_SOURCE, int(row["track_id"]))]
        native_indexes = [
            candidate_index[(NATIVE_SOURCE, int(candidate_id))]
            for candidate_id in row["native_member_candidate_ids"]
        ]
        evidence = {}
        for target in ("q", "valid25", "valid50"):
            track_value = float(quality[target][track_index])
            native_value = float(np.median(quality[target][native_indexes]))
            evidence[f"nested_track_{target}"] = track_value
            evidence[f"nested_native_{target}_median"] = native_value
            evidence[f"nested_{target}_delta_track_minus_native"] = track_value - native_value
        result.append({**evidence, **relation})
    return result


def _scene(scene: str, args, package: dict) -> tuple[list[dict], list[dict], dict]:
    masks, _scores, _classes, feature_rows, tracks, candidate_audit = load_safety_scene(
        scene, args.native_cache, args.quality_ledger_root, args.track_root,
    )
    relation_path = args.no_gt_relation_root / scene / "relation_features_no_gt.jsonl"
    relation_summary_path = args.no_gt_relation_root / scene / "summary.json"
    relation_rows = read_jsonl(relation_path)
    relation_summary = json.loads(relation_summary_path.read_text())
    if relation_summary.get("ground_truth_usage") != "none":
        raise ValueError(f"{scene}: relation summary violates no-GT contract")
    if relation_summary.get("ap_evaluation_run") is not False:
        raise ValueError(f"{scene}: relation source was mixed with AP evaluation")
    if candidate_audit != relation_summary.get("candidate_audit"):
        raise ValueError(f"{scene}: candidate inputs differ from frozen relation ledger")

    components = _components(relation_rows)
    candidates = _component_candidates(scene, components, relation_rows, feature_rows)
    quality, quality_audit = _quality_predictions(package, feature_rows)
    stacked = _stacked_relation_rows(relation_rows, feature_rows, quality, package)
    relation_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    stacked_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for raw, evidence in zip(relation_rows, stacked):
        key = (scene, int(raw["relation_component_id"]))
        relation_by_component[key].append(raw)
        stacked_by_component[key].append(evidence)
    model_rows = build_candidate_feature_rows(
        candidates, relation_by_component, stacked_by_component,
    )
    matrix, feature_names = _feature_matrix(model_rows)
    expected_names = package["metadata"]["feature_contract"]["list_candidate_feature_names"]
    if feature_names != expected_names:
        raise ValueError(f"{scene}: frozen list-head feature schema mismatch")
    keep = package["list_winner_model"].predict_proba(matrix)[:, 1]

    predictions = []
    related_track_predictions = {}
    for row, probability in zip(model_rows, keep):
        source = str(row["candidate_source"])
        candidate_id = int(row["candidate_id"])
        prediction = {
            "scene_name": scene,
            "relation_component_id": int(row["relation_component_id"]),
            "candidate_source": source,
            "candidate_id": candidate_id,
            "keep_probability": float(probability),
            "policy_applied": source == TRACK_SOURCE,
            "ground_truth_usage": "none",
        }
        predictions.append(prediction)
        if source == TRACK_SOURCE:
            related_track_predictions[candidate_id] = prediction

    score_plan = []
    native_count = 0
    track_count = 0
    changed_track_count = 0
    candidate_index = {
        (str(row["candidate_source"]), int(row["candidate_id"])): index
        for index, row in enumerate(feature_rows)
    }
    for row in feature_rows:
        source = str(row["candidate_source"])
        candidate_id = int(row["candidate_id"])
        source_score = float(row["original_source_score"])
        if source == NATIVE_SOURCE:
            native_count += 1
            coexist_score = source_score
            planned = coexist_score
            multiplier = 1.0
            probability = None
            component_id = None
            reason = "native_score_bit_for_bit_frozen"
        elif candidate_id in related_track_predictions:
            track_count += 1
            coexist_score = float(quality["q"][candidate_index[(source, candidate_id)]])
            prediction = related_track_predictions[candidate_id]
            probability = float(prediction["keep_probability"])
            multiplier, planned = fixed_track_score(coexist_score, probability)
            component_id = int(prediction["relation_component_id"])
            reason = POLICY
            changed_track_count += int(planned != coexist_score)
        else:
            track_count += 1
            coexist_score = float(quality["q"][candidate_index[(source, candidate_id)]])
            planned = coexist_score
            multiplier = 1.0
            probability = None
            component_id = None
            reason = "track_without_relation_component_frozen"
        score_plan.append({
            "scene_name": scene,
            "candidate_source": source,
            "candidate_id": candidate_id,
            "relation_component_id": component_id,
            "original_source_score": source_score,
            "frozen_coexist_score": coexist_score,
            "keep_probability": probability,
            "suppression_multiplier": multiplier,
            "planned_score": planned,
            "score_changed_vs_frozen_coexist": planned != coexist_score,
            "reason": reason,
            "candidate_removed": False,
            "geometry_modified": False,
            "class_modified": False,
            "ground_truth_usage": "none",
            "ap_evaluation_run": False,
        })
    if native_count != masks.shape[1] or track_count != len(tracks):
        raise AssertionError(f"{scene}: score plan does not cover every candidate")
    if any(row["planned_score"] != row["frozen_coexist_score"] for row in score_plan
           if row["candidate_source"] == NATIVE_SOURCE):
        raise AssertionError(f"{scene}: native score changed")
    summary = {
        "scene_name": scene,
        "native_candidate_count": native_count,
        "track_candidate_count": track_count,
        "relation_count": len(relation_rows),
        "relation_component_count": len(components),
        "component_candidate_prediction_count": len(predictions),
        "related_track_count": len(related_track_predictions),
        "changed_track_score_count": changed_track_count,
        "candidate_count_modification_count": 0,
        "candidate_geometry_modification_count": 0,
        "candidate_class_modification_count": 0,
        "native_score_modification_count": 0,
        "quality_prediction_audit": quality_audit,
        "candidate_audit": candidate_audit,
        "ground_truth_usage": "none",
        "ap_evaluation_run": False,
        "input_provenance": {
            "relation_features_no_gt_sha256": _sha256(relation_path),
            "relation_scene_summary_sha256": _sha256(relation_summary_path),
        },
    }
    return predictions, score_plan, summary


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    expected_count = args.expected_scene_count
    if len(scenes) != expected_count:
        raise ValueError(f"expected {expected_count} scenes, got {len(scenes)}")
    official_scenes = set(read_scene_list(args.official_scene_list))
    overlap = official_scenes & set(scenes)
    if overlap:
        raise ValueError(f"official100 and target scenes overlap: {sorted(overlap)}")
    relation_global_summary = json.loads(
        (args.no_gt_relation_root / "summary.json").read_text()
    )
    if relation_global_summary.get("ground_truth_usage") != "none":
        raise ValueError("global relation ledger violates no-GT contract")
    package_path = args.model_root / "model_package.pkl"
    metadata_path = args.model_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if _sha256(package_path) != metadata["model_package_sha256"]:
        raise ValueError("frozen model package SHA-256 mismatch")
    if metadata.get("frozen_policy") != POLICY:
        raise ValueError("model package policy differs from score-plan policy")
    with package_path.open("rb") as handle:
        package = pickle.load(handle)
    if package["metadata"].get("version") != metadata.get("version"):
        raise ValueError("embedded model metadata version mismatch")
    if package["metadata"].get("frozen_policy") != POLICY:
        raise ValueError("embedded model metadata policy mismatch")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    all_predictions = []
    all_score_plan = []
    scene_summaries = []
    try:
        for index, scene in enumerate(scenes, start=1):
            predictions, score_plan, summary = _scene(scene, args, package)
            scene_root = staging / scene
            scene_root.mkdir()
            _write_jsonl(scene_root / "component_candidate_predictions_no_gt.jsonl", predictions)
            _write_jsonl(scene_root / "frozen_score_plan.jsonl", score_plan)
            (scene_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            all_predictions.extend(predictions)
            all_score_plan.extend(score_plan)
            scene_summaries.append(summary)
            print(f"[frozen track-harm plan] {index}/{len(scenes)} {scene}", flush=True)
        _write_jsonl(staging / "component_candidate_predictions_no_gt.jsonl", all_predictions)
        _write_jsonl(staging / "frozen_score_plan.jsonl", all_score_plan)
        output = {
            "version": VERSION,
            "policy": POLICY,
            "score_contract": (
                "track = full_quality_q * (1 - (1 - P(keep))^3); "
                "native = exact original source score"
            ),
            "scene_count": len(scenes),
            "candidate_count": len(all_score_plan),
            "native_candidate_count": sum(row["native_candidate_count"] for row in scene_summaries),
            "track_candidate_count": sum(row["track_candidate_count"] for row in scene_summaries),
            "relation_count": sum(row["relation_count"] for row in scene_summaries),
            "relation_component_count": sum(row["relation_component_count"] for row in scene_summaries),
            "component_candidate_prediction_count": len(all_predictions),
            "related_track_count": sum(row["related_track_count"] for row in scene_summaries),
            "changed_track_score_count": sum(row["changed_track_score_count"] for row in scene_summaries),
            "score_reason_counts": dict(sorted(Counter(
                row["reason"] for row in all_score_plan
            ).items())),
            "official100_target_scene_overlap_count": 0,
            "feature_schema_verified": True,
            "plan_complete": len(all_score_plan) == sum(
                row["native_candidate_count"] + row["track_candidate_count"]
                for row in scene_summaries
            ),
            "native_scores_bit_for_bit_frozen": all(
                row["planned_score"] == row["frozen_coexist_score"]
                for row in all_score_plan if row["candidate_source"] == NATIVE_SOURCE
            ),
            "candidate_count_modification_count": 0,
            "candidate_geometry_modification_count": 0,
            "candidate_class_modification_count": 0,
            "candidate_removal_count": 0,
            "threshold_scanning": False,
            "continuous_weight_scanning": False,
            "ground_truth_usage": "none",
            "ap_evaluation_run": False,
            "ap_evaluation_run_count": 0,
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "official_scene_list_sha256": _sha256(args.official_scene_list),
                "model_package_sha256": metadata["model_package_sha256"],
                "model_metadata_sha256": _sha256(metadata_path),
                "no_gt_relation_global_summary_sha256": _sha256(
                    args.no_gt_relation_root / "summary.json"
                ),
                "scene_relation_features_sha256": {
                    row["scene_name"]: row["input_provenance"]["relation_features_no_gt_sha256"]
                    for row in scene_summaries
                },
            },
        }
        if not output["plan_complete"] or not output["native_scores_bit_for_bit_frozen"]:
            raise AssertionError("frozen score plan contract failed")
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
    parser.add_argument("--official-scene-list", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--no-gt-relation-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "official_scene_list", "model_root", "native_cache",
        "track_root", "quality_ledger_root", "no_gt_relation_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    for name in ("native_cache", "track_root", "quality_ledger_root", "no_gt_relation_root"):
        if "ground_truth" in str(getattr(args, name)).lower():
            raise SystemExit(f"refusing GT-like input path: {name}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
