#!/usr/bin/env python3
"""Run the one allowed safety60 AP replay for the frozen combined champion.

The first phase consumes only frozen official100 models and no-GT safety60
candidate/relation inputs.  It writes a score-plus-pair-union append plan with
all original candidate files untouched.  The second phase is gated by an
explicit flag and a persistent receipt, and makes exactly one class-agnostic
AP aggregation call.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
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
from tools.build_candidate_pair_union_oof_plan import (  # noqa: E402
    _geometry_digest,
    append_score,
)
from tools.diagnose_candidate_pair_union_threshold_cross_oof import (  # noqa: E402
    prior_correct_balanced_probability,
)
from tools.build_frozen_track_harm_score_plan import (  # noqa: E402
    _component_candidates,
    _components,
    _stacked_relation_rows,
    fixed_track_score,
)
from tools.build_train_candidate_pair_union_utility_ledger import (  # noqa: E402
    MIN_REGION_SIZE,
    _pure_inference_relation_features,
    proposal_points,
)
from tools.build_train_candidate_component_action_utility_ledger import _sha256  # noqa: E402
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    _evaluate,
    _load_track_points,
    _prediction,
    load_safety_scene,
)
from tools.evaluate_frozen_track_harm_score_plan_ap import validate_score_row  # noqa: E402
from tools.evaluate_official100_geometry_group_ranking_oof_ap import (  # noqa: E402
    geometry_groups_and_audit,
)
from tools.build_safety60_structured_action_plan import _quality_predictions  # noqa: E402
from tools.build_safety60_structured_action_plan import _relation_predictions  # noqa: E402
from tools.build_train_candidate_component_union_feature_ledger import component_track_features  # noqa: E402
from tools.train_candidate_component_list_calibration_head_oof import (  # noqa: E402
    _feature_matrix,
    build_candidate_feature_rows,
)
from tools.train_candidate_component_union_list_head_oof import augment_union_features  # noqa: E402


POLICY = "champion_track_suppression_plus_pair_union_append"
PLAN_VERSION = "frozen_safety60_champion_pair_union_plan_v2_prior_corrected"
AP_VERSION = "frozen_safety60_champion_pair_union_ap_v2_prior_corrected"
COMBINED_PACKAGE_VERSION = "official100_champion_pair_union_combined_full_v2_prior_corrected"
PAIR_UNION_PACKAGE_VERSION = "official100_pair_union_threshold_cross_full_v2_prior_corrected"
SOURCE_PLAN_VERSION = "frozen_safety60_champion_pair_union_plan_v1"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _load_combined_package(model_root: Path) -> tuple[dict, dict, dict]:
    metadata_path = model_root / "metadata.json"
    package_path = model_root / "model_package.pkl"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("frozen_policy") != POLICY:
        raise ValueError("model root is not the frozen combined champion")
    if _sha256(package_path) != metadata.get("model_package_sha256"):
        raise ValueError("combined model package SHA-256 mismatch")
    with package_path.open("rb") as handle:
        package = pickle.load(handle)
    if metadata.get("version") != COMBINED_PACKAGE_VERSION:
        raise ValueError("combined model package is not the prior-corrected version")
    if package.get("metadata", {}).get("version") != metadata.get("version"):
        raise ValueError("embedded combined model metadata differs from sidecar")
    track_package = package.get("track_suppression_package")
    union_package = package.get("pair_union_append_package")
    if not isinstance(track_package, dict) or not isinstance(union_package, dict):
        raise ValueError("combined package lacks one of its frozen components")
    if track_package.get("metadata", {}).get("frozen_policy") != "track_harm_focal_suppression":
        raise ValueError("unexpected embedded track-suppression package")
    if union_package.get("metadata", {}).get("version") != PAIR_UNION_PACKAGE_VERSION:
        raise ValueError("unexpected embedded pair-union package")
    prior = union_package.get("metadata", {}).get(
        "training_component_balanced_natural_positive_rate"
    )
    if not isinstance(prior, (int, float)) or not 0.0 < float(prior) < 1.0:
        raise ValueError("embedded pair-union package lacks a valid natural positive rate")
    return metadata, track_package, union_package


def corrected_pair_union_probability(raw_probability: float, metadata: dict) -> float:
    """Restore the natural proposal prior used by the official100 OOF policy."""
    prior = metadata.get("training_component_balanced_natural_positive_rate")
    if not isinstance(prior, (int, float)) or not 0.0 < float(prior) < 1.0:
        raise ValueError("pair-union metadata lacks a valid natural positive rate")
    return float(prior_correct_balanced_probability(
        np.asarray([raw_probability], dtype=np.float64), float(prior)
    )[0])


def _existing_geometry_digests(masks: np.ndarray, tracks: list[dict]) -> set[str]:
    """Return exact geometry digests for the entire frozen candidate pool."""
    digests = set()
    by_packed: dict[bytes, int] = {}
    for candidate_id in range(masks.shape[1]):
        column = np.asarray(masks[:, candidate_id], dtype=np.uint8)
        packed = np.packbits(column).tobytes()
        if packed in by_packed:
            continue
        by_packed[packed] = candidate_id
        digests.add(_geometry_digest(np.flatnonzero(column).astype(np.int64)))
    for track in tracks:
        points, _ = _load_track_points(track, masks.shape[0])
        digests.add(_geometry_digest(points))
    return digests


def _union_feature_row(relation: dict, track: np.ndarray, native: np.ndarray, existing: set[str]) -> tuple[dict, np.ndarray]:
    intersection = np.intersect1d(track, native, assume_unique=True)
    proposal = proposal_points("pair_union", track, native)
    features = _pure_inference_relation_features(relation["features"])
    # official100's pair-union ledger retained this historical alias while
    # the safety60 no-GT relation ledger kept the explicit descriptive name.
    # Both are the same raw source-score difference and neither uses GT.
    if "relation__delta_raw_original_score" not in features:
        features["relation__delta_raw_original_score"] = float(
            relation["features"]["original_score_delta_track_minus_native"]
        )
    features.update({
        "proposal__union_point_count": float(len(proposal)),
        "proposal__intersection_point_count": float(len(intersection)),
        "proposal__track_exclusive_point_count": float(len(track) - len(intersection)),
        "proposal__native_exclusive_point_count": float(len(native) - len(intersection)),
        "proposal__union_growth_over_track_fraction": float((len(proposal) - len(track)) / max(1, len(track))),
        "proposal__union_growth_over_native_fraction": float((len(proposal) - len(native)) / max(1, len(native))),
        "proposal__geometry_novel_vs_existing": float(_geometry_digest(proposal) not in existing),
        "proposal__accepted_min_region": float(len(proposal) >= MIN_REGION_SIZE),
    })
    return features, proposal


def _union_rows_for_scene(
    scene: str,
    relation_rows: list[dict],
    masks: np.ndarray,
    feature_rows: list[dict],
    tracks: list[dict],
    track_package: dict,
    union_package: dict,
) -> tuple[list[dict], list[dict], dict]:
    track_by_id = {int(row["track_id"]): row for row in tracks}
    expected_names = list(union_package["metadata"]["feature_names"])
    quality, _quality_audit = _quality_predictions(track_package, feature_rows)
    feature_index = {
        (str(row["candidate_source"]), int(row["candidate_id"])): index
        for index, row in enumerate(feature_rows)
    }
    existing = _existing_geometry_digests(masks, tracks)
    proposed: dict[str, list[dict]] = {}
    audit_rows = []
    for relation in relation_rows:
        if relation.get("contracts", {}).get("feature_ground_truth_usage") != "none":
            raise ValueError(f"{scene}: relation row is not a no-GT inference feature")
        track_id = int(relation["track_id"])
        members = [int(value) for value in relation["native_member_candidate_ids"]]
        track, _ = _load_track_points(track_by_id[track_id], masks.shape[0])
        native = np.flatnonzero(np.asarray(masks[:, members[0]], dtype=bool)).astype(np.int64)
        for member in members[1:]:
            if not np.array_equal(np.asarray(masks[:, member], dtype=bool), np.asarray(masks[:, members[0]], dtype=bool)):
                raise ValueError(f"{scene}: native exact geometry group is inconsistent")
        features, proposal = _union_feature_row(relation, track, native, existing)
        if sorted(features) != expected_names:
            missing = sorted(set(expected_names) - set(features))
            extra = sorted(set(features) - set(expected_names))
            raise ValueError(f"{scene}: pair-union feature schema mismatch; missing={missing}, extra={extra}")
        eligible = bool(features["proposal__geometry_novel_vs_existing"] and features["proposal__accepted_min_region"])
        raw_probability = None
        probability = None
        if eligible:
            matrix = np.asarray([[float(features[name]) for name in expected_names]], dtype=np.float64)
            raw_probability = float(union_package["model"].predict_proba(matrix)[0, 1])
            probability = corrected_pair_union_probability(
                raw_probability, union_package["metadata"]
            )
        digest = _geometry_digest(proposal)
        track_q = float(quality["q"][feature_index[(TRACK_SOURCE, track_id)]])
        native_q = float(np.median([
            quality["q"][feature_index[(NATIVE_SOURCE, candidate_id)]] for candidate_id in members
        ]))
        audit = {
            "scene_name": scene,
            "relation_component_id": int(relation["relation_component_id"]),
            "track_id": track_id,
            "native_exact_geometry_group_id": str(relation["native_exact_geometry_group_id"]),
            "native_member_candidate_ids": members,
            "proposal_geometry_sha256": digest,
            "proposal_point_count": int(len(proposal)),
            "eligible_novel_min_region": eligible,
            "balanced_fit_raw_probability": raw_probability,
            "threshold_cross_probability": probability,
            "track_quality_q": track_q,
            "native_group_median_quality_q": native_q,
            "ground_truth_usage": "none",
            "ap_evaluation_run": False,
        }
        audit_rows.append(audit)
        if eligible:
            proposed.setdefault(digest, []).append({**audit, "points": proposal})

    selected_rows = []
    for candidate_id, (digest, supports) in enumerate(sorted(proposed.items())):
        selected = max(supports, key=lambda row: (
            float(row["threshold_cross_probability"]),
            min(float(row["track_quality_q"]), float(row["native_group_median_quality_q"])),
            -int(row["track_id"]), str(row["native_exact_geometry_group_id"]),
        ))
        base = min(float(selected["track_quality_q"]), float(selected["native_group_median_quality_q"]))
        score = append_score(base, float(selected["threshold_cross_probability"]))
        selected_rows.append({
            **{key: value for key, value in selected.items() if key != "points"},
            "candidate_source": "pair_union_append",
            "candidate_id": candidate_id,
            "policy": "pair_union_threshold_cross_focal_append",
            "base_quality": base,
            "new_score": score,
            "support_relation_count": len(supports),
            "candidate_retained": True,
            "candidate_removed": False,
            "geometry_modified": False,
            "class_modified": False,
        })
        selected_rows[-1]["_points"] = selected["points"]
    summary = {
        "relation_count": len(relation_rows),
        "eligible_relation_proposal_count": sum(row["eligible_novel_min_region"] for row in audit_rows),
        "unique_materialized_candidate_count": len(selected_rows),
        "duplicate_relation_proposal_count": sum(row["eligible_novel_min_region"] for row in audit_rows) - len(selected_rows),
    }
    return audit_rows, selected_rows, summary


def _track_score_rows_for_scene(
    scene: str,
    masks: np.ndarray,
    feature_rows: list[dict],
    tracks: list[dict],
    relation_rows: list[dict],
    candidate_audit: dict,
    track_package: dict,
) -> tuple[list[dict], dict]:
    """Recreate the champion's score plan, including its union feature family."""
    components = _components(relation_rows)
    candidates = _component_candidates(scene, components, relation_rows, feature_rows)
    quality, quality_audit = _quality_predictions(track_package, feature_rows)
    stacked = _stacked_relation_rows(relation_rows, feature_rows, quality, track_package)
    relation_by_component, stacked_by_component = {}, {}
    for raw, evidence in zip(relation_rows, stacked):
        key = (scene, int(raw["relation_component_id"]))
        relation_by_component.setdefault(key, []).append(raw)
        stacked_by_component.setdefault(key, []).append(evidence)
    model_rows = build_candidate_feature_rows(candidates, relation_by_component, stacked_by_component)
    track_by_id = {int(row["track_id"]): row for row in tracks}
    union_lookup = {}
    for component in components:
        component_id = int(component["relation_component_id"])
        raw = relation_by_component[(scene, component_id)]
        group_members = {}
        for row in raw:
            group_id = str(row["native_exact_geometry_group_id"])
            members = [int(value) for value in row["native_member_candidate_ids"]]
            previous = group_members.setdefault(group_id, members)
            if previous != members:
                raise ValueError(f"{scene}: inconsistent exact native group in component")
        native_points = {
            group_id: np.flatnonzero(np.asarray(masks[:, members[0]], dtype=bool)).astype(np.int64)
            for group_id, members in group_members.items()
        }
        selected_track_points = {
            track_id: _load_track_points(track_by_id[track_id], masks.shape[0])[0]
            for track_id in component["track_ids"]
        }
        union_features = component_track_features(native_points, selected_track_points)
        for track_id, features in union_features.items():
            union_lookup[(scene, component_id, int(track_id))] = features
    union_names = list(track_package["metadata"]["feature_contract"]["component_union_feature_names"])
    model_rows = augment_union_features(model_rows, union_lookup, union_names)
    matrix, feature_names = _feature_matrix(model_rows)
    expected_names = track_package["metadata"]["feature_contract"]["list_candidate_feature_names"]
    if feature_names != expected_names:
        raise ValueError(f"{scene}: champion list-head feature schema mismatch")
    keep = track_package["list_winner_model"].predict_proba(matrix)[:, 1]
    related = {
        int(row["candidate_id"]): float(probability)
        for row, probability in zip(model_rows, keep)
        if str(row["candidate_source"]) == TRACK_SOURCE
    }
    candidate_index = {
        (str(row["candidate_source"]), int(row["candidate_id"])): index
        for index, row in enumerate(feature_rows)
    }
    score_rows = []
    for row in feature_rows:
        source = str(row["candidate_source"])
        candidate_id = int(row["candidate_id"])
        original = float(row["original_source_score"])
        if source == NATIVE_SOURCE:
            coexist, probability, multiplier, planned, reason = original, None, 1.0, original, "native_score_bit_for_bit_frozen"
        else:
            coexist = float(quality["q"][candidate_index[(source, candidate_id)]])
            probability = related.get(candidate_id)
            if probability is None:
                multiplier, planned, reason = 1.0, coexist, "track_without_relation_component_frozen"
            else:
                multiplier, planned = fixed_track_score(coexist, probability)
                reason = "track_harm_focal_suppression"
        score_rows.append({
            "scene_name": scene, "candidate_source": source, "candidate_id": candidate_id,
            "original_source_score": original, "frozen_coexist_score": coexist,
            "keep_probability": probability, "suppression_multiplier": multiplier,
            "planned_score": planned, "score_changed_vs_frozen_coexist": planned != coexist,
            "reason": reason, "candidate_removed": False, "geometry_modified": False,
            "class_modified": False, "ground_truth_usage": "none", "ap_evaluation_run": False,
        })
    if len(score_rows) != masks.shape[1] + len(tracks):
        raise AssertionError(f"{scene}: score plan lost candidates")
    return score_rows, {
        "scene_name": scene, "native_candidate_count": masks.shape[1], "track_candidate_count": len(tracks),
        "relation_count": len(relation_rows), "relation_component_count": len(components),
        "related_track_count": len(related),
        "changed_track_score_count": sum(row["score_changed_vs_frozen_coexist"] for row in score_rows if row["candidate_source"] == TRACK_SOURCE),
        "quality_prediction_audit": quality_audit, "candidate_audit": candidate_audit,
        "ground_truth_usage": "none", "ap_evaluation_run": False,
    }


def build_plan(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    official = read_scene_list(args.official_scene_list)
    if len(scenes) != 60 or len(official) != 100 or set(scenes) & set(official):
        raise ValueError("requires disjoint safety60 and official100 scene lists")
    metadata, track_package, union_package = _load_combined_package(args.model_root)
    relation_summary = json.loads((args.relation_root / "summary.json").read_text())
    if relation_summary.get("ground_truth_usage") != "none" or relation_summary.get("ap_evaluation_run") is not False:
        raise ValueError("relation root must be a frozen no-GT/no-AP ledger")
    # The execution environment may impose a short foreground-process limit.
    # Keep this *no-GT-only* staging area stable so an interrupted construction
    # can resume without recreating or altering any frozen candidate input.
    staging = args.plan_root.parent / f".{args.plan_root.name}.no_gt_build_staging"
    staging.mkdir(parents=True, exist_ok=True)
    scene_summaries = []
    all_scores, all_unions, all_union_audits = [], [], []
    try:
        # Reuse the frozen track-harm scorer exactly, but feed its embedded
        # official100 full-data package rather than the obsolete standalone root.
        built_now = 0
        for index, scene in enumerate(scenes, start=1):
            scene_root = staging / scene
            scene_summary_path = scene_root / "summary.json"
            if scene_summary_path.is_file():
                scene_summary = json.loads(scene_summary_path.read_text())
                score_rows = read_jsonl(scene_root / "frozen_score_plan.jsonl")
                union_audits = read_jsonl(scene_root / "pair_union_relation_audit_no_gt.jsonl")
                persisted_unions = read_jsonl(scene_root / "pair_union_append_candidates.jsonl")
                scene_summaries.append(scene_summary)
                all_scores.extend(score_rows)
                all_unions.extend(persisted_unions)
                all_union_audits.extend(union_audits)
                continue
            masks, _native_scores, _classes, feature_rows, tracks, candidate_audit = load_safety_scene(
                scene, args.native_cache, args.quality_ledger_root, args.track_root,
            )
            source_scene_root = args.relation_root / scene
            source_scene_summary = json.loads((source_scene_root / "summary.json").read_text())
            prior_audit = dict(source_scene_summary["candidate_audit"])
            comparable_current = dict(candidate_audit)
            previous_track_digest = prior_audit.pop("automatic_tracks_sha256", None)
            current_track_digest = comparable_current.pop("automatic_tracks_sha256", None)
            if prior_audit != comparable_current:
                raise ValueError(f"{scene}: no-GT relation ledger candidate contract changed")
            relation_rows = read_jsonl(source_scene_root / "relation_features_no_gt.jsonl")
            score_rows, score_summary = _track_score_rows_for_scene(
                scene, masks, feature_rows, tracks, relation_rows, candidate_audit, track_package,
            )
            union_audits, union_rows, union_summary = _union_rows_for_scene(
                scene, relation_rows, masks, feature_rows, tracks, track_package, union_package,
            )
            candidate_root = scene_root / "pair_union_candidates"
            candidate_root.mkdir(parents=True)
            persisted_unions = []
            for row in union_rows:
                points = row.pop("_points")
                path = candidate_root / f"union{int(row['candidate_id']):04d}_points.npz"
                np.savez_compressed(path, point_indices=points)
                persisted_unions.append({
                    **row,
                    "points_path": str((args.plan_root / scene / "pair_union_candidates" / path.name).resolve()),
                })
            _write_jsonl(scene_root / "frozen_score_plan.jsonl", score_rows)
            _write_jsonl(scene_root / "pair_union_relation_audit_no_gt.jsonl", union_audits)
            _write_jsonl(scene_root / "pair_union_append_candidates.jsonl", persisted_unions)
            scene_summary = {
                "scene_name": scene,
                "candidate_audit": candidate_audit,
                "track_score_plan": score_summary,
                "pair_union": union_summary,
                "ground_truth_usage": "none",
                "ap_evaluation_run": False,
                "candidate_file_modification_count": 0,
                "candidate_files_modified": False,
            }
            (scene_root / "summary.json").write_text(json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            scene_summaries.append(scene_summary)
            all_scores.extend(score_rows)
            all_unions.extend(persisted_unions)
            all_union_audits.extend(union_audits)
            built_now += 1
            print(f"[combined no-GT plan] {index}/60 {scene}: unions={len(persisted_unions)}", flush=True)
            del masks, feature_rows, tracks, relation_rows, union_audits, union_rows, score_rows
            gc.collect()
            if args.max_scenes_per_invocation and built_now >= args.max_scenes_per_invocation:
                return {
                    "no_gt_build_complete": False,
                    "completed_scene_count": len(scene_summaries),
                    "remaining_scene_count": len(scenes) - len(scene_summaries),
                    "ground_truth_read": False,
                    "ap_evaluation_run_count": 0,
                }
        _write_jsonl(staging / "frozen_score_plan.jsonl", all_scores)
        _write_jsonl(staging / "pair_union_relation_audit_no_gt.jsonl", all_union_audits)
        _write_jsonl(staging / "pair_union_append_candidates.jsonl", all_unions)
        output = {
            "version": PLAN_VERSION,
            "policy": POLICY,
            "scene_count": len(scenes),
            "candidate_count": len(all_scores),
            "track_score_changed_count": sum(row["score_changed_vs_frozen_coexist"] for row in all_scores),
            "pair_union_relation_count": len(all_union_audits),
            "pair_union_eligible_relation_proposal_count": sum(row["eligible_novel_min_region"] for row in all_union_audits),
            "pair_union_append_candidate_count": len(all_unions),
            "candidate_count_modification_count": 0,
            "candidate_geometry_modification_count": 0,
            "candidate_class_modification_count": 0,
            "candidate_removal_count": 0,
            "native_scores_bit_for_bit_frozen": all(
                row["planned_score"] == row["frozen_coexist_score"]
                for row in all_scores if row["candidate_source"] == NATIVE_SOURCE
            ),
            "pair_union_append_only": True,
            "pair_union_probability_prior_correction": {
                "applied": True,
                "natural_positive_rate": float(union_package["metadata"][
                    "training_component_balanced_natural_positive_rate"
                ]),
            },
            "threshold_scanning": False,
            "continuous_weight_scanning": False,
            "ground_truth_usage": "none",
            "ap_evaluation_run": False,
            "ap_evaluation_run_count": 0,
            "input_provenance": {
                "combined_model_package_sha256": metadata["model_package_sha256"],
                "combined_model_metadata_sha256": _sha256(args.model_root / "metadata.json"),
                "scene_list_sha256": _sha256(args.scene_list),
                "official_scene_list_sha256": _sha256(args.official_scene_list),
                "relation_root_summary_sha256": _sha256(args.relation_root / "summary.json"),
                "relation_contract_adapter": "automatic-tracks JSON digest refreshed only after exact candidate-audit parity excluding that digest",
                "scene_candidate_audits": {row["scene_name"]: row["candidate_audit"] for row in scene_summaries},
            },
        }
        if not output["native_scores_bit_for_bit_frozen"]:
            raise AssertionError("native score freeze contract failed")
        (staging / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.plan_root)
    except Exception:
        # Preserve the no-GT checkpoint for diagnosis/resume.  It never
        # becomes the formal plan root unless all 60 scenes finish.
        raise
    return output


def build_plan_from_v1(args: argparse.Namespace) -> dict:
    """Upgrade the frozen v1 no-GT plan by applying the missing prior correction.

    The correction is strictly monotonic, so it cannot change which support
    relation wins when duplicate proposal geometries are aggregated.  Existing
    track scores and union geometries therefore remain bit-for-bit frozen.
    """
    source = args.source_plan_root
    source_summary_path = source / "summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    if source_summary.get("version") != SOURCE_PLAN_VERSION:
        raise ValueError("source plan is not the frozen v1 no-GT plan")
    required = {
        "scene_count": 60,
        "candidate_count_modification_count": 0,
        "candidate_geometry_modification_count": 0,
        "candidate_class_modification_count": 0,
        "candidate_removal_count": 0,
        "native_scores_bit_for_bit_frozen": True,
        "pair_union_append_only": True,
        "ground_truth_usage": "none",
        "ap_evaluation_run": False,
        "ap_evaluation_run_count": 0,
    }
    mismatch = {key: [value, source_summary.get(key)] for key, value in required.items()
                if source_summary.get(key) != value}
    if mismatch:
        raise ValueError(f"source v1 plan contract mismatch: {mismatch}")
    scenes = read_scene_list(args.scene_list)
    metadata, _track_package, union_package = _load_combined_package(args.model_root)
    prior = float(union_package["metadata"][
        "training_component_balanced_natural_positive_rate"
    ])
    staging = args.plan_root.parent / f".{args.plan_root.name}.no_gt_build_staging"
    if staging.exists() and any(staging.iterdir()):
        raise ValueError(f"refusing to mix upgraded plan with existing staging: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    all_scores, all_unions, all_union_audits = [], [], []
    scene_summaries = []
    try:
        for index, scene in enumerate(scenes, start=1):
            source_scene = source / scene
            target_scene = staging / scene
            target_scene.mkdir(parents=True)
            score_rows = read_jsonl(source_scene / "frozen_score_plan.jsonl")
            audit_rows = read_jsonl(source_scene / "pair_union_relation_audit_no_gt.jsonl")
            union_rows = read_jsonl(source_scene / "pair_union_append_candidates.jsonl")
            for row in audit_rows:
                raw_value = row["threshold_cross_probability"]
                if raw_value is None:
                    row["balanced_fit_raw_probability"] = None
                    continue
                raw = float(raw_value)
                row["balanced_fit_raw_probability"] = raw
                row["threshold_cross_probability"] = corrected_pair_union_probability(
                    raw, union_package["metadata"]
                )
            candidate_root = target_scene / "pair_union_candidates"
            candidate_root.mkdir()
            for row in union_rows:
                raw = float(row["threshold_cross_probability"])
                corrected = corrected_pair_union_probability(raw, union_package["metadata"])
                row["balanced_fit_raw_probability"] = raw
                row["threshold_cross_probability"] = corrected
                row["new_score"] = append_score(float(row["base_quality"]), corrected)
                source_points = Path(row["points_path"])
                target_points = candidate_root / source_points.name
                try:
                    os.link(source_points, target_points)
                except OSError:
                    shutil.copy2(source_points, target_points)
                row["points_path"] = str((args.plan_root / scene / "pair_union_candidates" / target_points.name).resolve())
            _write_jsonl(target_scene / "frozen_score_plan.jsonl", score_rows)
            _write_jsonl(target_scene / "pair_union_relation_audit_no_gt.jsonl", audit_rows)
            _write_jsonl(target_scene / "pair_union_append_candidates.jsonl", union_rows)
            scene_summary = json.loads((source_scene / "summary.json").read_text())
            scene_summary["pair_union_probability_prior_correction"] = {
                "applied": True, "natural_positive_rate": prior,
            }
            (target_scene / "summary.json").write_text(
                json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            all_scores.extend(score_rows)
            all_union_audits.extend(audit_rows)
            all_unions.extend(union_rows)
            scene_summaries.append(scene_summary)
            print(f"[combined no-GT prior upgrade] {index}/60 {scene}: unions={len(union_rows)}", flush=True)
        _write_jsonl(staging / "frozen_score_plan.jsonl", all_scores)
        _write_jsonl(staging / "pair_union_relation_audit_no_gt.jsonl", all_union_audits)
        _write_jsonl(staging / "pair_union_append_candidates.jsonl", all_unions)
        output = dict(source_summary)
        output.update({
            "version": PLAN_VERSION,
            "pair_union_probability_prior_correction": {
                "applied": True, "natural_positive_rate": prior,
            },
            "ground_truth_usage": "none",
            "ap_evaluation_run": False,
            "ap_evaluation_run_count": 0,
        })
        provenance = dict(output.get("input_provenance", {}))
        provenance.update({
            "combined_model_package_sha256": metadata["model_package_sha256"],
            "combined_model_metadata_sha256": _sha256(args.model_root / "metadata.json"),
            "source_invalid_v1_plan_summary_sha256": _sha256(source_summary_path),
            "prior_upgrade_contract": "strictly monotonic probability transform; frozen support selection and geometry reused",
            "scene_candidate_audits": {
                row["scene_name"]: row["candidate_audit"] for row in scene_summaries
            },
        })
        output["input_provenance"] = provenance
        (staging / "summary.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.plan_root)
    except Exception:
        raise
    return output


def _official_eligibility(path: Path) -> dict:
    summary = json.loads(path.read_text())
    deltas = {key: float(value) for key, value in summary["delta_vs_frozen_coexist"].items()}
    if not all(deltas[key] > 0.0 for key in ("ap", "ap50", "ap25")):
        raise ValueError(f"official100 combined policy is not all-positive: {deltas}")
    if int(summary.get("main_ap_positive_fold_count_vs_champion", 0)) != 5:
        raise ValueError("official100 combined policy is not main-AP positive in every fold")
    return deltas


def _load_rows_by_scene(path: Path, scenes: list[str]) -> dict[str, list[dict]]:
    result = {scene: [] for scene in scenes}
    for row in read_jsonl(path):
        scene = str(row["scene_name"])
        if scene not in result:
            raise ValueError(f"plan contains a non-target scene: {scene}")
        result[scene].append(row)
    return result


def preflight(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    metadata, track_package, union_package = _load_combined_package(args.model_root)
    plan_summary_path = args.plan_root / "summary.json"
    plan = json.loads(plan_summary_path.read_text())
    preflight_success = args.plan_root.parent / f".{args.plan_root.name}.no_gt_preflight_passed.json"
    if preflight_success.is_file():
        persisted = json.loads(preflight_success.read_text())
        if persisted.get("plan_summary_sha256") == _sha256(plan_summary_path):
            audit = persisted.get("audit")
            if isinstance(audit, dict) and audit.get("preflight_passed") is True:
                return audit
    required = {
        "version": PLAN_VERSION, "policy": POLICY, "scene_count": 60,
        "candidate_count_modification_count": 0, "candidate_geometry_modification_count": 0,
        "candidate_class_modification_count": 0, "candidate_removal_count": 0,
        "native_scores_bit_for_bit_frozen": True, "pair_union_append_only": True,
        "pair_union_probability_prior_correction": {
            "applied": True,
            "natural_positive_rate": float(union_package["metadata"][
                "training_component_balanced_natural_positive_rate"
            ]),
        },
        "threshold_scanning": False, "continuous_weight_scanning": False,
        "ground_truth_usage": "none", "ap_evaluation_run": False, "ap_evaluation_run_count": 0,
    }
    mismatch = {key: [value, plan.get(key)] for key, value in required.items() if plan.get(key) != value}
    if mismatch:
        raise ValueError(f"combined plan contract mismatch: {mismatch}")
    if plan["input_provenance"]["combined_model_package_sha256"] != metadata["model_package_sha256"]:
        raise ValueError("combined plan/model package mismatch")
    score_by_scene = _load_rows_by_scene(args.plan_root / "frozen_score_plan.jsonl", scenes)
    union_by_scene = _load_rows_by_scene(args.plan_root / "pair_union_append_candidates.jsonl", scenes)
    old_score_by_scene = _load_rows_by_scene(args.frozen_coexist_plan_root / "frozen_score_plan.jsonl", scenes)
    old_baseline = json.loads(args.frozen_coexist_ap_summary.read_text())["frozen_coexist_baseline"]
    preflight_staging = args.plan_root.parent / f".{args.plan_root.name}.no_gt_preflight_staging.json"
    cached = json.loads(preflight_staging.read_text()) if preflight_staging.is_file() else {"version": PLAN_VERSION, "scenes": {}}
    if cached.get("version") != PLAN_VERSION:
        raise ValueError("preflight checkpoint version mismatch")
    cached_scenes = dict(cached.get("scenes", {}))
    built_now = 0
    for index, scene in enumerate(scenes, start=1):
        prior = cached_scenes.get(scene)
        if prior is not None:
            continue
        masks, native_scores, _classes, feature_rows, tracks, candidate_audit = load_safety_scene(
            scene, args.native_cache, args.quality_ledger_root, args.track_root,
        )
        if candidate_audit != plan["input_provenance"]["scene_candidate_audits"][scene]:
            raise ValueError(f"{scene}: candidate inputs changed after plan creation")
        rows = score_by_scene[scene]
        identities = {(str(row["candidate_source"]), int(row["candidate_id"])) for row in rows}
        if len(rows) != len(identities) or len(rows) != masks.shape[1] + len(tracks):
            raise ValueError(f"{scene}: frozen score plan coverage mismatch")
        for row in rows:
            validate_score_row(row)
        old_rows = {
            (str(row["candidate_source"]), int(row["candidate_id"])): row
            for row in old_score_by_scene[scene]
        }
        for row in rows:
            old = old_rows.get((str(row["candidate_source"]), int(row["candidate_id"])))
            if old is None or float(old["frozen_coexist_score"]) != float(row["frozen_coexist_score"]):
                raise ValueError(f"{scene}: frozen coexist score differs from the established safety60 baseline")
        quality, _ = _quality_predictions(track_package, feature_rows)
        feature_index = {(str(row["candidate_source"]), int(row["candidate_id"])): pos for pos, row in enumerate(feature_rows)}
        relation_lookup = {
            (int(row["track_id"]), str(row["native_exact_geometry_group_id"])): row
            for row in read_jsonl(args.relation_root / scene / "relation_features_no_gt.jsonl")
        }
        for row in union_by_scene[scene]:
            key = (int(row["track_id"]), str(row["native_exact_geometry_group_id"]))
            relation = relation_lookup.get(key)
            if relation is None:
                raise ValueError(f"{scene}: union relation missing from no-GT ledger")
            track = next(item for item in tracks if int(item["track_id"]) == key[0])
            track_points, _ = _load_track_points(track, masks.shape[0])
            members = [int(value) for value in row["native_member_candidate_ids"]]
            native = np.flatnonzero(np.asarray(masks[:, members[0]], dtype=bool)).astype(np.int64)
            expected = proposal_points("pair_union", track_points, native)
            path = Path(row["points_path"])
            with np.load(path) as payload:
                observed = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
            if not np.array_equal(observed, expected) or _geometry_digest(observed) != row["proposal_geometry_sha256"]:
                raise ValueError(f"{scene}: append union geometry differs from frozen relation")
            probability = float(row["threshold_cross_probability"])
            base = min(
                float(quality["q"][feature_index[(TRACK_SOURCE, key[0])]]),
                float(np.median([quality["q"][feature_index[(NATIVE_SOURCE, member)]] for member in members])),
            )
            if float(row["base_quality"]) != base or float(row["new_score"]) != append_score(base, probability):
                raise ValueError(f"{scene}: append union score differs from frozen formula")
        cached_scenes[scene] = {"score_row_count": len(rows), "union_count": len(union_by_scene[scene])}
        preflight_staging.write_text(json.dumps({"version": PLAN_VERSION, "scenes": cached_scenes}, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        built_now += 1
        print(f"[combined preflight] {index}/60 {scene}", flush=True)
        if args.max_scenes_per_invocation and built_now >= args.max_scenes_per_invocation:
            return {
                "preflight_passed": False,
                "preflight_complete": False,
                "validated_scene_count": len(cached_scenes),
                "remaining_scene_count": len(scenes) - len(cached_scenes),
                "ground_truth_read": False,
                "ap_evaluation_run_count": 0,
            }
    if set(cached_scenes) != set(scenes):
        raise AssertionError("preflight checkpoint has unexpected scene coverage")
    score_count = sum(int(cached_scenes[scene]["score_row_count"]) for scene in scenes)
    union_count = sum(int(cached_scenes[scene]["union_count"]) for scene in scenes)
    preflight_staging.unlink(missing_ok=True)
    result = {
        "preflight_passed": True,
        "scene_count": 60,
        "score_row_count": score_count,
        "pair_union_append_candidate_count": union_count,
        "official100_combined_eligibility": _official_eligibility(args.official_oof_summary),
        "frozen_coexist_baseline": old_baseline,
        "combined_model_package_sha256": metadata["model_package_sha256"],
        "plan_summary_sha256": _sha256(plan_summary_path),
        "pair_union_probability_prior_correction": plan[
            "pair_union_probability_prior_correction"
        ],
        "ground_truth_read": False,
        "ap_evaluation_run_count": 0,
    }
    preflight_success.write_text(json.dumps({
        "plan_summary_sha256": result["plan_summary_sha256"], "audit": result,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def run_once(args: argparse.Namespace, audit: dict) -> dict:
    scenes = read_scene_list(args.scene_list)
    score_by_scene = _load_rows_by_scene(args.plan_root / "frozen_score_plan.jsonl", scenes)
    union_by_scene = _load_rows_by_scene(args.plan_root / "pair_union_append_candidates.jsonl", scenes)
    matches = {}
    scene_counts = []
    match_staging = args.output_dir.parent / f".{args.output_dir.name}.ap_match_staging"
    match_staging.mkdir(parents=True, exist_ok=True)
    completed_scenes = {
        path.name[:-len(".matches.pkl")] for path in match_staging.glob("*.matches.pkl")
        if (match_staging / f"{path.name[:-len('.matches.pkl')]}.scene_count.json").is_file()
    }
    built_now = 0
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original_load_ids(path))
    try:
        for index, scene in enumerate(scenes, start=1):
            match_path = match_staging / f"{scene}.matches.pkl"
            count_path = match_staging / f"{scene}.scene_count.json"
            if scene in completed_scenes:
                with match_path.open("rb") as handle:
                    matches.update(pickle.load(handle))
                scene_counts.append(json.loads(count_path.read_text()))
                continue
            masks, native_scores, _classes, feature_rows, tracks, _audit = load_safety_scene(
                scene, args.native_cache, args.quality_ledger_root, args.track_root,
            )
            plan = {(str(row["candidate_source"]), int(row["candidate_id"])): row for row in score_by_scene[scene]}
            groups, group_audit = geometry_groups_and_audit(
                masks, [int(row["point_count"]) for row in feature_rows[:masks.shape[1]]], None,
            )
            native_ids = sorted(min(members, key=lambda candidate_id: (-float(native_scores[candidate_id]), candidate_id)) for members in groups)
            ordered_tracks = sorted(tracks, key=lambda row: int(row["track_id"]))
            track_masks = np.zeros((masks.shape[0], len(ordered_tracks)), dtype=bool)
            track_ids = []
            for column, track in enumerate(ordered_tracks):
                points, _ = _load_track_points(track, masks.shape[0])
                track_masks[points, column] = True
                track_ids.append(int(track["track_id"]))
            unions = sorted(union_by_scene[scene], key=lambda row: int(row["candidate_id"]))
            union_masks = np.zeros((masks.shape[0], len(unions)), dtype=bool)
            for column, row in enumerate(unions):
                with np.load(row["points_path"]) as payload:
                    union_masks[np.asarray(payload["point_indices"], dtype=np.int64), column] = True
            selected_masks = np.concatenate([np.asarray(masks[:, native_ids], dtype=bool), track_masks, union_masks], axis=1)
            selected_scores = np.asarray([
                *(float(plan[(NATIVE_SOURCE, candidate_id)]["planned_score"]) for candidate_id in native_ids),
                *(float(plan[(TRACK_SOURCE, track_id)]["planned_score"]) for track_id in track_ids),
                *(float(row["new_score"]) for row in unions),
            ], dtype=np.float64)
            gt_file = str(args.gt_dir / f"{scene}.txt")
            gt, pred = instance_eval.assign_instances_for_scan(_prediction(selected_masks, selected_scores, len(selected_scores)), gt_file)
            scene_match = {os.path.abspath(gt_file): {"gt": gt, "pred": pred}}
            scene_count = {
                "scene_name": scene,
                "native_geometry_representative_count": len(native_ids),
                "track_candidate_count": len(track_ids),
                "pair_union_append_candidate_count": len(unions),
                "candidate_count": len(selected_scores),
                "canonical_mask_element_sha256": group_audit["canonical_mask_element_sha256"],
                "candidate_files_modified": False,
            }
            # Persist the no-aggregation GT matching result atomically enough
            # for a later foreground invocation to resume.  `_evaluate` below
            # remains the sole AP aggregation call and is receipt-protected.
            with match_path.open("wb") as handle:
                pickle.dump(scene_match, handle, protocol=pickle.HIGHEST_PROTOCOL)
            count_path.write_text(json.dumps(scene_count, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            matches.update(scene_match)
            scene_counts.append(scene_count)
            built_now += 1
            print(f"[combined AP matching] {index}/60 {scene}: unions={len(unions)}", flush=True)
            if args.max_scenes_per_invocation and built_now >= args.max_scenes_per_invocation:
                return {
                    "ap_evaluation_run_count": 0,
                    "ap_matching_complete": False,
                    "matched_scene_count": len(completed_scenes) + built_now,
                    "remaining_scene_count": len(scenes) - len(completed_scenes) - built_now,
                }
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    staging = args.output_dir.parent / f".{args.output_dir.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    receipt = args.output_dir.parent / f".{args.output_dir.name}.ap_once_receipt.json"
    if receipt.exists():
        raise ValueError(f"one-shot AP receipt already exists: {receipt}")
    receipt.write_text(json.dumps({
        "version": AP_VERSION, "policy": POLICY,
        "ap_call_started_utc": datetime.now(timezone.utc).isoformat(),
        "ap_evaluation_run_count": 1, "completed": False,
        "plan_summary_sha256": audit["plan_summary_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    try:
        # This is deliberately the only AP aggregation call in the entrypoint.
        metrics = _evaluate(POLICY, matches, staging)
        baseline = audit["frozen_coexist_baseline"]
        summary = {
            "version": AP_VERSION,
            "policy": POLICY,
            "experiment": "one-shot safety60 class-agnostic AP for frozen official100 combined champion",
            "preflight": audit,
            "frozen_coexist_baseline": baseline,
            "selected_policy_ap": metrics,
            "delta_vs_frozen_coexist": {key: float(metrics[key] - baseline[key]) for key in ("ap", "ap50", "ap25")},
            "scene_candidate_counts": scene_counts,
            "candidate_count_modified": False,
            "candidate_geometry_modified": False,
            "candidate_class_modified": False,
            "candidate_files_modified": False,
            "candidate_file_modification_count": 0,
            "pair_union_append_only": True,
            "pair_union_probability_prior_correction": audit[
                "pair_union_probability_prior_correction"
            ],
            "ground_truth_usage": "one frozen AP evaluation only",
            "threshold_scanning": False,
            "continuous_weight_scanning": False,
            "evaluated_policy_count": 1,
            "ap_evaluation_run_count": 1,
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_dir)
        receipt.write_text(json.dumps({
            "version": AP_VERSION, "policy": POLICY, "ap_evaluation_run_count": 1,
            "completed": True, "output_summary_sha256": _sha256(args.output_dir / "summary.json"),
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(match_staging, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--official-scene-list", type=Path, required=True)
    parser.add_argument("--official-oof-summary", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--relation-root", type=Path, required=True)
    parser.add_argument("--frozen-coexist-plan-root", type=Path, required=True)
    parser.add_argument("--frozen-coexist-ap-summary", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--source-plan-root", type=Path,
                        help="Upgrade an existing frozen v1 no-GT plan by applying prior correction.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    parser.add_argument("--max-scenes-per-invocation", type=int, default=0,
                        help="checkpoint no-GT build/preflight after this many new scenes (0 means no limit)")
    args = parser.parse_args()
    for name in (
        "scene_list", "official_scene_list", "official_oof_summary", "model_root", "native_cache",
        "track_root", "quality_ledger_root", "relation_root", "frozen_coexist_plan_root",
        "frozen_coexist_ap_summary", "gt_dir", "plan_root", "source_plan_root", "output_dir",
    ):
        if getattr(args, name) is None:
            continue
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"output already exists; one-shot AP will not rerun: {args.output_dir}")
    if not args.plan_root.exists():
        build_status = (build_plan_from_v1(args) if args.source_plan_root is not None
                        else build_plan(args))
        if not build_status.get("version"):
            print(json.dumps(build_status, ensure_ascii=False, indent=2, sort_keys=True))
            return
    audit = preflight(args)
    if not audit.get("preflight_complete", True):
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.preflight_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not args.allow_gt_evaluation:
        raise SystemExit("must pass --allow-gt-evaluation after successful no-GT preflight")
    print(json.dumps(run_once(args, audit), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
