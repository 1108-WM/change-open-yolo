"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import json

import pickle

from pathlib import Path

import numpy as np

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE

from tools.build_candidate_pair_union_oof_plan import _geometry_digest, append_score

from tools.fi1_shared.diagnose_candidate_pair_union_threshold_cross_oof import prior_correct_balanced_probability

from tools.fi1_shared.build_frozen_track_harm_score_plan import _component_candidates, _components, _stacked_relation_rows, fixed_track_score

from tools.fi1_shared.build_train_candidate_pair_union_utility_ledger import MIN_REGION_SIZE, _pure_inference_relation_features, proposal_points

from tools.fi1_shared.build_train_candidate_component_action_utility_ledger import _sha256

from tools.fi1_shared.evaluate_candidate_quality_reranking_class_agnostic_ap import _load_track_points

from tools.fi1_shared.build_safety60_structured_action_plan import _quality_predictions

from tools.fi1_shared.build_train_candidate_component_union_feature_ledger import component_track_features

from tools.fi1_shared.train_candidate_component_list_calibration_head_oof import _feature_matrix, build_candidate_feature_rows

from tools.fi1_shared.train_candidate_component_union_list_head_oof import augment_union_features

POLICY = "champion_track_suppression_plus_pair_union_append"

COMBINED_PACKAGE_VERSION = "official100_champion_pair_union_combined_full_v2_prior_corrected"

PAIR_UNION_PACKAGE_VERSION = "official100_pair_union_threshold_cross_full_v2_prior_corrected"

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
