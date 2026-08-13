#!/usr/bin/env python3
"""Build the frozen structured action plan for safety60 without ground truth.

This tool reconstructs positive track--baseline geometry-group relations from
immutable candidate masks, builds the same inference features as official100,
applies the exported full-data model package, and writes one plan for the
single frozen ``structured_lower_relation_veto`` policy.  It never reads GT,
modifies candidate files, or evaluates AP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.build_automatic_sam_track_growth_ledger import _raw_superpoint_context  # noqa: E402
from tools.build_c1_gvc_paper_reference_quality_ledger import (  # noqa: E402
    _frame_contract,
    _load_2d_observations,
)
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    _sha256,
    build_component_actions,
)
from tools.build_train_candidate_relation_feature_ledger import (  # noqa: E402
    ADJACENCY_KNN,
    ADJACENCY_MAX_DISTANCE,
    MIN_CONTACT_POINTS,
    MIN_CONTACT_RATIO,
    _candidate_view_rows_with_box,
    build_inference_relation_rows,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    FEATURE_GROUP,
    PROTOCOL_NAME,
    load_safety_scene,
)
from tools.train_candidate_component_action_head_structured_oof import (  # noqa: E402
    RELATIVE_FEATURES,
    STACKED_RELATION_FIELDS,
    TARGET_FEATURES,
    UTILITY_SCALE,
    choose_structured_action,
    structured_action_feature_row,
)
from tools.train_candidate_quality_head_oof import canonicalize_predictions, feature_matrix  # noqa: E402


POLICY = "structured_lower_relation_veto"
VERSION = "safety60_structured_action_plan_v2"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _combined_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda value: str(value)):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _track_points(track: dict, point_count: int) -> tuple[np.ndarray, Path]:
    path = Path(track["points_path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen track points: {path}")
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points) or points[0] < 0 or points[-1] >= point_count:
        raise ValueError(f"invalid track points: {path}")
    if int(track.get("point_count", len(points))) != len(points):
        raise ValueError(f"track point count mismatch: {path}")
    return points, path


def _native_groups(scene: str, masks: np.ndarray, scores: np.ndarray) -> tuple[list[dict], dict[int, str]]:
    by_digest: dict[str, list[int]] = defaultdict(list)
    for candidate_id in range(masks.shape[1]):
        packed = np.packbits(np.asarray(masks[:, candidate_id], dtype=np.uint8)).tobytes()
        by_digest[hashlib.sha256(packed).hexdigest()].append(candidate_id)
    groups = []
    member_to_group = {}
    for index, members in enumerate(sorted(by_digest.values(), key=lambda values: values[0])):
        group_id = f"{scene}:native_geometry:{index:04d}"
        groups.append({
            "native_exact_geometry_group_id": group_id,
            "native_member_candidate_ids": members,
            "native_exact_geometry_group_size": len(members),
            "native_anchor_candidate_id": members[0],
            "native_original_score_median": float(np.median(scores[members])),
        })
        for member in members:
            member_to_group[member] = group_id
    if len(member_to_group) != masks.shape[1]:
        raise AssertionError(f"{scene}: exact geometry grouping lost candidates")
    return groups, member_to_group


def _positive_pairs(
    scene: str, masks: np.ndarray, scores: np.ndarray, tracks: list[dict],
) -> tuple[list[dict], dict[int, np.ndarray], dict[str, np.ndarray], list[Path]]:
    groups, _ = _native_groups(scene, masks, scores)
    group_by_id = {row["native_exact_geometry_group_id"]: row for row in groups}
    anchors = np.asarray([row["native_anchor_candidate_id"] for row in groups], dtype=np.int64)
    native_sizes = np.count_nonzero(masks[:, anchors], axis=0).astype(np.int64)
    native_points = {
        row["native_exact_geometry_group_id"]: np.flatnonzero(
            np.asarray(masks[:, row["native_anchor_candidate_id"]], dtype=bool)
        ).astype(np.int64)
        for row in groups
    }
    track_points = {}
    point_paths = []
    pairs = []
    for track in sorted(tracks, key=lambda row: int(row["track_id"])):
        track_id = int(track["track_id"])
        points, path = _track_points(track, masks.shape[0])
        point_paths.append(path)
        track_points[track_id] = points
        intersections = np.count_nonzero(masks[points][:, anchors], axis=0).astype(np.int64)
        for group_index in np.flatnonzero(intersections > 0):
            group_id = groups[int(group_index)]["native_exact_geometry_group_id"]
            group = group_by_id[group_id]
            intersection = int(intersections[group_index])
            native_size = int(native_sizes[group_index])
            union = len(points) + native_size - intersection
            track_coverage = float(intersection / len(points))
            native_coverage = float(intersection / max(1, native_size))
            pairs.append({
                "scene_name": scene,
                "track_id": track_id,
                "native_exact_geometry_group_id": group_id,
                "native_member_candidate_ids": group["native_member_candidate_ids"],
                "native_exact_geometry_group_size": group["native_exact_geometry_group_size"],
                "track_original_score": float(track["mean_node_quality"]),
                "native_original_score_median": group["native_original_score_median"],
                "point_iou": float(intersection / max(1, union)),
                "track_inside_native_ratio": track_coverage,
                "native_inside_track_ratio": native_coverage,
                "mutual_duplicate_strict_099": bool(
                    track_coverage > 0.99 and native_coverage > 0.99
                ),
            })
    involved_groups = {str(row["native_exact_geometry_group_id"]) for row in pairs}
    involved_tracks = {int(row["track_id"]) for row in pairs}
    return (
        pairs,
        {key: value for key, value in track_points.items() if key in involved_tracks},
        {key: value for key, value in native_points.items() if key in involved_groups},
        point_paths,
    )


def _quality_predictions(package: dict, feature_rows: list[dict]) -> tuple[dict, dict]:
    matrix, names = feature_matrix(feature_rows, FEATURE_GROUP, PROTOCOL_NAME)
    expected = package["metadata"]["feature_contract"]["candidate_quality_feature_names"]
    if names != expected:
        raise ValueError("safety60 candidate quality feature schema differs from frozen model")
    predictions = {}
    for target, model in package["quality_models"].items():
        raw = model.predict_proba(matrix)[:, 1] if hasattr(model, "predict_proba") else model.predict(matrix)
        predictions[target] = canonicalize_predictions(target, raw)
    return predictions, {"feature_names": names, "candidate_count": len(feature_rows)}


def _relation_predictions(package: dict, rows: list[dict]) -> list[dict]:
    target_names = package["metadata"]["feature_contract"]["target_relation_feature_names"]
    relative_names = package["metadata"]["feature_contract"]["relative_relation_feature_names"]
    if target_names != list(TARGET_FEATURES) or relative_names != list(RELATIVE_FEATURES):
        raise ValueError("frozen relation schema differs from code contract")
    target_matrix = np.asarray([
        [float(row["features"][name]) for name in target_names] for row in rows
    ], dtype=np.float64)
    relative_matrix = np.asarray([
        [float(row["features"][name]) for name in relative_names] for row in rows
    ], dtype=np.float64)
    same = package["target_relation_model"].predict_proba(target_matrix)[:, 1]
    better = package["relative_relation_model"].predict_proba(relative_matrix)[:, 1]
    return [{
        "same_target_score": float(same[index]),
        "different_target_score": float(1.0 - same[index]),
        "track_better_score": float(better[index]),
        "baseline_better_score": float(1.0 - better[index]),
        "track_win_relation_score": float(same[index] * better[index]),
        "baseline_win_relation_score": float(same[index] * (1.0 - better[index])),
        "coexist_relation_score": float(1.0 - same[index]),
    } for index in range(len(rows))]


def _action_prediction(models: dict, matrix: np.ndarray) -> dict:
    probabilities = models["state"].predict_proba(matrix)[0]
    by_state = {int(state): float(probabilities[index]) for index, state in enumerate(models["state"].classes_)}
    probability = {
        "harmful": by_state.get(0, 0.0),
        "neutral": by_state.get(1, 0.0),
        "positive": by_state.get(2, 0.0),
    }
    gain_mean = max(0.0, float(models["gain_mean"].predict(matrix)[0])) / UTILITY_SCALE
    gain_lower = max(0.0, float(models["gain_lower25"].predict(matrix)[0])) / UTILITY_SCALE
    cost_mean = max(0.0, float(models["cost_mean"].predict(matrix)[0])) / UTILITY_SCALE
    cost_upper = max(0.0, float(models["cost_upper75"].predict(matrix)[0])) / UTILITY_SCALE
    return {
        "state_probability": probability,
        "predicted_positive_gain_mean": gain_mean,
        "predicted_positive_gain_lower25": gain_lower,
        "predicted_harm_cost_mean": cost_mean,
        "predicted_harm_cost_upper75": cost_upper,
        "predicted_mean_utility": probability["positive"] * gain_mean - probability["harmful"] * cost_mean,
        "predicted_lower_utility": probability["positive"] * gain_lower - probability["harmful"] * cost_upper,
    }


def _scene(scene: str, args, package: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    from utils import WORLD_2_CAM

    masks, scores, _classes, feature_rows, tracks, candidate_audit = load_safety_scene(
        scene, args.native_cache, args.quality_ledger_root, args.track_root,
    )
    quality, quality_audit = _quality_predictions(package, feature_rows)
    pairs, track_points, native_points, point_paths = _positive_pairs(scene, masks, scores, tracks)
    scene_stem = scene[5:] if scene.startswith("scene") else scene
    processed_path = args.dataset_root / scene / f"{scene_stem}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10 or len(processed) != masks.shape[0]:
        raise ValueError(f"{scene}: processed scene contract differs from candidates")
    context = _raw_superpoint_context(
        processed, ADJACENCY_KNN, ADJACENCY_MAX_DISTANCE,
        MIN_CONTACT_POINTS, MIN_CONTACT_RATIO,
    )
    automatic_scene = args.automatic_root / scene
    frame_indices, frame_id_to_index = _frame_contract(automatic_scene)
    observations = _load_2d_observations(args.yoloworld_sam_root / scene)
    world = WORLD_2_CAM(str(args.dataset_root / scene), args.depth_scale, args.config)
    projections_t, visibility_t = world.get_mesh_projections()
    projections = projections_t.detach().cpu().numpy().astype(np.float64)
    visibility = visibility_t.detach().cpu().numpy().astype(bool)
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    camera_centers = {
        frame: np.asarray(np.loadtxt(world.poses[frame]), dtype=np.float64)[:3, 3]
        for frame in frame_indices
    }
    track_by_id = {int(row["track_id"]): row for row in tracks}
    track_views = {}
    for track_id, points in track_points.items():
        raw_frames = {str(value) for value in track_by_id[track_id].get("frame_ids", [])}
        source_frames = {frame_id_to_index[value] for value in raw_frames if value in frame_id_to_index}
        all_rows = _candidate_view_rows_with_box(
            points, frame_indices, projections, visibility, scaling, observations,
        )
        track_views[track_id] = {
            frame: row for frame, row in all_rows.items() if frame not in source_frames
        }
    native_views = {
        group_id: _candidate_view_rows_with_box(
            points, frame_indices, projections, visibility, scaling, observations,
        )
        for group_id, points in native_points.items()
    }
    relation_rows, components, relation_summary = build_inference_relation_rows(
        scene, pairs, processed, context, track_points, native_points,
        track_views, native_views, camera_centers,
    )

    native_count = masks.shape[1]
    track_order = [int(row["track_id"]) for row in sorted(tracks, key=lambda value: int(value["track_id"]))]
    track_quality_index = {track_id: native_count + index for index, track_id in enumerate(track_order)}
    relation_prediction = _relation_predictions(package, relation_rows)
    stacked_rows = []
    for row, relation_scores in zip(relation_rows, relation_prediction):
        members = list(map(int, row["native_member_candidate_ids"]))
        track_index = track_quality_index[int(row["track_id"])]
        evidence = {}
        for target in ("q", "valid25", "valid50"):
            track_value = float(quality[target][track_index])
            native_value = float(np.median(quality[target][members]))
            evidence[f"nested_track_{target}"] = track_value
            evidence[f"nested_native_{target}_median"] = native_value
            evidence[f"nested_{target}_delta_track_minus_native"] = track_value - native_value
        stacked = {**evidence, **relation_scores}
        if set(stacked) != set(STACKED_RELATION_FIELDS):
            raise AssertionError("stacked relation inference fields differ from frozen contract")
        stacked_rows.append(stacked)

    relations_by_component: dict[int, list[dict]] = defaultdict(list)
    stacked_by_component: dict[int, list[dict]] = defaultdict(list)
    for raw, stacked in zip(relation_rows, stacked_rows):
        component_id = int(raw["relation_component_id"])
        relations_by_component[component_id].append(raw)
        stacked_by_component[component_id].append(stacked)
    predictions = {}
    model_rows = {}
    all_actions = {}
    expected_action_names = package["metadata"]["feature_contract"]["action_feature_names"]
    for component in components:
        component_id = int(component["relation_component_id"])
        actions = [{"scene_name": scene, **row} for row in build_component_actions(component)]
        all_actions[(scene, component_id)] = actions
        for action in actions:
            if action["action_kind"] == "coexist":
                continue
            key = (scene, component_id, str(action["action_name"]))
            features = structured_action_feature_row(
                action, relations_by_component[component_id], stacked_by_component[component_id],
            )
            names = sorted(features)
            if names != expected_action_names:
                missing = sorted(set(expected_action_names) - set(names))
                extra = sorted(set(names) - set(expected_action_names))
                raise ValueError(f"{scene}: action feature schema mismatch; missing={missing}, extra={extra}")
            matrix = np.asarray([[float(features[name]) for name in names]], dtype=np.float64)
            predictions[key] = _action_prediction(
                package["action_models"][str(action["action_kind"])], matrix,
            )
            model_rows[key] = {"model_features": features}
    selected = {
        key: choose_structured_action(actions, predictions, model_rows, POLICY)
        for key, actions in all_actions.items()
    }
    plan_rows = []
    for (scene_name, component_id), action in sorted(selected.items()):
        plan_rows.append({
            "policy": POLICY,
            "scene_name": scene_name,
            "relation_component_id": component_id,
            "component_native_exact_geometry_group_ids": action["component_native_exact_geometry_group_ids"],
            "component_track_ids": action["component_track_ids"],
            "selected_action_name": action["action_name"],
            "selected_action_kind": action["action_kind"],
            "selected_track_id": action.get("selected_track_id"),
            "kept_native_exact_geometry_group_ids": action["kept_native_exact_geometry_group_ids"],
            "kept_track_ids": action["kept_track_ids"],
            "ground_truth_usage": "none",
            "candidate_materialized": False,
            "ap_evaluation_run": False,
        })
    prediction_rows = [{
        "scene_name": key[0],
        "relation_component_id": key[1],
        "action_name": key[2],
        **value,
    } for key, value in sorted(predictions.items())]
    input_paths = [
        args.native_cache / f"{scene}_pred_masks.npy",
        args.native_cache / f"{scene}_pred_scores.npy",
        args.native_cache / f"{scene}_pred_classes.npy",
        args.track_root / scene / "automatic_tracks.json",
        args.quality_ledger_root / scene / "c1_gvc_quality_ledger.json",
        processed_path,
        *point_paths,
    ]
    summary = {
        **relation_summary,
        "native_candidate_count": int(masks.shape[1]),
        "track_candidate_count": len(tracks),
        "quality_prediction_audit": quality_audit,
        "planned_component_count": len(plan_rows),
        "planned_action_kind_counts": dict(sorted(Counter(
            row["selected_action_kind"] for row in plan_rows
        ).items())),
        "candidate_input_combined_sha256": _combined_digest(input_paths),
        "candidate_audit": candidate_audit,
        "ground_truth_usage": "none",
        "candidate_files_modified": False,
        "ap_evaluation_run": False,
    }
    del world, projections_t, visibility_t, projections, visibility, masks, processed
    return relation_rows, prediction_rows, plan_rows, summary


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 60:
        raise ValueError("frozen safety protocol requires exactly 60 scenes")
    official_scenes = set(read_scene_list(args.official_scene_list))
    overlap = official_scenes & set(scenes)
    if overlap:
        raise ValueError(f"official100 and safety60 overlap: {sorted(overlap)}")
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
    all_plan_rows = []
    scene_summaries = []
    try:
        for index, scene in enumerate(scenes, start=1):
            relation_rows, prediction_rows, plan_rows, summary = _scene(scene, args, package)
            scene_root = staging / scene
            scene_root.mkdir()
            _write_jsonl(scene_root / "relation_features_no_gt.jsonl", relation_rows)
            _write_jsonl(scene_root / "action_predictions_no_gt.jsonl", prediction_rows)
            _write_jsonl(scene_root / "frozen_action_plan.jsonl", plan_rows)
            (scene_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            all_plan_rows.extend(plan_rows)
            scene_summaries.append(summary)
            print(f"[safety60 no-GT plan] {index}/60 {scene}", flush=True)
        _write_jsonl(staging / "frozen_action_plan.jsonl", all_plan_rows)
        output = {
            "version": VERSION,
            "policy": POLICY,
            "scene_count": len(scenes),
            "relation_count": sum(row["relation_count"] for row in scene_summaries),
            "relation_component_count": sum(row["relation_component_count"] for row in scene_summaries),
            "planned_action_kind_counts": dict(sorted(Counter(
                row["selected_action_kind"] for row in all_plan_rows
            ).items())),
            "official100_safety60_scene_overlap_count": 0,
            "feature_schema_verified": True,
            "plan_complete": len(all_plan_rows) == sum(
                row["relation_component_count"] for row in scene_summaries
            ),
            "ground_truth_usage": "none",
            "candidate_files_modified": False,
            "candidate_file_modification_count": 0,
            "ap_evaluation_run": False,
            "ap_evaluation_run_count": 0,
            "threshold_scanning": False,
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "official_scene_list_sha256": _sha256(args.official_scene_list),
                "model_package_sha256": metadata["model_package_sha256"],
                "scene_candidate_input_sha256": {
                    row["scene_name"]: row["candidate_input_combined_sha256"]
                    for row in scene_summaries
                },
            },
        }
        if not output["plan_complete"]:
            raise AssertionError("frozen plan does not cover every relation component")
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
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--yoloworld-sam-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "official_scene_list", "model_root", "native_cache", "track_root",
        "quality_ledger_root", "automatic_root", "yoloworld_sam_root", "dataset_root",
        "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if any("ground_truth" in str(getattr(args, name)).lower() for name in (
        "native_cache", "track_root", "quality_ledger_root", "automatic_root",
        "yoloworld_sam_root", "dataset_root",
    )):
        raise SystemExit("S2 refuses any input path containing ground_truth")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root exists and is non-empty: {args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
