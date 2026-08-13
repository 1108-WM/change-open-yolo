#!/usr/bin/env python3
"""Audit the frozen official100 semantic-guided geometry action space.

Z5a is deliberately read-only: it joins already materialized pair-union
geometry to its frozen native/track parents and to the Z1/Z2c/Z3 semantic
ledgers.  It never reads ground truth, computes AP, trains a model, selects a
threshold, or creates/modifies a candidate geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_train_candidate_component_union_feature_ledger import _track_points  # noqa: E402


VERSION = "z5a_semantic_geometry_action_space_official100_v1"
EXPECTED_SPLIT_SHA256 = "aa657449965bc76164a1a1b77c7785aa705a0295eeed1307163b325f7233fe3e"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise ValueError("Z5a requires exactly 100 unique official-train scenes")
    return scenes


def _points(path: Path, point_count: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points) or points[0] < 0 or (point_count is not None and points[-1] >= point_count):
        raise ValueError(f"invalid point indices: {path}")
    return points


def _geometry_sha1(points: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(points, dtype=np.int64).tobytes()).hexdigest()


def _geometry_sha256(points: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(len(points).to_bytes(8, "little"))
    digest.update(np.ascontiguousarray(points, dtype=np.int64).tobytes())
    return digest.hexdigest()


def pair_geometry_contract(track: np.ndarray, native: np.ndarray, child: np.ndarray) -> dict:
    intersection = np.intersect1d(track, native, assume_unique=True)
    expected_union = np.union1d(track, native)
    track_only = len(track) - len(intersection)
    native_only = len(native) - len(intersection)
    exact_union = bool(np.array_equal(expected_union, child))
    if len(intersection) == len(track) == len(native):
        overlap_type = "equal"
    elif len(intersection) == len(track):
        overlap_type = "track_subset_of_native"
    elif len(intersection) == len(native):
        overlap_type = "native_subset_of_track"
    elif len(intersection):
        overlap_type = "partial_overlap"
    else:
        overlap_type = "disjoint"
    return {
        "track_point_count": int(len(track)),
        "native_point_count": int(len(native)),
        "child_point_count": int(len(child)),
        "shared_point_count": int(len(intersection)),
        "track_exclusive_point_count": int(track_only),
        "native_exclusive_point_count": int(native_only),
        "expected_union_point_count": int(len(expected_union)),
        "point_iou": float(len(intersection) / max(1, len(expected_union))),
        "track_inside_native_ratio": float(len(intersection) / max(1, len(track))),
        "native_inside_track_ratio": float(len(intersection) / max(1, len(native))),
        "overlap_type": overlap_type,
        "point_membership_contact": bool(len(intersection) > 0),
        "child_exactly_equals_parent_union": exact_union,
        "child_contains_track": bool(np.intersect1d(child, track, assume_unique=True).size == len(track)),
        "child_contains_native": bool(np.intersect1d(child, native, assume_unique=True).size == len(native)),
        "contact_contract": "shared point membership only; no new distance or adjacency threshold",
    }


def _top_stats(distribution: np.ndarray, available: bool) -> dict:
    if not available or not np.isfinite(distribution).all() or float(distribution.sum()) <= 0:
        return {
            "available": False, "top1_class_index": None, "top1_probability": None,
            "top2_class_index": None, "top2_probability": None, "top1_top2_margin": None,
            "normalized_entropy": None,
        }
    values = np.asarray(distribution, dtype=np.float64)
    values = values / values.sum()
    order = np.argsort(-values, kind="stable")
    positive = values[values > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(len(values))
    return {
        "available": True,
        "top1_class_index": int(order[0]), "top1_probability": float(values[order[0]]),
        "top2_class_index": int(order[1]), "top2_probability": float(values[order[1]]),
        "top1_top2_margin": float(values[order[0]] - values[order[1]]),
        "normalized_entropy": float(entropy),
    }


def semantic_summary(node: dict, arrays: dict[str, np.ndarray]) -> dict:
    index = int(node["node_index"])
    result = {
        "semantic_evidence_node_key": str(node["semantic_evidence_node_key"]),
        "candidate_source": str(node["candidate_source"]),
        "geometry_hash": str(node["geometry_hash"]),
        "point_count": int(node["point_count"]),
        "geometry_yolo": _top_stats(arrays["geometry_yolo"][index], bool(node["geometry_yolo_available"])),
        "inherited_yolo": _top_stats(arrays["inherited_yolo"][index], bool(node["inherited_yolo_available"])),
        "geometry_alpha": _top_stats(arrays["geometry_alpha"][index], bool(node["geometry_alpha_available"])),
        "inherited_alpha": _top_stats(arrays["inherited_alpha"][index], bool(node["inherited_alpha_available"])),
        "geometry_yolo_alpha_js": node.get("geometry_yolo_alpha_js"),
        "geometry_yolo_alpha_top1_agreement": node.get("geometry_yolo_alpha_top1_agreement"),
        "inherited_yolo_alpha_js": node.get("inherited_yolo_alpha_js"),
        "inherited_yolo_alpha_top1_agreement": node.get("inherited_yolo_alpha_top1_agreement"),
        "geometry_inherited_alpha_js": node.get("geometry_inherited_alpha_js"),
        "geometry_alpha_view_count": int(node.get("geometry_alpha_view_count", 0)),
        "inherited_alpha_view_count": int(node.get("inherited_alpha_view_count", 0)),
    }
    return result


def frozen_oof_summary(rows: list[dict], source: str) -> dict:
    safe = []
    for row in rows:
        predictions = row.get("oof_predictions", {})
        if "C_joint_yolo_alpha" not in predictions:
            raise ValueError("missing frozen C_joint_yolo_alpha OOF prediction")
        model_score = float(predictions["C_joint_yolo_alpha"])
        original_score = float(row["original_score"])
        safe.append({
            "candidate_id": int(row["candidate_id"]),
            "class_index": int(row["class_index"]),
            "model_oof_score": model_score,
            "original_score": original_score,
            "current_hybrid_score": original_score if source == "pair_union" else model_score,
        })
    if not safe:
        return {
            "available": False, "hypothesis_count": 0, "current_hybrid_score_source": None,
            "top_class_index": None, "top_score": None,
        }
    ordered = sorted(safe, key=lambda row: (-row["current_hybrid_score"], row["class_index"], row["candidate_id"]))
    return {
        "available": True,
        "hypothesis_count": len(safe),
        "current_hybrid_score_source": (
            "frozen_original_score" if source == "pair_union" else "scene_isolated_C_joint_yolo_alpha_OOF"
        ),
        "top_class_index": int(ordered[0]["class_index"]),
        "top_score": float(ordered[0]["current_hybrid_score"]),
        "score_min": float(min(row["current_hybrid_score"] for row in safe)),
        "score_max": float(max(row["current_hybrid_score"] for row in safe)),
    }


def _selected_frames(evidence: dict) -> set[str]:
    return {
        str(row["frame_id"])
        for row in evidence["distribution"].get("views", [])
        if bool(row.get("supported"))
    }


def common_visibility_summary(child: dict, track: dict, native: dict) -> dict:
    child_frames = _selected_frames(child)
    track_frames = _selected_frames(track)
    native_frames = _selected_frames(native)
    all_three = child_frames & track_frames & native_frames
    return {
        "child_selected_evidence_view_count": len(child_frames),
        "track_selected_evidence_view_count": len(track_frames),
        "native_selected_evidence_view_count": len(native_frames),
        "child_track_common_view_count": len(child_frames & track_frames),
        "child_native_common_view_count": len(child_frames & native_frames),
        "track_native_common_view_count": len(track_frames & native_frames),
        "all_three_common_view_count": len(all_three),
        "all_three_common_frame_ids": sorted(all_three, key=lambda value: int(value)),
        "child_track_support_frame_count": len(
            set(map(str, child.get("track_support_frame_ids", [])))
            & set(map(str, track.get("track_support_frame_ids", [])))
        ),
        "contract": "intersection of already selected Z1 supported views; no view threshold selected",
    }


def _audit_unmaterialized_family(
    family: str, roots: list[Path], required_tokens: tuple[str, ...], legacy_tokens: tuple[str, ...]
) -> dict:
    named_paths = []
    compatible = []
    legacy = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                display_path = path.relative_to(PROJECT_ROOT)
            except ValueError:
                display_path = path
            lower = str(display_path).lower()
            if not any(token in lower for token in required_tokens):
                continue
            named_paths.append(str(path))
            if "official100" in lower and not any(token in lower for token in ("oracle_gt", "safety60", "even48", "test60")):
                compatible.append(str(path))
            if any(token in lower for token in legacy_tokens):
                legacy.append(str(path))
    # A compatible name alone is not enough: Z5a requires an explicit action
    # ledger plus materialized child geometry.  No such contract currently
    # exists for split or boundary-owner.
    return {
        "action_type": family,
        "status": "unavailable",
        "action_count": 0,
        "searched_roots": [str(root) for root in roots],
        "name_match_count": len(named_paths),
        "official100_nonlegacy_name_match_count": len(compatible),
        "legacy_name_match_count": len(legacy),
        "reason": (
            "no official100 action ledger with traceable parent identity and already materialized child point sets"
            if family == "split" else
            "no official100 superpoint owner-action ledger with traceable parent/child geometry; legacy safety60 MV3DIS assets are excluded"
        ),
        "candidate_generation_applied": False,
        "threshold_scanning_applied": False,
    }


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    scene_set = set(scenes)
    split_sha = _sha256(args.split_manifest)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"frozen split manifest SHA-256 mismatch: {split_sha}")

    z1_summary = json.loads((args.z1_root / "summary.json").read_text())
    z2c_summary = json.loads((args.z2c_root / "summary.json").read_text())
    z3_summary = json.loads((args.z3_oof_root / "summary.json").read_text())
    plan_summary = json.loads((args.combined_plan_root / "summary.json").read_text())
    if z1_summary.get("ground_truth_usage") != "none" or z1_summary.get("candidate_mutation") is not False:
        raise ValueError("Z1 input is not a frozen no-GT ledger")
    if z2c_summary.get("ground_truth_usage") != "none" or z2c_summary.get("candidate_mutation") is not False:
        raise ValueError("Z2c input is not a frozen no-GT ledger")
    if int(z2c_summary.get("node_count", -1)) != 9708:
        raise ValueError("unexpected Z2c node count")
    if z3_summary.get("split_manifest", {}).get("sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("Z3 OOF split contract mismatch")
    if int(plan_summary.get("pair_union_append_candidate_count", -1)) != 1501:
        raise ValueError("unexpected combined pair-union count")
    if plan_summary.get("ground_truth_usage_for_plan_generation", "").startswith("none") is False:
        raise ValueError("pair-union plan was not generated under a no-GT contract")

    bindings = _read_jsonl(args.z1_root / "candidate_bindings.jsonl")
    evidence_rows = _read_jsonl(args.z1_root / "semantic_evidence_nodes.jsonl")
    nodes = _read_jsonl(args.z2c_root / "nodes.jsonl")
    plan_rows = _read_jsonl(args.combined_plan_root / "pair_union_append_candidates.jsonl")
    oof_rows = _read_jsonl(args.z3_oof_root / "oof_predictions.jsonl")
    arrays_payload = np.load(args.z2c_root / "semantic_distributions.npz")
    arrays = {name: arrays_payload[name] for name in arrays_payload.files}

    binding_by_candidate: dict[tuple[str, str, int], dict] = {}
    for row in bindings:
        scene = str(row["scene_name"])
        if scene not in scene_set:
            continue
        key = (scene, str(row["candidate_source"]), int(row["candidate_id"]))
        if key in binding_by_candidate:
            raise ValueError(f"duplicate Z1 candidate binding: {key}")
        binding_by_candidate[key] = row
    evidence_by_key = {str(row["semantic_evidence_node_key"]): row for row in evidence_rows}
    node_by_key = {str(row["semantic_evidence_node_key"]): row for row in nodes}
    if len(evidence_by_key) != len(evidence_rows) or len(node_by_key) != len(nodes):
        raise ValueError("duplicate semantic evidence node key")
    oof_by_key: dict[str, list[dict]] = defaultdict(list)
    for row in oof_rows:
        # Deliberately project to inference-safe fields later; label_* fields
        # that coexist in this frozen training artifact are never emitted or
        # used for an action decision.
        oof_by_key[str(row["semantic_evidence_node_key"])].append(row)

    by_scene_plan: dict[str, list[dict]] = defaultdict(list)
    for row in plan_rows:
        by_scene_plan[str(row["scene_name"])].append(row)
    if set(by_scene_plan) - scene_set or len(plan_rows) != 1501:
        raise ValueError("pair-union plan scene/count mismatch")

    actions = []
    error_counts = Counter()
    source_counts = Counter()
    semantic_join_counts = Counter()
    geometry_relation_counts = Counter()
    for scene in scenes:
        cache_root = args.records_root / scene / "native_cache"
        masks = np.load(cache_root / f"{scene}_pred_masks.npy", mmap_mode="r")
        track_payload = json.loads((
            args.records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
        ).read_text())
        track_by_id = {int(row["track_id"]): row for row in track_payload["tracks"]}
        for plan in sorted(by_scene_plan.get(scene, []), key=lambda row: int(row["candidate_id"])):
            action_id = f"{scene}:merge_pair_union:{int(plan['candidate_id']):04d}"
            track_id = int(plan["selected_track_id"])
            member_ids = [int(value) for value in plan["selected_native_member_candidate_ids"]]
            representative_id = member_ids[0]
            child_binding = binding_by_candidate.get((scene, "pair_union", int(plan["candidate_id"])))
            track_binding = binding_by_candidate.get((scene, "track", track_id))
            native_bindings = [binding_by_candidate.get((scene, "native", member)) for member in member_ids]
            if child_binding is None or track_binding is None or any(row is None for row in native_bindings):
                error_counts["missing_candidate_binding"] += 1
                continue
            native_keys = {str(row["semantic_evidence_node_key"]) for row in native_bindings if row is not None}
            if len(native_keys) != 1:
                error_counts["native_exact_group_multiple_semantic_nodes"] += 1
                continue
            child_key = str(child_binding["semantic_evidence_node_key"])
            track_key = str(track_binding["semantic_evidence_node_key"])
            native_key = next(iter(native_keys))
            if not all(key in node_by_key and key in evidence_by_key for key in (child_key, track_key, native_key)):
                error_counts["missing_semantic_node"] += 1
                continue

            native = np.flatnonzero(np.asarray(masks[:, representative_id], dtype=bool)).astype(np.int64)
            if any(not np.array_equal(
                np.asarray(masks[:, representative_id], dtype=bool),
                np.asarray(masks[:, member], dtype=bool),
            ) for member in member_ids[1:]):
                error_counts["invalid_native_exact_geometry_group"] += 1
                continue
            track = _track_points(track_by_id[track_id], masks.shape[0])
            child = _points(Path(plan["points_path"]), masks.shape[0])
            geometry = pair_geometry_contract(track, native, child)
            if not geometry["child_exactly_equals_parent_union"]:
                error_counts["child_not_exact_parent_union"] += 1
                continue
            if _geometry_sha256(child) != str(plan["geometry_sha256"]):
                error_counts["pair_union_sha256_mismatch"] += 1
                continue
            expected_hashes = {
                "child": (child_key, child), "track": (track_key, track), "native": (native_key, native),
            }
            hash_mismatch = False
            for role, (key, points) in expected_hashes.items():
                if _geometry_sha1(points) != str(node_by_key[key]["geometry_hash"]):
                    error_counts[f"{role}_semantic_geometry_hash_mismatch"] += 1
                    hash_mismatch = True
            if hash_mismatch:
                continue

            child_semantic = semantic_summary(node_by_key[child_key], arrays)
            track_semantic = semantic_summary(node_by_key[track_key], arrays)
            native_semantic = semantic_summary(node_by_key[native_key], arrays)
            child_semantic["frozen_oof"] = frozen_oof_summary(oof_by_key[child_key], "pair_union")
            track_semantic["frozen_oof"] = frozen_oof_summary(oof_by_key[track_key], "track")
            native_semantic["frozen_oof"] = frozen_oof_summary(oof_by_key[native_key], "native")
            semantic_available = all(
                semantic["geometry_yolo"]["available"]
                and semantic["inherited_yolo"]["available"]
                and semantic["frozen_oof"]["available"]
                for semantic in (child_semantic, track_semantic, native_semantic)
            )
            missing_reasons = []
            for role, semantic in (
                ("child", child_semantic), ("track", track_semantic), ("native", native_semantic)
            ):
                if not semantic["geometry_yolo"]["available"]:
                    missing_reasons.append(f"{role}:geometry_yolo")
                if not semantic["geometry_alpha"]["available"]:
                    missing_reasons.append(f"{role}:geometry_alpha")
                if not semantic["frozen_oof"]["available"]:
                    missing_reasons.append(f"{role}:frozen_oof")
            action = {
                "action_id": action_id,
                "action_type": "merge",
                "action_subtype": "frozen_pair_union_append",
                "scene_name": scene,
                "source": "output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1",
                "child": {
                    "candidate_source": "pair_union", "candidate_id": int(plan["candidate_id"]),
                    "geometry_key": child_key, "geometry_sha1": _geometry_sha1(child),
                    "geometry_sha256": str(plan["geometry_sha256"]), "point_count": len(child),
                    "points_path": str(plan["points_path"]),
                },
                "parents": {
                    "track": {"candidate_id": track_id, "geometry_key": track_key, "point_count": len(track)},
                    "native_exact_group": {
                        "group_id": str(plan["selected_native_exact_geometry_group_id"]),
                        "member_candidate_ids": member_ids, "representative_candidate_id": representative_id,
                        "geometry_key": native_key, "point_count": len(native),
                    },
                },
                "geometry_relation": geometry,
                "frozen_plan_evidence": {
                    "support_relation_count": int(plan["support_relation_count"]),
                    "threshold_cross_probability": float(plan["threshold_cross_probability"]),
                    "base_quality": float(plan["base_quality"]),
                    "append_score": float(plan["new_score"]),
                    "candidate_retained": bool(plan["candidate_retained"]),
                },
                "common_visibility": common_visibility_summary(
                    evidence_by_key[child_key], evidence_by_key[track_key], evidence_by_key[native_key]
                ),
                "semantics": {
                    "child": child_semantic, "track_parent": track_semantic, "native_parent": native_semantic,
                },
                "semantic_available": semantic_available,
                "semantic_missing_reasons": missing_reasons,
                "ground_truth_usage": "none",
                "candidate_mutation": False,
                "action_selected_or_applied": False,
            }
            actions.append(action)
            source_counts["frozen_pair_union_append"] += 1
            geometry_relation_counts[geometry["overlap_type"]] += 1
            semantic_join_counts["complete"] += int(semantic_available)
            semantic_join_counts["incomplete"] += int(not semantic_available)

    arrays_payload.close()
    split_audit = _audit_unmaterialized_family(
        "split", args.asset_search_roots, ("split",), ("oracle_gt", "safety60")
    )
    owner_audit = _audit_unmaterialized_family(
        "boundary_owner", args.asset_search_roots,
        ("boundary_owner", "boundary-owner", "boundary_assignment", "owner"),
        ("mv3dis", "safety60", "even48"),
    )
    family_rows = [
        {
            "action_type": "merge", "status": "available", "action_count": len(actions),
            "source": "frozen pair-union append plan", "candidate_generation_applied": False,
            "threshold_scanning_applied": False,
        },
        split_audit,
        owner_audit,
    ]

    if error_counts or len(actions) != 1501:
        raise RuntimeError(
            f"Z5a merge contract failed: actions={len(actions)} errors={dict(error_counts)}"
        )
    action_scenes = {row["scene_name"] for row in actions}
    summary = {
        "version": VERSION,
        "diagnostic_type": "Z5a frozen semantic-geometry action-space contract audit",
        "scene_count": len(scenes),
        "scene_list_coverage_count": len(scene_set),
        "merge_action_scene_coverage_count": len(action_scenes),
        "merge_action_scene_missing_count": len(scene_set - action_scenes),
        "merge_action_count": len(actions),
        "action_family_status": {row["action_type"]: row["status"] for row in family_rows},
        "source_action_counts": dict(sorted(source_counts.items())),
        "geometry_relation_counts": dict(sorted(geometry_relation_counts.items())),
        "semantic_join_counts": dict(sorted(semantic_join_counts.items())),
        "semantic_node_join_success_count": len(actions) * 3,
        "semantic_node_join_expected_count": len(actions) * 3,
        "semantic_node_join_success_rate": 1.0,
        "missing_or_duplicate_counts": dict(sorted(error_counts.items())),
        "ground_truth_usage": "none",
        "ap_computed": False,
        "model_trained": False,
        "candidate_mutation": False,
        "candidate_generation_applied": False,
        "threshold_or_weight_scanning": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "contracts": {
            "merge": "only the 1,501 already materialized frozen pair-union actions are audited",
            "split": "unavailable unless an official100 traceable materialized child-mask ledger already exists",
            "boundary_owner": "unavailable unless an official100 traceable superpoint owner ledger already exists",
            "z3_oof_projection": "only candidate/class/original-score/C_joint OOF prediction fields are used; label fields are ignored and never emitted",
            "contact": "shared point membership only; no distance, adjacency, IoU, or semantic threshold selected",
        },
        "input_provenance": {
            "scene_list": str(args.scene_list), "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest": str(args.split_manifest), "split_manifest_sha256": split_sha,
            "z1_summary_sha256": _sha256(args.z1_root / "summary.json"),
            "z1_bindings_sha256": _sha256(args.z1_root / "candidate_bindings.jsonl"),
            "z2c_summary_sha256": _sha256(args.z2c_root / "summary.json"),
            "z2c_nodes_sha256": _sha256(args.z2c_root / "nodes.jsonl"),
            "z2c_distributions_sha256": _sha256(args.z2c_root / "semantic_distributions.npz"),
            "z3_oof_summary_sha256": _sha256(args.z3_oof_root / "summary.json"),
            "z3_oof_predictions_sha256": _sha256(args.z3_oof_root / "oof_predictions.jsonl"),
            "z3_hybrid_ap_control_summary_reference": str(args.z3_hybrid_ap_summary),
            "z3_hybrid_ap_control_summary_sha256": _sha256(args.z3_hybrid_ap_summary),
            "combined_plan_summary_sha256": _sha256(args.combined_plan_root / "summary.json"),
            "combined_plan_rows_sha256": _sha256(args.combined_plan_root / "pair_union_append_candidates.jsonl"),
        },
    }

    staging = args.output_dir.parent / f".{args.output_dir.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "actions.jsonl", actions)
        _write_jsonl(staging / "action_families.jsonl", family_rows)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--z2c-root", type=Path, required=True)
    parser.add_argument("--z3-oof-root", type=Path, required=True)
    parser.add_argument("--z3-hybrid-ap-summary", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--asset-search-root", dest="asset_search_roots", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "split_manifest", "records_root", "z1_root", "z2c_root",
        "z3_oof_root", "z3_hybrid_ap_summary", "combined_plan_root", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    args.asset_search_roots = [
        _resolve(path) for path in (args.asset_search_roots or [Path("output"), Path("docs/diagnostics")])
    ]
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
