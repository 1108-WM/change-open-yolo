#!/usr/bin/env python3
"""Build the preregistered stage-B relation ledger and no-deletion rerank plan."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    GeometryResolver,
    _read_jsonl,
    _resolve,
    _sha256,
)


VERSION = "ncs_fi1_stage_b_relation_rerank_plan_v1"
SOURCE_ORDER = {"native": 0, "track": 1, "pair_union": 2}
RELATION_TYPES = (
    "exact_duplicate", "near_duplicate", "containment",
    "complementary", "conflict", "uncertain",
)


def classify_relation(
    *, same_hash: bool, pair_union_parent: bool, union_new_fraction: float,
    iou: float, coverage_a_in_b: float, coverage_b_in_a: float,
    cross_source: bool, direct_evidence: bool, same_fraction: float,
    different_fraction: float,
) -> str:
    if same_hash:
        return "exact_duplicate"
    if pair_union_parent and union_new_fraction >= 0.10:
        return "complementary"
    minimum_coverage = min(coverage_a_in_b, coverage_b_in_a)
    maximum_coverage = max(coverage_a_in_b, coverage_b_in_a)
    if minimum_coverage >= 0.90:
        return "near_duplicate"
    if maximum_coverage >= 0.90:
        return "containment"
    if iou < 0.05 or (direct_evidence and different_fraction > same_fraction):
        return "conflict"
    if cross_source and iou >= 0.05:
        return "complementary"
    return "uncertain"


def relation_decay_action(relation: dict, candidate_a: dict, candidate_b: dict) -> dict | None:
    """Return one candidate-local continuous action, or None for no-op."""
    kind = relation["relation_type"]
    quality_a = float(candidate_a["stage_a_quality"])
    quality_b = float(candidate_b["stage_a_quality"])
    source_a = str(candidate_a["candidate_source"])
    source_b = str(candidate_b["candidate_source"])
    if kind == "near_duplicate":
        rank_a = (quality_a, -SOURCE_ORDER[source_a], -int(candidate_a["candidate_id"]))
        rank_b = (quality_b, -SOURCE_ORDER[source_b], -int(candidate_b["candidate_id"]))
        lower = candidate_a if rank_a < rank_b else candidate_b
        if lower["candidate_source"] == "native":
            return None
        factor = 1.0 - 0.5 * float(relation["minimum_bidirectional_coverage"])
        return {"geometry_key": lower["geometry_key"], "factor": factor, "reason": kind}
    if kind == "containment":
        coverage_a = float(relation["coverage_a_in_b"])
        coverage_b = float(relation["coverage_b_in_a"])
        if coverage_a >= 0.90 and coverage_b < 0.90:
            contained, container = candidate_a, candidate_b
        elif coverage_b >= 0.90 and coverage_a < 0.90:
            contained, container = candidate_b, candidate_a
        else:
            return None
        if (
            contained["candidate_source"] == "native"
            or float(contained["stage_a_quality"]) > float(container["stage_a_quality"])
        ):
            return None
        factor = 1.0 - 0.5 * float(relation["maximum_bidirectional_coverage"])
        return {"geometry_key": contained["geometry_key"], "factor": factor, "reason": kind}
    if kind == "conflict":
        if source_a == "native" and source_b != "native" and quality_b > quality_a:
            factor = quality_a / quality_b if quality_b > 0.0 else 1.0
            return {"geometry_key": candidate_b["geometry_key"], "factor": factor, "reason": "conflict_native_cap"}
        if source_b == "native" and source_a != "native" and quality_a > quality_b:
            factor = quality_b / quality_a if quality_a > 0.0 else 1.0
            return {"geometry_key": candidate_a["geometry_key"], "factor": factor, "reason": "conflict_native_cap"}
    return None


def _geometry_relation(points_a: np.ndarray, points_b: np.ndarray) -> dict:
    intersection = int(np.intersect1d(points_a, points_b, assume_unique=True).size)
    union = len(points_a) + len(points_b) - intersection
    return {
        "intersection_point_count": intersection,
        "union_point_count": union,
        "point_iou": intersection / max(1, union),
        "coverage_a_in_b": intersection / max(1, len(points_a)),
        "coverage_b_in_a": intersection / max(1, len(points_b)),
        "point_count_ratio_min_over_max": min(len(points_a), len(points_b)) / max(len(points_a), len(points_b)),
    }


def _direct_fields(row: dict | None) -> dict:
    raw = row["features"] if row is not None else {}
    return {
        "direct_relation_evidence": row is not None,
        "public_common_selected_view_count": int(raw.get("public_common_selected_view_count", 0)),
        "public_same_matched_observation_fraction": float(raw.get("public_same_matched_observation_fraction", 0.0)),
        "public_different_matched_observation_fraction": float(raw.get("public_different_matched_observation_fraction", 0.0)),
        "public_projected_box_iou_mean": float(raw.get("public_projected_box_iou_mean", 0.0)),
        "public_native_gvc_mean": float(raw.get("public_native_gvc_mean", 0.0)),
        "public_track_gvc_mean": float(raw.get("public_track_gvc_mean", 0.0)),
        "public_track_minus_native_gvc": float(raw.get("public_track_minus_native_gvc", 0.0)),
        "exclusive_boundary_contact_ratio_mean": float(raw.get("exclusive_boundary_contact_ratio_mean", 0.0)),
        "mean_rgb_distance": float(raw.get("mean_rgb_distance", 0.0)),
        "mean_normal_difference": float(raw.get("mean_normal_difference", 0.0)),
    }


def run(args: argparse.Namespace) -> dict:
    for name in (
        "unique_geometry_root", "relation_root", "champion_plan_root",
        "stage_a_oof_root", "stage_a_oof_audit_root", "preregistration", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    oof_summary_path = args.stage_a_oof_root / "summary.json"
    oof_summary = json.loads(oof_summary_path.read_text())
    oof_audit_path = args.stage_a_oof_audit_root / "summary.json"
    oof_audit = json.loads(oof_audit_path.read_text())
    if oof_audit.get("audit_valid") is not True or int(oof_audit.get("error_count", -1)) != 0:
        raise ValueError("stage-A OOF audit is not valid")
    if oof_summary.get("advancement_gate", {}).get("advancement_authorized") is not True:
        raise ValueError("stage-A advancement gate did not authorize stage B")
    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    predictions = _read_jsonl(args.stage_a_oof_root / oof_summary["prediction_file"])
    prediction_by_key = {str(row["geometry_key"]): row for row in predictions}
    if len(prediction_by_key) != len(nodes):
        raise ValueError("stage-A prediction count differs from unique geometry count")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    if set(node_by_key) != set(prediction_by_key):
        raise ValueError("stage-A predictions do not exactly cover unique geometries")

    native_member_to_geometry = {}
    track_to_geometry = {}
    union_to_geometry = {}
    candidate_rows = {}
    for node in nodes:
        geometry_key = str(node["geometry_key"])
        source = str(node["canonical_candidate_source"])
        prediction = prediction_by_key[geometry_key]
        candidate_rows[geometry_key] = {
            "geometry_key": geometry_key,
            "geometry_hash": str(node["geometry_hash"]),
            "scene_name": str(node["scene_name"]),
            "fold_index": int(prediction["fold_index"]),
            "candidate_source": source,
            "candidate_id": int(node["canonical_candidate_id"]),
            "point_count": int(node["point_count"]),
            "stage_a_quality": float(prediction["oof_unified_quality"]),
            "geometry_locator": node["canonical_geometry_locator"],
        }
        for member in node["members"]:
            member_source = str(member["candidate_source"])
            member_id = int(member["candidate_id"])
            if member_source == "native":
                previous = native_member_to_geometry.setdefault((node["scene_name"], member_id), geometry_key)
            elif member_source == "track":
                previous = track_to_geometry.setdefault((node["scene_name"], member_id), geometry_key)
            elif member_source == "pair_union":
                previous = union_to_geometry.setdefault((node["scene_name"], member_id), geometry_key)
            else:
                raise ValueError(f"unsupported member source: {member_source}")
            if previous != geometry_key:
                raise ValueError("member maps to multiple unique geometries")

    union_rows = _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl")
    union_by_geometry = {}
    components: dict[tuple[str, int], set[str]] = defaultdict(set)
    geometry_component = {}
    direct_relations = {}
    for scene_root in sorted(path for path in args.relation_root.iterdir() if path.is_dir()):
        scene = scene_root.name
        relation_path = scene_root / "relation_features_no_gt.jsonl"
        if not relation_path.is_file():
            continue
        for row in _read_jsonl(relation_path):
            component = (scene, int(row["relation_component_id"]))
            track_key = track_to_geometry[(scene, int(row["track_id"]))]
            native_keys = {
                native_member_to_geometry[(scene, int(candidate_id))]
                for candidate_id in row["native_member_candidate_ids"]
            }
            if len(native_keys) != 1:
                raise ValueError(f"{scene}: native exact group maps to multiple geometries")
            native_key = next(iter(native_keys))
            components[component].update((track_key, native_key))
            direct_key = tuple(sorted((track_key, native_key)))
            if direct_key in direct_relations:
                raise ValueError(f"duplicate direct relation geometry pair: {direct_key}")
            direct_relations[direct_key] = row
    for row in union_rows:
        scene = str(row["scene_name"])
        component = (scene, int(row["relation_component_id"]))
        union_key = union_to_geometry[(scene, int(row["candidate_id"]))]
        track_key = track_to_geometry[(scene, int(row["track_id"]))]
        native_keys = {
            native_member_to_geometry[(scene, int(candidate_id))]
            for candidate_id in row["native_member_candidate_ids"]
        }
        if len(native_keys) != 1:
            raise ValueError(f"{scene}: union native parent maps to multiple geometries")
        native_key = next(iter(native_keys))
        components[component].update((union_key, track_key, native_key))
        union_by_geometry[union_key] = {
            "row": row,
            "parent_geometry_keys": {track_key, native_key},
        }
    for component, geometry_keys in components.items():
        for geometry_key in geometry_keys:
            previous = geometry_component.setdefault(geometry_key, component)
            if previous != component:
                raise ValueError(f"geometry occurs in multiple components: {geometry_key}")
    for geometry_key, candidate in candidate_rows.items():
        if geometry_key not in geometry_component:
            component = (candidate["scene_name"], -(len(geometry_component) + 1))
            components[component].add(geometry_key)
            geometry_component[geometry_key] = component

    resolver = GeometryResolver()
    relation_rows = []
    component_rows = []
    relation_counts = Counter()
    source_pair_counts = Counter()
    strongest = {
        key: {"factor": 1.0, "relation_id": None, "reason": "no_relation_decay"}
        for key in candidate_rows
    }
    for component_index, ((scene, source_component_id), geometry_keys) in enumerate(
        sorted(components.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        ordered_keys = sorted(
            geometry_keys,
            key=lambda key: (
                -candidate_rows[key]["stage_a_quality"],
                SOURCE_ORDER[candidate_rows[key]["candidate_source"]],
                candidate_rows[key]["candidate_id"],
                key,
            ),
        )
        point_cache = {}
        for key in ordered_keys:
            candidate = candidate_rows[key]
            point_cache[key] = resolver.points(
                candidate["geometry_locator"], candidate["point_count"], 2**63 - 1
            )
        local_relation_counts = Counter()
        for pair_index, (key_a, key_b) in enumerate(itertools.combinations(ordered_keys, 2)):
            candidate_a, candidate_b = candidate_rows[key_a], candidate_rows[key_b]
            geometry = _geometry_relation(point_cache[key_a], point_cache[key_b])
            direct = direct_relations.get(tuple(sorted((key_a, key_b))))
            direct_fields = _direct_fields(direct)
            union_parent = False
            union_new_fraction = 0.0
            for union_key, other_key in ((key_a, key_b), (key_b, key_a)):
                info = union_by_geometry.get(union_key)
                if info is not None and other_key in info["parent_geometry_keys"]:
                    union_parent = True
                    union_new_fraction = max(
                        union_new_fraction,
                        1.0 - geometry[
                            "coverage_a_in_b" if union_key == key_a else "coverage_b_in_a"
                        ],
                    )
            relation_type = classify_relation(
                same_hash=candidate_a["geometry_hash"] == candidate_b["geometry_hash"],
                pair_union_parent=union_parent,
                union_new_fraction=union_new_fraction,
                iou=geometry["point_iou"],
                coverage_a_in_b=geometry["coverage_a_in_b"],
                coverage_b_in_a=geometry["coverage_b_in_a"],
                cross_source=candidate_a["candidate_source"] != candidate_b["candidate_source"],
                direct_evidence=direct_fields["direct_relation_evidence"],
                same_fraction=direct_fields["public_same_matched_observation_fraction"],
                different_fraction=direct_fields["public_different_matched_observation_fraction"],
            )
            relation_id = f"{scene}:component:{source_component_id}:pair:{pair_index}"
            relation = {
                "relation_id": relation_id,
                "scene_name": scene,
                "fold_index": int(candidate_a["fold_index"]),
                "source_relation_component_id": source_component_id,
                "component_index": component_index,
                "geometry_key_a": key_a,
                "geometry_key_b": key_b,
                "source_a": candidate_a["candidate_source"],
                "source_b": candidate_b["candidate_source"],
                "candidate_id_a_metadata_only": candidate_a["candidate_id"],
                "candidate_id_b_metadata_only": candidate_b["candidate_id"],
                "stage_a_quality_a": candidate_a["stage_a_quality"],
                "stage_a_quality_b": candidate_b["stage_a_quality"],
                **geometry,
                "minimum_bidirectional_coverage": min(
                    geometry["coverage_a_in_b"], geometry["coverage_b_in_a"]
                ),
                "maximum_bidirectional_coverage": max(
                    geometry["coverage_a_in_b"], geometry["coverage_b_in_a"]
                ),
                "pair_union_parent_relation": union_parent,
                "union_new_fraction_over_parent": union_new_fraction,
                **direct_fields,
                "relation_type": relation_type,
                "candidate_mutation": False,
                "geometry_mutation": False,
                "candidate_deletion": False,
                "ap_computed": False,
            }
            relation_rows.append(relation)
            relation_counts[relation_type] += 1
            local_relation_counts[relation_type] += 1
            source_pair_counts["+".join(sorted((candidate_a["candidate_source"], candidate_b["candidate_source"])))] += 1
            action = relation_decay_action(relation, candidate_a, candidate_b)
            if action is not None:
                current = strongest[action["geometry_key"]]
                factor = max(0.0, min(1.0, float(action["factor"])))
                if factor < current["factor"]:
                    strongest[action["geometry_key"]] = {
                        "factor": factor,
                        "relation_id": relation_id,
                        "reason": action["reason"],
                    }
        component_rows.append({
            "component_index": component_index,
            "scene_name": scene,
            "fold_index": int(candidate_rows[ordered_keys[0]]["fold_index"]),
            "source_relation_component_id": source_component_id,
            "geometry_count": len(ordered_keys),
            "geometry_keys": ordered_keys,
            "source_counts": dict(sorted(Counter(
                candidate_rows[key]["candidate_source"] for key in ordered_keys
            ).items())),
            "relation_count": sum(local_relation_counts.values()),
            "relation_type_counts": dict(sorted(local_relation_counts.items())),
            "singleton": len(ordered_keys) == 1,
        })

    plan_rows = []
    changed_counts = Counter()
    fold_changed_counts = Counter()
    for geometry_key, candidate in sorted(candidate_rows.items()):
        action = strongest[geometry_key]
        if candidate["candidate_source"] == "native":
            action = {"factor": 1.0, "relation_id": None, "reason": "native_safety_frozen"}
        stage_a_quality = float(candidate["stage_a_quality"])
        factor = float(action["factor"])
        stage_b_score = stage_a_quality * factor
        changed = stage_b_score < stage_a_quality - 1e-15
        if changed:
            changed_counts[candidate["candidate_source"]] += 1
            fold_changed_counts[int(candidate["fold_index"])] += 1
        plan_rows.append({
            "geometry_key": geometry_key,
            "geometry_hash": candidate["geometry_hash"],
            "scene_name": candidate["scene_name"],
            "fold_index": candidate["fold_index"],
            "candidate_source": candidate["candidate_source"],
            "candidate_id_metadata_only": candidate["candidate_id"],
            "source_relation_component_id": geometry_component[geometry_key][1],
            "stage_a_quality": stage_a_quality,
            "strongest_decay_factor": factor,
            "strongest_relation_id": action["relation_id"],
            "stage_b_score": stage_b_score,
            "score_changed": changed,
            "reason": action["reason"],
            "candidate_deletion": False,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "frozen_cache_write": False,
            "ap_computed": False,
        })

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        paths = {
            "components": staging / "relation_components.jsonl",
            "relations": staging / "candidate_relations.jsonl",
            "plan": staging / "stage_b_rerank_plan.jsonl",
        }
        for name, rows in (
            ("components", component_rows), ("relations", relation_rows), ("plan", plan_rows)
        ):
            paths[name].write_text("".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ))
        advancement_checks = {
            "all_geometry_covered_once": len(plan_rows) == len(candidate_rows),
            "native_scores_unchanged": all(
                row["stage_b_score"] == row["stage_a_quality"]
                for row in plan_rows if row["candidate_source"] == "native"
            ),
            "no_score_increase": all(row["stage_b_score"] <= row["stage_a_quality"] for row in plan_rows),
            "pair_union_parent_complementary_exists": any(
                row["pair_union_parent_relation"] and row["relation_type"] == "complementary"
                for row in relation_rows
            ),
            "near_duplicate_or_containment_exists": (
                relation_counts["near_duplicate"] + relation_counts["containment"] > 0
            ),
            **{
                f"fold_{fold}_has_non_native_continuous_decay": fold_changed_counts[fold] > 0
                for fold in range(5)
            },
        }
        summary = {
            "version": VERSION,
            "scene_count": len(set(row["scene_name"] for row in plan_rows)),
            "geometry_count": len(plan_rows),
            "component_count": len(component_rows),
            "singleton_component_count": sum(row["singleton"] for row in component_rows),
            "relation_count": len(relation_rows),
            "relation_type_counts": {name: int(relation_counts[name]) for name in RELATION_TYPES},
            "source_pair_counts": dict(sorted(source_pair_counts.items())),
            "changed_candidate_counts": dict(sorted(changed_counts.items())),
            "fold_changed_candidate_counts": {str(fold): int(fold_changed_counts[fold]) for fold in range(5)},
            "advancement_gate": {
                "checks": advancement_checks,
                "advancement_authorized_pending_independent_audit": all(advancement_checks.values()),
            },
            "files": {name: path.name for name, path in paths.items()},
            "hashes": {name: _sha256(path) for name, path in paths.items()},
            "ground_truth_usage": "stage-A OOF unified quality only; no raw GT read",
            "candidate_deletion_count": 0,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "score_increase_count": 0,
            "frozen_cache_write": False,
            "ap_computed": False,
            "validation60_read": False,
            "val312_read": False,
            "input_provenance": {
                "preregistration_sha256": _sha256(args.preregistration),
                "unique_geometry_summary_sha256": _sha256(args.unique_geometry_root / "summary.json"),
                "relation_summary_sha256": _sha256(args.relation_root / "summary.json"),
                "champion_plan_summary_sha256": _sha256(args.champion_plan_root / "summary.json"),
                "stage_a_oof_summary_sha256": _sha256(oof_summary_path),
                "stage_a_oof_audit_sha256": _sha256(oof_audit_path),
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique-geometry-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"
    ))
    parser.add_argument("--relation-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/relation_ledger"
    ))
    parser.add_argument("--champion-plan-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/champion_plan"
    ))
    parser.add_argument("--stage-a-oof-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_a_unified_quality_oof_train100_20260822"
    ))
    parser.add_argument("--stage-a-oof-audit-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_a_unified_quality_oof_train100_audit_20260822"
    ))
    parser.add_argument("--preregistration", type=Path, default=Path(
        "docs/NCS_FI1_STAGE_B_PREREGISTRATION_20260822.md"
    ))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output_root": str(_resolve(args.output_root)),
        "geometry_count": result["geometry_count"],
        "relation_count": result["relation_count"],
        "relation_type_counts": result["relation_type_counts"],
        "changed_candidate_counts": result["changed_candidate_counts"],
        "advancement_authorized_pending_independent_audit": result["advancement_gate"]["advancement_authorized_pending_independent_audit"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
