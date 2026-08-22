#!/usr/bin/env python3
"""Independently recompute and audit the stage-B relation rerank plan."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    GeometryResolver,
    _read_jsonl,
    _resolve,
    _sha256,
)
from tools.build_ncs_fi1_stage_b_relation_rerank_plan import (  # noqa: E402
    RELATION_TYPES,
    _geometry_relation,
    classify_relation,
    relation_decay_action,
)


VERSION = "ncs_fi1_stage_b_relation_rerank_plan_audit_v1"


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def run(args: argparse.Namespace) -> dict:
    for name in (
        "unique_geometry_root", "champion_plan_root", "stage_a_oof_root",
        "stage_b_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    summary_path = args.stage_b_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    components_path = args.stage_b_root / summary["files"]["components"]
    relations_path = args.stage_b_root / summary["files"]["relations"]
    plan_path = args.stage_b_root / summary["files"]["plan"]
    components = _read_jsonl(components_path)
    relations = _read_jsonl(relations_path)
    plan = _read_jsonl(plan_path)
    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    oof_summary = json.loads((args.stage_a_oof_root / "summary.json").read_text())
    predictions = _read_jsonl(args.stage_a_oof_root / oof_summary["prediction_file"])
    prediction_by_key = {str(row["geometry_key"]): row for row in predictions}
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    errors = Counter()
    if len(node_by_key) != len(nodes):
        errors["duplicate_unique_geometry_key"] += 1
    if set(node_by_key) != set(prediction_by_key):
        errors["stage_a_coverage_mismatch"] += 1

    candidates = {}
    native_member_to_geometry = {}
    track_to_geometry = {}
    union_to_geometry = {}
    for key, node in node_by_key.items():
        prediction = prediction_by_key[key]
        candidates[key] = {
            "geometry_key": key,
            "geometry_hash": str(node["geometry_hash"]),
            "scene_name": str(node["scene_name"]),
            "fold_index": int(prediction["fold_index"]),
            "candidate_source": str(node["canonical_candidate_source"]),
            "candidate_id": int(node["canonical_candidate_id"]),
            "point_count": int(node["point_count"]),
            "stage_a_quality": float(prediction["oof_unified_quality"]),
            "geometry_locator": node["canonical_geometry_locator"],
        }
        for member in node["members"]:
            source = str(member["candidate_source"])
            lookup_key = (str(node["scene_name"]), int(member["candidate_id"]))
            mapping = {
                "native": native_member_to_geometry,
                "track": track_to_geometry,
                "pair_union": union_to_geometry,
            }[source]
            previous = mapping.setdefault(lookup_key, key)
            if previous != key:
                errors["member_to_multiple_geometry"] += 1

    union_parent_sets = {}
    for row in _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl"):
        scene = str(row["scene_name"])
        union_key = union_to_geometry[(scene, int(row["candidate_id"]))]
        track_key = track_to_geometry[(scene, int(row["track_id"]))]
        native_keys = {
            native_member_to_geometry[(scene, int(candidate_id))]
            for candidate_id in row["native_member_candidate_ids"]
        }
        if len(native_keys) != 1:
            errors["union_native_parent_geometry_count"] += 1
            continue
        union_parent_sets[union_key] = {track_key, next(iter(native_keys))}

    component_by_geometry = {}
    component_keys = {}
    for component in components:
        component_index = int(component["component_index"])
        keys = list(component["geometry_keys"])
        if len(keys) != len(set(keys)) or len(keys) != int(component["geometry_count"]):
            errors["component_geometry_count_mismatch"] += 1
        component_keys[component_index] = set(keys)
        for key in keys:
            if key not in candidates:
                errors["component_unknown_geometry"] += 1
            previous = component_by_geometry.setdefault(key, component_index)
            if previous != component_index:
                errors["geometry_in_multiple_components"] += 1
    if set(component_by_geometry) != set(candidates):
        errors["component_coverage_mismatch"] += 1

    resolver = GeometryResolver()
    point_cache = {}
    strongest = {
        key: {"factor": 1.0, "relation_id": None, "reason": "no_relation_decay"}
        for key in candidates
    }
    relation_ids = set()
    relation_counts = Counter()
    for row in relations:
        relation_id = str(row["relation_id"])
        if relation_id in relation_ids:
            errors["duplicate_relation_id"] += 1
        relation_ids.add(relation_id)
        key_a, key_b = str(row["geometry_key_a"]), str(row["geometry_key_b"])
        candidate_a, candidate_b = candidates.get(key_a), candidates.get(key_b)
        if candidate_a is None or candidate_b is None:
            errors["relation_unknown_geometry"] += 1
            continue
        component_index = int(row["component_index"])
        if key_a not in component_keys.get(component_index, set()) or key_b not in component_keys.get(component_index, set()):
            errors["relation_component_mismatch"] += 1
        if candidate_a["scene_name"] != candidate_b["scene_name"] or candidate_a["fold_index"] != candidate_b["fold_index"]:
            errors["relation_scene_or_fold_mismatch"] += 1
        for key, candidate in ((key_a, candidate_a), (key_b, candidate_b)):
            if key not in point_cache:
                point_cache[key] = resolver.points(
                    candidate["geometry_locator"], candidate["point_count"], 2**63 - 1
                )
        geometry = _geometry_relation(point_cache[key_a], point_cache[key_b])
        for name, value in geometry.items():
            if not _close(row.get(name), value):
                errors[f"geometry_{name}_mismatch"] += 1
        union_parent = False
        union_new_fraction = 0.0
        for union_key, other_key in ((key_a, key_b), (key_b, key_a)):
            if other_key in union_parent_sets.get(union_key, set()):
                union_parent = True
                union_new_fraction = max(
                    union_new_fraction,
                    1.0 - geometry[
                        "coverage_a_in_b" if union_key == key_a else "coverage_b_in_a"
                    ],
                )
        if bool(row["pair_union_parent_relation"]) != union_parent:
            errors["union_parent_flag_mismatch"] += 1
        if not _close(row["union_new_fraction_over_parent"], union_new_fraction):
            errors["union_new_fraction_mismatch"] += 1
        expected_type = classify_relation(
            same_hash=candidate_a["geometry_hash"] == candidate_b["geometry_hash"],
            pair_union_parent=union_parent,
            union_new_fraction=union_new_fraction,
            iou=geometry["point_iou"],
            coverage_a_in_b=geometry["coverage_a_in_b"],
            coverage_b_in_a=geometry["coverage_b_in_a"],
            cross_source=candidate_a["candidate_source"] != candidate_b["candidate_source"],
            direct_evidence=bool(row["direct_relation_evidence"]),
            same_fraction=float(row["public_same_matched_observation_fraction"]),
            different_fraction=float(row["public_different_matched_observation_fraction"]),
        )
        if str(row["relation_type"]) != expected_type:
            errors["relation_type_mismatch"] += 1
        relation_counts[expected_type] += 1
        audited_relation = {
            **row,
            **geometry,
            "minimum_bidirectional_coverage": min(
                geometry["coverage_a_in_b"], geometry["coverage_b_in_a"]
            ),
            "maximum_bidirectional_coverage": max(
                geometry["coverage_a_in_b"], geometry["coverage_b_in_a"]
            ),
            "relation_type": expected_type,
        }
        action = relation_decay_action(audited_relation, candidate_a, candidate_b)
        if action is not None:
            current = strongest[action["geometry_key"]]
            factor = max(0.0, min(1.0, float(action["factor"])))
            if factor < current["factor"]:
                strongest[action["geometry_key"]] = {
                    "factor": factor, "relation_id": relation_id, "reason": action["reason"]
                }
        for name in ("candidate_mutation", "geometry_mutation", "candidate_deletion"):
            if row.get(name) is not False:
                errors[name] += 1
        if row.get("ap_computed") is not False:
            errors["ap_computed"] += 1

    plan_by_key = {}
    changed_counts = Counter()
    fold_changed = Counter()
    for row in plan:
        key = str(row["geometry_key"])
        if key in plan_by_key:
            errors["duplicate_plan_geometry"] += 1
        plan_by_key[key] = row
        candidate = candidates.get(key)
        if candidate is None:
            errors["plan_unknown_geometry"] += 1
            continue
        expected = strongest[key]
        if candidate["candidate_source"] == "native":
            expected = {"factor": 1.0, "relation_id": None, "reason": "native_safety_frozen"}
        expected_score = candidate["stage_a_quality"] * expected["factor"]
        if not _close(row["stage_a_quality"], candidate["stage_a_quality"]):
            errors["plan_stage_a_quality_mismatch"] += 1
        if not _close(row["strongest_decay_factor"], expected["factor"]):
            errors["plan_factor_mismatch"] += 1
        if row.get("strongest_relation_id") != expected["relation_id"]:
            errors["plan_relation_id_mismatch"] += 1
        if row.get("reason") != expected["reason"]:
            errors["plan_reason_mismatch"] += 1
        if not _close(row["stage_b_score"], expected_score):
            errors["plan_stage_b_score_mismatch"] += 1
        if float(row["stage_b_score"]) > float(row["stage_a_quality"]) + 1e-15:
            errors["score_increase"] += 1
        if candidate["candidate_source"] == "native" and not _close(row["stage_b_score"], row["stage_a_quality"]):
            errors["native_score_changed"] += 1
        changed = float(row["stage_b_score"]) < float(row["stage_a_quality"]) - 1e-15
        if bool(row["score_changed"]) != changed:
            errors["score_changed_flag_mismatch"] += 1
        if changed:
            changed_counts[candidate["candidate_source"]] += 1
            fold_changed[candidate["fold_index"]] += 1
        for name in ("candidate_deletion", "candidate_mutation", "geometry_mutation", "class_mutation", "frozen_cache_write"):
            if row.get(name) is not False:
                errors[name] += 1
        if row.get("ap_computed") is not False:
            errors["ap_computed"] += 1
    if set(plan_by_key) != set(candidates):
        errors["plan_coverage_mismatch"] += 1

    for name, path in (("components", components_path), ("relations", relations_path), ("plan", plan_path)):
        if _sha256(path) != summary["hashes"][name]:
            errors[f"{name}_sha256_mismatch"] += 1
    expected_relation_counts = {name: int(relation_counts[name]) for name in RELATION_TYPES}
    if summary.get("relation_type_counts") != expected_relation_counts:
        errors["relation_type_summary_mismatch"] += 1
    if summary.get("changed_candidate_counts") != dict(sorted(changed_counts.items())):
        errors["changed_candidate_summary_mismatch"] += 1

    advancement_checks = {
        "all_geometry_covered_once": len(plan_by_key) == len(candidates),
        "native_scores_unchanged": errors.get("native_score_changed", 0) == 0,
        "no_score_increase": errors.get("score_increase", 0) == 0,
        "pair_union_parent_complementary_exists": any(
            row["pair_union_parent_relation"] and row["relation_type"] == "complementary"
            for row in relations
        ),
        "near_duplicate_or_containment_exists": (
            relation_counts["near_duplicate"] + relation_counts["containment"] > 0
        ),
        **{f"fold_{fold}_has_non_native_continuous_decay": fold_changed[fold] > 0 for fold in range(5)},
    }
    output = {
        "version": VERSION,
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "error_counts": dict(sorted(errors.items())),
        "geometry_count": len(candidates),
        "component_count": len(components),
        "relation_count": len(relations),
        "plan_count": len(plan),
        "relation_type_counts": expected_relation_counts,
        "changed_candidate_counts": dict(sorted(changed_counts.items())),
        "fold_changed_candidate_counts": {str(fold): int(fold_changed[fold]) for fold in range(5)},
        "advancement_gate": {
            "checks": advancement_checks,
            "advancement_authorized": not errors and all(advancement_checks.values()),
        },
        "candidate_deletion_count": 0,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "class_mutation": False,
        "score_increase_count": int(errors.get("score_increase", 0)),
        "frozen_cache_write": False,
        "ap_computed": False,
        "validation60_read": False,
        "val312_read": False,
        "input_provenance": {
            "stage_b_summary_sha256": _sha256(summary_path),
            "components_sha256": _sha256(components_path),
            "relations_sha256": _sha256(relations_path),
            "plan_sha256": _sha256(plan_path),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique-geometry-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"
    ))
    parser.add_argument("--champion-plan-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/champion_plan"
    ))
    parser.add_argument("--stage-a-oof-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_a_unified_quality_oof_train100_20260822"
    ))
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
