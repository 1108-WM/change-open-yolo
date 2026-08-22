#!/usr/bin/env python3
"""Independently audit the stage-C member dataset and atom conservation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    GeometryResolver, _quality_target, _read_jsonl, _resolve, _sha256,
)
from tools.build_ncs_fi1_stage_c_member_dataset_gt import (  # noqa: E402
    FEATURE_NAMES, ROLES, _geometry_sha256, decompose_union_atoms,
)
from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402


VERSION = "ncs_fi1_stage_c_member_dataset_audit_v1"


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def run(args: argparse.Namespace) -> dict:
    for name in (
        "dataset_root", "prepared_root", "ground_truth_root", "unique_geometry_root",
        "champion_plan_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    summary_path = args.dataset_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    atoms_path = args.dataset_root / summary["files"]["atoms"]
    unions_path = args.dataset_root / summary["files"]["unions"]
    atom_rows = _read_jsonl(atoms_path)
    union_rows = _read_jsonl(unions_path)
    atom_by_id = {str(row["atom_id"]): row for row in atom_rows}
    union_by_key = {
        (str(row["scene_name"]), int(row["union_candidate_id"])): row
        for row in union_rows
    }
    errors = Counter()
    if len(atom_by_id) != len(atom_rows):
        errors["duplicate_atom_id"] += 1
    if len(union_by_key) != len(union_rows):
        errors["duplicate_union_key"] += 1
    if _sha256(atoms_path) != summary["hashes"]["atoms"]:
        errors["atoms_sha256_mismatch"] += 1
    if _sha256(unions_path) != summary["hashes"]["unions"]:
        errors["unions_sha256_mismatch"] += 1

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    unions = _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl")
    if set(union_by_key) != {
        (str(row["scene_name"]), int(row["candidate_id"])) for row in unions
    }:
        errors["union_coverage_mismatch"] += 1
    resolver = GeometryResolver()
    scene_cache = {}
    audited_atom_ids = set()
    role_counts = Counter()
    fold_counts = Counter()
    shared_point_count = 0
    for source in unions:
        scene = str(source["scene_name"])
        union_id = int(source["candidate_id"])
        ledger = union_by_key.get((scene, union_id))
        if ledger is None:
            continue
        cached = scene_cache.get(scene)
        if cached is None:
            prepared_stem = scene[len("scene"):] if scene.startswith("scene") else scene
            processed = np.load(
                args.prepared_root / scene / f"{prepared_stem}.npy", mmap_mode="r"
            )
            gt, gt_meta = _load_gt(
                args.ground_truth_root / f"{scene}.txt", args.min_gt_points
            )
            cached = (processed, gt, gt_meta)
            scene_cache[scene] = cached
        processed, gt, gt_meta = cached
        union_node = node_by_key[str(ledger["union_geometry_key"])]
        track_node = node_by_key[str(ledger["track_geometry_key"])]
        native_node = node_by_key[str(ledger["native_geometry_key"])]
        union_points = resolver.points(union_node["canonical_geometry_locator"], int(union_node["point_count"]), len(gt))
        track_points = resolver.points(track_node["canonical_geometry_locator"], int(track_node["point_count"]), len(gt))
        native_points = resolver.points(native_node["canonical_geometry_locator"], int(native_node["point_count"]), len(gt))
        atoms = decompose_union_atoms(
            union_points, track_points, native_points,
            np.asarray(processed[:, 9], dtype=np.int64),
        )
        best = _best_gt(union_points, gt, gt_meta)
        target = best["best_gt"]
        encoded = int(target["encoded_id"]) if target else None
        if int(ledger["atom_count"]) != len(atoms):
            errors["union_atom_count_mismatch"] += 1
        if int(ledger["atom_point_count_sum"]) != len(union_points):
            errors["union_atom_point_sum_mismatch"] += 1
        if str(ledger["original_union_sha256"]) != _geometry_sha256(union_points):
            errors["original_union_hash_mismatch"] += 1
        if not _close(ledger["label_original_union_best_iou"], best["best_iou"]):
            errors["original_union_iou_mismatch"] += 1
        if not _close(ledger["label_original_union_quality_q"], _quality_target(best["best_iou"])):
            errors["original_union_quality_mismatch"] += 1
        local_roles = Counter()
        for atom_index, atom in enumerate(atoms):
            atom_id = f"{scene}:union:{union_id}:atom:{atom_index}"
            row = atom_by_id.get(atom_id)
            if row is None:
                errors["missing_atom"] += 1
                continue
            audited_atom_ids.add(atom_id)
            points = atom["points"]
            expected_purity = float(np.mean(gt[points] == encoded)) if encoded is not None else 0.0
            checks = {
                "atom_role_mismatch": str(row["role"]) == atom["role"],
                "atom_raw_superpoint_mismatch": int(row["raw_superpoint_id"]) == int(atom["raw_superpoint_id"]),
                "atom_point_count_mismatch": int(row["point_count"]) == len(points),
                "atom_hash_mismatch": str(row["point_sha256"]) == _geometry_sha256(points),
                "atom_target_mismatch": _close(row["label_retention_probability"], expected_purity),
                "atom_feature_schema_mismatch": sorted(row["features"]) == sorted(FEATURE_NAMES),
                "atom_nonfinite_feature": all(math.isfinite(float(value)) for value in row["features"].values()),
            }
            for name, valid in checks.items():
                if not valid:
                    errors[name] += 1
            if atom["role"] == "shared":
                shared_point_count += len(points)
            role_counts[atom["role"]] += 1
            local_roles[atom["role"]] += 1
            fold_counts[int(row["fold_index"])] += 1
            for name in ("candidate_mutation", "frozen_geometry_mutation", "ap_computed"):
                if row.get(name) is not False:
                    errors[name] += 1
        if ledger.get("role_atom_counts") != {role: int(local_roles[role]) for role in ROLES}:
            errors["union_role_count_mismatch"] += 1
        if ledger.get("original_union_retained") is not True or ledger.get("append_only") is not True:
            errors["original_union_retention_contract"] += 1
    if audited_atom_ids != set(atom_by_id):
        errors["atom_coverage_mismatch"] += 1
    if summary.get("role_atom_counts") != {role: int(role_counts[role]) for role in ROLES}:
        errors["summary_role_counts_mismatch"] += 1
    if summary.get("fold_atom_counts") != {str(fold): int(fold_counts[fold]) for fold in range(5)}:
        errors["summary_fold_counts_mismatch"] += 1

    checks = {
        "all_pair_unions_covered": (
            len(union_by_key) == len(unions) == int(summary.get("union_count", -1))
        ),
        "all_atoms_recomputed": len(audited_atom_ids) == len(atom_rows),
        "shared_atoms_exist": role_counts["shared"] > 0 and shared_point_count > 0,
        "all_folds_have_atoms": all(fold_counts[fold] > 0 for fold in range(5)),
        "original_unions_retained": int(summary.get("original_union_retained_count", -1)) == len(unions),
    }
    output = {
        "version": VERSION,
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "error_counts": dict(sorted(errors.items())),
        "union_count": len(union_rows),
        "atom_count": len(atom_rows),
        "role_atom_counts": {role: int(role_counts[role]) for role in ROLES},
        "fold_atom_counts": {str(fold): int(fold_counts[fold]) for fold in range(5)},
        "shared_point_count": shared_point_count,
        "advancement_gate": {
            "checks": checks,
            "advancement_authorized": not errors and all(checks.values()),
        },
        "candidate_mutation": False,
        "frozen_geometry_mutation": False,
        "frozen_cache_write": False,
        "ap_computed": False,
        "validation60_read": False,
        "val312_read": False,
        "input_provenance": {
            "dataset_summary_sha256": _sha256(summary_path),
            "atoms_sha256": _sha256(atoms_path),
            "unions_sha256": _sha256(unions_path),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"))
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--champion-plan-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/champion_plan"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
