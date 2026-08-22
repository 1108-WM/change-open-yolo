#!/usr/bin/env python3
"""Audit C-v2 member evidence, continuous labels, and independent reproduction."""

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
    GeometryResolver, _read_jsonl, _resolve, _sha256,
)
from tools.build_ncs_fi1_stage_c_member_dataset_gt import (  # noqa: E402
    _geometry_sha256, decompose_union_atoms,
)
from tools.build_ncs_fi1_stage_c_v2_member_dataset_gt import (  # noqa: E402
    MEMBER_EVIDENCE_FEATURE_NAMES, fixed_target_removal_labels,
)
from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402


VERSION = "ncs_fi1_stage_c_v2_member_dataset_audit_v1"


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def run(args: argparse.Namespace) -> dict:
    for name in (
        "dataset_root", "reproduction_root", "base_dataset_root", "prepared_root",
        "ground_truth_root", "unique_geometry_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    summary_path = args.dataset_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    atoms_path = args.dataset_root / summary["files"]["atoms"]
    unions_path = args.dataset_root / summary["files"]["unions"]
    atoms = _read_jsonl(atoms_path)
    unions = _read_jsonl(unions_path)
    errors = Counter()
    if _sha256(atoms_path) != summary["hashes"]["atoms"]:
        errors["atoms_sha256_mismatch"] += 1
    if _sha256(unions_path) != summary["hashes"]["unions"]:
        errors["unions_sha256_mismatch"] += 1
    reproduction_summary = json.loads((args.reproduction_root / "summary.json").read_text())
    reproduction_atoms = args.reproduction_root / reproduction_summary["files"]["atoms"]
    reproduction_unions = args.reproduction_root / reproduction_summary["files"]["unions"]
    if _sha256(atoms_path) != _sha256(reproduction_atoms):
        errors["independent_reproduction_atoms_mismatch"] += 1
    if _sha256(unions_path) != _sha256(reproduction_unions):
        errors["independent_reproduction_unions_mismatch"] += 1

    base_summary = json.loads((args.base_dataset_root / "summary.json").read_text())
    base_atoms = _read_jsonl(args.base_dataset_root / base_summary["files"]["atoms"])
    base_by_id = {str(row["atom_id"]): row for row in base_atoms}
    atom_by_id = {str(row["atom_id"]): row for row in atoms}
    if len(atom_by_id) != len(atoms):
        errors["duplicate_atom_id"] += 1
    if set(atom_by_id) != set(base_by_id):
        errors["base_atom_coverage_mismatch"] += 1
    feature_names = list(summary["feature_names"])
    base_feature_names = list(base_summary["feature_names"])
    if feature_names != base_feature_names + list(MEMBER_EVIDENCE_FEATURE_NAMES):
        errors["feature_summary_schema_mismatch"] += 1
    for atom_id, row in atom_by_id.items():
        base = base_by_id.get(atom_id)
        if base is None:
            continue
        if any(not _close(row["features"].get(name), base["features"].get(name)) for name in base_feature_names):
            errors["base_feature_mutation"] += 1
        if sorted(row["features"]) != sorted(feature_names):
            errors["feature_schema_mismatch"] += 1
        if not all(math.isfinite(float(value)) for value in row["features"].values()):
            errors["nonfinite_feature"] += 1
        for name in MEMBER_EVIDENCE_FEATURE_NAMES:
            value = float(row["features"][name])
            if name.endswith(("fraction", "coverage_mean", "coverage_min", "coverage_max")) and not 0.0 <= value <= 1.0 + 1e-12:
                errors["bounded_member_feature_out_of_range"] += 1
        for name in ("candidate_mutation", "frozen_geometry_mutation", "ap_computed"):
            if row.get(name) is not False:
                errors[name] += 1

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    atoms_by_union = {}
    for row in atoms:
        atoms_by_union.setdefault((str(row["scene_name"]), int(row["union_candidate_id"])), []).append(row)
    for rows in atoms_by_union.values():
        rows.sort(key=lambda row: int(str(row["atom_id"]).rsplit(":", 1)[1]))
    resolver = GeometryResolver()
    scene_cache = {}
    role_counts = Counter()
    fold_counts = Counter()
    distinct_delta = set()
    for union in unions:
        scene = str(union["scene_name"])
        union_id = int(union["union_candidate_id"])
        cached = scene_cache.get(scene)
        if cached is None:
            stem = scene[len("scene"):] if scene.startswith("scene") else scene
            processed = np.load(args.prepared_root / scene / f"{stem}.npy", mmap_mode="r")
            gt, gt_meta = _load_gt(args.ground_truth_root / f"{scene}.txt", args.min_gt_points)
            cached = (processed, gt, gt_meta)
            scene_cache[scene] = cached
        processed, gt, gt_meta = cached
        union_node = node_by_key[str(union["union_geometry_key"])]
        track_node = node_by_key[str(union["track_geometry_key"])]
        native_node = node_by_key[str(union["native_geometry_key"])]
        union_points = resolver.points(union_node["canonical_geometry_locator"], int(union_node["point_count"]), len(gt))
        track_points = resolver.points(track_node["canonical_geometry_locator"], int(track_node["point_count"]), len(gt))
        native_points = resolver.points(native_node["canonical_geometry_locator"], int(native_node["point_count"]), len(gt))
        decomposed = decompose_union_atoms(
            union_points, track_points, native_points,
            np.asarray(processed[:, 9], dtype=np.int64),
        )
        rows = atoms_by_union.get((scene, union_id), [])
        if len(rows) != len(decomposed):
            errors["union_atom_count_mismatch"] += 1
            continue
        best = _best_gt(union_points, gt, gt_meta)
        target = best["best_gt"]
        encoded = int(target["encoded_id"]) if target else None
        target_count = int(target["point_count"]) if target else 0
        union_intersection = int(np.count_nonzero(gt[union_points] == encoded)) if target else 0
        if union.get("fixed_target_gt_encoded_id") != encoded:
            errors["union_fixed_target_mismatch"] += 1
        for atom, row in zip(decomposed, rows):
            if _geometry_sha256(atom["points"]) != row["point_sha256"]:
                errors["atom_hash_mismatch"] += 1
            atom_intersection = int(np.count_nonzero(gt[atom["points"]] == encoded)) if target else 0
            expected = fixed_target_removal_labels(
                union_point_count=len(union_points), atom_point_count=len(atom["points"]),
                target_point_count=target_count, union_target_intersection=union_intersection,
                atom_target_intersection=atom_intersection,
            )
            if int(row["label_atom_target_intersection"]) != atom_intersection:
                errors["atom_target_intersection_mismatch"] += 1
            for name, value in expected.items():
                if not _close(row[f"label_{name}"], value):
                    errors[f"label_{name}_mismatch"] += 1
            role_counts[str(row["role"])] += 1
            fold_counts[int(row["fold_index"])] += 1
            if row["role"] != "shared":
                distinct_delta.add(round(expected["delta_iou_remove"], 6))

    if summary.get("role_atom_counts") != dict(sorted(role_counts.items())):
        errors["summary_role_counts_mismatch"] += 1
    if summary.get("fold_atom_counts") != {str(fold): int(fold_counts[fold]) for fold in range(5)}:
        errors["summary_fold_counts_mismatch"] += 1
    if int(summary.get("exclusive_distinct_delta_iou_remove_rounded_6_count", -1)) != len(distinct_delta):
        errors["summary_distinct_delta_count_mismatch"] += 1
    checks = {
        "dataset_scene_count_is_100": int(summary.get("dataset_scene_count", -1)) == 100,
        "all_pair_unions_covered": len(unions) == int(summary.get("union_count", -1)),
        "all_atoms_covered": len(atoms) == int(summary.get("atom_count", -1)),
        "continuous_target_not_binary": len(distinct_delta) >= 100,
        "independent_reproduction_identical": (
            errors.get("independent_reproduction_atoms_mismatch", 0) == 0
            and errors.get("independent_reproduction_unions_mismatch", 0) == 0
        ),
        "all_folds_have_atoms": all(fold_counts[fold] > 0 for fold in range(5)),
    }
    output = {
        "version": VERSION,
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "error_counts": dict(sorted(errors.items())),
        "dataset_scene_count": int(summary["dataset_scene_count"]),
        "involved_union_scene_count": int(summary["involved_union_scene_count"]),
        "union_count": len(unions),
        "atom_count": len(atoms),
        "role_atom_counts": dict(sorted(role_counts.items())),
        "fold_atom_counts": {str(fold): int(fold_counts[fold]) for fold in range(5)},
        "exclusive_distinct_delta_iou_remove_rounded_6_count": len(distinct_delta),
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
            "reproduction_atoms_sha256": _sha256(reproduction_atoms),
            "reproduction_unions_sha256": _sha256(reproduction_unions),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reproduction-root", type=Path, required=True)
    parser.add_argument("--base-dataset-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_c_member_dataset_train100_20260822"))
    parser.add_argument("--prepared-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"))
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
