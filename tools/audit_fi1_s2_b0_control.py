#!/usr/bin/env python3
"""Independently audit the no-GT S2 B0 semantic-control cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.fi1_geometry_semantic_separation_core import read_jsonl, sha256_file  # noqa: E402


VERSION = "fi1_geometry_semantic_separation_s2_b0_v1_audit"
RESULT_VERSION = "fi1_geometry_semantic_separation_s2_b0_v1"
MANIFEST_NAME = "b0_semantic_control_manifest.jsonl"
CACHE_MANIFEST_NAME = "prediction_cache_manifest.jsonl"
Z3_SCORE_NAME = "C_joint_yolo_alpha"
INVALID_TRACK_KEYS = {
    ("scene0307_00", 40),
    ("scene0580_01", 27),
    ("scene0643_00", 85),
    ("scene0655_00", 37),
    ("scene0663_01", 38),
    ("scene0678_00", 74),
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _error(errors: Counter, name: str, condition: bool) -> None:
    if condition:
        errors[name] += 1


def _float32(value: float) -> float:
    return float(np.float32(value))


def _points(path: Path, point_count: int) -> np.ndarray:
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points) or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"invalid point indices: {path}")
    return points


def _expected_record(row: dict, z3_row: dict | None) -> dict:
    scene = str(row["scene_name"])
    source = str(row["candidate_source"])
    candidate_id = int(row["candidate_id"])
    origin = str(row["origin_kind"])
    base = {
        "manifest_index": int(row["hypothesis_index"]),
        "hypothesis_index": int(row["hypothesis_index"]),
        "semantic_hypothesis_key": str(row["semantic_hypothesis_key"]),
        "geometry_evidence_key": str(row["geometry_evidence_key"]),
        "scene_name": scene,
        "geometry_hash": str(row["geometry_hash"]),
        "origin_kind": origin,
        "candidate_source": source,
        "candidate_id": candidate_id,
        "s1_class_index": int(row["class_index"]),
        "s1_class_valid": bool(row["class_valid"]),
        "s1_legacy_frozen_score": row["legacy_frozen_score"],
        "fi1_geometry_score": float(row["fi1_geometry_score"]),
        "geometry_locator_read_only": dict(row["geometry_locator_read_only"]),
        "b0_column_index": None,
        "candidate_retained_in_semantic_ledger": True,
        "candidate_deletion": False,
        "geometry_mutation": False,
        "s1_class_mutation": False,
        "s1_score_mutation": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    if origin == "fi1_refined_union":
        return {
            **base, "b0_in_scope": False, "b0_materialized": False,
            "b0_class_index": None, "b0_score": None, "b0_class_source": None,
            "b0_score_source": None, "b0_class_differs_from_s1": False,
            "b0_boundary_reason": "b2_append_only_refined_union_out_of_scope",
        }
    if origin != "legacy_member":
        raise ValueError(f"unexpected S1 origin: {origin}")
    if z3_row is None:
        if source == "native" and int(row["class_index"]) == 198:
            return {
                **base, "b0_in_scope": True, "b0_materialized": True,
                "b0_class_index": 198,
                "b0_score": _float32(float(row["legacy_frozen_score"])),
                "b0_class_source": "frozen_native_background_sentinel",
                "b0_score_source": "frozen_native_original_score",
                "b0_class_differs_from_s1": False,
                "b0_boundary_reason": "native_background_sentinel_retained",
            }
        if source == "track" and int(row["class_index"]) == -1 and (scene, candidate_id) in INVALID_TRACK_KEYS:
            return {
                **base, "b0_in_scope": True, "b0_materialized": False,
                "b0_class_index": None, "b0_score": None, "b0_class_source": None,
                "b0_score_source": None, "b0_class_differs_from_s1": False,
                "b0_boundary_reason": "historical_invalid_track_class_minus1_omitted",
            }
        raise ValueError(f"unexpected missing Z3 identity: {(scene, source, candidate_id)}")
    if abs(float(z3_row["original_score"]) - float(row["legacy_frozen_score"])) > 1e-7:
        raise ValueError(f"S1/Z3 original score differs: {(scene, source, candidate_id)}")
    b0_class = int(z3_row["class_index"])
    if not 0 <= b0_class < 198:
        raise ValueError(f"invalid Z3 class: {(scene, source, candidate_id)}")
    if source == "native" and b0_class != int(row["class_index"]):
        raise ValueError(f"native Z3 class differs: {(scene, candidate_id)}")
    if source == "pair_union":
        score = _float32(float(z3_row["original_score"]))
        score_source = "frozen_pair_union_original_score"
    else:
        score = _float32(float(z3_row["oof_predictions"][Z3_SCORE_NAME]))
        score_source = "frozen_full_official100_z3_C_joint_yolo_alpha"
    return {
        **base, "b0_in_scope": True, "b0_materialized": True,
        "b0_class_index": b0_class, "b0_score": score,
        "b0_class_source": "frozen_native_class" if source == "native" else "frozen_z3_transfer_class",
        "b0_score_source": score_source,
        "b0_class_differs_from_s1": b0_class != int(row["class_index"]),
        "b0_boundary_reason": None,
    }


def audit(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "s1_root", "z3_root", "stream_records_root", "combined_plan_root",
        "config_path", "preregistration_path", "result_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = [line.strip() for line in args.scene_list.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("invalid audit scene list")
    scene_set = set(scenes)
    paths = {
        "scene_list": args.scene_list,
        "s1_summary": args.s1_root / "summary.json",
        "s1_semantic_ledger": args.s1_root / "semantic_hypothesis_ledger.jsonl",
        "z3_summary": args.z3_root / "summary.json",
        "z3_rows": args.z3_root / "oof_predictions.jsonl",
        "stream_summary": args.stream_records_root / "summary.json",
        "combined_plan_summary": args.combined_plan_root / "summary.json",
        "pair_union_rows": args.combined_plan_root / "pair_union_append_candidates.jsonl",
        "config": args.config_path,
        "preregistration": args.preregistration_path,
        "result_summary": args.result_root / "summary.json",
        "result_manifest": args.result_root / MANIFEST_NAME,
        "result_cache_manifest": args.result_root / CACHE_MANIFEST_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing audit inputs: " + ", ".join(missing))
    prompts = yaml.safe_load(args.config_path.read_text())["network2d"]["text_prompts"]
    if len(prompts) != args.class_count:
        raise ValueError("frozen class configuration differs")

    s1_rows = read_jsonl(paths["s1_semantic_ledger"])
    z3_by_key = {}
    for row in read_jsonl(paths["z3_rows"]):
        key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        if key in z3_by_key:
            raise ValueError(f"duplicate Z3 audit key: {key}")
        z3_by_key[key] = row
    expected = []
    by_scene_source = defaultdict(list)
    for row in s1_rows:
        key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        record = _expected_record(row, z3_by_key.get(key))
        expected.append(record)
        if record["b0_materialized"]:
            by_scene_source[(record["scene_name"], record["candidate_source"])].append(record)
    used_z3 = {
        (row["scene_name"], row["candidate_source"], row["candidate_id"])
        for row in expected if row["origin_kind"] == "legacy_member" and
        (row["scene_name"], row["candidate_source"], row["candidate_id"]) in z3_by_key
    }
    if used_z3 != set(z3_by_key):
        raise ValueError("independent S1/Z3 join coverage differs")
    for scene in scenes:
        ordered = []
        for source in ("native", "track", "pair_union"):
            ordered.extend(sorted(by_scene_source[(scene, source)], key=lambda row: row["candidate_id"]))
        for column, row in enumerate(ordered):
            row["b0_column_index"] = column

    observed = read_jsonl(paths["result_manifest"])
    observed_cache_rows = read_jsonl(paths["result_cache_manifest"])
    result_summary = json.loads(paths["result_summary"].read_text())
    errors = Counter()
    _error(errors, "semantic_manifest_count", len(observed) != len(expected))
    for index in range(min(len(observed), len(expected))):
        _error(errors, "semantic_manifest_record_mismatch", observed[index] != expected[index])
    observed_keys = [str(row.get("semantic_hypothesis_key", "")) for row in observed]
    expected_keys = [str(row["semantic_hypothesis_key"]) for row in expected]
    _error(errors, "semantic_key_order_or_coverage", observed_keys != expected_keys)
    _error(errors, "semantic_key_not_unique", len(observed_keys) != len(set(observed_keys)))

    source_counts = Counter(row["candidate_source"] for row in expected if row["b0_materialized"])
    boundary_counts = Counter(row["b0_boundary_reason"] for row in expected if row["b0_boundary_reason"])
    class_differences = Counter(row["candidate_source"] for row in expected if row["b0_class_differs_from_s1"])
    expected_counts = {
        "scene_count": len(scenes),
        "semantic_hypothesis_count": len(expected),
        "b0_in_scope_count": sum(bool(row["b0_in_scope"]) for row in expected),
        "b2_only_refined_union_count": sum(not bool(row["b0_in_scope"]) for row in expected),
        "b0_materialized_count": sum(source_counts.values()),
        "b0_native_count": source_counts["native"],
        "b0_track_count": source_counts["track"],
        "b0_pair_union_count": source_counts["pair_union"],
        "native_background_sentinel_count": boundary_counts["native_background_sentinel_retained"],
        "invalid_track_boundary_exclusion_count": boundary_counts["historical_invalid_track_class_minus1_omitted"],
        "track_class_difference_count": class_differences["track"],
        "pair_union_class_difference_count": class_differences["pair_union"],
        "z3_row_count": len(z3_by_key),
    }
    for field, actual in expected_counts.items():
        registered = getattr(args, f"expected_{field}")
        _error(errors, f"frozen_count::{field}", int(actual) != int(registered))
        _error(errors, f"summary_count::{field}", int(result_summary.get(field, -1)) != int(actual))

    track_sources = {}
    union_sources = defaultdict(dict)
    for scene in scenes:
        track_path = args.stream_records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
        tracks = json.loads(track_path.read_text()).get("tracks", [])
        for row in tracks:
            track_sources[(scene, int(row["track_id"]))] = Path(row["points_path"])
    for row in read_jsonl(paths["pair_union_rows"]):
        scene = str(row["scene_name"])
        if scene in scene_set:
            candidate_id = int(row["candidate_id"])
            if candidate_id in union_sources[scene]:
                raise ValueError(f"duplicate union audit identity: {(scene, candidate_id)}")
            union_sources[scene][candidate_id] = Path(row["points_path"])

    expected_cache_rows = []
    cache_row_by_scene = {str(row.get("scene_name", "")): row for row in observed_cache_rows}
    _error(errors, "cache_manifest_scene_count", len(cache_row_by_scene) != len(scenes))
    for scene_index, scene in enumerate(scenes):
        prefix = args.stream_records_root / scene / "native_cache" / f"{scene}_pred_"
        native_masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
        native_classes = np.asarray(np.load(str(prefix) + "classes.npy"), dtype=np.int64)
        native_scores = np.asarray(np.load(str(prefix) + "scores.npy"), dtype=np.float32)
        native_rows = sorted(by_scene_source[(scene, "native")], key=lambda row: row["candidate_id"])
        track_rows = sorted(by_scene_source[(scene, "track")], key=lambda row: row["candidate_id"])
        union_rows = sorted(by_scene_source[(scene, "pair_union")], key=lambda row: row["candidate_id"])
        ordered = native_rows + track_rows + union_rows
        expected_classes = np.asarray([row["b0_class_index"] for row in ordered], dtype=np.int64)
        expected_scores = np.asarray([row["b0_score"] for row in ordered], dtype=np.float32)
        cache_root = args.result_root / "prediction_cache"
        masks_path = cache_root / f"{scene}_pred_masks.npy"
        classes_path = cache_root / f"{scene}_pred_classes.npy"
        scores_path = cache_root / f"{scene}_pred_scores.npy"
        if not masks_path.is_file() or not classes_path.is_file() or not scores_path.is_file():
            errors["missing_cache_file"] += 1
            continue
        masks = np.load(masks_path, mmap_mode="r")
        classes = np.asarray(np.load(classes_path), dtype=np.int64)
        scores = np.asarray(np.load(scores_path), dtype=np.float32)
        _error(errors, "cache_mask_shape", list(masks.shape) != [int(native_masks.shape[0]), len(ordered)])
        _error(errors, "cache_mask_dtype", masks.dtype != np.dtype(bool))
        _error(errors, "cache_class_shape", list(classes.shape) != [len(ordered)])
        _error(errors, "cache_score_shape", list(scores.shape) != [len(ordered)])
        _error(errors, "cache_classes", not np.array_equal(classes, expected_classes))
        _error(errors, "cache_scores", not np.array_equal(scores, expected_scores))
        if masks.shape == (native_masks.shape[0], len(ordered)):
            _error(
                errors, "cache_native_masks",
                not np.array_equal(np.asarray(masks[:, :len(native_rows)], dtype=bool), np.asarray(native_masks, dtype=bool)),
            )
            for row in track_rows:
                expected_points = _points(track_sources[(scene, int(row["candidate_id"]))], native_masks.shape[0])
                actual_points = np.flatnonzero(masks[:, int(row["b0_column_index"])])
                _error(errors, "cache_track_mask", not np.array_equal(actual_points, expected_points))
            for row in union_rows:
                expected_points = _points(union_sources[scene][int(row["candidate_id"])], native_masks.shape[0])
                actual_points = np.flatnonzero(masks[:, int(row["b0_column_index"])])
                _error(errors, "cache_pair_union_mask", not np.array_equal(actual_points, expected_points))
        hashes = {
            "masks": sha256_file(masks_path), "classes": sha256_file(classes_path),
            "scores": sha256_file(scores_path),
        }
        expected_cache_row = {
            "scene_index": scene_index, "scene_name": scene,
            "point_count": int(native_masks.shape[0]), "column_count": len(ordered),
            "native_count": len(native_rows), "track_count": len(track_rows),
            "pair_union_count": len(union_rows),
            "masks_shape": [int(native_masks.shape[0]), len(ordered)], "masks_dtype": "bool",
            "classes_shape": [len(ordered)], "classes_dtype": "int64",
            "scores_shape": [len(ordered)], "scores_dtype": "float32",
            "hashes": hashes, "ground_truth_read": False, "ap_computed": False,
        }
        expected_cache_rows.append(expected_cache_row)
        _error(errors, "cache_manifest_record", cache_row_by_scene.get(scene) != expected_cache_row)
        _error(errors, "native_source_classes", not np.array_equal(expected_classes[:len(native_rows)], native_classes))
        sentinel = native_classes == 198
        _error(
            errors, "native_sentinel_scores",
            not np.array_equal(expected_scores[:len(native_rows)][sentinel], native_scores[sentinel]),
        )

    input_provenance = {
        name: sha256_file(path) for name, path in paths.items()
        if name not in {"result_summary", "result_manifest", "result_cache_manifest"}
    }
    output_hashes = {
        "b0_semantic_control_manifest": sha256_file(paths["result_manifest"]),
        "prediction_cache_manifest": sha256_file(paths["result_cache_manifest"]),
    }
    _error(errors, "summary_version", result_summary.get("version") != RESULT_VERSION)
    _error(errors, "summary_input_provenance", result_summary.get("input_provenance") != input_provenance)
    _error(errors, "summary_output_hashes", result_summary.get("output_hashes") != output_hashes)
    _error(errors, "summary_candidate_deletion", int(result_summary.get("candidate_deletion_count", -1)) != 0)
    for field in ("geometry_mutation", "s1_class_mutation", "s1_score_mutation", "ground_truth_read", "ap_computed"):
        _error(errors, f"forbidden_summary_flag::{field}", result_summary.get(field) is not False)

    output = {
        "version": VERSION,
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "errors": dict(sorted(errors.items())),
        **expected_counts,
        "b0_source_counts": dict(sorted(source_counts.items())),
        "b0_class_difference_counts": dict(sorted(class_differences.items())),
        "boundary_counts": dict(sorted(boundary_counts.items())),
        "candidate_deletion_count": 0,
        "geometry_mutation": False,
        "s1_class_mutation": False,
        "s1_score_mutation": False,
        "ground_truth_read": False,
        "ap_computed": False,
        "input_provenance": {
            **input_provenance,
            "result_summary": sha256_file(paths["result_summary"]),
            **output_hashes,
        },
    }
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"audit output is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--z3-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--preregistration-path", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--expected-scene-count", type=int, default=312)
    parser.add_argument("--expected-semantic-hypothesis-count", type=int, default=213339)
    parser.add_argument("--expected-b0-in-scope-count", type=int, default=213233)
    parser.add_argument("--expected-b2-only-refined-union-count", type=int, default=106)
    parser.add_argument("--expected-b0-materialized-count", type=int, default=213227)
    parser.add_argument("--expected-b0-native-count", type=int, default=187200)
    parser.add_argument("--expected-b0-track-count", type=int, default=18135)
    parser.add_argument("--expected-b0-pair-union-count", type=int, default=7892)
    parser.add_argument("--expected-native-background-sentinel-count", type=int, default=295)
    parser.add_argument("--expected-invalid-track-boundary-exclusion-count", type=int, default=6)
    parser.add_argument("--expected-track-class-difference-count", type=int, default=1950)
    parser.add_argument("--expected-pair-union-class-difference-count", type=int, default=1137)
    parser.add_argument("--expected-z3-row-count", type=int, default=212932)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
