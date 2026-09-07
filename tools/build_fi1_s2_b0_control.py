#!/usr/bin/env python3
"""Build the preregistered no-GT S2 B0 semantic-control cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.fi1_geometry_semantic_separation_core import read_jsonl, sha256_file  # noqa: E402


VERSION = "fi1_geometry_semantic_separation_s2_b0_v1"
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


def _optional_int(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def _scenes(path: Path, expected: int | None) -> list[str]:
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or contains duplicates")
    if expected is not None and len(rows) != expected:
        raise ValueError(f"scene_count differs: {len(rows)} != {expected}")
    return rows


def _points(path: Path, point_count: int) -> np.ndarray:
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points) or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"invalid point indices: {path}")
    return points


def _float32(value: float) -> float:
    return float(np.float32(value))


def _validate_count(actual: int, expected: int | None, name: str) -> None:
    if expected is not None and int(actual) != int(expected):
        raise ValueError(f"{name} differs: {actual} != {expected}")


def _load_z3(root: Path, scenes: set[str]) -> tuple[dict[tuple[str, str, int], dict], dict]:
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("ground_truth_usage") != "none":
        raise ValueError("Z3 input is not a no-GT transfer output")
    rows = {}
    for row in read_jsonl(root / "oof_predictions.jsonl"):
        key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        if key[0] not in scenes or key[1] not in {"native", "track", "pair_union"}:
            raise ValueError(f"invalid Z3 key: {key}")
        if key in rows:
            raise ValueError(f"duplicate Z3 key: {key}")
        score = row.get("oof_predictions", {}).get(Z3_SCORE_NAME)
        if score is None:
            raise ValueError(f"missing frozen Z3 score: {key}")
        rows[key] = row
    return rows, summary


def _manifest_record(row: dict, z3_row: dict | None) -> dict:
    source = str(row["candidate_source"])
    scene = str(row["scene_name"])
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
            **base,
            "b0_in_scope": False,
            "b0_materialized": False,
            "b0_class_index": None,
            "b0_score": None,
            "b0_class_source": None,
            "b0_score_source": None,
            "b0_class_differs_from_s1": False,
            "b0_boundary_reason": "b2_append_only_refined_union_out_of_scope",
        }
    if origin != "legacy_member":
        raise ValueError(f"unsupported semantic origin: {origin}")
    if z3_row is None:
        if source == "native" and int(row["class_index"]) == 198:
            return {
                **base,
                "b0_in_scope": True,
                "b0_materialized": True,
                "b0_class_index": 198,
                "b0_score": _float32(float(row["legacy_frozen_score"])),
                "b0_class_source": "frozen_native_background_sentinel",
                "b0_score_source": "frozen_native_original_score",
                "b0_class_differs_from_s1": False,
                "b0_boundary_reason": "native_background_sentinel_retained",
            }
        if source == "track" and int(row["class_index"]) == -1 and (scene, candidate_id) in INVALID_TRACK_KEYS:
            return {
                **base,
                "b0_in_scope": True,
                "b0_materialized": False,
                "b0_class_index": None,
                "b0_score": None,
                "b0_class_source": None,
                "b0_score_source": None,
                "b0_class_differs_from_s1": False,
                "b0_boundary_reason": "historical_invalid_track_class_minus1_omitted",
            }
        raise ValueError(f"unexpected semantic hypothesis without Z3 row: {(scene, source, candidate_id)}")

    if abs(float(z3_row["original_score"]) - float(row["legacy_frozen_score"])) > 1e-7:
        raise ValueError(f"legacy/Z3 original score mismatch: {(scene, source, candidate_id)}")
    b0_class = int(z3_row["class_index"])
    if not 0 <= b0_class < 198:
        raise ValueError(f"Z3 row has invalid B0 class: {(scene, source, candidate_id)}")
    if source == "native" and b0_class != int(row["class_index"]):
        raise ValueError(f"native B0 class differs from S1: {(scene, candidate_id)}")
    if source == "pair_union":
        b0_score = _float32(float(z3_row["original_score"]))
        score_source = "frozen_pair_union_original_score"
    else:
        b0_score = _float32(float(z3_row["oof_predictions"][Z3_SCORE_NAME]))
        score_source = "frozen_full_official100_z3_C_joint_yolo_alpha"
    return {
        **base,
        "b0_in_scope": True,
        "b0_materialized": True,
        "b0_class_index": b0_class,
        "b0_score": b0_score,
        "b0_class_source": "frozen_native_class" if source == "native" else "frozen_z3_transfer_class",
        "b0_score_source": score_source,
        "b0_class_differs_from_s1": b0_class != int(row["class_index"]),
        "b0_boundary_reason": None,
    }


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "s1_root", "z3_root", "stream_records_root",
        "combined_plan_root", "config_path", "preregistration_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list, args.expected_scene_count)
    scene_set = set(scenes)
    required = (
        args.s1_root / "summary.json",
        args.s1_root / "semantic_hypothesis_ledger.jsonl",
        args.z3_root / "summary.json",
        args.z3_root / "oof_predictions.jsonl",
        args.stream_records_root / "summary.json",
        args.combined_plan_root / "summary.json",
        args.combined_plan_root / "pair_union_append_candidates.jsonl",
        args.config_path,
        args.preregistration_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    s1_summary = json.loads((args.s1_root / "summary.json").read_text())
    if s1_summary.get("ground_truth_read") is not False or s1_summary.get("ap_computed") is not False:
        raise ValueError("S1 input is not no-GT/no-AP")
    for root_name, root in (("stream", args.stream_records_root), ("combined", args.combined_plan_root)):
        summary = json.loads((root / "summary.json").read_text())
        if summary.get("ground_truth_usage") != "none":
            raise ValueError(f"{root_name} input is not no-GT")
    prompts = yaml.safe_load(args.config_path.read_text())["network2d"]["text_prompts"]
    if len(prompts) != args.class_count:
        raise ValueError(f"class prompt count differs: {len(prompts)} != {args.class_count}")

    semantic_rows = read_jsonl(args.s1_root / "semantic_hypothesis_ledger.jsonl")
    if [int(row["hypothesis_index"]) for row in semantic_rows] != list(range(len(semantic_rows))):
        raise ValueError("S1 hypothesis indexes are not contiguous and ordered")
    if {str(row["scene_name"]) for row in semantic_rows} != scene_set:
        raise ValueError("S1 scene coverage differs")
    z3_by_key, z3_summary = _load_z3(args.z3_root, scene_set)

    manifest_rows = []
    by_scene_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    semantic_keys = set()
    for row in semantic_rows:
        key = str(row["semantic_hypothesis_key"])
        if key in semantic_keys:
            raise ValueError(f"duplicate semantic hypothesis key: {key}")
        semantic_keys.add(key)
        join_key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        record = _manifest_record(row, z3_by_key.get(join_key))
        manifest_rows.append(record)
        if record["b0_materialized"]:
            by_scene_source[(record["scene_name"], record["candidate_source"])].append(record)

    used_z3 = {
        (row["scene_name"], row["candidate_source"], row["candidate_id"])
        for row in manifest_rows if row["origin_kind"] == "legacy_member" and
        (row["scene_name"], row["candidate_source"], row["candidate_id"]) in z3_by_key
    }
    if used_z3 != set(z3_by_key):
        raise ValueError("Z3 rows are not exactly consumed by the S1 join")

    source_counts = Counter()
    class_difference_counts = Counter()
    b0_in_scope_count = 0
    b2_only_count = 0
    boundary_counts = Counter()
    for record in manifest_rows:
        if record["b0_in_scope"]:
            b0_in_scope_count += 1
        else:
            b2_only_count += 1
        if record["b0_materialized"]:
            source_counts[record["candidate_source"]] += 1
        if record["b0_class_differs_from_s1"]:
            class_difference_counts[record["candidate_source"]] += 1
        if record["b0_boundary_reason"]:
            boundary_counts[record["b0_boundary_reason"]] += 1

    frozen_counts = {
        "semantic_hypothesis_count": len(manifest_rows),
        "b0_in_scope_count": b0_in_scope_count,
        "b2_only_refined_union_count": b2_only_count,
        "b0_materialized_count": sum(source_counts.values()),
        "b0_native_count": source_counts["native"],
        "b0_track_count": source_counts["track"],
        "b0_pair_union_count": source_counts["pair_union"],
        "native_background_sentinel_count": boundary_counts["native_background_sentinel_retained"],
        "invalid_track_boundary_exclusion_count": boundary_counts["historical_invalid_track_class_minus1_omitted"],
        "track_class_difference_count": class_difference_counts["track"],
        "pair_union_class_difference_count": class_difference_counts["pair_union"],
        "z3_row_count": len(z3_by_key),
    }
    for field, actual in frozen_counts.items():
        _validate_count(actual, getattr(args, f"expected_{field}"), field)
    if int(z3_summary.get("row_count", -1)) != len(z3_by_key):
        raise ValueError("Z3 summary row count differs")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    cache_stage = staging / "prediction_cache"
    cache_stage.mkdir(parents=True)
    cache_rows = []
    try:
        for ordinal, scene in enumerate(scenes, 1):
            source_prefix = args.stream_records_root / scene / "native_cache" / f"{scene}_pred_"
            native_masks = np.load(str(source_prefix) + "masks.npy", mmap_mode="r")
            native_classes = np.asarray(np.load(str(source_prefix) + "classes.npy"), dtype=np.int64)
            native_scores = np.asarray(np.load(str(source_prefix) + "scores.npy"), dtype=np.float32)
            native_records = sorted(by_scene_source[(scene, "native")], key=lambda row: row["candidate_id"])
            track_records = sorted(by_scene_source[(scene, "track")], key=lambda row: row["candidate_id"])
            union_records = sorted(by_scene_source[(scene, "pair_union")], key=lambda row: row["candidate_id"])
            if [row["candidate_id"] for row in native_records] != list(range(len(native_classes))):
                raise ValueError(f"{scene}: native IDs do not exactly cover source columns")
            if native_masks.ndim != 2 or native_masks.shape[1] != len(native_classes) or len(native_scores) != len(native_classes):
                raise ValueError(f"{scene}: native cache dimensions differ")
            ordered = native_records + track_records + union_records
            for column, record in enumerate(ordered):
                record["b0_column_index"] = column
            point_count = int(native_masks.shape[0])
            column_count = len(ordered)
            masks_path = cache_stage / f"{scene}_pred_masks.npy"
            classes_path = cache_stage / f"{scene}_pred_classes.npy"
            scores_path = cache_stage / f"{scene}_pred_scores.npy"
            output_masks = np.lib.format.open_memmap(
                masks_path, mode="w+", dtype=bool, shape=(point_count, column_count)
            )
            output_masks[:, :len(native_records)] = np.asarray(native_masks, dtype=bool)
            if column_count > len(native_records):
                output_masks[:, len(native_records):] = False
            classes = np.asarray([row["b0_class_index"] for row in ordered], dtype=np.int64)
            scores = np.asarray([row["b0_score"] for row in ordered], dtype=np.float32)
            if not np.array_equal(classes[:len(native_records)], native_classes):
                raise ValueError(f"{scene}: B0 native classes differ from frozen cache")
            sentinel = native_classes == 198
            if not np.array_equal(scores[:len(native_records)][sentinel], native_scores[sentinel]):
                raise ValueError(f"{scene}: B0 sentinel native scores differ from frozen cache")
            for record in track_records + union_records:
                points_path = Path(record["geometry_locator_read_only"]["points_path"])
                points = _points(points_path, point_count)
                output_masks[points, int(record["b0_column_index"])] = True
            output_masks.flush()
            del output_masks
            np.save(classes_path, classes)
            np.save(scores_path, scores)
            cache_rows.append({
                "scene_index": ordinal - 1,
                "scene_name": scene,
                "point_count": point_count,
                "column_count": column_count,
                "native_count": len(native_records),
                "track_count": len(track_records),
                "pair_union_count": len(union_records),
                "masks_shape": [point_count, column_count],
                "masks_dtype": "bool",
                "classes_shape": [column_count],
                "classes_dtype": "int64",
                "scores_shape": [column_count],
                "scores_dtype": "float32",
                "hashes": {
                    "masks": sha256_file(masks_path),
                    "classes": sha256_file(classes_path),
                    "scores": sha256_file(scores_path),
                },
                "ground_truth_read": False,
                "ap_computed": False,
            })
            print(f"[S2 B0] {ordinal}/{len(scenes)} {scene}: columns={column_count}", flush=True)

        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows))
        cache_manifest_path = staging / CACHE_MANIFEST_NAME
        cache_manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cache_rows))
        input_paths = {
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
        }
        summary = {
            "version": VERSION,
            "contract": "historical B0 control reproduced without collapsing S1 semantic hypotheses",
            "scene_count": len(scenes),
            **frozen_counts,
            "b0_source_counts": dict(sorted(source_counts.items())),
            "b0_class_difference_counts": dict(sorted(class_difference_counts.items())),
            "boundary_counts": dict(sorted(boundary_counts.items())),
            "candidate_deletion_count": 0,
            "geometry_mutation": False,
            "s1_class_mutation": False,
            "s1_score_mutation": False,
            "ground_truth_read": False,
            "ap_computed": False,
            "input_provenance": {name: sha256_file(path) for name, path in input_paths.items()},
            "output_hashes": {
                "b0_semantic_control_manifest": sha256_file(manifest_path),
                "prediction_cache_manifest": sha256_file(cache_manifest_path),
            },
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--z3-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--preregistration-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--expected-scene-count", type=_optional_int, default=312)
    parser.add_argument("--expected-semantic-hypothesis-count", type=_optional_int, default=213339)
    parser.add_argument("--expected-b0-in-scope-count", type=_optional_int, default=213233)
    parser.add_argument("--expected-b2-only-refined-union-count", type=_optional_int, default=106)
    parser.add_argument("--expected-b0-materialized-count", type=_optional_int, default=213227)
    parser.add_argument("--expected-b0-native-count", type=_optional_int, default=187200)
    parser.add_argument("--expected-b0-track-count", type=_optional_int, default=18135)
    parser.add_argument("--expected-b0-pair-union-count", type=_optional_int, default=7892)
    parser.add_argument("--expected-native-background-sentinel-count", type=_optional_int, default=295)
    parser.add_argument("--expected-invalid-track-boundary-exclusion-count", type=_optional_int, default=6)
    parser.add_argument("--expected-track-class-difference-count", type=_optional_int, default=1950)
    parser.add_argument("--expected-pair-union-class-difference-count", type=_optional_int, default=1137)
    parser.add_argument("--expected-z3-row-count", type=_optional_int, default=212932)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
