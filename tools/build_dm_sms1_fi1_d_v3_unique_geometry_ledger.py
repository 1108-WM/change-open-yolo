#!/usr/bin/env python3
"""Adapt an audited FI1-D-v3 challenge plan into a DM-SMS-1 geometry ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_dm_sms1_unique_geometry_ledger import GeometryResolver  # noqa: E402
from tools.dm_sms_core import geometry_hash  # noqa: E402


VERSION = "dm_sms1_fi1_d_v3_unique_geometry_ledger_v1"
SOURCES = ("native", "track", "pair_union", "refined_union")


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


def _scenes(path: Path) -> list[str]:
    rows = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or contains duplicates")
    return rows


def _valid_class(value: int, class_count: int) -> bool:
    return 0 <= int(value) < class_count


def _plan_path(inference_root: Path) -> Path:
    summary = json.loads((inference_root / "complete_plan" / "summary.json").read_text())
    path = inference_root / "complete_plan" / str(summary["files"]["plan"])
    if _sha256(path) != str(summary["hashes"]["plan"]):
        raise ValueError("FI1-D-v3 complete-plan SHA-256 mismatch")
    return path


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "inference_root", "inference_audit_root",
        "legacy_unique_geometry_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list)
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError("scene count differs from the frozen joint contract")
    inference_summary = json.loads((args.inference_root / "summary.json").read_text())
    inference_audit = json.loads((args.inference_audit_root / "summary.json").read_text())
    if (
        int(inference_summary.get("scene_count", -1)) != len(scenes)
        or inference_summary.get("ground_truth_usage") != "none"
        or inference_summary.get("ap_computed") is not False
        or inference_summary.get("candidate_deletion_count") != 0
        or inference_summary.get("geometry_mutation") is not False
        or inference_summary.get("class_mutation") is not False
        or inference_audit.get("audit_valid") is not True
        or int(inference_audit.get("error_count", -1)) != 0
        or inference_audit.get("advancement_gate", {}).get("advancement_authorized") is not True
    ):
        raise ValueError("FI1-D-v3 frozen inference/audit contract is invalid")

    legacy_rows = _read_jsonl(args.legacy_unique_geometry_root / "unique_geometry_ledger.jsonl")
    legacy_by_key = {str(row["geometry_key"]): row for row in legacy_rows}
    if len(legacy_by_key) != len(legacy_rows):
        raise ValueError("legacy unique geometry ledger has duplicate geometry_key")
    if {str(row["scene_name"]) for row in legacy_rows} != set(scenes):
        raise ValueError("legacy unique geometry scene coverage differs")

    plan_path = _plan_path(args.inference_root)
    plan = _read_jsonl(plan_path)
    if len({str(row.get("plan_key", "")) for row in plan}) != len(plan):
        raise ValueError("FI1-D-v3 complete plan has empty or duplicate plan_key")
    if {str(row["scene_name"]) for row in plan} != set(scenes):
        raise ValueError("FI1-D-v3 complete plan scene coverage differs")

    resolver = GeometryResolver()
    grouped: dict[tuple[str, str], dict] = {}
    counts = Counter()
    seen_original = set()
    seen_plan_keys = set()
    for plan_index, plan_row in enumerate(plan):
        scene = str(plan_row["scene_name"])
        plan_key = str(plan_row.get("plan_key", ""))
        if not plan_key or plan_key in seen_plan_keys:
            raise ValueError("FI1-D-v3 complete plan has empty or duplicate plan_key")
        seen_plan_keys.add(plan_key)
        source = str(plan_row["candidate_source"])
        if source not in SOURCES:
            raise ValueError(f"unsupported FI1-D-v3 candidate source: {source}")
        if (
            plan_row.get("candidate_retained") is not True
            or plan_row.get("candidate_deletion") is not False
            or plan_row.get("geometry_mutation") is not False
            or plan_row.get("class_mutation") is not False
        ):
            raise ValueError(f"{plan_row.get('plan_key')}: FI1-D-v3 mutation contract failed")
        locator = dict(plan_row["geometry_locator_read_only"])
        points = resolver.points(locator)
        if len(points) != int(plan_row["point_count"]):
            raise ValueError(f"{plan_row.get('plan_key')}: point count differs")
        digest = geometry_hash(points)
        digest_algorithm = str(plan_row.get("geometry_digest_algorithm", ""))
        if digest_algorithm == "sha1_point_indices":
            expected_plan_digest = digest
        elif digest_algorithm == "sha256_point_indices":
            expected_plan_digest = hashlib.sha256(
                np.unique(np.asarray(points, dtype=np.int64)).tobytes()
            ).hexdigest()
        else:
            raise ValueError(f"{plan_row.get('plan_key')}: unsupported geometry digest algorithm")
        if str(plan_row.get("geometry_digest", "")) != expected_plan_digest:
            raise ValueError(f"{plan_row.get('plan_key')}: FI1-D-v3 geometry digest differs")
        identity = (scene, digest)
        frozen_class = int(plan_row["frozen_class_index"])
        frozen_valid = _valid_class(frozen_class, args.class_count)
        score = float(plan_row["challenger_score"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"{plan_row.get('plan_key')}: invalid challenger score")

        if source == "refined_union":
            if plan_row.get("append_only") is not True:
                raise ValueError(f"{plan_row.get('plan_key')}: refined union is not append-only")
            original_key = str(plan_row.get("original_union_geometry_key", ""))
            original = legacy_by_key.get(original_key)
            if original is None or str(original["canonical_candidate_source"]) != "pair_union":
                raise ValueError(f"{plan_row.get('plan_key')}: refined union parent is invalid")
            if int(original["canonical_frozen_class_index"]) != frozen_class:
                raise ValueError(f"{plan_row.get('plan_key')}: refined union class inheritance differs")
            candidate_id = int(original["canonical_candidate_id"])
            append_only = True
        else:
            if plan_row.get("append_only") is not False:
                raise ValueError(f"{plan_row.get('plan_key')}: original candidate append-only flag differs")
            key = str(plan_row.get("geometry_key", ""))
            original = legacy_by_key.get(key)
            if original is None or key in seen_original:
                raise ValueError(f"{plan_row.get('plan_key')}: original geometry join failed")
            seen_original.add(key)
            if (
                str(original["scene_name"]) != scene
                or str(original["canonical_candidate_source"]) != source
                or str(original["geometry_hash"]) != digest
                or int(original["canonical_frozen_class_index"]) != frozen_class
            ):
                raise ValueError(f"{plan_row.get('plan_key')}: original FI1-D-v3/Legacy identity mismatch")
            candidate_id = int(original["canonical_candidate_id"])
            append_only = False

        member = {
            "plan_index": plan_index,
            "plan_key": plan_key,
            "candidate_source": source,
            "candidate_id": candidate_id,
            "frozen_class_index": frozen_class,
            "frozen_class_valid": frozen_valid,
            "challenger_score": score,
            "frozen_score": score,
            "geometry_hash": digest,
            "point_count": len(points),
            "geometry_locator_read_only": locator,
            "geometry_locator": locator,
            "append_only": append_only,
            "fi1_d_v3_plan_key": plan_key,
            "fi1_d_v3_append_only": append_only,
            "candidate_retained": True,
            "candidate_deletion": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
        }
        row = grouped.get(identity)
        if row is None:
            row = {
                "scene_name": scene,
                "geometry_key": f"{scene}:visual_geometry:{digest}",
                "geometry_hash": digest,
                "point_count": len(points),
                "member_count": 0,
                "member_sources": [],
                "members": [],
                "canonical_member_index": 0,
                "canonical_candidate_source": source,
                "canonical_candidate_id": candidate_id,
                "canonical_frozen_class_index": frozen_class,
                "canonical_frozen_class_valid": frozen_valid,
                "canonical_frozen_score": score,
                "canonical_geometry_locator": locator,
                "ground_truth_usage": "none",
                "ground_truth_read": False,
                "ap_computed": False,
                "embedding_computed": False,
                "candidate_mutation": False,
                "geometry_mutation": False,
                "class_mutation": False,
                "score_mutation": False,
            }
            grouped[identity] = row
        else:
            canonical_points = resolver.points(row["canonical_geometry_locator"])
            if not np.array_equal(canonical_points, points):
                raise ValueError(f"{plan_key}: duplicate geometry hash has different point indices")
        row["members"].append(member)
        row["member_count"] = len(row["members"])
        row["member_sources"] = sorted({item["candidate_source"] for item in row["members"]})
        counts[f"source::{source}"] += 1
        counts["invalid_class"] += int(not frozen_valid)

    if seen_original != set(legacy_by_key):
        raise ValueError("FI1-D-v3 challenge does not cover every frozen Legacy geometry exactly once")
    rows = sorted(grouped.values(), key=lambda row: (row["scene_name"], row["geometry_hash"]))
    duplicate_rows = [row for row in rows if int(row["member_count"]) > 1]
    duplicate_scene_count = len({row["scene_name"] for row in duplicate_rows})
    duplicate_different_class_count = sum(
        len({int(member["frozen_class_index"]) for member in row["members"]}) > 1
        for row in duplicate_rows
    )
    duplicate_different_score_count = sum(
        len({float(member["frozen_score"]) for member in row["members"]}) > 1
        for row in duplicate_rows
    )
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        ledger_path = staging / "unique_geometry_ledger.jsonl"
        ledger_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ))
        source_counts = {source: int(counts[f"source::{source}"]) for source in SOURCES}
        canonical_source_counts = Counter(row["canonical_candidate_source"] for row in rows)
        summary = {
            "version": "dm_sms1_unique_geometry_ledger_v1",
            "adapter_version": VERSION,
            "dataset": args.dataset_name,
            "scene_count": len(scenes),
            "member_count": len(plan),
            "unique_geometry_count": len(rows),
            "duplicate_member_count": len(plan) - len(rows),
            "cross_source_duplicate_geometry_count": sum(
                len(row["member_sources"]) > 1 for row in duplicate_rows
            ),
            "duplicate_geometry_output_count": 0,
            "duplicate_geometry_group_count": len(duplicate_rows),
            "duplicate_geometry_scene_count": duplicate_scene_count,
            "duplicate_group_different_class_count": duplicate_different_class_count,
            "duplicate_group_different_score_count": duplicate_different_score_count,
            "invalid_frozen_class_member_count": int(counts["invalid_class"]),
            "invalid_canonical_frozen_class_count": sum(
                not bool(row["canonical_frozen_class_valid"]) for row in rows
            ),
            "source_member_counts": {key: value for key, value in source_counts.items() if value},
            "canonical_source_counts": dict(sorted(canonical_source_counts.items())),
            "fi1_d_v3_control_candidate_count": len(legacy_rows),
            "fi1_d_v3_challenger_candidate_count": len(plan),
            "fi1_d_v3_refined_union_candidate_count": source_counts["refined_union"],
            "candidate_deletion_count": 0,
            "contract_valid": True,
            "ground_truth_usage": "none",
            "ground_truth_read": False,
            "ap_computed": False,
            "embedding_computed": False,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
            "candidate_identity_contract": "(scene_name, plan_key)",
            "visual_geometry_identity_contract": "(scene_name, geometry_hash)",
            "visual_evidence_computed_once_per_unique_geometry": True,
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "fi1_d_v3_inference_summary_sha256": _sha256(args.inference_root / "summary.json"),
                "fi1_d_v3_inference_audit_sha256": _sha256(args.inference_audit_root / "summary.json"),
                "fi1_d_v3_complete_plan_sha256": _sha256(plan_path),
                "legacy_unique_geometry_ledger_sha256": _sha256(
                    args.legacy_unique_geometry_root / "unique_geometry_ledger.jsonl"
                ),
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
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--inference-audit-root", type=Path, required=True)
    parser.add_argument("--legacy-unique-geometry-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=312)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--dataset-name", default="ScanNet200-val312")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
