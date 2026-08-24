#!/usr/bin/env python3
"""Independently audit the no-GT FI1-D-v3 semantic prediction cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_dm_sms1_unique_geometry_ledger import GeometryResolver  # noqa: E402
from tools.dm_sms_core import geometry_hash  # noqa: E402


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict:
    for name in ("cache_root", "ledger_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    decision_root = getattr(args, "decision_root", None)
    if decision_root is not None:
        args.decision_root = _resolve(decision_root)
    summary = json.loads((args.cache_root / "summary.json").read_text())
    ledger = _rows(args.ledger_root / "unique_geometry_ledger.jsonl")
    errors = Counter()
    decisions_by_key = {}
    if decision_root is not None:
        try:
            decisions = _rows(args.decision_root / "safe_decisions.jsonl")
            decision_keys = [
                (str(row.get("scene_name", "")), str(row.get("plan_key", "")))
                for row in decisions
            ]
            if (any(not scene or not key for scene, key in decision_keys)
                    or len(decision_keys) != len(set(decision_keys))):
                errors["decision_identity_contract"] += 1
            decisions_by_key = dict(zip(decision_keys, decisions))
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            errors["decision_ledger_read_error"] += 1
    by_scene = defaultdict(list)
    expected_plan_keys = set()
    for row in ledger:
        members = row.get("members", [])
        if not members or int(row.get("member_count", -1)) != len(members):
            errors["ledger_member_contract"] += 1
            continue
        for member in members:
            plan_key = str(member.get("plan_key") or member.get("fi1_d_v3_plan_key") or "")
            if plan_key in expected_plan_keys or not plan_key:
                errors["ledger_plan_key_contract"] += 1
            expected_plan_keys.add(plan_key)
            by_scene[str(row["scene_name"])].append((row, member))
    cache_scene_root = args.cache_root / "prediction_cache"
    observed_scene_dirs = {
        path.name for path in cache_scene_root.iterdir() if path.is_dir()
    } if cache_scene_root.is_dir() else set()
    if observed_scene_dirs != set(by_scene):
        errors["cache_scene_coverage_mismatch"] += 1
    resolver = GeometryResolver()
    geometry_count = 0
    for scene, rows in sorted(by_scene.items()):
        rows.sort(key=lambda item: int(item[1]["plan_index"]))
        root = args.cache_root / "prediction_cache" / scene
        try:
            scene_summary = json.loads((root / "summary.json").read_text())
            for name, digest in scene_summary["file_sha256"].items():
                if _sha256(root / name) != digest:
                    errors["file_sha256_mismatch"] += 1
            masks = np.load(root / "masks.npy", mmap_mode="r")
            classes = np.load(root / "frozen_classes.npy", mmap_mode="r")
            scores = np.load(root / "frozen_scores.npy", mmap_mode="r")
            hashes = json.loads((root / "geometry_hashes.json").read_text())
            plan_keys = json.loads((root / "plan_keys.json").read_text())
            sources = json.loads((root / "sources.json").read_text())
            plan_indices = json.loads((root / "plan_indices.json").read_text())
            if not (
                masks.ndim == 2
                and masks.shape[1] == len(rows)
                and len(classes) == len(rows)
                and len(scores) == len(rows)
                and len(hashes) == len(rows)
                and len(plan_keys) == len(rows)
                and len(sources) == len(rows)
                and len(plan_indices) == len(rows)
            ):
                errors["shape_mismatch"] += 1
                continue
            if (
                int(scene_summary.get("point_count", -1)) != int(masks.shape[0])
                or int(scene_summary.get("candidate_count", -1)) != len(rows)
                or int(scene_summary.get("unique_geometry_count", -1))
                != len({row[0]["geometry_hash"] for row in rows})
                or scene_summary.get("ground_truth_read") is not False
                or scene_summary.get("ap_computed") is not False
            ):
                errors["scene_summary_mismatch"] += 1
            for column, (row, member) in enumerate(rows):
                points = resolver.points(member["geometry_locator_read_only"])
                observed = np.flatnonzero(np.asarray(masks[:, column], dtype=bool)).astype(np.int64)
                if not np.array_equal(points, observed) or geometry_hash(observed) != str(row["geometry_hash"]):
                    errors["geometry_mismatch"] += 1
                if (
                    hashes[column] != str(row["geometry_hash"])
                    or plan_keys[column] != str(member["plan_key"])
                    or sources[column] != str(member["candidate_source"])
                    or int(plan_indices[column]) != int(member["plan_index"])
                    or int(classes[column]) != int(member["frozen_class_index"])
                    or not np.isclose(float(scores[column]), float(member["challenger_score"]), rtol=0.0, atol=1e-7)
                ):
                    errors["metadata_mismatch"] += 1
                if decision_root is not None:
                    decision = decisions_by_key.get((scene, str(member["plan_key"])))
                    if (
                        decision is None
                        or str(decision.get("geometry_hash", "")) != str(row["geometry_hash"])
                        or int(decision.get("canonical_frozen_class_index", -999))
                        != int(member["frozen_class_index"])
                        or decision.get("candidate_source") != member.get("candidate_source")
                        or float(decision.get("challenger_score", -1.0))
                        != float(member.get("challenger_score", -2.0))
                        or decision.get("append_only") != member.get("append_only")
                        or not isinstance(decision.get("arbitrated_class_index"), int)
                        or any(decision.get(key) is not False for key in (
                            "candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion",
                        ))
                    ):
                        errors["challenge_class_overlay_contract"] += 1
            geometry_count += len(rows)
        except (FileNotFoundError, KeyError, ValueError, OSError, json.JSONDecodeError):
            errors["scene_cache_read_error"] += 1
    if (
        summary.get("cache_valid") is not True
        or int(summary.get("scene_count", -1)) != len(by_scene)
        or int(summary.get("candidate_count", -1)) != geometry_count
        or int(summary.get("unique_geometry_count", -1)) != len(ledger)
        or summary.get("duplicate_geometry_columns_preserved") is not True
        or int(summary.get("candidate_deletion_count", -1)) != 0
        or summary.get("plan_key_unique") is not True
        or summary.get("ground_truth_read") is not False
        or summary.get("ap_computed") is not False
    ):
        errors["summary_mismatch"] += 1
    expected_candidate_count = getattr(args, "expected_candidate_count", None)
    expected_unique_geometry_count = getattr(args, "expected_unique_geometry_count", None)
    if expected_candidate_count is not None and int(expected_candidate_count) != len(expected_plan_keys):
        errors["frozen_contract::candidate_count"] += 1
    if expected_unique_geometry_count is not None and int(expected_unique_geometry_count) != len(ledger):
        errors["frozen_contract::unique_geometry_count"] += 1
    if decision_root is not None and set(decisions_by_key) != {
        (scene, str(member["plan_key"]))
        for scene, rows in by_scene.items() for _row, member in rows
    }:
        errors["decision_coverage_mismatch"] += 1
    if str(summary.get("input_provenance", {}).get("unique_geometry_ledger_sha256", "")) != _sha256(
        args.ledger_root / "unique_geometry_ledger.jsonl"
    ):
        errors["summary_ledger_provenance_mismatch"] += 1
    audit_valid = sum(errors.values()) == 0
    audit = {
        "version": "dm_sms1_fi1_d_v3_prediction_cache_audit_v1",
        "audit_valid": audit_valid,
        "error_count": int(sum(errors.values())),
        "errors": dict(sorted(errors.items())),
        "scene_count": len(by_scene),
        "candidate_count": geometry_count,
        "geometry_count": geometry_count,
        "unique_geometry_count": len(ledger),
        "candidate_deletion_count": len(expected_plan_keys) - geometry_count,
        "plan_key_coverage_complete": geometry_count == len(expected_plan_keys),
        "duplicate_geometry_columns_preserved": geometry_count >= len(ledger),
        "control_challenge_masks_identical": audit_valid,
        "control_challenge_scores_identical": audit_valid,
        "control_challenge_order_identical": audit_valid,
        "challenge_relative_to_control_class_only": audit_valid and decision_root is not None,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "input_provenance": {
            "cache_summary_sha256": _sha256(args.cache_root / "summary.json"),
            "unique_geometry_ledger_sha256": _sha256(args.ledger_root / "unique_geometry_ledger.jsonl"),
            "decision_ledger_sha256": (
                _sha256(args.decision_root / "safe_decisions.jsonl") if decision_root is not None else None
            ),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return audit
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=39304)
    parser.add_argument("--expected-unique-geometry-count", type=int, default=39250)
    parser.add_argument("--output-root", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
