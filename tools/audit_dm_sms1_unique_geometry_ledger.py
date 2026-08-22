#!/usr/bin/env python3
"""Independently audit a DM-SMS-1 unique-geometry ledger without GT or AP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dm_sms_core import canonical_member, geometry_hash


SOURCE_ORDER = ("native", "track", "pair_union")
SOURCE_RANK = {source: index for index, source in enumerate(SOURCE_ORDER)}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_scenes(path: Path) -> list[str]:
    scenes = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {path}:{line_number}") from error
    return rows


class GeometryResolver:
    def __init__(self) -> None:
        self.native_masks: dict[Path, np.ndarray] = {}

    def points(self, locator: dict) -> np.ndarray:
        kind = str(locator.get("kind"))
        if kind == "native_mask_column":
            path = _resolve(Path(locator["masks_path"]))
            masks = self.native_masks.get(path)
            if masks is None:
                masks = np.load(path, mmap_mode="r")
                if masks.ndim != 2:
                    raise ValueError(f"native mask cache is not two-dimensional: {path}")
                self.native_masks[path] = masks
            column = int(locator["column_index"])
            if column < 0 or column >= masks.shape[1]:
                raise ValueError(f"native mask column is out of range: {path}:{column}")
            points = np.flatnonzero(np.asarray(masks[:, column], dtype=bool)).astype(np.int64)
        elif kind == "point_indices_npz":
            path = _resolve(Path(locator["points_path"]))
            key = str(locator.get("array_key", "point_indices"))
            with np.load(path) as payload:
                if key not in payload:
                    raise ValueError(f"{key} is missing from {path}")
                points = np.unique(np.asarray(payload[key], dtype=np.int64))
        else:
            raise ValueError(f"unsupported geometry locator kind: {kind!r}")
        if not len(points) or np.any(points < 0):
            raise ValueError("resolved geometry is empty or contains negative indices")
        return points


def _audit_row(row: dict, resolver: GeometryResolver, class_count: int) -> Counter:
    counts = Counter()
    scene = str(row["scene_name"])
    digest = str(row["geometry_hash"])
    if row.get("ground_truth_usage") != "none" or row.get("ap_computed") is not False:
        raise ValueError(f"{scene}/{digest}: row violates no-GT/no-AP contract")
    members = list(row["members"])
    if not members or int(row["member_count"]) != len(members):
        raise ValueError(f"{scene}/{digest}: member count mismatch")
    if int(row["point_count"]) <= 0:
        raise ValueError(f"{scene}/{digest}: invalid point count")
    sources = []
    for member in members:
        source = str(member["candidate_source"])
        if source not in SOURCE_RANK:
            raise ValueError(f"{scene}/{digest}: invalid source {source!r}")
        class_index = int(member["frozen_class_index"])
        if class_index < -1 or class_index > class_count:
            raise ValueError(f"{scene}/{digest}: class outside frozen class space")
        if bool(member["frozen_class_valid"]) != (0 <= class_index < class_count):
            raise ValueError(f"{scene}/{digest}: class-valid flag mismatch")
        points = resolver.points(member["geometry_locator"])
        if len(points) != int(member["point_count"]) or len(points) != int(row["point_count"]):
            raise ValueError(f"{scene}/{digest}: locator point-count mismatch")
        if geometry_hash(points) != digest or str(member["geometry_hash"]) != digest:
            raise ValueError(f"{scene}/{digest}: locator geometry hash mismatch")
        sources.append(source)
        counts[f"member_source::{source}"] += 1
        counts["invalid_class_member"] += int(not member["frozen_class_valid"])

    expected_sources = sorted(set(sources), key=SOURCE_RANK.get)
    if list(row["member_sources"]) != expected_sources:
        raise ValueError(f"{scene}/{digest}: member source list mismatch")
    scores = np.asarray([float(member["frozen_score"]) for member in members])
    ranks = np.asarray([SOURCE_RANK[source] for source in sources], dtype=np.int64)
    candidate_ids = np.asarray([int(member["candidate_id"]) for member in members], dtype=np.int64)
    canonical_index = canonical_member(list(range(len(members))), scores, ranks, candidate_ids)
    canonical = members[canonical_index]
    expected = {
        "canonical_member_index": canonical_index,
        "canonical_candidate_source": str(canonical["candidate_source"]),
        "canonical_candidate_id": int(canonical["candidate_id"]),
        "canonical_frozen_class_index": int(canonical["frozen_class_index"]),
        "canonical_frozen_class_valid": bool(canonical["frozen_class_valid"]),
        "canonical_frozen_score": float(canonical["frozen_score"]),
        "canonical_geometry_locator": canonical["geometry_locator"],
    }
    mismatch = {key: [value, row.get(key)] for key, value in expected.items() if row.get(key) != value}
    if mismatch:
        raise ValueError(f"{scene}/{digest}: canonical fields mismatch: {mismatch}")
    counts[f"canonical_source::{canonical['candidate_source']}"] += 1
    counts["invalid_canonical_class"] += int(not canonical["frozen_class_valid"])
    counts["cross_source_duplicate"] += int(len(expected_sources) > 1)
    counts["duplicate_member"] += len(members) - 1
    counts["member"] += len(members)
    counts["geometry"] += 1
    return counts


def run(args: argparse.Namespace) -> dict:
    args.ledger_root = _resolve(args.ledger_root)
    args.scene_list = _resolve(args.scene_list)
    args.output_dir = _resolve(args.output_dir)
    scenes = _read_scenes(args.scene_list)
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError(
            f"scene count {len(scenes)} differs from expected {args.expected_scene_count}"
        )
    input_summary_path = args.ledger_root / "summary.json"
    ledger_path = args.ledger_root / "unique_geometry_ledger.jsonl"
    input_summary = json.loads(input_summary_path.read_text())
    required_summary = {
        "version": "dm_sms1_unique_geometry_ledger_v1",
        "scene_count": len(scenes),
        "duplicate_geometry_output_count": 0,
        "contract_valid": True,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "embedding_computed": False,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "class_mutation": False,
        "score_mutation": False,
    }
    mismatch = {
        key: [expected, input_summary.get(key)]
        for key, expected in required_summary.items()
        if input_summary.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"input ledger summary contract mismatch: {mismatch}")

    rows = _read_jsonl(ledger_path)
    order = [(str(row["scene_name"]), str(row["geometry_hash"])) for row in rows]
    if order != sorted(order) or len(order) != len(set(order)):
        raise ValueError("ledger is not uniquely ordered by scene_name, geometry_hash")
    if {scene for scene, _ in order} != set(scenes):
        raise ValueError("ledger scene coverage differs from frozen scene list")
    resolver = GeometryResolver()
    counts = Counter()
    for row in rows:
        counts.update(_audit_row(row, resolver, args.class_count))

    source_member_counts = {
        source: int(counts[f"member_source::{source}"])
        for source in SOURCE_ORDER if counts[f"member_source::{source}"]
    }
    canonical_source_counts = {
        source: int(counts[f"canonical_source::{source}"])
        for source in SOURCE_ORDER if counts[f"canonical_source::{source}"]
    }
    derived = {
        "member_count": int(counts["member"]),
        "unique_geometry_count": int(counts["geometry"]),
        "duplicate_member_count": int(counts["duplicate_member"]),
        "cross_source_duplicate_geometry_count": int(counts["cross_source_duplicate"]),
        "invalid_frozen_class_member_count": int(counts["invalid_class_member"]),
        "invalid_canonical_frozen_class_count": int(counts["invalid_canonical_class"]),
        "source_member_counts": source_member_counts,
        "canonical_source_counts": canonical_source_counts,
    }
    summary_mismatch = {
        key: [value, input_summary.get(key)]
        for key, value in derived.items() if input_summary.get(key) != value
    }
    if summary_mismatch:
        raise ValueError(f"ledger aggregate mismatch: {summary_mismatch}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = {
        "version": "dm_sms1_unique_geometry_audit_v1",
        "audit_valid": True,
        "scene_count": len(scenes),
        **derived,
        "duplicate_geometry_output_count": 0,
        "locator_reconstruction_error_count": 0,
        "canonical_rule_error_count": 0,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "embedding_computed": False,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "class_mutation": False,
        "score_mutation": False,
        "input_provenance": {
            "scene_list": str(args.scene_list),
            "scene_list_sha256": _sha256(args.scene_list),
            "ledger_path": str(ledger_path),
            "ledger_sha256": _sha256(ledger_path),
            "ledger_summary_path": str(input_summary_path),
            "ledger_summary_sha256": _sha256(input_summary_path),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/diagnostics/dm_sms1_unique_geometry_audit_ncs_train100_20260817"),
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--class-count", type=int, default=198)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
