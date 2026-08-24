#!/usr/bin/env python3
"""Materialize the no-GT FI1-D-v3 frozen semantic prediction cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
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


def _scenes(path: Path) -> list[str]:
    values = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("scene list is empty or contains duplicates")
    return values


def _prepared_point_count(root: Path, scene: str) -> int:
    stem = scene[len("scene"):] if scene.startswith("scene") else scene
    path = root / scene / f"{stem}.npy"
    return int(np.load(path, mmap_mode="r").shape[0])


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "ledger_root", "ledger_audit_root", "prepared_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list)
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError("scene count differs from the joint contract")
    ledger_summary = json.loads((args.ledger_root / "summary.json").read_text())
    ledger_audit = json.loads((args.ledger_audit_root / "summary.json").read_text())
    if (
        int(ledger_summary.get("scene_count", -1)) != len(scenes)
        or ledger_summary.get("contract_valid") is not True
        or ledger_summary.get("ground_truth_read") is not False
        or ledger_summary.get("ap_computed") is not False
        or ledger_audit.get("audit_valid") is not True
        or int(ledger_audit.get("error_count", -1)) != 0
    ):
        raise ValueError("joint unique geometry ledger is not fully audited")
    ledger_path = args.ledger_root / "unique_geometry_ledger.jsonl"
    by_scene = defaultdict(list)
    for row in _rows(ledger_path):
        by_scene[str(row["scene_name"])].append(row)
    if set(by_scene) != set(scenes):
        raise ValueError("joint unique geometry scene coverage differs")
    for scene in scenes:
        by_scene[scene].sort(key=lambda row: str(row["geometry_hash"]))

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    (staging / "prediction_cache").mkdir(parents=True)
    resolver = GeometryResolver()
    scene_summaries = []
    try:
        for index, scene in enumerate(scenes, 1):
            rows = by_scene[scene]
            point_count = _prepared_point_count(args.prepared_root, scene)
            root = staging / "prediction_cache" / scene
            root.mkdir()
            masks = np.zeros((point_count, len(rows)), dtype=bool)
            classes = np.empty(len(rows), dtype=np.int64)
            scores = np.empty(len(rows), dtype=np.float32)
            hashes, plan_keys, sources = [], [], []
            for column, row in enumerate(rows):
                points = resolver.points(row["canonical_geometry_locator"])
                if (
                    len(points) != int(row["point_count"])
                    or points[-1] >= point_count
                    or geometry_hash(points) != str(row["geometry_hash"])
                ):
                    raise ValueError(f"{scene}/{row['geometry_hash']}: geometry cache mismatch")
                masks[points, column] = True
                classes[column] = int(row["canonical_frozen_class_index"])
                scores[column] = float(row["canonical_frozen_score"])
                hashes.append(str(row["geometry_hash"]))
                plan_keys.append(str(row["fi1_d_v3_plan_key"]))
                sources.append(str(row["canonical_candidate_source"]))
            np.save(root / "masks.npy", masks, allow_pickle=False)
            np.save(root / "frozen_classes.npy", classes, allow_pickle=False)
            np.save(root / "frozen_scores.npy", scores, allow_pickle=False)
            (root / "geometry_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
            (root / "plan_keys.json").write_text(json.dumps(plan_keys, indent=2) + "\n")
            (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
            files = {
                name: _sha256(root / name) for name in (
                    "masks.npy", "frozen_classes.npy", "frozen_scores.npy",
                    "geometry_hashes.json", "plan_keys.json", "sources.json",
                )
            }
            scene_summary = {
                "scene_name": scene,
                "point_count": point_count,
                "geometry_count": len(rows),
                "file_sha256": files,
                "ground_truth_read": False,
                "ap_computed": False,
            }
            (root / "summary.json").write_text(
                json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            scene_summaries.append(scene_summary)
            print(f"[DM-SMS-1 D-v3 cache] {index}/{len(scenes)} {scene}: {len(rows)}", flush=True)
        summary = {
            "version": "dm_sms1_fi1_d_v3_prediction_cache_v1",
            "dataset": args.dataset_name,
            "scene_count": len(scenes),
            "geometry_count": sum(row["geometry_count"] for row in scene_summaries),
            "cache_valid": True,
            "ground_truth_usage": "none",
            "ground_truth_read": False,
            "ap_computed": False,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "score_mutation": False,
            "proposal_deletion": False,
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "unique_geometry_ledger_sha256": _sha256(ledger_path),
                "unique_geometry_audit_sha256": _sha256(args.ledger_audit_root / "summary.json"),
            },
            "scene_summaries": scene_summaries,
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
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--ledger-audit-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=312)
    parser.add_argument("--dataset-name", default="ScanNet200-val312")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
