#!/usr/bin/env python3
"""Audit a Z2 Alpha-CLIP track ledger without reading ground truth."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _track_path(root: Path, scene: str) -> Path:
    choices = (
        root / scene / "automatic_tracks.json",
        root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json",
    )
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError(f"{scene}: automatic_tracks.json not found under {root}")


def _finite(values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _audit_record(record: dict, class_count: int, max_views: int) -> list[str]:
    key = f"{record.get('scene_name')}:{record.get('track_id')}"
    errors = []
    views = record.get("views", [])
    class_index = int(record.get("alphaclip_class_index", -1))
    logits = record.get("clip_logits", [])
    probability = float(record.get("alphaclip_top_probability", 0.0))
    margin = float(record.get("alphaclip_logit_margin", 0.0))
    if len(views) > max_views:
        errors.append(f"{key}: view count {len(views)} exceeds {max_views}")
    if class_index < 0:
        if views or logits or probability != 0.0 or margin != 0.0:
            errors.append(f"{key}: empty semantic record violates zero/empty contract")
        return errors
    if not views:
        errors.append(f"{key}: semantic record has no views")
    if not 0 <= class_index < class_count:
        errors.append(f"{key}: class index {class_index} is out of range")
    if len(logits) != class_count or not _finite(logits):
        errors.append(f"{key}: aggregate logits are not {class_count} finite values")
    elif 0 <= class_index < class_count:
        logit_array = np.asarray(logits, dtype=np.float64)
        if not np.isclose(logit_array[class_index], logit_array.max(), rtol=0.0, atol=1e-12):
            errors.append(f"{key}: aggregate class is not in the maximum-logit tie set")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        errors.append(f"{key}: invalid top probability {probability}")
    if not math.isfinite(margin) or margin < -1e-7:
        errors.append(f"{key}: invalid logit margin {margin}")
    for view_index, view in enumerate(views):
        view_logits = view.get("clip_logits", [])
        view_top = int(view.get("clip_top_class_id", -1))
        if len(view_logits) != class_count or not _finite(view_logits):
            errors.append(f"{key}: view {view_index} logits are not {class_count} finite values")
        elif int(np.argmax(np.asarray(view_logits, dtype=np.float64))) != view_top:
            errors.append(f"{key}: view {view_index} top-1 disagrees with logits")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--ledger_root", type=Path, required=True)
    parser.add_argument("--expected_class_count", type=int, default=198)
    parser.add_argument("--max_views", type=int, default=3)
    parser.add_argument("--expected_crop_padding_ratio", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "ledger_root", "output"):
        setattr(args, name, _resolve(getattr(args, name)))

    scenes = _read_scenes(args.scene_list)
    expected_keys = set()
    expected_rows = {}
    for scene in scenes:
        tracks = json.loads(_track_path(args.track_root, scene).read_text()).get("tracks", [])
        for track in tracks:
            key = (scene, int(track["track_id"]))
            if key in expected_keys:
                raise ValueError(f"duplicate source track key: {scene}:{track['track_id']}")
            expected_keys.add(key)
            expected_rows[key] = track

    observed_keys = set()
    records = []
    errors = []
    for scene in scenes:
        path = args.ledger_root / scene / "automatic_track_alphaclip_semantics.json"
        if not path.is_file():
            errors.append(f"missing scene ledger: {path}")
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            errors.append(f"{scene}: scene ledger is not a list")
            continue
        for record in rows:
            key = (str(record.get("scene_name")), int(record.get("track_id", -1)))
            if key in observed_keys:
                errors.append(f"duplicate ledger key: {key[0]}:{key[1]}")
            observed_keys.add(key)
            source = expected_rows.get(key)
            if source is None:
                errors.append(f"unexpected ledger key: {key[0]}:{key[1]}")
            else:
                for field in ("support_view_count", "point_count"):
                    if int(record.get(field, -1)) != int(source[field]):
                        errors.append(f"{key[0]}:{key[1]}: {field} disagrees with source")
            errors.extend(_audit_record(record, args.expected_class_count, args.max_views))
            if args.expected_crop_padding_ratio is not None and "crop_contract" in record:
                actual = float(record["crop_contract"].get("crop_padding_ratio", -1.0))
                if not math.isclose(actual, args.expected_crop_padding_ratio, rel_tol=0.0, abs_tol=1e-12):
                    errors.append(f"{key[0]}:{key[1]}: crop padding ratio {actual} disagrees with contract")
            records.append(record)

    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing:
        errors.append(f"missing {len(missing)} source track keys")
    if extra:
        errors.append(f"found {len(extra)} extra ledger keys")
    semantic = [record for record in records if int(record.get("alphaclip_class_index", -1)) >= 0]
    empty = [record for record in records if int(record.get("alphaclip_class_index", -1)) < 0]
    view_histogram = Counter(len(record.get("views", [])) for record in records)
    empty_point_counts = [int(record["point_count"]) for record in empty]
    empty_support_counts = [int(record["support_view_count"]) for record in empty]
    summary_path = args.ledger_root / "automatic_track_alphaclip_semantic_summary.json"
    source_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    if source_summary is None:
        errors.append("missing root semantic summary")
    else:
        if int(source_summary.get("track_count", -1)) != len(records):
            errors.append("root summary track_count disagrees with ledger")
        if int(source_summary.get("with_semantics_count", -1)) != len(semantic):
            errors.append("root summary with_semantics_count disagrees with ledger")

    payload = {
        "audit_contract": "no GT; source-track identity and Alpha-CLIP distribution integrity only",
        "valid": not errors,
        "scene_count": len(scenes),
        "expected_track_count": len(expected_keys),
        "ledger_track_count": len(records),
        "unique_track_count": len(observed_keys),
        "with_semantics_count": len(semantic),
        "without_semantics_count": len(empty),
        "semantic_coverage_percent": 100.0 * len(semantic) / len(records) if records else 0.0,
        "view_count_histogram": {str(key): value for key, value in sorted(view_histogram.items())},
        "empty_semantic_source_stats": {
            "point_count_min": min(empty_point_counts) if empty_point_counts else None,
            "point_count_median": float(np.median(empty_point_counts)) if empty_point_counts else None,
            "support_view_count_min": min(empty_support_counts) if empty_support_counts else None,
            "support_view_count_median": float(np.median(empty_support_counts)) if empty_support_counts else None,
        },
        "expected_class_count": args.expected_class_count,
        "expected_crop_padding_ratio": args.expected_crop_padding_ratio,
        "error_count": len(errors),
        "errors": errors[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
