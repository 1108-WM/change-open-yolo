#!/usr/bin/env python3
"""Audit GT-free native/pair-union geometry Alpha-CLIP ledgers against Z1."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ("native", "pair_union")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _finite(values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _audit_alpha(record: dict, class_count: int, max_views: int) -> list[str]:
    key = str(record.get("semantic_evidence_node_key"))
    errors = []
    views = record.get("views", [])
    class_index = int(record.get("alphaclip_class_index", -1))
    logits = record.get("clip_logits", [])
    probability = float(record.get("alphaclip_top_probability", 0.0))
    margin = float(record.get("alphaclip_logit_margin", 0.0))
    if len(views) > max_views:
        errors.append(f"{key}: view count exceeds {max_views}")
    if class_index < 0:
        if views or logits or probability != 0.0 or margin != 0.0:
            errors.append(f"{key}: empty semantic record violates zero/empty contract")
        return errors
    if not views:
        errors.append(f"{key}: semantic record has no views")
    if not 0 <= class_index < class_count:
        errors.append(f"{key}: class index is out of range")
    if len(logits) != class_count or not _finite(logits):
        errors.append(f"{key}: aggregate logits are not {class_count} finite values")
    elif 0 <= class_index < class_count:
        values = np.asarray(logits, dtype=np.float64)
        if not np.isclose(values[class_index], values.max(), rtol=0.0, atol=1e-12):
            errors.append(f"{key}: aggregate class is not in maximum-logit tie set")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        errors.append(f"{key}: invalid top probability")
    if not math.isfinite(margin) or margin < -1e-7:
        errors.append(f"{key}: invalid logit margin")
    for view_index, view in enumerate(views):
        view_logits = view.get("clip_logits", [])
        if len(view_logits) != class_count or not _finite(view_logits):
            errors.append(f"{key}: view {view_index} logits are invalid")
        elif int(np.argmax(np.asarray(view_logits))) != int(view.get("clip_top_class_id", -1)):
            errors.append(f"{key}: view {view_index} top1 disagrees with logits")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "z1_root", "ledger_root", "output"):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    scene_set = set(scenes)

    expected = defaultdict(list)
    with (args.z1_root / "candidate_bindings.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["scene_name"]) in scene_set and str(row["candidate_source"]) in SOURCE_NAMES:
                expected[str(row["semantic_evidence_node_key"])].append(row)

    errors = []
    records = []
    observed = set()
    for scene in scenes:
        path = args.ledger_root / scene / "geometry_node_alphaclip_semantics.json"
        if not path.is_file():
            errors.append(f"missing scene ledger: {path}")
            continue
        for record in json.loads(path.read_text()):
            key = str(record.get("semantic_evidence_node_key"))
            if key in observed:
                errors.append(f"duplicate ledger evidence key: {key}")
            observed.add(key)
            rows = expected.get(key)
            if not rows:
                errors.append(f"unexpected ledger evidence key: {key}")
            else:
                source = str(record.get("candidate_source"))
                if {str(row["candidate_source"]) for row in rows} != {source}:
                    errors.append(f"{key}: candidate source disagrees with Z1")
                if int(record.get("bound_candidate_count", -1)) != len(rows):
                    errors.append(f"{key}: bound candidate count disagrees with Z1")
                if str(record.get("geometry_hash")) != str(rows[0]["geometry_hash"]):
                    errors.append(f"{key}: geometry hash disagrees with Z1")
            contract = record.get("crop_contract", {})
            if contract.get("crop_mode") != "limited_context" or not math.isclose(
                float(contract.get("crop_padding_ratio", -1.0)), 0.50, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(f"{key}: limited-context crop contract is invalid")
            errors.extend(_audit_alpha(record, args.class_count, args.max_views))
            records.append(record)

    missing = set(expected) - observed
    extra = observed - set(expected)
    if missing:
        errors.append(f"missing {len(missing)} expected Z1 evidence keys")
    if extra:
        errors.append(f"found {len(extra)} extra evidence keys")
    source_counts = Counter(str(row["candidate_source"]) for row in records)
    semantic_counts = Counter(
        str(row["candidate_source"])
        for row in records if int(row.get("alphaclip_class_index", -1)) >= 0
    )
    view_histogram = Counter(len(row.get("views", [])) for row in records)
    summary_path = args.ledger_root / "summary.json"
    root_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    if root_summary is None:
        errors.append("missing root summary")
    elif int(root_summary.get("record_count", -1)) != len(records):
        errors.append("root summary record_count disagrees with ledger")
    payload = {
        "audit_contract": "no GT; Z1 evidence identity and Alpha-CLIP distribution integrity",
        "valid": not errors,
        "scene_count": len(scenes),
        "expected_evidence_node_count": len(expected),
        "ledger_record_count": len(records),
        "unique_evidence_node_count": len(observed),
        "source_record_counts": dict(sorted(source_counts.items())),
        "source_with_semantics_counts": dict(sorted(semantic_counts.items())),
        "view_count_histogram": {str(key): value for key, value in sorted(view_histogram.items())},
        "class_count": args.class_count,
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
