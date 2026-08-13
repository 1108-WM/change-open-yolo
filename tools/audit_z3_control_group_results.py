#!/usr/bin/env python3
"""Audit completed official100 Z3 summaries and cross-run invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z0-summary", type=Path, required=True)
    parser.add_argument("--yolo-summary", type=Path, required=True)
    parser.add_argument("--alpha-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    z0 = _load(args.z0_summary)
    yolo = _load(args.yolo_summary)
    alpha = _load(args.alpha_summary)
    z0_pair = z0["sources"]["pair_union"]["evaluations"]["current_class_current_score"]
    yolo_frozen = yolo["variants"]["frozen_current"]["sources"]["pair_union"]
    alpha_context = alpha["variants"]["frozen_context_equal_top1"]["sources"]["pair_union"]

    yolo_variants = yolo["variants"]
    yolo_native = [variant["sources"]["native_only"] for variant in yolo_variants.values()]
    native_invariant = all(row == yolo_native[0] for row in yolo_native[1:])
    alpha_native = alpha["native_only_reference"]
    alpha_native_matches_z0 = all(
        _close(alpha_native[key], z0["sources"]["native_only"]["evaluations"]["current_class_current_score"][key])
        for key in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
    )
    frozen_matches_z0 = all(
        _close(yolo_frozen[key], z0_pair[key])
        for key in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
    )
    context_delta = {
        key: float(alpha_context[key] - z0_pair[key])
        for key in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
    }
    payload = {
        "audit_contract": "official100 explicit GT evaluator summaries only; no new tuning or training",
        "valid": bool(
            yolo.get("diagnostic_only")
            and alpha.get("diagnostic_only")
            and yolo.get("geometry_unchanged")
            and alpha.get("geometry_unchanged")
            and yolo.get("candidate_membership_unchanged")
            and alpha.get("candidate_membership_unchanged")
            and native_invariant
            and alpha_native_matches_z0
            and frozen_matches_z0
        ),
        "yolo_variant_count": len(yolo_variants),
        "yolo_csv_count": len(list(args.yolo_summary.parent.glob("*.csv"))),
        "alpha_variant_count": len(alpha["variants"]),
        "alpha_csv_count": len(list(args.alpha_summary.parent.glob("*.csv"))),
        "native_only_invariant_across_yolo_variants": native_invariant,
        "alpha_native_reference_matches_z0": alpha_native_matches_z0,
        "yolo_frozen_current_matches_z0": frozen_matches_z0,
        "z0_pair_union": z0_pair,
        "yolo_frozen_pair_union": yolo_frozen,
        "best_fixed_alpha_variant": "frozen_context_equal_top1",
        "best_fixed_alpha_pair_union": alpha_context,
        "best_fixed_alpha_delta_vs_z0": context_delta,
        "errors": [],
    }
    if payload["yolo_variant_count"] != 7:
        payload["errors"].append("unexpected YOLO variant count")
    if payload["yolo_csv_count"] != 28:
        payload["errors"].append("unexpected YOLO CSV count")
    if payload["alpha_variant_count"] != 6:
        payload["errors"].append("unexpected Alpha variant count")
    if payload["alpha_csv_count"] != 19:
        payload["errors"].append("unexpected Alpha CSV count")
    payload["valid"] = payload["valid"] and not payload["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
