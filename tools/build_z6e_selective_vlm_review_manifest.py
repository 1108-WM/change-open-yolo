#!/usr/bin/env python3
"""Build a frozen, GT-free budgeted manifest for selective multi-view VLM review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "improvement_gated_semantic_only"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _candidate_name(review: dict, class_index: int) -> str:
    matches = [
        str(row["class_name"]) for row in review["candidate_classes"]
        if int(row["class_index"]) == class_index
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate class {class_index} missing or duplicated")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-root", type=Path,
        default=Path("docs/diagnostics/z6c_semantic_review_input_official100_20260812"),
    )
    parser.add_argument(
        "--view-root", type=Path,
        default=Path("docs/diagnostics/z6b_object_view_manifest_official100_20260812"),
    )
    parser.add_argument(
        "--selector-root", type=Path,
        default=Path("docs/diagnostics/z6d_nested_improvement_gate_oof_official100_20260812"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/diagnostics/z6e_selective_vlm_review_manifest_official100_20260812"),
    )
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--min-pairwise-cosine-mean", type=float, default=0.855297)
    parser.add_argument("--safety60-transfer", action="store_true")
    args = parser.parse_args()
    for name in ("review_root", "view_root", "selector_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    reviews = {
        str(row["semantic_evidence_node_key"]): row
        for row in _read_jsonl(args.review_root / "semantic_review_inputs.jsonl")
    }
    views = {
        str(row["semantic_evidence_node_key"]): row
        for row in _read_jsonl(args.view_root / "object_view_manifest.jsonl")
    }
    eligible = []
    seen_nodes = set()
    for row in _read_jsonl(args.selector_root / "oof_selections.jsonl"):
        decision = row["selectors"][SYSTEM]
        node_key = str(row["semantic_evidence_node_key"])
        review = reviews[node_key]
        view = views[node_key]
        current_class = int(row["current_class_index"])
        proposed_class = int(decision["proposed_class_index"])
        if (
            node_key in seen_nodes
            or decision["gate_state"] != "accepted_new_class"
            or current_class == proposed_class
            or not bool(review["yolo_alpha_top1_disagreement"])
            or str(review["dino_embedding_state"]) != "available"
            or int(review["view_count"]) != 3
            or float(review["dino_pairwise_cosine_mean"]) < args.min_pairwise_cosine_mean
            or int(view["view_count"]) != 3
        ):
            continue
        seen_nodes.add(node_key)
        eligible.append({
            "scene_name": str(row["scene_name"]),
            "prediction_index": int(row["prediction_index"]),
            "semantic_evidence_node_key": node_key,
            "candidate_source": str(row["candidate_source"]),
            "candidate_id": int(row["candidate_id"]),
            "current_class_index": current_class,
            "current_class_name": str(review["representative_current_class_name"]),
            "proposed_class_index": proposed_class,
            "proposed_class_name": _candidate_name(review, proposed_class),
            "representative_current_score": float(review["representative_current_score"]),
            "selector_improve_probability": float(decision["gate_improve_probability"]),
            "selector_margin": float(decision["selector_margin"]),
            "yolo_top1_class_index": int(review["yolo_top1_class_index"]),
            "alpha_top1_class_index": int(review["alpha_top1_class_index"]),
            "dino_pairwise_cosine_mean": float(review["dino_pairwise_cosine_mean"]),
            "dino_dispersion": float(review["dino_dispersion"]),
            "views": [{
                "view_rank": int(item["view_rank"]),
                "rgb_path": str(item["rgb_path"]),
                "bbox_xyxy": [int(value) for value in item["bbox_xyxy"]],
                "visible_point_count": int(item["visible_point_count"]),
            } for item in view["views"]],
            "allowed_outputs": ["CURRENT", "PROPOSED", "ABSTAIN"],
            "fallback": "CURRENT",
        })
    eligible.sort(key=lambda row: (
        -float(row["representative_current_score"]),
        -float(row["selector_improve_probability"]),
        str(row["scene_name"]), int(row["prediction_index"]),
    ))
    selected = eligible[:args.budget]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "review_manifest.jsonl"
    with output_path.open("w") as handle:
        for review_index, row in enumerate(selected):
            handle.write(json.dumps({"review_index": review_index, **row}, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    payload = {
        "diagnostic_type": "frozen budgeted selective multi-view VLM review manifest",
        "eligible_count": len(eligible), "selected_count": len(selected), "maximum_budget": args.budget,
        "scene_count": len({row["scene_name"] for row in selected}),
        "selector_system": SYSTEM,
        "routing_contract": {
            "selector_state": "accepted_new_class",
            "current_hypothesis": "prediction-specific current class; no result propagation to sibling predictions",
            "semantic_conflict": "YOLO top1 != Alpha top1",
            "views": "exact frozen top-3 views",
            "appearance_consistency": f"DINO pairwise cosine mean >= frozen ledger median {args.min_pairwise_cosine_mean}",
            "ordering": "descending representative current hybrid score, then selector probability, then stable keys",
            "node_cap": 1,
        },
        "decision_contract": "VLM must output CURRENT, PROPOSED, or ABSTAIN; only PROPOSED mutates class; invalid/ABSTAIN fall back to CURRENT",
        "review_manifest_sha256": digest,
        "ground_truth_usage": "none", "model_invoked": False,
        "candidate_mutation": False, "geometry_mutation": False, "class_mutation": False,
        "score_mutation": False, "inference_plan_written": False,
        "safety60_read": bool(args.safety60_transfer), "even48_read": False, "test60_read": False,
        "params": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
