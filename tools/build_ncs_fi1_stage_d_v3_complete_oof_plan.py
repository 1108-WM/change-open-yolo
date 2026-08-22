#!/usr/bin/env python3
"""Build the frozen complete control/challenger plan for the single official100 AP run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import _read_jsonl, _resolve, _sha256  # noqa: E402

VERSION = "ncs_fi1_stage_d_v3_complete_oof_plan_v1"


def run(args: argparse.Namespace) -> dict:
    for name in ("unique_geometry_root", "stage_b_root", "stage_d_v3_dataset_root", "stage_d_v3_root", "stage_d_v3_audit_root", "preregistration", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    model_audit = json.loads((args.stage_d_v3_audit_root / "summary.json").read_text())
    if model_audit.get("audit_valid") is not True or model_audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        raise ValueError("stage-D-v3 model audit did not authorize the AP plan")
    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    stage_b = {str(row["geometry_key"]): row for row in _read_jsonl(args.stage_b_root / "stage_b_rerank_plan.jsonl")}
    dataset = {str(row["candidate_key"]): row for row in _read_jsonl(args.stage_d_v3_dataset_root / "full_rank_marginal_gain_dataset.jsonl")}
    scores = {str(row["candidate_key"]): row for row in _read_jsonl(args.stage_d_v3_root / "stage_d_v3_append_score_plan.jsonl")}
    expected_unique = len(nodes)
    expected_baseline = sum(str(row["canonical_candidate_source"]) in {"native", "track"} for row in nodes)
    expected_original = sum(row["candidate_variant"] == "original" for row in dataset.values())
    expected_refined = sum(row["candidate_variant"] == "refined" for row in dataset.values())
    if len(dataset) != expected_original + expected_refined or set(dataset) != set(scores):
        raise ValueError("complete-plan input coverage differs from preregistration")
    original_by_geometry = {
        str(row["original_union_geometry_key"]): row
        for row in dataset.values() if row["candidate_variant"] == "original"
    }
    if len(original_by_geometry) != expected_original:
        raise ValueError("original union geometry mapping is incomplete")
    if expected_unique != expected_baseline + expected_original:
        raise ValueError("unique geometry source partition differs from complete-plan inventory")

    plan = []
    source_counts = Counter()
    for node in nodes:
        source = str(node["canonical_candidate_source"])
        geometry_key = str(node["geometry_key"])
        if source in {"native", "track"}:
            challenge_score = float(stage_b[geometry_key]["stage_b_score"])
            score_source = "stage_b_oof"
        elif source == "pair_union":
            candidate = original_by_geometry[geometry_key]
            challenge_score = float(scores[str(candidate["candidate_key"])]["stage_d_v3_append_score"])
            score_source = "stage_d_v3_original_union_oof"
        else:
            raise ValueError(f"unexpected canonical source: {source}")
        plan.append({
            "plan_key": geometry_key, "scene_name": str(node["scene_name"]),
            "fold_index": int(stage_b[geometry_key]["fold_index"]),
            "candidate_source": source, "geometry_key": geometry_key,
            "geometry_locator_read_only": node["canonical_geometry_locator"],
            "geometry_digest": str(node["geometry_hash"]), "geometry_digest_algorithm": "sha1_point_indices",
            "point_count": int(node["point_count"]),
            "frozen_class_index": int(node["canonical_frozen_class_index"]),
            "control_score": float(node["canonical_frozen_score"]),
            "challenger_score": challenge_score, "challenger_score_source": score_source,
            "candidate_retained": True, "candidate_deletion": False, "geometry_mutation": False,
            "class_mutation": False, "append_only": False,
        })
        source_counts[source] += 1
    for row in dataset.values():
        if row["candidate_variant"] != "refined":
            continue
        original_node = next(node for node in nodes if str(node["geometry_key"]) == str(row["original_union_geometry_key"]))
        score_row = scores[str(row["candidate_key"])]
        plan.append({
            "plan_key": str(row["candidate_key"]), "scene_name": str(row["scene_name"]),
            "fold_index": int(row["fold_index"]), "candidate_source": "refined_union",
            "geometry_key": None, "original_union_geometry_key": str(row["original_union_geometry_key"]),
            "geometry_locator_read_only": row["geometry_locator_read_only"],
            "geometry_digest": str(row["geometry_sha256"]), "geometry_digest_algorithm": "sha256_point_indices",
            "point_count": int(row["point_count"]),
            "frozen_class_index": int(original_node["canonical_frozen_class_index"]),
            "class_inheritance": "original_union_canonical_frozen_class_index",
            "control_score": None,
            "challenger_score": float(score_row["stage_d_v3_append_score"]),
            "challenger_score_source": "stage_d_v3_refined_union_oof",
            "candidate_retained": True, "candidate_deletion": False, "geometry_mutation": False,
            "class_mutation": False, "append_only": True,
        })
        source_counts["refined_union"] += 1
    plan.sort(key=lambda row: (row["scene_name"], row["plan_key"]))
    if len({row["plan_key"] for row in plan}) != len(plan):
        raise ValueError("duplicate complete-plan key")
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        plan_path = staging / "complete_oof_ap_plan.jsonl"
        plan_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in plan))
        summary = {
            "version": VERSION, "scene_count": len({row["scene_name"] for row in plan}),
            "control_candidate_count": expected_unique,
            "challenger_candidate_count": expected_unique + expected_refined,
            "baseline_candidate_count": expected_baseline,
            "original_union_candidate_count": expected_original,
            "refined_union_candidate_count": expected_refined,
            "source_counts": dict(sorted(source_counts.items())),
            "files": {"plan": plan_path.name}, "hashes": {"plan": _sha256(plan_path)},
            "contract": (
                f"control={expected_unique} canonical frozen; challenger=all retained with "
                f"stage-B native/track, D-v3 original, plus {expected_refined} D-v3 refined"
            ),
            "candidate_deletion_count": 0, "geometry_mutation": False, "class_mutation": False,
            "ap_computed": False, "validation60_read": False, "val312_read": False,
            "input_provenance": {
                "preregistration_sha256": _sha256(args.preregistration),
                "unique_geometry_summary_sha256": _sha256(args.unique_geometry_root / "summary.json"),
                "stage_b_summary_sha256": _sha256(args.stage_b_root / "summary.json"),
                "stage_d_v3_dataset_summary_sha256": _sha256(args.stage_d_v3_dataset_root / "summary.json"),
                "stage_d_v3_summary_sha256": _sha256(args.stage_d_v3_root / "summary.json"),
                "stage_d_v3_audit_summary_sha256": _sha256(args.stage_d_v3_audit_root / "summary.json"),
            },
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--stage-b-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_20260822"))
    parser.add_argument("--stage-d-v3-dataset-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_d_v3_full_rank_marginal_dataset_train100_20260822"))
    parser.add_argument("--stage-d-v3-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_d_v3_full_rank_marginal_oof_train100_20260822"))
    parser.add_argument("--stage-d-v3-audit-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_d_v3_full_rank_marginal_oof_train100_audit_20260822"))
    parser.add_argument("--preregistration", type=Path, default=Path("docs/NCS_FI1_STAGE_D_V3_PREREGISTRATION_20260822.md"))
    parser.add_argument("--output-root", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
