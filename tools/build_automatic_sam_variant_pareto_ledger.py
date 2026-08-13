#!/usr/bin/env python3
"""为同一自动 SAM 轨迹的几何变体建立无权重 Pareto 竞争账本。

仅在语义证据最高类别相同的变体之间，按类别无关 GVC 较高、语义间隔较高、
语义熵较低判定严格支配。输出保留全部非支配变体，不选择最终候选或类别。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def dominates(left, right, epsilon=1e-12):
    """left 在三项连续证据上均不差且至少一项更优时，才严格支配 right。"""
    objectives = (
        (float(left["gvc_score"]), float(right["gvc_score"]), True),
        (float(left["semantic_vote_margin"]), float(right["semantic_vote_margin"]), True),
        (float(left["semantic_normalized_entropy"]), float(right["semantic_normalized_entropy"]), False),
    )
    no_worse = all(
        left_value >= right_value - epsilon if maximize else left_value <= right_value + epsilon
        for left_value, right_value, maximize in objectives
    )
    strictly_better = any(
        left_value > right_value + epsilon if maximize else left_value < right_value - epsilon
        for left_value, right_value, maximize in objectives
    )
    return bool(no_worse and strictly_better)


def _family_records(quality_rows, semantic_by_id):
    families = defaultdict(list)
    skipped = []
    for quality in quality_rows:
        semantic = semantic_by_id.get(str(quality["variant_id"]))
        if semantic is None:
            skipped.append({"variant_id": quality["variant_id"], "reason": "missing_semantic_evidence"})
            continue
        record = {
            "variant_id": str(quality["variant_id"]),
            "variant_type": str(quality["variant_type"]),
            "source_track_id": int(quality["source_track_id"]),
            "semantic_evidence_top_class_index": int(semantic["semantic_evidence_top_class_index"]),
            "gvc_score": float(quality["gvc_score"]),
            "semantic_vote_margin": float(semantic["semantic_vote_margin"]),
            "semantic_normalized_entropy": float(semantic["semantic_normalized_entropy"]),
            "native_top_candidate_id": int(quality["native_top_candidate_id"]),
            "native_top_iou": float(quality["native_top_iou"]),
            "variant_inside_top_native_ratio": float(quality["variant_inside_top_native_ratio"]),
            "variant_point_count": int(quality["variant_point_count"]),
        }
        families[int(record["source_track_id"])].append(record)
    return families, skipped


def build_pareto_records(quality_rows, semantic_rows):
    semantic_by_id = {str(row["variant_id"]): row for row in semantic_rows}
    families, skipped = _family_records(quality_rows, semantic_by_id)
    output = []
    for track_id, members in sorted(families.items()):
        base = next((item for item in members if item["variant_type"] == "seed_superpoint_closure"), None)
        by_semantic_evidence = defaultdict(list)
        for member in members:
            by_semantic_evidence[int(member["semantic_evidence_top_class_index"])].append(member)
        for top_class, group in sorted(by_semantic_evidence.items()):
            for member in sorted(group, key=lambda item: item["variant_id"]):
                dominators = sorted(
                    other["variant_id"] for other in group
                    if other["variant_id"] != member["variant_id"] and dominates(other, member)
                )
                output.append({
                    "source_track_id": track_id,
                    "variant_id": member["variant_id"],
                    "variant_type": member["variant_type"],
                    "semantic_evidence_top_class_index": top_class,
                    "same_semantic_evidence_family_size": len(group),
                    "pareto_non_dominated": not dominators,
                    "dominating_variant_ids": dominators,
                    "semantic_class_changed_from_seed": bool(
                        base is not None
                        and top_class >= 0
                        and int(base["semantic_evidence_top_class_index"]) >= 0
                        and top_class != int(base["semantic_evidence_top_class_index"])
                    ),
                    "objectives": {
                        "gvc_score": member["gvc_score"],
                        "semantic_vote_margin": member["semantic_vote_margin"],
                        "semantic_normalized_entropy": member["semantic_normalized_entropy"],
                    },
                    "native_relation": {
                        "native_top_candidate_id": member["native_top_candidate_id"],
                        "native_top_iou": member["native_top_iou"],
                        "variant_inside_top_native_ratio": member["variant_inside_top_native_ratio"],
                    },
                    "variant_point_count": member["variant_point_count"],
                    "decision_state": "仅标记同语义证据组内的 Pareto 关系；不选最终版本或类别。",
                })
    return output, skipped


def _build_scene(scene_name, args):
    quality_rows = json.loads((args.quality_ledger_root / scene_name / "automatic_sam_variant_quality_ledger.json").read_text())
    semantic_rows = json.loads((args.semantic_ledger_root / scene_name / "automatic_sam_variant_semantic_ledger.json").read_text())
    records, skipped = build_pareto_records(quality_rows, semantic_rows)
    scene_root = args.output_root / scene_name
    scene_root.mkdir(parents=True, exist_ok=False)
    (scene_root / "automatic_sam_variant_pareto_ledger.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "scene_name": scene_name,
        "variant_count": len(records),
        "pareto_non_dominated_count": sum(row["pareto_non_dominated"] for row in records),
        "dominated_count": sum(not row["pareto_non_dominated"] for row in records),
        "semantic_changed_from_seed_count": sum(row["semantic_class_changed_from_seed"] for row in records),
        "skipped_count": len(skipped),
        "decision_state": "不读取 GT；不选最终版本或类别，不输出候选或最终分数。",
    }
    (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--quality_ledger_root", type=Path, required=True)
    parser.add_argument("--semantic_ledger_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "quality_ledger_root", "semantic_ledger_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"非支配 {summary['pareto_non_dominated_count']}/{summary['variant_count']}",
            flush=True,
        )
    payload = {
        "purpose": "在候选输出前保留自动 SAM 变体的无权重 Pareto 竞争关系。",
        "gt_usage": "不读取 GT；不选最终版本或类别，不输出候选或最终分数。",
        "scene_count": len(summaries),
        "variant_count": sum(item["variant_count"] for item in summaries),
        "pareto_non_dominated_count": sum(item["pareto_non_dominated_count"] for item in summaries),
        "dominated_count": sum(item["dominated_count"] for item in summaries),
        "params": vars(args),
    }
    (args.output_root / "automatic_sam_variant_pareto_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
