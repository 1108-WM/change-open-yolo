#!/usr/bin/env python3
"""记录自动 SAM Pareto 候选之间的无 GT 竞争关系，不做筛选或融合。"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path):
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("场景列表含重复项")
    return values


def _iou_and_coverages(left, right):
    intersection = np.intersect1d(left, right, assume_unique=True).size
    return (
        float(intersection / max(1, len(left) + len(right) - intersection)),
        float(intersection / max(1, len(left))),
        float(intersection / max(1, len(right))),
    )


def build_scene_records(candidates, scene_root):
    """候选的同类重复和跨类几何冲突均只作为连续关系写入。"""
    points = [np.unique(np.asarray(np.load(scene_root / item["seed_points_path"])["point_indices"], dtype=np.int64)) for item in candidates]
    relations = [[] for _ in candidates]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            iou, left_coverage, right_coverage = _iou_and_coverages(points[left], points[right])
            if iou == 0.0:
                continue
            same_class = int(candidates[left]["class_id"]) == int(candidates[right]["class_id"])
            relations[left].append((right, same_class, iou, left_coverage, right_coverage))
            relations[right].append((left, same_class, iou, right_coverage, left_coverage))
    records = []
    for index, candidate in enumerate(candidates):
        same = [item for item in relations[index] if item[1]]
        conflict = [item for item in relations[index] if not item[1]]
        best_same = max(same, key=lambda item: (item[2], item[3], -item[0]), default=None)
        best_conflict = max(conflict, key=lambda item: (item[2], item[3], -item[0]), default=None)
        native = candidate.get("native_relation_diagnostic", {})
        records.append({
            "scene_name": candidate["scene_name"],
            "candidate_id": int(candidate["candidate_id"]),
            "source_track_id": int(candidate["source_track_id"]),
            "class_id": int(candidate["class_id"]),
            "class_name": candidate["class_name"],
            "selected_variant_id": candidate["selected_variant_id"],
            "gvc_score": float(candidate["gvc_score"]),
            "semantic_vote_margin": float(candidate["semantic_vote_margin"]),
            "semantic_normalized_entropy": float(candidate["semantic_normalized_entropy"]),
            "num_seed_points": int(candidate["num_seed_points"]),
            "same_class_relation_count": len(same),
            "same_class_best_iou": float(best_same[2]) if best_same else 0.0,
            "same_class_best_self_coverage": float(best_same[3]) if best_same else 0.0,
            "same_class_best_candidate_id": int(candidates[best_same[0]]["candidate_id"]) if best_same else -1,
            "cross_class_relation_count": len(conflict),
            "cross_class_best_iou": float(best_conflict[2]) if best_conflict else 0.0,
            "cross_class_best_self_coverage": float(best_conflict[3]) if best_conflict else 0.0,
            "cross_class_best_candidate_id": int(candidates[best_conflict[0]]["candidate_id"]) if best_conflict else -1,
            "cross_class_best_class_id": int(candidates[best_conflict[0]]["class_id"]) if best_conflict else -1,
            "native_top_iou": float(native.get("native_top_iou", 0.0)),
            "variant_inside_top_native_ratio": float(native.get("variant_inside_top_native_ratio", 0.0)),
            "decision_state": "仅记录跨轨迹连续竞争关系；不按阈值保留、删除、融合或改写 native。",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "candidate_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    all_records = []
    for index, scene in enumerate(_scenes(args.scene_list), start=1):
        scene_root = args.candidate_root / scene
        candidates = json.loads((scene_root / "backprojection_candidates.json").read_text())["candidates"]
        records = build_scene_records(candidates, scene_root)
        target = args.output_root / scene
        target.mkdir()
        (target / "automatic_sam_pareto_candidate_competition_ledger.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        all_records.extend(records)
        print(f"[场景完成] {index}: {scene}，候选 {len(records)}", flush=True)
    summary = {
        "gt_usage": "none",
        "scene_count": len(_scenes(args.scene_list)),
        "candidate_count": len(all_records),
        "same_class_iou_ge_050_count": sum(item["same_class_best_iou"] >= .5 for item in all_records),
        "cross_class_iou_ge_050_count": sum(item["cross_class_best_iou"] >= .5 for item in all_records),
        "native_inside_ratio_ge_070_count": sum(item["variant_inside_top_native_ratio"] >= .7 for item in all_records),
        "decision_state": "无 GT 竞争关系账本，不产生筛选、融合或最终预测。",
    }
    (args.output_root / "automatic_sam_pareto_candidate_competition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
