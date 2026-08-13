#!/usr/bin/env python3
"""从自动 SAM 轨迹生长账本展开可审计的一跳几何变体计划。

每条轨迹先保留其原始 superpoint 闭包，再为每个具有跨视角正证据的相邻原子
记录一个独立的一跳扩张选项。输出不是最终候选 mask，不包含类别、分数或接受决定。
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def expand_track_variant_plan(track_record):
    """保留全部一跳动作，后续质量账本而非此处的阈值决定何者可进入竞争。"""
    track_id = int(track_record["track_id"])
    base_superpoints = sorted({int(item["superpoint_id"]) for item in track_record["supported_superpoints"]})
    common = {
        "scene_name": track_record["scene_name"],
        "source_track_id": track_id,
        "source_observation_ids": [int(item) for item in track_record["source_observation_ids"]],
        "source_view_count": int(track_record["source_view_count"]),
        "base_superpoint_ids": base_superpoints,
        "base_superpoint_count": len(base_superpoints),
        "internal_cross_view_edge_count": int(track_record["internal_cross_view_edge_count"]),
        "mean_internal_reprojection_support": float(track_record["mean_internal_reprojection_support"]),
        "decision_state": "仅为后续无 GT 质量账本保留几何动作；不生成最终候选。",
    }
    variants = [{
        **common,
        "variant_id": f"track{track_id:04d}_seed_closure",
        "variant_type": "seed_superpoint_closure",
        "added_superpoint_ids": [],
        "proposed_superpoint_count": len(base_superpoints),
    }]
    for frontier in track_record["growth_frontier_superpoints"]:
        superpoint_id = int(frontier["superpoint_id"])
        if superpoint_id in base_superpoints:
            continue
        variants.append({
            **common,
            "variant_id": f"track{track_id:04d}_add_sp{superpoint_id}",
            "variant_type": "one_hop_positive_evidence_addition",
            "added_superpoint_ids": [superpoint_id],
            "proposed_superpoint_count": len(base_superpoints) + 1,
            "adjacent_seed_superpoint_ids": [int(item) for item in frontier["adjacent_seed_superpoint_ids"]],
            "adjacent_seed_superpoint_count": int(frontier["adjacent_seed_superpoint_count"]),
            "boundary_contact_count_sum": int(frontier["boundary_contact_count_sum"]),
            "mean_normal_difference": float(frontier["mean_normal_difference"]),
            "mean_color_difference": float(frontier["mean_color_difference"]),
            "mean_boundary_distance": float(frontier["mean_boundary_distance"]),
            "cross_view_link_count": int(frontier["cross_view_link_count"]),
            "mean_cross_view_reprojection_support": float(frontier["mean_cross_view_reprojection_support"]),
            "linked_seed_node_count": int(frontier["linked_seed_node_count"]),
        })
    return variants


def _build_scene(scene_name, args):
    source_path = args.growth_ledger_root / scene_name / "automatic_sam_track_growth_ledger.json"
    track_records = json.loads(source_path.read_text())
    variants = []
    for track_record in track_records:
        variants.extend(expand_track_variant_plan(track_record))
    scene_root = args.output_root / scene_name
    scene_root.mkdir(parents=True, exist_ok=False)
    with (scene_root / "automatic_sam_growth_variant_plan.jsonl").open("w") as handle:
        for variant in variants:
            handle.write(json.dumps(variant, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "scene_name": scene_name,
        "track_count": len(track_records),
        "seed_closure_variant_count": sum(item["variant_type"] == "seed_superpoint_closure" for item in variants),
        "one_hop_variant_count": sum(item["variant_type"] == "one_hop_positive_evidence_addition" for item in variants),
        "variant_count": len(variants),
        "decision_state": "不读取 GT；不生成最终候选、不赋类别、不融合、不评分、不评测。",
    }
    (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--growth_ledger_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "growth_ledger_root", "output_root"):
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
            f"基础闭包 {summary['seed_closure_variant_count']}，一跳变体 {summary['one_hop_variant_count']}",
            flush=True,
        )
    payload = {
        "purpose": "为自动 SAM 的受约束局部生长登记基础闭包和单原子扩张变体。",
        "gt_usage": "不读取 GT；不生成最终候选、不赋类别、不融合、不评分、不评测。",
        "scene_count": len(summaries),
        "seed_closure_variant_count": sum(item["seed_closure_variant_count"] for item in summaries),
        "one_hop_variant_count": sum(item["one_hop_variant_count"] for item in summaries),
        "variant_count": sum(item["variant_count"] for item in summaries),
        "params": vars(args),
    }
    (args.output_root / "automatic_sam_growth_variant_plan_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
