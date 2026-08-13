#!/usr/bin/env python3
"""为几何已固定的自动 SAM 变体汇聚冻结 YOLO-World+SAM 多视图语义证据。

输出类别分布、间隔、熵和跨视角冲突，供后续候选竞争使用。最高票类别仅是证据
摘要，不是最终类别；本工具不读取 GT、不写候选、不融合、不评分、不评测。
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _raw_superpoint_points(processed):
    raw_ids = np.asarray(processed[:, 9], dtype=np.int64)
    order = np.argsort(raw_ids, kind="mergesort")
    ids, starts = np.unique(raw_ids[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    return {
        int(superpoint_id): np.asarray(order[start:end], dtype=np.int64)
        for superpoint_id, start, end in zip(ids, starts, ends)
    }


def _variant_points(variant, superpoint_points):
    ids = {int(item) for item in variant["base_superpoint_ids"]}
    ids.update(int(item) for item in variant.get("added_superpoint_ids", []))
    chunks = [superpoint_points[item] for item in sorted(ids) if item in superpoint_points]
    return np.concatenate(chunks).astype(np.int64, copy=False) if chunks else np.empty(0, dtype=np.int64)


def _load_semantic_observations(scene_root):
    by_frame = defaultdict(list)
    with (scene_root / "observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            point_path = Path(row["point_indices_path"])
            if not point_path.is_absolute():
                point_path = scene_root / point_path
            by_frame[int(row["frame_index"])].append({
                "observation_id": int(row["observation_id"]),
                "class_id": int(row["class_id"]),
                "quality": float(row["score"]) * float(row["sam_score"]),
                "points": np.unique(np.asarray(np.load(point_path)["point_indices"], dtype=np.int64)),
            })
    return by_frame


def _selected_frames(automatic_scene_root):
    payload = json.loads((automatic_scene_root / "summary.json").read_text())
    return sorted(int(item["frame_index"]) for item in payload["frames"])


def frame_class_mask_votes(visible_points, observations):
    """每帧每类仅保留最强 mask 对可见变体点的支持，避免同类检测重复加票。"""
    visible_points = np.unique(np.asarray(visible_points, dtype=np.int64))
    votes = {}
    for observation in observations:
        shared = int(np.intersect1d(visible_points, observation["points"], assume_unique=True).size)
        if shared == 0:
            continue
        value = float(observation["quality"]) * float(shared / len(visible_points))
        class_id = int(observation["class_id"])
        candidate = (value, int(observation["observation_id"]), shared)
        previous = votes.get(class_id)
        if previous is None or candidate[0] > previous[0] or (candidate[0] == previous[0] and candidate[1] < previous[1]):
            votes[class_id] = candidate
    return votes


def summarize_semantic_votes(votes, support_views, used_frame_count, labels, topk=16):
    ordered = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    total = float(sum(value for _, value in ordered))
    top_class, top_vote = ordered[0] if ordered else (-1, 0.0)
    second_vote = ordered[1][1] if len(ordered) > 1 else 0.0
    probabilities = [value / total for _, value in ordered] if total > 0.0 else []
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
    normalized_entropy = float(entropy / math.log(max(2, len(labels)))) if probabilities else 0.0
    return {
        "semantic_evidence_top_class_index": int(top_class),
        "semantic_evidence_top_class_name": labels[top_class] if 0 <= top_class < len(labels) else "无语义证据",
        "semantic_vote_total": total,
        "semantic_vote_margin": float((top_vote - second_vote) / max(top_vote, 1e-6)),
        "semantic_normalized_entropy": normalized_entropy,
        "semantic_top_class_view_ratio": float(support_views.get(top_class, 0) / max(1, used_frame_count)),
        "semantic_class_distribution": [
            {
                "class_index": int(class_id),
                "class_name": labels[class_id] if 0 <= class_id < len(labels) else "未知类别",
                "vote": float(vote),
                "probability": float(vote / total) if total > 0.0 else 0.0,
                "support_views": int(support_views[class_id]),
            }
            for class_id, vote in ordered[:topk]
        ],
    }


def _scene_records(scene_name, args):
    from utils import WORLD_2_CAM

    variants = []
    with (args.variant_plan_root / scene_name / "automatic_sam_growth_variant_plan.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                variants.append(json.loads(line))
    if args.max_variants_per_scene is not None:
        variants = variants[: args.max_variants_per_scene]
    quality_records = json.loads((args.quality_ledger_root / scene_name / "automatic_sam_variant_quality_ledger.json").read_text())
    quality_by_variant = {str(item["variant_id"]): item for item in quality_records}
    processed = np.load(args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r")
    superpoint_points = _raw_superpoint_points(processed)
    semantic_observations = _load_semantic_observations(args.yoloworld_sam_root / scene_name)
    frames = _selected_frames(args.automatic_root / scene_name)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, visibility = world.get_mesh_projections()
    visibility = visibility.detach().cpu().numpy().astype(bool)
    records = []
    for variant in variants:
        points = _variant_points(variant, superpoint_points)
        votes = defaultdict(float)
        support_views = defaultdict(int)
        used_frames = 0
        for frame_index in frames:
            visible_points = points[visibility[frame_index, points]]
            if len(visible_points) < args.min_visible_points:
                continue
            frame_votes = frame_class_mask_votes(visible_points, semantic_observations.get(frame_index, []))
            if not frame_votes:
                continue
            used_frames += 1
            for class_id, (value, _, _) in frame_votes.items():
                votes[class_id] += float(value)
                support_views[class_id] += 1
        quality = quality_by_variant.get(str(variant["variant_id"]), {})
        records.append({
            "scene_name": scene_name,
            "variant_id": variant["variant_id"],
            "variant_type": variant["variant_type"],
            "source_track_id": int(variant["source_track_id"]),
            "variant_point_count": int(len(points)),
            "semantic_evidence_frame_count": used_frames,
            "semantic_state": "几何固定后的冻结 YOLO-World+SAM 分布；最高类别不是最终赋类。",
            "geometry_quality_reference": {
                key: quality[key] for key in (
                    "gvc_score", "native_top_iou", "variant_inside_top_native_ratio",
                    "internal_cross_view_edge_count", "mean_internal_reprojection_support",
                ) if key in quality
            },
            **summarize_semantic_votes(votes, support_views, used_frames, args.labels, args.topk),
        })
    del world, visibility, processed
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--variant_plan_root", type=Path, required=True)
    parser.add_argument("--quality_ledger_root", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--yoloworld_sam_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--min_visible_points", type=int, default=30)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--max_variants_per_scene", type=int)
    parser.add_argument("--max_scenes", type=int)
    args = parser.parse_args()
    for name in (
        "scene_list", "variant_plan_root", "quality_ledger_root", "automatic_root", "yoloworld_sam_root",
        "dataset_root", "processed_scene_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.min_visible_points <= 0 or args.topk <= 0:
        raise SystemExit("--min_visible_points 与 --topk 必须为正数。")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    args.labels = list(args.config["network2d"]["text_prompts"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        records = _scene_records(scene_name, args)
        scene_root = args.output_root / scene_name
        scene_root.mkdir()
        (scene_root / "automatic_sam_variant_semantic_ledger.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "scene_name": scene_name,
            "variant_count": len(records),
            "with_semantic_evidence_count": sum(row["semantic_evidence_top_class_index"] >= 0 for row in records),
        }
        summaries.append(summary)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(records)} 条变体语义记录", flush=True)
    payload = {
        "purpose": "为几何已固定的自动 SAM 变体汇聚多视图语义分布，而不是提前赋类。",
        "gt_usage": "不读取 GT；不生成最终候选、不融合、不评分、不评测。",
        "scene_count": len(summaries),
        "variant_count": sum(item["variant_count"] for item in summaries),
        "with_semantic_evidence_count": sum(item["with_semantic_evidence_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "automatic_sam_variant_semantic_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
