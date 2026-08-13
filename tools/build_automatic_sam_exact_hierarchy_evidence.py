#!/usr/bin/env python3
"""将自动 SAM 的真实同帧二值 mask 关系提升为跨轨迹层级证据，不作决策。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复项")
    return scenes


def _jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rle_area(payload):
    """RLE 从背景零值开始，奇数段为前景。"""
    size = payload.get("size", [])
    counts = [int(value) for value in payload.get("counts", [])]
    if len(size) != 2 or any(int(value) <= 0 for value in size) or any(value < 0 for value in counts):
        raise ValueError("RLE size 或 counts 非法")
    if sum(counts) != int(size[0]) * int(size[1]):
        raise ValueError("RLE counts 与二维尺寸不一致")
    return int(sum(counts[1::2]))


def validate_observation_rles(observations):
    """验证每条观测都有完整 RLE，且前景面积与 SAM 保存面积相同。"""
    validated = {}
    for observation in observations:
        observation_id = int(observation["observation_id"])
        if "mask_rle" not in observation:
            raise ValueError(f"观测 {observation_id} 缺少 mask_rle；请用 --save-mask-rle 重导出")
        area = rle_area(observation["mask_rle"])
        if area != int(observation["area"]):
            raise ValueError(f"观测 {observation_id} 的 RLE 面积 {area} 与 SAM 面积 {observation['area']} 不一致")
        validated[observation_id] = area
    return validated


def build_track_pair_evidence(nodes, same_frame_relations):
    """仅按真实二维 mask 连续关系聚合轨迹对，保留方向和连续值。"""
    tracks_by_observation = {
        int(node["observation_id"]): sorted(set(int(track) for track in node.get("existing_track_ids", [])))
        for node in nodes
    }
    evidence = defaultdict(list)
    for relation in same_frame_relations:
        left_tracks = tracks_by_observation.get(int(relation["left_observation_id"]), [])
        right_tracks = tracks_by_observation.get(int(relation["right_observation_id"]), [])
        for left_track in left_tracks:
            for right_track in right_tracks:
                if left_track == right_track:
                    continue
                pair = tuple(sorted((left_track, right_track)))
                left_is_pair_left = left_track == pair[0]
                evidence[pair].append({
                    "iou": float(relation["iou"]),
                    "pair_left_coverage": float(relation["left_coverage"] if left_is_pair_left else relation["right_coverage"]),
                    "pair_right_coverage": float(relation["right_coverage"] if left_is_pair_left else relation["left_coverage"]),
                    "intersection_pixel_count": int(relation["intersection_pixel_count"]),
                    "frame_index": int(relation["frame_index"]),
                })
    records = []
    for (left_track, right_track), rows in sorted(evidence.items()):
        records.append({
            "relation_kind": "automatic_track_pair_exact_2d_mask_evidence",
            "left_source_track_id": left_track,
            "right_source_track_id": right_track,
            "same_frame_exact_mask_relation_count": len(rows),
            "distinct_frame_count": len({row["frame_index"] for row in rows}),
            "mean_exact_mask_iou": float(np.mean([row["iou"] for row in rows])),
            "max_exact_mask_iou": float(max(row["iou"] for row in rows)),
            "mean_left_coverage": float(np.mean([row["pair_left_coverage"] for row in rows])),
            "mean_right_coverage": float(np.mean([row["pair_right_coverage"] for row in rows])),
            "max_left_coverage": float(max(row["pair_left_coverage"] for row in rows)),
            "max_right_coverage": float(max(row["pair_right_coverage"] for row in rows)),
            "relation_state": "真实二维 mask 的连续层级证据；不定义同实例、parent/child、聚合、删除或候选选择。",
            "gt_usage": "none",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--evidence-graph-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "automatic_root", "evidence_graph_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    scenes = _scenes(args.scene_list)
    total_observations, total_pairs = 0, 0
    for index, scene in enumerate(scenes, start=1):
        observations = _jsonl(args.automatic_root / scene / "automatic_observations.jsonl")
        observation_areas = validate_observation_rles(observations)
        relations_path = args.automatic_root / scene / "same_frame_mask_relations.jsonl"
        if not relations_path.is_file():
            raise FileNotFoundError(f"缺少 {relations_path}；请用 --save-exact-same-frame-relations 重导出")
        records = build_track_pair_evidence(
            _jsonl(args.evidence_graph_root / scene / "nodes.jsonl"), _jsonl(relations_path)
        )
        target = args.output_root / scene
        target.mkdir()
        _write_jsonl(target / "automatic_track_pair_exact_hierarchy_evidence.jsonl", records)
        (target / "rle_validation_summary.json").write_text(json.dumps({
            "scene_name": scene, "observation_count": len(observation_areas),
            "rle_area_sum": sum(observation_areas.values()), "track_pair_count": len(records), "gt_usage": "none",
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        total_observations += len(observation_areas)
        total_pairs += len(records)
        print(f"[场景完成] {index}: {scene}，RLE 观测 {len(observation_areas)}，轨迹对 {len(records)}", flush=True)
    payload = {
        "purpose": "把真实同帧自动 SAM 二值 mask 关系提升为自动轨迹对的连续层级证据。",
        "gt_usage": "none",
        "decision_state": "不定义同实例、不聚合、不抑制、不删除、不改类别、不输出预测或 AP。",
        "scene_count": len(scenes), "observation_count": total_observations,
        "track_pair_count": total_pairs, "params": vars(args),
    }
    (args.output_root / "automatic_sam_exact_hierarchy_evidence_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
