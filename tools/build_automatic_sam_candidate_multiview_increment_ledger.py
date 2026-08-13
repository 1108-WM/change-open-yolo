#!/usr/bin/env python3
"""记录自动 SAM 候选在其轨迹观测中的相对 native 多视图增量，不作筛选。"""

import argparse
import json
from collections import Counter, defaultdict
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


def observation_increment(candidate_points, observation_points, native_union, same_class_native_union):
    """返回一个观测内的精确集合计数；不把比例转为接受或删除决定。"""
    candidate_points = np.unique(np.asarray(candidate_points, dtype=np.int64))
    observation_points = np.unique(np.asarray(observation_points, dtype=np.int64))
    observed_candidate = np.intersect1d(candidate_points, observation_points, assume_unique=True)
    independent = observed_candidate[~native_union[observed_candidate]]
    same_class_explained = observed_candidate[same_class_native_union[observed_candidate]]
    any_class_explained = observed_candidate[native_union[observed_candidate]]
    return {
        "sam_observation_point_count": int(len(observation_points)),
        "candidate_observation_point_count": int(len(observed_candidate)),
        "candidate_observation_ratio": float(len(observed_candidate) / max(1, len(observation_points))),
        "candidate_independent_point_count": int(len(independent)),
        "candidate_independent_ratio": float(len(independent) / max(1, len(observed_candidate))),
        "candidate_same_class_native_explained_point_count": int(len(same_class_explained)),
        "candidate_same_class_native_explained_ratio": float(len(same_class_explained) / max(1, len(observed_candidate))),
        "candidate_any_native_explained_point_count": int(len(any_class_explained)),
        "candidate_any_native_explained_ratio": float(len(any_class_explained) / max(1, len(observed_candidate))),
    }


def _native_masks(cache_root, scene_name):
    prefix = cache_root / f"{scene_name}_pred_"
    masks = np.asarray(np.load(str(prefix) + "masks.npy", mmap_mode="r"), dtype=bool)
    classes = np.asarray(np.load(str(prefix) + "classes.npy", mmap_mode="r"), dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != len(classes):
        raise ValueError(f"{scene_name} 的 native mask/class 维度不一致")
    return masks, classes


def _track_nodes(nodes):
    by_track = defaultdict(list)
    for node in nodes:
        for track_id in node.get("existing_track_ids", []):
            by_track[int(track_id)].append(node)
    return by_track


def _numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(values)), "mean": float(values.mean()), "p10": float(np.quantile(values, .10)),
        "p50": float(np.quantile(values, .50)), "p90": float(np.quantile(values, .90)), "max": float(values.max()),
    }


def build_scene_ledger(candidates, scene_root, nodes, native_cache_root, scene_name):
    """在每个候选自身的自动 SAM 轨迹观测内计算 native 未解释点。"""
    tracks = _track_nodes(nodes)
    observation_points = {}
    for node in nodes:
        path = Path(node["point_indices_path"])
        observation_points[int(node["node_id"])] = np.unique(
            np.asarray(np.load(path)["point_indices"], dtype=np.int64)
        )
    # native cache defines the full point universe; node point ids may omit trailing unobserved points.
    native_masks, native_classes = _native_masks(native_cache_root, scene_name)
    point_count = int(native_masks.shape[0])
    native_union = np.any(native_masks, axis=1)
    same_class_unions = {}

    rows = []
    for candidate in candidates:
        candidate_id = int(candidate["candidate_id"])
        class_id = int(candidate["class_id"])
        points = np.unique(np.asarray(np.load(scene_root / candidate["seed_points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < point_count)]
        same_class_native_union = same_class_unions.setdefault(
            class_id, np.any(native_masks[:, native_classes == class_id], axis=1)
        )
        views = []
        for node in sorted(tracks.get(int(candidate["source_track_id"]), []), key=lambda item: (item["frame_index"], item["node_id"])):
            metrics = observation_increment(points, observation_points[int(node["node_id"])], native_union, same_class_native_union)
            views.append({
                "node_id": int(node["node_id"]), "observation_id": int(node["observation_id"]),
                "frame_index": int(node["frame_index"]), **metrics,
            })
        rows.append({
            "scene_name": scene_name,
            "candidate_id": candidate_id,
            "source_track_id": int(candidate["source_track_id"]),
            "class_id": class_id,
            "class_name": str(candidate["class_name"]),
            "candidate_point_count": int(len(points)),
            "track_observation_count": int(len(views)),
            "candidate_supported_observation_count": int(sum(item["candidate_observation_point_count"] > 0 for item in views)),
            "candidate_independent_observation_count": int(sum(item["candidate_independent_point_count"] > 0 for item in views)),
            "candidate_observation_ratio": _numeric_summary([item["candidate_observation_ratio"] for item in views]),
            "candidate_independent_ratio": _numeric_summary([item["candidate_independent_ratio"] for item in views]),
            "candidate_same_class_native_explained_ratio": _numeric_summary([item["candidate_same_class_native_explained_ratio"] for item in views]),
            "candidate_any_native_explained_ratio": _numeric_summary([item["candidate_any_native_explained_ratio"] for item in views]),
            "view_records": views,
            "evidence_state": "逐轨迹观测的连续 native 增量证据；不等同于候选接受、删除或最终分数。",
            "gt_usage": "none",
            "decision_state": "不筛除、不合并、不改类别、不改分数、不改 native、不输出预测或 AP。",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--evidence-graph-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "candidate_root", "evidence_graph_root", "native_prediction_cache", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    totals = Counter()
    for index, scene_name in enumerate(_scenes(args.scene_list), start=1):
        scene_root = args.candidate_root / scene_name
        candidates = json.loads((scene_root / "backprojection_candidates.json").read_text())["candidates"]
        nodes = _jsonl(args.evidence_graph_root / scene_name / "nodes.jsonl")
        rows = build_scene_ledger(candidates, scene_root, nodes, args.native_prediction_cache, scene_name)
        target = args.output_root / scene_name
        target.mkdir()
        _write_jsonl(target / "automatic_sam_candidate_multiview_increment_ledger.jsonl", rows)
        totals["candidate_count"] += len(rows)
        totals["track_observation_count"] += sum(row["track_observation_count"] for row in rows)
        totals["candidate_with_independent_observation_count"] += sum(row["candidate_independent_observation_count"] > 0 for row in rows)
        print(f"[场景完成] {index}: {scene_name}，候选 {len(rows)}", flush=True)
    payload = {
        "purpose": "记录自动候选在其自身多视图 SAM 观测中相对冻结 native 的连续增量证据。",
        "gt_usage": "none",
        "decision_state": "不筛除、不合并、不改类别、不改分数、不改 native、不输出预测或 AP。",
        "scene_count": len(_scenes(args.scene_list)), **dict(totals), "params": vars(args),
    }
    (args.output_root / "automatic_sam_candidate_multiview_increment_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
