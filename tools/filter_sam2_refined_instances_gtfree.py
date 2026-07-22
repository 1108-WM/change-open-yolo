#!/usr/bin/env python3
"""GT-free quality guard for SAM2 refined superpoint instances.

该工具只使用 mesh、IBSp superpoint、已有 refined instances 以及可选的
MVPDist 候选元数据。它不读取 GT，输出一个新的 refined root，供后续
MVPDist 导出、离线诊断或三场景受控实验使用。
"""

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from plyfile import PlyData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _scene_names(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _scene_array_path(root, scene_name):
    scene_id = scene_name.removeprefix("scene")
    return Path(root) / scene_name / f"{scene_id}.npy"


def _load_candidate_metadata(root, scene_name):
    if root is None:
        return {}
    path = Path(root) / scene_name / "backprojection_candidates.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {int(item["candidate_id"]): item for item in payload.get("candidates", [])}


def _load_faces(dataset_root, scene_name):
    path = Path(dataset_root) / scene_name / f"{scene_name}_vh_clean_2.ply"
    return PlyData.read(path)["face"].data["vertex_indices"]


def _superpoint_adjacency(superpoints, faces):
    adjacency = defaultdict(set)
    for face in faces:
        segment_ids = {int(superpoints[int(vertex_id)]) for vertex_id in face}
        if len(segment_ids) < 2:
            continue
        for segment_id in segment_ids:
            adjacency[segment_id].update(other for other in segment_ids if other != segment_id)
    return adjacency


def _component_stats(segment_ids, adjacency, superpoint_sizes):
    segment_ids = set(int(item) for item in segment_ids)
    seen = set()
    components = []
    for start in sorted(segment_ids):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component = []
        while queue:
            segment_id = queue.popleft()
            component.append(segment_id)
            for neighbor in adjacency.get(segment_id, ()):
                if neighbor in segment_ids and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        point_count = int(sum(int(superpoint_sizes[item]) for item in component))
        components.append({"superpoint_ids": sorted(component), "point_count": point_count})
    components.sort(key=lambda item: (-item["point_count"], item["superpoint_ids"]))
    return components


def _kept_components(components, args):
    if not components:
        return [], []
    total_points = sum(item["point_count"] for item in components)
    kept = []
    dropped = []
    for index, component in enumerate(components):
        keep = index == 0
        if not keep:
            keep = (
                component["point_count"] >= int(args.min_component_points)
                and component["point_count"] / max(1, total_points) >= float(args.min_component_point_fraction)
            )
        (kept if keep else dropped).append(component)
    return kept, dropped


def _candidate_indices(record, num_points):
    path = Path(record["point_indices_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    indices = np.load(path)["point_indices"].astype(np.int64)
    indices = np.unique(indices[(indices >= 0) & (indices < num_points)])
    return indices.astype(np.int32)


def _passes_quality(record, metadata, original_indices, kept_indices, components, args):
    reasons = []
    original_points = int(len(original_indices))
    kept_points = int(len(kept_indices))
    original_superpoints = int(record.get("superpoint_count", len(record.get("superpoint_ids", []))))
    scene_fraction = kept_points / max(1, int(args._scene_point_count))
    largest_ratio = 0.0
    if components:
        largest_ratio = components[0]["point_count"] / max(1, sum(item["point_count"] for item in components))

    if kept_points < int(args.min_points):
        reasons.append("too_few_points")
    if scene_fraction > float(args.max_scene_point_fraction):
        reasons.append("scene_fraction_too_large")
    if original_superpoints > int(args.max_superpoints):
        reasons.append("too_many_superpoints")
    if len(components) > int(args.max_components):
        reasons.append("too_many_connected_components")
    if largest_ratio < float(args.min_largest_component_ratio):
        reasons.append("largest_component_ratio_too_low")
    if original_points and kept_points / original_points < float(args.min_kept_point_ratio):
        reasons.append("trimmed_too_much")
    if metadata and float(metadata.get("score", 0.0)) < float(args.min_mvpdist_score):
        reasons.append("mvpdist_score_too_low")
    return not reasons, reasons


def _process_scene(scene_name, args):
    refined_scene = Path(args.refined_root) / scene_name
    output_scene = Path(args.output_root) / scene_name
    output_scene.mkdir(parents=True, exist_ok=True)

    source_array = _scene_array_path(args.superpoint_root, scene_name)
    superpoints = np.load(source_array, mmap_mode="r")[:, 9].astype(np.int64)
    args._scene_point_count = int(len(superpoints))
    superpoint_sizes = np.bincount(superpoints, minlength=int(superpoints.max(initial=-1)) + 1).astype(np.int64)
    adjacency = _superpoint_adjacency(superpoints, _load_faces(args.dataset_root, scene_name))
    metadata_by_id = _load_candidate_metadata(args.mvpdist_candidates_root, scene_name)

    records = []
    dropped_records = []
    for source_record in _read_jsonl(refined_scene / "instances.jsonl"):
        source_id = int(source_record.get("instance_id", len(records)))
        original_indices = _candidate_indices(source_record, len(superpoints))
        components = _component_stats(source_record.get("superpoint_ids", []), adjacency, superpoint_sizes)
        kept_components, dropped_components = _kept_components(components, args)
        kept_superpoints = sorted(
            segment_id for component in kept_components for segment_id in component["superpoint_ids"]
        )
        kept_mask = np.isin(superpoints, kept_superpoints)
        kept_indices = np.flatnonzero(kept_mask).astype(np.int32)
        metadata = metadata_by_id.get(source_id, {})
        passed, drop_reasons = _passes_quality(
            source_record, metadata, original_indices, kept_indices, components, args
        )
        quality = {
            "source_instance_id": source_id,
            "original_point_count": int(len(original_indices)),
            "original_superpoint_count": int(source_record.get("superpoint_count", len(source_record.get("superpoint_ids", [])))),
            "scene_point_fraction": float(len(kept_indices) / max(1, len(superpoints))),
            "component_count": len(components),
            "component_point_counts": [int(item["point_count"]) for item in components],
            "largest_component_ratio": 0.0
            if not components
            else float(components[0]["point_count"] / max(1, sum(item["point_count"] for item in components))),
            "dropped_component_count": len(dropped_components),
            "dropped_component_points": int(sum(item["point_count"] for item in dropped_components)),
            "mvpdist_score": float(metadata.get("score", 0.0)) if metadata else None,
            "drop_reasons": drop_reasons,
        }
        if not passed:
            dropped_records.append({**quality, "source_track_ids": source_record.get("source_track_ids", [])})
            continue

        instance_id = len(records)
        point_path = output_scene / f"instance{instance_id:04d}_points.npz"
        np.savez_compressed(point_path, point_indices=kept_indices)
        records.append(
            {
                "instance_id": instance_id,
                "source_instance_id": source_id,
                "source_track_ids": [int(item) for item in source_record.get("source_track_ids", [])],
                "superpoint_ids": kept_superpoints,
                "superpoint_count": len(kept_superpoints),
                "point_count": int(len(kept_indices)),
                "support_score": float(source_record.get("support_score", 0.0)),
                "point_indices_path": str(point_path),
                "quality_guard": quality,
            }
        )

    (output_scene / "instances.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    summary = {
        "scene_name": scene_name,
        "gt_usage": "none",
        "source_refined_root": str(refined_scene),
        "source_superpoint_array": str(source_array),
        "input_instance_count": len(_read_jsonl(refined_scene / "instances.jsonl")),
        "output_instance_count": len(records),
        "dropped_instance_count": len(dropped_records),
        "dropped_instances": dropped_records,
    }
    (output_scene / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--superpoint_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--mvpdist_candidates_root", type=Path, default=None)
    parser.add_argument("--scene_names", required=True)
    parser.add_argument("--min_points", type=int, default=100)
    parser.add_argument("--max_scene_point_fraction", type=float, default=1.0)
    parser.add_argument("--max_superpoints", type=int, default=10**9)
    parser.add_argument("--max_components", type=int, default=10**9)
    parser.add_argument("--min_largest_component_ratio", type=float, default=0.0)
    parser.add_argument("--min_component_points", type=int, default=100)
    parser.add_argument("--min_component_point_fraction", type=float, default=0.02)
    parser.add_argument("--min_kept_point_ratio", type=float, default=0.0)
    parser.add_argument("--min_mvpdist_score", type=float, default=0.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [_process_scene(scene_name, args) for scene_name in _scene_names(args.scene_names)]
    payload = {
        "gt_usage": "none",
        "params": {key: _jsonable(value) for key, value in vars(args).items() if not key.startswith("_")},
        "scenes": summaries,
        "input_instance_count": sum(item["input_instance_count"] for item in summaries),
        "output_instance_count": sum(item["output_instance_count"] for item in summaries),
        "dropped_instance_count": sum(item["dropped_instance_count"] for item in summaries),
    }
    (args.output_root / "quality_guard_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
