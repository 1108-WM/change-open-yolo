#!/usr/bin/env python3
"""建立仅由固定正反可见性证据驱动的保守 superpoint 几何变体。

输入是已经生成的自动 SAM 轨迹和无 GT 正反证据账本。对每条轨迹，只有同时
满足“在至少一个视角可见”且“正反证据边际为负”的 superpoint 才被移除；从未
观测到、或证据并不为负的部分一律保留。该输出只是原轨迹的几何变体，既不新建
候选，也不赋类别、打分、融合或评测 AP。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _kept_superpoints(superpoint_ids, visible_view_counts, evidence_margins):
    """保留未知部分；只剔除实际可见且反证严格占优的 superpoint。"""
    superpoint_ids = np.asarray(superpoint_ids, dtype=np.int64)
    visible_view_counts = np.asarray(visible_view_counts, dtype=np.int64)
    evidence_margins = np.asarray(evidence_margins, dtype=np.float32)
    if not (len(superpoint_ids) == len(visible_view_counts) == len(evidence_margins)):
        raise ValueError("superpoint 证据数组长度不一致")
    return superpoint_ids[~((visible_view_counts > 0) & (evidence_margins < 0.0))]


def _variant_points(points, point_superpoints, kept_superpoints):
    points = np.unique(np.asarray(points, dtype=np.int64))
    kept_superpoints = np.asarray(kept_superpoints, dtype=np.int64)
    return points[np.isin(point_superpoints, kept_superpoints, assume_unique=False)]


def _scene_variants(scene_name, args):
    tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    evidence_path = args.evidence_root / scene_name / "visibility_counterevidence_ledger.json"
    evidence_by_id = {int(row["track_id"]): row for row in json.loads(evidence_path.read_text())}
    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    superpoints = np.asarray(np.load(processed_path, mmap_mode="r")[:, 9], dtype=np.int64)
    scene_root = args.output_root / scene_name
    points_root = scene_root / "variant_points"
    points_root.mkdir(parents=True, exist_ok=False)
    records = []
    for track in tracks:
        track_id = int(track["track_id"])
        evidence = evidence_by_id.get(track_id)
        if evidence is None:
            raise ValueError(f"{scene_name} track {track_id} 缺少固定证据账本")
        points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < len(superpoints))]
        npz = np.load(evidence["superpoint_evidence_path"])
        superpoint_ids = np.asarray(npz["superpoint_ids"], dtype=np.int64)
        if len(superpoint_ids) == 0:
            kept = np.unique(superpoints[points])
            removed = np.empty(0, dtype=np.int64)
        else:
            kept = _kept_superpoints(
                superpoint_ids, npz["visible_view_counts"], npz["evidence_margins"],
            )
            removed = np.setdiff1d(superpoint_ids, kept, assume_unique=True)
        variant = _variant_points(points, superpoints[points], kept)
        path = points_root / f"track{track_id:04d}_counterevidence_variant.npz"
        np.savez_compressed(path, point_indices=variant)
        records.append({
            "scene_name": scene_name,
            "track_id": track_id,
            "original_track_point_count": int(len(points)),
            "variant_track_point_count": int(len(variant)),
            "removed_point_count": int(len(points) - len(variant)),
            "original_superpoint_count": int(len(np.unique(superpoints[points]))),
            "removed_negative_margin_superpoint_count": int(len(removed)),
            "variant_points_path": str(path),
            "selection_rule": "只移除可见次数大于零且正反证据边际小于零的 superpoint；未知部分保留。",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--evidence_root", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "evidence_root", "processed_scene_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_records = []
    for index, scene_name in enumerate(scenes, start=1):
        records = _scene_variants(scene_name, args)
        all_records.extend(records)
        scene_root = args.output_root / scene_name
        (scene_root / "counterevidence_superpoint_variants.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(records)} 条几何变体", flush=True)
    payload = {
        "gt_usage": "不读取 GT；不生成候选、不赋类别、不打分、不融合、不评测 AP。",
        "decision_state": "这是固定轨迹的保守几何选择变体，尚未连接到候选形成。",
        "selection_rule": "仅移除可见次数大于零且正反证据边际小于零的 superpoint；未知部分保留。",
        "scene_count": len(scenes),
        "track_count": len(all_records),
        "changed_track_count": sum(row["removed_point_count"] > 0 for row in all_records),
        "removed_point_count": sum(row["removed_point_count"] for row in all_records),
        "params": {key: value for key, value in vars(args).items()},
    }
    (args.output_root / "counterevidence_superpoint_variant_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
