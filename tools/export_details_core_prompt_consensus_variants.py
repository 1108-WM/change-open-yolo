#!/usr/bin/env python3
"""物化固定提示账本中的保守 superpoint 增量变体。

每条 v1.2 轨迹只追加“同帧三个 SAM 假设全部包含、且至少两个独立提示帧支持”的
superpoint。原 v1.2 mask 始终是逐轨迹回退；无稳定增量的轨迹逐点不变。工具不
读取 GT、类别、语义或 native，不改变既有分数，也不做轨迹间合并、删除或 NMS。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from refine_details_automatic_tracks_consensus import _superpoint_points
from refine_details_consensus_all_view_reobservation import _read_scenes, _resolve


def conservative_variant_superpoints(base_superpoints, stable_added_superpoints):
    base = set(int(item) for item in base_superpoints)
    added = set(int(item) for item in stable_added_superpoints) - base
    return sorted(base | added), sorted(added)


def _export_scene(scene_name, args):
    track_payload = json.loads(
        (args.track_root / scene_name / "automatic_tracks.json").read_text()
    )
    tracks = track_payload.get("tracks", [])
    track_ids = [int(track["track_id"]) for track in tracks]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError(f"{scene_name} 含重复 track_id")
    ledger_rows = json.loads(
        (args.ledger_root / scene_name / "prompt_track_ledger.json").read_text()
    )
    ledger = {int(row["track_id"]): row for row in ledger_rows}
    if len(ledger) != len(ledger_rows) or not set(ledger).issubset(set(track_ids)):
        raise ValueError(f"{scene_name} 的提示账本 track_id 异常")

    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    points_by_superpoint = _superpoint_points(superpoints)

    staging_root = args.output_root / f".{scene_name}.writing"
    published_root = args.output_root / scene_name
    if staging_root.exists() or published_root.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published_root}")
    points_root = staging_root / "track_points"
    points_root.mkdir(parents=True)
    output_tracks = []
    changed_track_count = 0
    added_superpoint_count = 0
    added_point_count = 0
    for track in tracks:
        track_id = int(track["track_id"])
        stable = ledger.get(track_id, {}).get(
            "stable_new_superpoints_all_hypotheses", []
        )
        variant_superpoints, added = conservative_variant_superpoints(
            track.get("superpoint_ids", []), stable
        )
        missing = [item for item in variant_superpoints if item not in points_by_superpoint]
        if missing:
            raise ValueError(f"{scene_name} track {track_id} 引用未知 superpoint：{missing[:5]}")
        point_indices = np.concatenate(
            [points_by_superpoint[item] for item in variant_superpoints]
        ).astype(np.int64, copy=False)
        filename = f"track{track_id:04d}_points.npz"
        np.savez_compressed(points_root / filename, point_indices=point_indices)
        added_points = sum(len(points_by_superpoint[item]) for item in added)
        changed_track_count += int(bool(added))
        added_superpoint_count += len(added)
        added_point_count += added_points
        record = dict(track)
        record.update({
            "points_path": str(
                args.output_root / scene_name / "track_points" / filename
            ),
            "point_count": int(len(point_indices)),
            "superpoint_ids": variant_superpoints,
            "superpoint_count": len(variant_superpoints),
            "core_prompt_variant_changed": bool(added),
            "core_prompt_added_superpoint_ids": added,
            "core_prompt_added_superpoint_count": len(added),
            "core_prompt_added_point_count": int(added_points),
            "core_prompt_variant_rule": (
                "同帧三个点提示SAM假设全部支持，且至少两个独立提示帧支持"
            ),
        })
        output_tracks.append(record)

    summary = {
        "scene_name": scene_name,
        "track_count": len(output_tracks),
        "changed_track_count": changed_track_count,
        "unchanged_track_count": len(output_tracks) - changed_track_count,
        "added_superpoint_count": added_superpoint_count,
        "added_point_count": added_point_count,
    }
    output_payload = dict(track_payload)
    output_payload.update(summary)
    output_payload["tracks"] = output_tracks
    (staging_root / "automatic_tracks.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging_root, published_root)
    del processed
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list", "track_root", "ledger_root", "processed_scene_root", "output_root"
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "automatic_tracks.json"
        if existing.is_file() and args.resume:
            payload = json.loads(existing.read_text())
            summary = {
                key: payload[key]
                for key in (
                    "scene_name", "track_count", "changed_track_count",
                    "unchanged_track_count", "added_superpoint_count", "added_point_count",
                )
            }
        else:
            summary = _export_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"改变 {summary['changed_track_count']} 条，"
            f"新增 {summary['added_superpoint_count']} 个SP",
            flush=True,
        )

    root_summary = {
        "gt_usage": "不读取GT；不使用类别、语义或native候选。",
        "decision_state": "固定提示三假设交集的跨帧稳定superpoint并集变体。",
        "scene_count": len(summaries),
        "track_count": sum(item["track_count"] for item in summaries),
        "changed_track_count": sum(item["changed_track_count"] for item in summaries),
        "unchanged_track_count": sum(item["unchanged_track_count"] for item in summaries),
        "added_superpoint_count": sum(item["added_superpoint_count"] for item in summaries),
        "added_point_count": sum(item["added_point_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items()},
    }
    (args.output_root / "core_prompt_consensus_variant_summary.json").write_text(
        json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
