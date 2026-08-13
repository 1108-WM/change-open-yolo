#!/usr/bin/env python3
"""Build the no-GT, source-unified C1 GVC-Seg reference quality ledger.

The ledger leaves every native mask and frozen D2b track unchanged.  Both
sources are evaluated against the same frozen uniform30 YOLO-World+SAM 2D
observations: for each depth-consistent visible view, select the observation
with greatest projected-box IoU and record box-IoU times visible-point mask
support.  Tracks additionally receive a source-frame-excluded aggregation so
their own construction frames cannot verify them in the primary score.

This produces continuous evidence only.  It does not read GT, choose a score,
apply NMS, suppress proposals, materialize predictions, or run AP.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或包含重复场景")
    return scenes


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    inter = width * height
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - inter)
    return float(inter / max(union, 1e-8))


def _summary(values: list[float]) -> dict[str, float]:
    values_np = np.asarray(values, dtype=np.float64)
    if len(values_np) == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "variance": 0.0}
    return {
        "mean": float(values_np.mean()), "min": float(values_np.min()),
        "max": float(values_np.max()), "variance": float(values_np.var()),
    }


def _load_2d_observations(scene_root: Path) -> dict[int, list[dict]]:
    """Load frozen YOLO-World boxes with their frozen SAM point support."""
    by_frame: dict[int, list[dict]] = defaultdict(list)
    path = scene_root / "observations.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结 YOLO-World+SAM 观测：{path}")
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        points = np.unique(np.asarray(
            np.load(row["point_indices_path"])["point_indices"], dtype=np.int64
        ))
        by_frame[int(row["frame_index"])].append({
            "observation_id": int(row["observation_id"]),
            "bbox": np.asarray(row["bbox_xyxy"], dtype=np.float64),
            "points": points,
        })
    return by_frame


def _frame_contract(scene_root: Path) -> tuple[list[int], dict[str, int]]:
    payload = json.loads((scene_root / "summary.json").read_text())
    rows = payload.get("frames", [])
    # The streaming D1 writer preserves the source observations but only
    # stores aggregate counts in summary.json. Recover the exact uniform30
    # frame contract from those immutable observation records in that case.
    if not rows:
        observations_path = scene_root / "automatic_observations.jsonl"
        if observations_path.is_file():
            recovered: dict[str, int] = {}
            for line in observations_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                frame_id = str(row["frame_id"])
                frame_index = int(row["frame_index"])
                previous = recovered.setdefault(frame_id, frame_index)
                if previous != frame_index:
                    raise ValueError(f"{scene_root}: frame id maps to multiple indices")
            rows = [
                {"frame_id": frame_id, "frame_index": frame_index}
                for frame_id, frame_index in recovered.items()
            ]
    frame_indices = [int(row["frame_index"]) for row in rows]
    if not frame_indices or len(frame_indices) != len(set(frame_indices)):
        raise ValueError(f"{scene_root}: uniform30 帧合同为空或重复")
    return sorted(frame_indices), {str(row["frame_id"]): int(row["frame_index"]) for row in rows}


def _load_tracks(scene_root: Path) -> list[dict]:
    payload = json.loads((scene_root / "automatic_tracks.json").read_text())
    tracks = payload.get("tracks", [])
    ids = [int(row["track_id"]) for row in tracks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{scene_root}: track_id 重复")
    return tracks


def _candidate_view_rows(
    points: np.ndarray,
    frame_indices: list[int],
    projections: np.ndarray,
    visibility: np.ndarray,
    scaling: tuple[float, float],
    observations: dict[int, list[dict]],
    min_visible_points: int,
) -> list[dict]:
    rows = []
    for frame_index in frame_indices:
        visible_points = points[visibility[frame_index, points]]
        if len(visible_points) < min_visible_points:
            continue
        coords = projections[frame_index, visible_points]
        projected_box = np.asarray([
            coords[:, 0].min() / scaling[1], coords[:, 1].min() / scaling[0],
            coords[:, 0].max() / scaling[1], coords[:, 1].max() / scaling[0],
        ], dtype=np.float64)
        candidates = observations.get(frame_index, [])
        if candidates:
            chosen = max(
                candidates,
                key=lambda row: (_box_iou(projected_box, row["bbox"]), -row["observation_id"]),
            )
            box_iou = _box_iou(projected_box, chosen["bbox"])
            support = float(len(np.intersect1d(
                visible_points, chosen["points"], assume_unique=True
            )) / len(visible_points))
            observation_id = int(chosen["observation_id"])
        else:
            box_iou, support, observation_id = 0.0, 0.0, -1
        rows.append({
            "frame_index": int(frame_index),
            "depth_consistent": True,
            "visible_point_count": int(len(visible_points)),
            "visible_point_fraction": float(len(visible_points) / max(1, len(points))),
            "matched_observation_id": observation_id,
            "box_iou": box_iou,
            "mask_point_support": support,
            "gvc_frame_score": float(box_iou * support),
        })
    return rows


def _aggregate(rows: list[dict], max_views: int) -> dict:
    selected = sorted(rows, key=lambda row: (-row["visible_point_count"], row["frame_index"]))[:max_views]
    score = _summary([row["gvc_frame_score"] for row in selected])
    box = _summary([row["box_iou"] for row in selected])
    support = _summary([row["mask_point_support"] for row in selected])
    visible = _summary([float(row["visible_point_count"]) for row in selected])
    return {
        "eligible_depth_consistent_view_count": len(rows),
        "selected_view_count": len(selected),
        "matched_selected_view_count": sum(row["matched_observation_id"] >= 0 for row in selected),
        "zero_support_selected_view_fraction": float(sum(
            row["mask_point_support"] == 0.0 for row in selected
        ) / max(1, len(selected))),
        "gvc": score,
        "projected_box_iou": box,
        "visible_point_mask_support": support,
        "selected_visible_point_count": visible,
        "selected_frames": selected,
    }


def _scene_records(scene: str, args) -> tuple[list[dict], dict]:
    from utils import WORLD_2_CAM

    automatic_scene = args.automatic_root / scene
    frame_indices, frame_id_to_index = _frame_contract(automatic_scene)
    observations = _load_2d_observations(args.yoloworld_sam_root / scene)
    native_masks = np.load(args.native_prediction_cache / f"{scene}_pred_masks.npy", mmap_mode="r")
    native_scores = np.asarray(np.load(args.native_prediction_cache / f"{scene}_pred_scores.npy"), dtype=np.float32)
    if native_masks.ndim != 2 or native_masks.shape[1] != len(native_scores):
        raise ValueError(f"{scene}: native 缓存维度异常")
    tracks = _load_tracks(args.track_root / scene)
    world = WORLD_2_CAM(str(args.dataset_root / scene), args.depth_scale, args.config)
    projections_t, visibility_t = world.get_mesh_projections()
    projections = projections_t.detach().cpu().numpy().astype(np.float64)
    visibility = visibility_t.detach().cpu().numpy().astype(bool)
    if native_masks.shape[0] != visibility.shape[1]:
        raise ValueError(f"{scene}: native 点数与投影点数不一致")
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    records = []

    def append_record(source: str, candidate_id: int, points: np.ndarray,
                      original_score: float, source_frames: set[int], source_missing: int) -> None:
        rows = _candidate_view_rows(
            points, frame_indices, projections, visibility, scaling, observations,
            args.min_visible_points,
        )
        heldout_rows = [row for row in rows if row["frame_index"] not in source_frames]
        records.append({
            "scene_name": scene,
            "candidate_source": source,
            "candidate_id": candidate_id,
            "point_count": int(len(points)),
            "point_fraction_of_scene": float(len(points) / visibility.shape[1]),
            "original_source_score": float(original_score),
            "source_frame_count": len(source_frames),
            "unresolved_source_frame_id_count": source_missing,
            "gvc_including_source_frames": _aggregate(rows, args.max_views),
            "gvc_source_frame_excluded": _aggregate(heldout_rows, args.max_views),
            "decision_state": "连续质量账本；未设阈值、未统一排序、未删除或物化候选。",
        })

    for candidate_id in range(native_masks.shape[1]):
        points = np.flatnonzero(native_masks[:, candidate_id]).astype(np.int64)
        append_record("native_mask3d_yoloworld", candidate_id, points, native_scores[candidate_id], set(), 0)
    for track in tracks:
        points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < visibility.shape[1])]
        raw_frames = {str(value) for value in track.get("frame_ids", [])}
        source_frames = {frame_id_to_index[value] for value in raw_frames if value in frame_id_to_index}
        append_record(
            "d2b_track", int(track["track_id"]), points,
            float(track.get("mean_node_quality", 0.0)), source_frames,
            len(raw_frames - set(frame_id_to_index)),
        )
    summary = {
        "scene_name": scene,
        "native_candidate_count": int(native_masks.shape[1]),
        "track_candidate_count": len(tracks),
        "uniform30_frame_count": len(frame_indices),
        "two_d_observation_count": sum(len(value) for value in observations.values()),
    }
    del world, projections_t, visibility_t, projections, visibility, native_masks
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--yoloworld-sam-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=10)
    parser.add_argument("--min-visible-points", type=int, default=30)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list", "track_root", "native_prediction_cache", "automatic_root",
        "yoloworld_sam_root", "dataset_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_views < 1 or args.min_visible_points < 1:
        raise SystemExit("--max-views 与 --min-visible-points 必须为正数")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    summaries = []
    for index, scene in enumerate(scenes, start=1):
        scene_root = args.output_root / scene
        record_path = scene_root / "c1_gvc_quality_ledger.json"
        if record_path.is_file():
            if not args.resume:
                raise SystemExit(f"输出场景已存在：{scene}")
            summary = json.loads((scene_root / "summary.json").read_text())
            summaries.append(summary)
            print(f"[跳过已有] {index}/{len(scenes)} {scene}", flush=True)
            continue
        records, summary = _scene_records(scene, args)
        scene_root.mkdir(parents=True, exist_ok=False)
        record_path.write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        summaries.append(summary)
        print(f"[场景完成] {index}/{len(scenes)} {scene}: native={summary['native_candidate_count']}, track={summary['track_candidate_count']}", flush=True)
    payload = {
        "diagnostic_type": "GVC-Seg paper-reference source-unified quality ledger",
        "ground_truth_usage": "none",
        "proposal_materialization_applied": False,
        "nms_applied": False,
        "ranking_or_score_rewrite_applied": False,
        "track_primary_verification": "source-frame-excluded",
        "track_including_source_frames": "diagnostic ablation only",
        "scene_count": len(summaries),
        "native_candidate_count": sum(row["native_candidate_count"] for row in summaries),
        "track_candidate_count": sum(row["track_candidate_count"] for row in summaries),
        "params": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
