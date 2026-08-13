#!/usr/bin/env python3
"""Build candidate and pair supervision ledgers for one official train scene.

The candidate ledger keeps GT-derived values in label fields only. The pair
ledger contains geometry/source features and GT preference labels for real
native-track overlaps. It never writes predictions, scores, or candidate
geometry and is intended for format validation before multi-scene training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _load_gt(path: Path, min_points: int) -> tuple[np.ndarray, dict[int, dict]]:
    values = np.loadtxt(path, dtype=np.int64)
    if values.ndim != 1:
        values = values.reshape(-1)
    counts = {}
    for encoded in np.unique(values):
        encoded = int(encoded)
        instance_id = encoded % 1000 - 1
        if instance_id < 0:
            continue
        count = int(np.count_nonzero(values == encoded))
        if count >= min_points:
            counts[encoded] = {
                "encoded_id": encoded,
                "instance_id": instance_id,
                "semantic_id": int(encoded // 1000),
                "point_count": count,
            }
    return values, counts


def _best_gt(points: np.ndarray, gt: np.ndarray, gt_meta: dict[int, dict]) -> dict:
    if len(points) == 0:
        return {"best_iou": 0.0, "best_intersection": 0, "best_gt": None}
    labels, intersections = np.unique(gt[points], return_counts=True)
    best = None
    for encoded, intersection in zip(labels, intersections):
        encoded = int(encoded)
        meta = gt_meta.get(encoded)
        if meta is None:
            continue
        intersection = int(intersection)
        union = len(points) + meta["point_count"] - intersection
        iou = float(intersection / max(1, union))
        candidate = (iou, intersection, -encoded, meta)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return {"best_iou": 0.0, "best_intersection": 0, "best_gt": None}
    return {
        "best_iou": float(best[0]),
        "best_intersection": int(best[1]),
        "best_gt": dict(best[3]),
    }


def _load_points(row: dict, point_count: int) -> np.ndarray:
    path = Path(row["points_path"])
    points = np.unique(np.asarray(np.load(path)["point_indices"], dtype=np.int64))
    if len(points) != int(row.get("point_count", len(points))):
        raise ValueError(f"candidate {row.get('candidate_id', row.get('proposal_id'))} point count differs")
    if len(points) and (points[0] < 0 or points[-1] >= point_count):
        raise ValueError("candidate points are outside the scene")
    return points


def _flatten_gvc(row: dict) -> dict:
    source = row.get("gvc_source_frame_excluded", {})
    gvc = source.get("gvc", {})
    return {
        "gvc_excluded_mean": float(gvc.get("mean", 0.0)),
        "gvc_excluded_max": float(gvc.get("max", 0.0)),
        "gvc_excluded_variance": float(gvc.get("variance", 0.0)),
        "gvc_excluded_selected_view_count": int(source.get("selected_view_count", 0)),
        "gvc_excluded_matched_view_count": int(source.get("matched_selected_view_count", 0)),
        "gvc_excluded_zero_support_fraction": float(source.get("zero_support_selected_view_fraction", 0.0)),
    }


def _candidate_features(source: str, row: dict, point_count: int, gvc_row: dict | None) -> dict:
    features = {
        "scene_name": row.get("scene_name"),
        "candidate_source": source,
        "candidate_id": int(row.get("candidate_id", row.get("track_id", row.get("proposal_id")))),
        "point_count": int(row["point_count"]),
        "point_fraction_of_scene": float(row["point_count"] / max(1, point_count)),
        "original_source_score": float(row.get("original_source_score", row.get("mean_node_quality", 0.0))),
    }
    if source == "d2b_track":
        for key in ("support_view_count", "superpoint_count", "merge_action_count", "mean_consensus_rate", "mean_edge_score", "mean_node_quality"):
            if key in row:
                features[key] = float(row[key]) if "count" not in key else int(row[key])
        features["source_frame_count"] = int(len(row.get("frame_ids", [])))
    if gvc_row is not None:
        features.update(_flatten_gvc(gvc_row))
    return features


def _scene(scene: str, args) -> dict:
    prepared_scene = args.dataset_root / scene
    processed = np.load(prepared_scene / f"{scene.removeprefix('scene')}.npy", mmap_mode="r")
    point_count = int(processed.shape[0])
    gt, gt_meta = _load_gt(args.ground_truth_root / f"{scene}.txt", args.min_gt_points)
    native_masks = np.load(args.native_prediction_cache / f"{scene}_pred_masks.npy", mmap_mode="r")
    native_classes = np.load(args.native_prediction_cache / f"{scene}_pred_classes.npy", mmap_mode="r")
    native_scores = np.load(args.native_prediction_cache / f"{scene}_pred_scores.npy", mmap_mode="r")
    if native_masks.shape[0] != point_count or len(native_classes) != native_masks.shape[1] or len(native_scores) != native_masks.shape[1]:
        raise ValueError(f"{scene}: native dimensions differ from prepared scene")
    tracks_payload = json.loads((args.track_root / scene / "automatic_tracks.json").read_text())
    tracks = tracks_payload.get("tracks", [])
    gvc_rows = json.loads((args.gvc_root / scene / "c1_gvc_quality_ledger.json").read_text())
    gvc_by_key = {(str(row["candidate_source"]), int(row["candidate_id"])): row for row in gvc_rows}
    candidate_rows = []
    point_sets = {}
    for candidate_id in range(native_masks.shape[1]):
        points = np.flatnonzero(native_masks[:, candidate_id]).astype(np.int64)
        point_sets[("native_mask3d_yoloworld", candidate_id)] = points
        row = _candidate_features(
            "native_mask3d_yoloworld",
            {"scene_name": scene, "candidate_id": candidate_id, "point_count": len(points), "original_source_score": native_scores[candidate_id]},
            point_count,
            gvc_by_key.get(("native_mask3d_yoloworld", candidate_id)),
        )
        row["native_class_id"] = int(native_classes[candidate_id])
        candidate_rows.append(row)
    for track in tracks:
        candidate_id = int(track["track_id"])
        points = _load_points(track, point_count)
        point_sets[("d2b_track", candidate_id)] = points
        candidate_rows.append(_candidate_features("d2b_track", {**track, "scene_name": scene}, point_count, gvc_by_key.get(("d2b_track", candidate_id))))

    for row in candidate_rows:
        points = point_sets[(row["candidate_source"], int(row["candidate_id"]))]
        label = _best_gt(points, gt, gt_meta)
        best_gt = label["best_gt"]
        row.update({
            "label_best_gt_iou": label["best_iou"],
            "label_best_gt_intersection": label["best_intersection"],
            "label_best_gt_instance_id": best_gt["instance_id"] if best_gt else None,
            "label_best_gt_semantic_id": best_gt["semantic_id"] if best_gt else None,
            "label_valid_iou25": bool(label["best_iou"] >= 0.25),
            "label_valid_iou50": bool(label["best_iou"] >= 0.50),
            "ground_truth_usage": "label_only",
        })

    candidate_by_key = {(row["candidate_source"], int(row["candidate_id"])): row for row in candidate_rows}
    relation_rows = []
    relation_path = args.competition_root / scene / "track_native_relations.jsonl"
    for line in relation_path.read_text().splitlines():
        relation = json.loads(line)
        track_key = ("d2b_track", int(relation["proposal_id"]))
        native_key = ("native_mask3d_yoloworld", int(relation["native_candidate_id"]))
        if track_key not in candidate_by_key:
            continue
        track_label = candidate_by_key[track_key]
        native_label = candidate_by_key[native_key]
        track_iou = float(track_label["label_best_gt_iou"])
        native_iou = float(native_label["label_best_gt_iou"])
        if track_iou > native_iou:
            preferred = "track"
        elif native_iou > track_iou:
            preferred = "native"
        else:
            preferred = "equivalent"
        relation_rows.append({
            "scene_name": scene,
            "track_id": int(relation["proposal_id"]),
            "native_candidate_id": int(relation["native_candidate_id"]),
            "point_iou": float(relation["point_iou"]),
            "track_inside_native_ratio": float(relation["track_inside_native_ratio"]),
            "native_inside_track_ratio": float(relation["native_inside_track_ratio"]),
            "mutual_duplicate_strict_099": bool(relation["mutual_duplicate_strict_099"]),
            "track_original_score": float(relation["track_score"]),
            "native_original_score": float(relation["native_score"]),
            "track_label_best_gt_iou": track_iou,
            "native_label_best_gt_iou": native_iou,
            "label_iou_margin_track_minus_native": track_iou - native_iou,
            "label_preferred_source": preferred,
            "ground_truth_usage": "label_only",
        })

    exact_groups = defaultdict(list)
    for row in candidate_rows:
        if row["candidate_source"] == "native_mask3d_yoloworld":
            mask = native_masks[:, int(row["candidate_id"])]
            exact_groups[np.packbits(mask).tobytes()].append(int(row["candidate_id"]))
    for row in candidate_rows:
        row["native_exact_geometry_group_size"] = 1
        if row["candidate_source"] == "native_mask3d_yoloworld":
            mask = native_masks[:, int(row["candidate_id"])]
            row["native_exact_geometry_group_size"] = len(exact_groups[np.packbits(mask).tobytes()])

    summary = {
        "scene_name": scene,
        "point_count": point_count,
        "gt_instance_count": len(gt_meta),
        "native_candidate_count": int(native_masks.shape[1]),
        "track_candidate_count": len(tracks),
        "candidate_record_count": len(candidate_rows),
        "overlap_pair_record_count": len(relation_rows),
        "ground_truth_usage": "label_only",
        "feature_contract": "geometry/source/GVC only; GT fields are labels",
        "candidate_label_fields": [
            "label_best_gt_iou", "label_best_gt_intersection", "label_best_gt_instance_id",
            "label_best_gt_semantic_id", "label_valid_iou25", "label_valid_iou50",
        ],
    }
    return candidate_rows, relation_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--gvc-root", type=Path, required=True)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    args = parser.parse_args()
    for name in vars(args):
        if isinstance(getattr(args, name), Path):
            setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene in scenes:
        candidates, relations, summary = _scene(scene, args)
        staging = args.output_root / f".{scene}.tmp.{os.getpid()}"
        published = args.output_root / scene
        staging.mkdir()
        try:
            (staging / "candidate_labels.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in candidates) + "\n")
            (staging / "pair_labels.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in relations) + "\n")
            (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            os.replace(staging, published)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        summaries.append(summary)
        print(f"[done] {scene}: candidates={len(candidates)}, overlap_pairs={len(relations)}", flush=True)
    payload = {
        "scene_count": len(summaries),
        "candidate_record_count": sum(row["candidate_record_count"] for row in summaries),
        "overlap_pair_record_count": sum(row["overlap_pair_record_count"] for row in summaries),
        "ground_truth_usage": "label_only",
        "feature_contract": "geometry/source/GVC only; GT fields are labels",
        "training_ready": False,
        "scene_summaries": summaries,
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
