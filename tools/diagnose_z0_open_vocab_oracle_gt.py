#!/usr/bin/env python3
"""Z0: fixed-geometry open-vocabulary classification/ranking oracle audit.

This is deliberately an evaluation-only tool.  It expands the frozen native,
track and pair-union candidates into four source views and evaluates the five
registered class/score combinations described in the project hand-off:

* current class + current score;
* GT class + current score;
* current class + GT-only ideal score;
* GT class + GT-only ideal score;
* GT class + one-to-one duplicate-competition ideal score.

GT is read only inside the evaluator/oracle calculations.  No prediction cache,
candidate file, class, score, or inference ledger is written or modified.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402


SOURCE_NAMES = ("native_only", "track_only", "native_plus_track", "pair_union")
EVALUATION_NAMES = (
    "current_class_current_score",
    "gt_class_current_score",
    "current_class_gt_score",
    "gt_class_gt_score",
    "gt_class_one_to_one_score",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _load_gt(path: Path, min_region_size: int) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in instance_eval.VALID_CLASS_IDS_200_INST}
    rows = []
    for instance_id in np.unique(ids):
        instance_id = int(instance_id)
        class_id = instance_id // 1000
        if instance_id <= 0 or class_id not in valid:
            continue
        points = np.flatnonzero(ids == instance_id).astype(np.int32)
        if len(points) >= min_region_size:
            rows.append((class_id, points))
    return ids, rows


def _load_native(root: Path, scene: str, stream_records_root: Path | None = None) -> dict[str, np.ndarray]:
    if stream_records_root is not None:
        root = stream_records_root / scene / "native_cache"
    prefix = root / f"{scene}_pred_"
    masks = np.asarray(np.load(str(prefix) + "masks.npy", mmap_mode="r"), dtype=bool)
    scores = np.asarray(np.load(str(prefix) + "scores.npy"), dtype=np.float32)
    classes = np.asarray(np.load(str(prefix) + "classes.npy"), dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != len(scores) or len(scores) != len(classes):
        raise ValueError(f"{scene}: native prediction dimensions disagree")
    return {"pred_masks": masks, "pred_scores": scores, "pred_classes": classes}


def _load_tracks(
    track_root: Path, semantic_root: Path | None, scene: str, point_count: int,
    stream_records_root: Path | None = None, semantic_rows: list[dict] | None = None,
) -> dict[str, np.ndarray]:
    if stream_records_root is not None:
        track_root = stream_records_root / scene / "d2b_tracks_filtered"
    tracks = json.loads((track_root / scene / "automatic_tracks.json").read_text()).get("tracks", [])
    if semantic_rows is not None:
        semantics = semantic_rows
    elif semantic_root is not None:
        semantics = json.loads(
            (semantic_root / scene / "automatic_track_yoloworld_semantics.json").read_text()
        )
    else:
        raise ValueError("track semantics require --semantic-root or explicit in-memory reconstruction")
    by_id = {int(row["track_id"]): row for row in semantics}
    if len(by_id) != len(semantics):
        raise ValueError(f"{scene}: duplicate track semantic IDs")
    masks, classes, scores, track_ids = [], [], [], []
    skipped_invalid_class = 0
    for track in sorted(tracks, key=lambda row: int(row["track_id"])):
        track_id = int(track["track_id"])
        semantic = by_id.get(track_id)
        if semantic is None:
            raise ValueError(f"{scene}: missing YOLO-World semantic for track {track_id}")
        class_index = int(semantic.get("voted_class_index", -1))
        if class_index not in instance_eval.PRED_ID_TO_ID or int(instance_eval.PRED_ID_TO_ID[class_index]) < 0:
            skipped_invalid_class += 1
            continue
        with np.load(Path(track["points_path"])) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        if len(points) == 0 or np.any(points < 0) or np.any(points >= point_count):
            raise ValueError(f"{scene}: invalid points for track {track_id}")
        mask = np.zeros(point_count, dtype=bool)
        mask[points] = True
        masks.append(mask)
        classes.append(class_index)
        scores.append(max(0.0, float(track.get("mean_node_quality", 0.0))))
        track_ids.append(track_id)
    return {
        "pred_masks": np.stack(masks, axis=1) if masks else np.zeros((point_count, 0), dtype=bool),
        "pred_scores": np.asarray(scores, dtype=np.float32),
        "pred_classes": np.asarray(classes, dtype=np.int64),
        "track_ids": track_ids,
        "skipped_invalid_class": skipped_invalid_class,
    }


def _reconstruct_track_semantics_in_memory(
    scene: str, tracks: list[dict], stream_records_root: Path, prepared_dataset_root: Path,
    config_path: Path,
) -> list[dict]:
    """Recompute the already-frozen top-1 vote without writing an inference ledger."""
    import torch
    import yaml

    from tools.annotate_automatic_mask_tracks_yoloworld import _as_numpy, frame_class_votes
    from utils import WORLD_2_CAM

    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    depth_scale = float(config["openyolo3d"]["depth_scale"])
    cache_path = stream_records_root / scene / "yoloworld_bboxes_2d" / f"{scene}.pt"
    payload = torch.load(cache_path, map_location="cpu")
    # official100 caches are signed payloads.  Treating the top-level wrapper
    # as a frame mapping silently turns every semantic lookup into a miss.
    if not isinstance(payload, Mapping) or "metadata" not in payload or "predictions" not in payload:
        raise ValueError(
            f"{scene}: unsupported YOLO-World cache format at {cache_path}; "
            "expected {'metadata', 'predictions'}"
        )
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping) or metadata.get("scene_name") != scene:
        raise ValueError(f"{scene}: YOLO-World cache metadata scene mismatch at {cache_path}")
    predictions = payload["predictions"]
    if not isinstance(predictions, Mapping):
        raise ValueError(f"{scene}: YOLO-World cache predictions are not a mapping")
    world = WORLD_2_CAM(str(prepared_dataset_root / scene), depth_scale, config)
    mesh_projections, mesh_visibility = world.get_mesh_projections()
    projections = mesh_projections.detach().cpu().numpy().astype(np.int64)
    visibility = mesh_visibility.detach().cpu().numpy().astype(bool)
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    frame_lookup = {Path(path).stem: index for index, path in enumerate(world.color_paths)}
    missing_cache_frames = sorted(set(frame_lookup) - {str(key) for key in predictions})
    if missing_cache_frames:
        raise ValueError(
            f"{scene}: YOLO-World cache is missing prepared frames "
            f"{missing_cache_frames[:5]}{'...' if len(missing_cache_frames) > 5 else ''}"
        )
    rows = []
    for track in tracks:
        with np.load(Path(track["points_path"])) as payload:
            points = np.asarray(payload["point_indices"], dtype=np.int64)
        votes = {}
        for frame_id in track["frame_ids"]:
            frame_index = frame_lookup.get(str(frame_id))
            prediction = predictions.get(str(frame_id))
            if frame_index is None or prediction is None:
                continue
            visible = points[visibility[frame_index, points]]
            if not len(visible):
                continue
            coords = projections[frame_index, visible].astype(np.float32)
            local = frame_class_votes(
                coords[:, 0] / float(scaling[1]), coords[:, 1] / float(scaling[0]),
                _as_numpy(prediction["bbox"]).astype(np.float32),
                _as_numpy(prediction["labels"]).astype(np.int64),
                _as_numpy(prediction["scores"]).astype(np.float32),
            )
            for label, value in local.items():
                votes[label] = votes.get(label, 0.0) + float(value)
        best = min(votes, key=lambda label: (-votes[label], label)) if votes else -1
        rows.append({"track_id": int(track["track_id"]), "voted_class_index": int(best)})
    del world, predictions, projections, visibility
    return rows


def _load_union_rows(plan_root: Path, scene: str, point_count: int) -> dict[str, np.ndarray]:
    path = plan_root / "pair_union_append_candidates.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = sorted((row for row in rows if str(row["scene_name"]) == scene), key=lambda row: int(row["candidate_id"]))
    masks, classes, scores = [], [], []
    for row in rows:
        with np.load(Path(row["points_path"])) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        if len(points) != int(row["point_count"]) or np.any(points < 0) or np.any(points >= point_count):
            raise ValueError(f"{scene}: invalid pair-union points")
        mask = np.zeros(point_count, dtype=bool)
        mask[points] = True
        # The frozen append contract has no independent union semantic source.
        # Current-class evaluation therefore inherits the selected track vote.
        masks.append(mask)
        classes.append(None)
        scores.append(float(row["new_score"]))
    return {
        "pred_masks": np.stack(masks, axis=1) if masks else np.zeros((point_count, 0), dtype=bool),
        "pred_scores": np.asarray(scores, dtype=np.float32),
        "pred_classes": np.asarray(classes, dtype=object),
        "rows": rows,
    }


def _load_track_score_overrides(plan_root: Path, scene: str) -> dict[int, float]:
    path = plan_root / "champion_track_score_overrides.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row["scene_name"]) != scene:
            continue
        track_id = int(row["candidate_id"])
        if track_id in result:
            raise ValueError(f"{scene}: duplicate frozen track score override {track_id}")
        original = float(row["original_score"])
        new_score = float(row["new_score"])
        inherited_zero_clip = original == 0.0 and new_score <= 1e-6 + 1e-12
        if (new_score < -1e-12 or new_score > original + 1e-12) and not inherited_zero_clip:
            raise ValueError(f"{scene}: track override is not continuous suppression {track_id}")
        result[track_id] = max(0.0, new_score)
    return result


def _best_gt(masks: np.ndarray, gt_rows: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = masks.shape[1]
    sizes = masks.sum(axis=0).astype(np.int64)
    best_iou = np.zeros(count, dtype=np.float32)
    best_class = np.full(count, -1, dtype=np.int64)
    best_gt = np.full(count, -1, dtype=np.int64)
    for gt_index, (class_id, points) in enumerate(gt_rows):
        intersection = np.asarray(masks[points].sum(axis=0), dtype=np.int64)
        iou = intersection / np.maximum(1, sizes + len(points) - intersection)
        update = iou > best_iou
        best_iou[update] = iou[update]
        best_class[update] = class_id
        best_gt[update] = gt_index
    return best_iou, best_class, best_gt


def _same_class_gt_scores(masks: np.ndarray, classes: np.ndarray, gt_rows: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Best same-semantic-class GT IoU for ranking-only diagnostics."""
    sizes = masks.sum(axis=0).astype(np.int64)
    scores = np.zeros(masks.shape[1], dtype=np.float32)
    for class_id, points in gt_rows:
        selected = np.flatnonzero(classes == class_id)
        if not len(selected):
            continue
        inter = np.asarray(masks[points][:, selected].sum(axis=0), dtype=np.int64)
        iou = inter / np.maximum(1, sizes[selected] + len(points) - inter)
        scores[selected] = np.maximum(scores[selected], iou.astype(np.float32))
    return scores


def _one_to_one_scores(
    masks: np.ndarray, classes: np.ndarray, gt_rows: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    """Assign each GT to at most one same-semantic-class prediction."""
    scores = np.zeros(masks.shape[1], dtype=np.float32)
    sizes = masks.sum(axis=0).astype(np.int64)
    for class_id in sorted({row[0] for row in gt_rows}):
        pred = np.flatnonzero(classes == class_id)
        gt = [(idx, points) for idx, (label, points) in enumerate(gt_rows) if label == class_id]
        if not len(pred) or not gt:
            continue
        matrix = np.zeros((len(pred), len(gt)), dtype=np.float32)
        for j, (_, points) in enumerate(gt):
            inter = np.asarray(masks[points][:, pred].sum(axis=0), dtype=np.int64)
            matrix[:, j] = inter / np.maximum(1, sizes[pred] + len(points) - inter)
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Z0 一对一 oracle 需要 scipy.optimize") from exc
        rows, cols = linear_sum_assignment(-matrix)
        for row, col in zip(rows, cols):
            if matrix[row, col] > 0.0:
                scores[pred[row]] = matrix[row, col]
    return scores


class _Predictions(Mapping):
    """Lazily reconstruct one scene at a time so Z0 never retains 100 masks."""

    def __init__(self, scenes: list[str], build_prediction):
        self.scenes, self.build_prediction = scenes, build_prediction

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)

    def __getitem__(self, scene):
        return self.build_prediction(scene)

    def items(self):
        for index, scene in enumerate(self.scenes, 1):
            yield scene, self.build_prediction(scene)
            print(f"[z0 evaluator] {index}/{len(self.scenes)} {scene}", flush=True)
            gc.collect()


def _evaluate(predictions: Mapping, gt_root: Path, out_csv: Path) -> dict:
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        averages, _, _, _ = instance_eval.evaluate(predictions, str(gt_root), str(out_csv), dataset="scannet200")
    return {
        "ap": float(averages["all_ap"]), "ap50": float(averages["all_ap_50%"]),
        "ap25": float(averages["all_ap_25%"]), "head_ap": float(averages["head_ap"]),
        "common_ap": float(averages["common_ap"]), "tail_ap": float(averages["tail_ap"]),
    }


def _scene_prediction(
    scene: str, source: str, evaluation: str, args: argparse.Namespace,
    inverse_class_map: dict[int, int], semantic_rows: list[dict] | None,
) -> dict[str, np.ndarray]:
    """Build one GT-only evaluation prediction; all arrays die after this scene."""
    native = _load_native(args.native_prediction_cache, scene, args.stream_records_root)
    tracks = _load_tracks(
        args.track_root, args.semantic_root, scene, native["pred_masks"].shape[0],
        args.stream_records_root, semantic_rows,
    )
    overrides = _load_track_score_overrides(args.combined_plan_root, scene)
    track_ids = tracks.pop("track_ids")
    tracks["pred_scores"] = np.asarray(
        [overrides.get(track_id, float(score)) for track_id, score in zip(track_ids, tracks["pred_scores"])],
        dtype=np.float32,
    )
    unions = _load_union_rows(args.combined_plan_root, scene, native["pred_masks"].shape[0])
    track_by_id = {track_id: cls for track_id, cls in zip(track_ids, tracks["pred_classes"])}
    valid_union = [i for i, row in enumerate(unions["rows"])
                   if int(row["selected_track_id"]) in track_by_id]
    unions["pred_masks"] = unions["pred_masks"][:, valid_union]
    unions["pred_scores"] = unions["pred_scores"][valid_union]
    unions["rows"] = [unions["rows"][i] for i in valid_union]
    unions["pred_classes"] = np.asarray(
        [track_by_id[int(row["selected_track_id"])] for row in unions["rows"]], dtype=np.int64
    )
    pieces = {
        "native_only": [native], "track_only": [tracks],
        "native_plus_track": [native, tracks], "pair_union": [native, tracks, unions],
    }[source]
    masks = np.concatenate([piece["pred_masks"] for piece in pieces], axis=1)
    classes = np.concatenate([piece["pred_classes"] for piece in pieces]).astype(np.int64)
    scores = np.concatenate([piece["pred_scores"] for piece in pieces]).astype(np.float32)
    semantic_classes = np.asarray(
        [int(instance_eval.PRED_ID_TO_ID.get(int(value), -1)) for value in classes],
        dtype=np.int64,
    )
    if evaluation == "current_class_current_score":
        return {"pred_masks": masks, "pred_classes": classes, "pred_scores": scores}
    gt_rows = _load_gt(args.gt_instance_dir / f"{scene}.txt", args.min_region_size)[1]
    best_iou, best_class, _ = _best_gt(masks, gt_rows)
    oracle_classes = classes.copy()
    valid = (best_iou >= args.oracle_min_iou) & (best_class >= 0)
    for index in np.flatnonzero(valid):
        if int(best_class[index]) in inverse_class_map:
            oracle_classes[index] = inverse_class_map[int(best_class[index])]
    if evaluation == "gt_class_current_score":
        oracle_scores, output_classes = scores, oracle_classes
    elif evaluation == "current_class_gt_score":
        oracle_scores, output_classes = _same_class_gt_scores(masks, semantic_classes, gt_rows), classes
    elif evaluation == "gt_class_gt_score":
        oracle_scores, output_classes = best_iou, oracle_classes
    elif evaluation == "gt_class_one_to_one_score":
        oracle_semantic_classes = np.asarray(
            [int(instance_eval.PRED_ID_TO_ID.get(int(value), -1)) for value in oracle_classes],
            dtype=np.int64,
        )
        oracle_scores, output_classes = _one_to_one_scores(
            masks, oracle_semantic_classes, gt_rows
        ), oracle_classes
    else:  # pragma: no cover - parser-level invariant
        raise ValueError(f"unknown Z0 evaluation: {evaluation}")
    return {"pred_masks": masks, "pred_classes": output_classes, "pred_scores": oracle_scores}


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise ValueError("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inverse = {int(semantic): int(index) for index, semantic in instance_eval.PRED_ID_TO_ID.items() if int(semantic) >= 0}
    semantic_rows_by_scene: dict[str, list[dict] | None] = {}
    semantic_coverage: list[dict] = []
    for index, scene in enumerate(scenes, 1):
        if args.reconstruct_track_semantics_from_stream:
            track_path = args.stream_records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
            frozen_tracks = json.loads(track_path.read_text()).get("tracks", [])
            if args.isolate_track_semantics_process:
                command = [
                    sys.executable, str(PROJECT_ROOT / "tools" / "reconstruct_z0_track_semantics_stdout.py"),
                    "--scene", scene, "--stream-records-root", str(args.stream_records_root),
                    "--prepared-dataset-root", str(args.prepared_dataset_root), "--config-path", str(args.config_path),
                ]
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                semantic_rows_by_scene[scene] = json.loads(result.stdout)
            else:
                semantic_rows_by_scene[scene] = _reconstruct_track_semantics_in_memory(
                    scene, frozen_tracks, args.stream_records_root, args.prepared_dataset_root, args.config_path,
                )
            valid_votes = sum(
                int(int(row.get("voted_class_index", -1)) in instance_eval.PRED_ID_TO_ID
                    and int(instance_eval.PRED_ID_TO_ID[int(row["voted_class_index"])]) >= 0)
                for row in semantic_rows_by_scene[scene]
            )
            semantic_coverage.append({
                "scene_name": scene,
                "track_count": len(frozen_tracks),
                "valid_voted_class_count": valid_votes,
                "invalid_voted_class_count": len(frozen_tracks) - valid_votes,
            })
        else:
            semantic_rows_by_scene[scene] = None
        print(f"[z0 semantics] {index}/{len(scenes)} {scene}", flush=True)

    results = {}
    total_tracks = 0
    total_valid_votes = 0
    if args.reconstruct_track_semantics_from_stream:
        total_tracks = sum(row["track_count"] for row in semantic_coverage)
        total_valid_votes = sum(row["valid_voted_class_count"] for row in semantic_coverage)
        if total_tracks > 0 and total_valid_votes == 0:
            raise RuntimeError(
                "Z0 semantic preflight failed: all reconstructed tracks lack valid YOLO-World classes"
            )
    for source in SOURCE_NAMES:
        source_result = {"ground_truth_usage": "evaluation_only/oracle_only", "evaluations": {}}
        for evaluation in EVALUATION_NAMES:
            mapping = _Predictions(scenes, lambda scene, source=source, evaluation=evaluation: _scene_prediction(
                scene, source, evaluation, args, inverse, semantic_rows_by_scene[scene],
            ))
            csv_path = args.output_dir / f"{source}__{evaluation}.csv"
            source_result["evaluations"][evaluation] = _evaluate(mapping, args.gt_instance_dir, csv_path)
        results[source] = source_result
        gc.collect()
    summary = {
        "diagnostic_type": "Z0 fixed-geometry open-vocabulary class/ranking oracle decomposition",
        "diagnostic_only": True,
        "ground_truth_usage": "evaluation_only/oracle_only",
        "scene_count": len(scenes),
        "oracle_min_iou": args.oracle_min_iou,
        "sources": results,
        "candidate_contract": {
            "native_unchanged": True, "track_append_only": True, "pair_union_append_only": True,
            "pair_union_current_class": "selected_track_YOLO-World_vote",
            "track_semantics": (
                "in_memory_reconstruction_from_frozen_2D_YOLO-World_detections"
                if args.reconstruct_track_semantics_from_stream else "frozen_semantic_ledger"
            ),
            "prediction_cache_modified": False,
            "inference_ledger_written": False,
        },
        "semantic_coverage_summary": {
            "track_count": total_tracks,
            "valid_voted_class_count": total_valid_votes,
            "invalid_voted_class_count": total_tracks - total_valid_votes,
            "valid_fraction": float(total_valid_votes / total_tracks) if total_tracks else None,
            "scene_count_with_invalid_votes": sum(
                row["invalid_voted_class_count"] > 0 for row in semantic_coverage
            ),
        },
        "semantic_coverage": semantic_coverage,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path)
    parser.add_argument("--track-root", type=Path)
    parser.add_argument("--semantic-root", type=Path)
    parser.add_argument("--stream-records-root", type=Path,
                        help="official100 stream records root; supplies per-scene native/F2 inputs")
    parser.add_argument("--prepared-dataset-root", type=Path,
                        help="prepared ScanNet scene root; required only for in-memory semantic reconstruction")
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--reconstruct-track-semantics-from-stream", action="store_true",
                        help="read frozen 2D detections and reconstruct votes in memory; writes no semantic ledger")
    parser.add_argument("--isolate-track-semantics-process", action="store_true",
                        help="run each no-write semantic reconstruction in a child process to bound memory")
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--oracle-min-iou", type=float, default=0.25)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not (args.allow_gt_evaluation and args.allow_gt_diagnostics):
        raise SystemExit("Z0 requires both --allow-gt-evaluation and --allow-gt-diagnostics")
    if not 0.0 < args.oracle_min_iou <= 1.0:
        raise SystemExit("--oracle-min-iou must be in (0, 1]")
    if args.stream_records_root is None and (args.native_prediction_cache is None or args.track_root is None):
        raise SystemExit("provide --stream-records-root or both --native-prediction-cache and --track-root")
    if args.reconstruct_track_semantics_from_stream:
        if args.stream_records_root is None or args.prepared_dataset_root is None:
            raise SystemExit("in-memory semantic reconstruction requires --stream-records-root and --prepared-dataset-root")
    elif args.semantic_root is None:
        raise SystemExit("provide --semantic-root unless reconstructing frozen track semantics in memory")
    for name in ("scene_list", "native_prediction_cache", "track_root", "semantic_root", "stream_records_root", "prepared_dataset_root", "config_path", "combined_plan_root", "gt_instance_dir", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, _resolve(value))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
