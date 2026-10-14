#!/usr/bin/env python3
"""Build the GT-free Z1 YOLO-World multi-view 200-class distribution ledger.

The ledger is read-only with respect to frozen candidates and caches.  For each
native, track, and pair-union geometry it selects the most visible prepared
views, keeps every non-zero per-class YOLO-World evidence value, and records a
common sparse distribution contract.  Pair-union rows additionally retain the
currently frozen selected-track semantic inheritance; the geometry-projected
distribution is diagnostic evidence and never changes inference labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _load_yoloworld_cache(path: Path, scene: str) -> Mapping:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "metadata" not in payload or "predictions" not in payload:
        raise ValueError(f"{scene}: YOLO-World cache is not a signed metadata/predictions payload: {path}")
    metadata = payload["metadata"]
    predictions = payload["predictions"]
    if not isinstance(metadata, Mapping) or metadata.get("scene_name") != scene:
        raise ValueError(f"{scene}: YOLO-World cache metadata scene mismatch: {path}")
    if not isinstance(predictions, Mapping):
        raise ValueError(f"{scene}: YOLO-World cache predictions are not a mapping: {path}")
    return predictions


def _points(path: Path, point_count: int) -> np.ndarray:
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if len(points) == 0 or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"invalid point indices in {path}")
    return points


def _geometry_hash(points: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(points, dtype=np.int64).tobytes()).hexdigest()


def _distribution_stats(evidence: Mapping[int, float], class_count: int, top_k: int) -> dict:
    clean = {int(key): max(0.0, float(value)) for key, value in evidence.items() if float(value) > 0.0}
    total = float(sum(clean.values()))
    ordered = sorted(clean.items(), key=lambda item: (-item[1], item[0]))
    probabilities = [(int(label), float(value / total)) for label, value in ordered] if total > 0 else []
    top1 = probabilities[0][1] if probabilities else 0.0
    top2 = probabilities[1][1] if len(probabilities) > 1 else 0.0
    if probabilities:
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for _, prob in probabilities)
        entropy /= max(math.log(max(2, class_count)), 1e-12)
    else:
        entropy = 0.0
    return {
        "evidence_total": total,
        "positive_class_count": len(probabilities),
        "normalized_entropy_class_space": float(entropy),
        "top1_class_index": int(probabilities[0][0]) if probabilities else -1,
        "top1_probability": float(top1),
        "top2_class_index": int(probabilities[1][0]) if len(probabilities) > 1 else -1,
        "top2_probability": float(top2),
        "top1_top2_margin": float(top1 - top2),
        "distribution": [
            {"class_index": int(label), "probability": float(prob)}
            for label, prob in probabilities
        ],
        "top_k": [
            {"class_index": int(label), "probability": float(prob)}
            for label, prob in probabilities[:top_k]
        ],
    }


def _frame_evidence(
    points: np.ndarray,
    frame_index: int,
    projections: np.ndarray,
    visibility: np.ndarray,
    scaling: tuple[float, float],
    prediction: Mapping,
    class_count: int,
) -> tuple[dict, dict[int, float]]:
    visible = points[visibility[frame_index, points]]
    view = {
        "frame_id": str(frame_index),
        "visible_point_count": int(len(visible)),
        "visible_point_ratio": float(len(visible) / max(1, len(points))),
        "detection_count": 0,
        "supported": False,
        "class_evidence": [],
    }
    if len(visible) == 0:
        return view, {}
    coords = projections[frame_index, visible].astype(np.float32)
    xs = coords[:, 0] / float(scaling[1])
    ys = coords[:, 1] / float(scaling[0])
    boxes = _as_numpy(prediction["bbox"]).astype(np.float32)
    labels = _as_numpy(prediction["labels"]).astype(np.int64)
    scores = _as_numpy(prediction["scores"]).astype(np.float32)
    if len(boxes) != len(labels) or len(labels) != len(scores):
        raise ValueError("YOLO-World frame bbox/label/score dimensions disagree")
    view["detection_count"] = int(len(labels))
    frame_evidence: dict[int, float] = {}
    for box, label, score in zip(boxes, labels, scores):
        label = int(label)
        if label < 0 or label >= class_count:
            continue
        x1, y1, x2, y2 = box
        inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
        if not inside.any():
            continue
        value = max(0.0, float(score)) * float(inside.mean())
        frame_evidence[label] = max(frame_evidence.get(label, 0.0), value)
    view["supported"] = bool(frame_evidence)
    view["class_evidence"] = [
        {"class_index": int(label), "evidence": float(value)}
        for label, value in sorted(frame_evidence.items(), key=lambda item: (-item[1], item[0]))
    ]
    return view, frame_evidence


def _candidate_distribution(
    points: np.ndarray,
    frame_ids: list[str],
    frame_lookup: Mapping[str, int],
    predictions: Mapping,
    projections: np.ndarray,
    visibility: np.ndarray,
    scaling: tuple[float, float],
    max_views: int,
    min_visible_points: int,
    top_k: int,
    class_count: int,
    support_frame_ids: set[str] | None,
) -> dict:
    eligible = []
    missing_prediction = 0
    missing_frame = 0
    for frame_id in frame_ids:
        key = str(frame_id)
        index = frame_lookup.get(key)
        if index is None:
            missing_frame += 1
            continue
        if key not in predictions:
            missing_prediction += 1
            continue
        count = int(visibility[index, points].sum())
        if count >= min_visible_points:
            role = (
                "native_prepared_view" if support_frame_ids is None
                else "track_support_view" if key in support_frame_ids
                else "independent_review_view"
            )
            eligible.append((count, key, index, role))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    if support_frame_ids is None:
        selected = eligible[:max_views]
    else:
        support = [item for item in eligible if item[3] == "track_support_view"]
        independent = [item for item in eligible if item[3] == "independent_review_view"]
        support_limit = max(1, max_views // 2)
        independent_limit = max_views - support_limit
        selected = support[:support_limit] + independent[:independent_limit]
        if len(selected) < max_views:
            used = {(item[1], item[2]) for item in selected}
            remainder = [item for item in eligible if (item[1], item[2]) not in used]
            selected.extend(remainder[: max_views - len(selected)])
        selected.sort(key=lambda item: (-item[0], item[1]))
    aggregate: dict[int, float] = defaultdict(float)
    aggregate_by_role: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    support_views: Counter[int] = Counter()
    views = []
    for _, frame_id, frame_index, role in selected:
        view, evidence = _frame_evidence(
            points, frame_index, projections, visibility, scaling, predictions[frame_id], class_count
        )
        view["frame_id"] = str(frame_id)
        view["frame_index"] = int(frame_index)
        view["view_role"] = role
        views.append(view)
        for label, value in evidence.items():
            aggregate[label] += value
            aggregate_by_role[role][label] += value
            support_views[label] += 1
    stats = _distribution_stats(aggregate, class_count, top_k)
    role_distributions = {
        role: _distribution_stats(values, class_count, top_k)
        for role, values in sorted(aggregate_by_role.items())
    }
    for role in (
        "native_prepared_view", "track_support_view", "independent_review_view",
    ):
        role_distributions.setdefault(role, _distribution_stats({}, class_count, top_k))
    stats.update({
        "candidate_point_count": int(len(points)),
        "eligible_view_count": int(len(eligible)),
        "selected_view_count": int(len(selected)),
        "evidence_view_count": int(sum(bool(row["supported"]) for row in views)),
        "zero_support_view_count": int(sum(not row["supported"] for row in views)),
        "selected_view_role_counts": dict(sorted(Counter(row["view_role"] for row in views).items())),
        "evidence_view_role_counts": dict(sorted(Counter(
            row["view_role"] for row in views if row["supported"]
        ).items())),
        "missing_frame_count": int(missing_frame),
        "missing_prediction_count": int(missing_prediction),
        "class_support_views": [
            {"class_index": int(label), "support_view_count": int(count)}
            for label, count in sorted(support_views.items(), key=lambda item: (-item[1], item[0]))
        ],
        "role_distributions": role_distributions,
        "views": views,
    })
    return stats


def _frozen_support_vote(
    points: np.ndarray,
    support_frame_ids: set[str],
    frame_lookup: Mapping[str, int],
    predictions: Mapping,
    projections: np.ndarray,
    visibility: np.ndarray,
    scaling: tuple[float, float],
    class_count: int,
    top_k: int,
) -> dict:
    """Reproduce the frozen Z0 top-1 vote over every track support frame."""
    aggregate: dict[int, float] = defaultdict(float)
    visible_views = 0
    evidence_views = 0
    missing_frames = 0
    missing_predictions = 0
    for frame_id in sorted(support_frame_ids, key=int):
        frame_index = frame_lookup.get(frame_id)
        if frame_index is None:
            missing_frames += 1
            continue
        prediction = predictions.get(frame_id)
        if prediction is None:
            missing_predictions += 1
            continue
        if not visibility[frame_index, points].any():
            continue
        visible_views += 1
        _, evidence = _frame_evidence(
            points, frame_index, projections, visibility, scaling, prediction, class_count
        )
        if evidence:
            evidence_views += 1
        for label, value in evidence.items():
            aggregate[label] += value
    stats = _distribution_stats(aggregate, class_count, top_k)
    stats.update({
        "input_support_view_count": len(support_frame_ids),
        "visible_support_view_count": visible_views,
        "evidence_support_view_count": evidence_views,
        "missing_frame_count": missing_frames,
        "missing_prediction_count": missing_predictions,
    })
    return stats


def _load_native(root: Path, scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = root / scene / "native_cache" / f"{scene}_pred_"
    masks = np.asarray(np.load(str(prefix) + "masks.npy", mmap_mode="r"), dtype=bool)
    classes = np.asarray(np.load(str(prefix) + "classes.npy"), dtype=np.int64)
    scores = np.asarray(np.load(str(prefix) + "scores.npy"), dtype=np.float32)
    if masks.ndim != 2 or masks.shape[1] != len(classes) or len(classes) != len(scores):
        raise ValueError(f"{scene}: native cache dimensions disagree")
    return masks, classes, scores


def _load_tracks(root: Path, scene: str) -> list[dict]:
    path = root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
    return json.loads(path.read_text()).get("tracks", [])


def _load_union_rows(plan_root: Path, scene: str) -> list[dict]:
    path = plan_root / "pair_union_append_candidates.jsonl"
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row["scene_name"]) == scene:
                    rows.append(row)
    return sorted(rows, key=lambda row: int(row["candidate_id"]))


def _scene_records(scene: str, args: argparse.Namespace, prompts: list[str]) -> tuple[list[dict], list[dict], dict]:
    from utils import WORLD_2_CAM

    masks, native_classes, native_scores = _load_native(args.stream_records_root, scene)
    point_count = int(masks.shape[0])
    tracks = _load_tracks(args.stream_records_root, scene)
    track_by_id = {int(row["track_id"]): row for row in tracks}
    union_rows = _load_union_rows(args.combined_plan_root, scene)
    predictions = _load_yoloworld_cache(
        args.stream_records_root / scene / "yoloworld_bboxes_2d" / f"{scene}.pt", scene
    )
    config = args.config
    world = WORLD_2_CAM(str(args.prepared_dataset_root / scene), args.depth_scale, config)
    projection, visibility = world.get_mesh_projections()
    projection = projection.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    frame_lookup = {Path(path).stem: index for index, path in enumerate(world.color_paths)}
    prediction_keys = {str(key) for key in predictions}
    if set(frame_lookup) - prediction_keys:
        raise ValueError(f"{scene}: YOLO-World cache does not cover all prepared color frames")
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    all_frame_ids = sorted(frame_lookup, key=lambda value: int(value))
    records = []
    geometry_nodes: dict[str, int] = {}
    semantic_evidence_nodes: dict[tuple, int] = {}
    evidence_records: list[dict] = []

    def append_record(
        source: str, candidate_id: int, points: np.ndarray, frame_ids: list[str],
        support_frame_ids: set[str] | None, extra: dict,
    ) -> dict:
        geometry_key = _geometry_hash(points)
        geometry_node_id = geometry_nodes.setdefault(geometry_key, len(geometry_nodes))
        frame_contract = (
            "all_prepared_frames"
            if source == "native"
            else "all_prepared_frames_with_support_independent_roles"
        )
        evidence_key = (
            geometry_key, frame_contract, tuple(map(str, frame_ids)),
            None if support_frame_ids is None else tuple(sorted(support_frame_ids, key=int)),
        )
        semantic_evidence_node_id = semantic_evidence_nodes.get(evidence_key)
        if semantic_evidence_node_id is None:
            semantic_evidence_node_id = len(semantic_evidence_nodes)
            semantic_evidence_nodes[evidence_key] = semantic_evidence_node_id
            distribution = _candidate_distribution(
                points, frame_ids, frame_lookup, predictions, projection, visibility, scaling,
                args.max_views, args.min_visible_points, args.top_k,
                len(prompts),
                support_frame_ids,
            )
            frozen_support_vote = (
                None if support_frame_ids is None else _frozen_support_vote(
                    points, support_frame_ids, frame_lookup, predictions, projection, visibility,
                    scaling, len(prompts), args.top_k,
                )
            )
            evidence_records.append({
                "scene_name": scene,
                "semantic_evidence_node_id": int(semantic_evidence_node_id),
                "semantic_evidence_node_key": f"{scene}:{semantic_evidence_node_id}",
                "geometry_node_id": int(geometry_node_id),
                "geometry_hash": geometry_key,
                "frame_contract": frame_contract,
                "input_frame_ids": [str(value) for value in frame_ids],
                "track_support_frame_ids": (
                    [] if support_frame_ids is None else sorted(support_frame_ids, key=int)
                ),
                "frozen_support_vote": frozen_support_vote,
                "distribution": distribution,
            })
        record = {
            "scene_name": scene,
            "candidate_source": source,
            "candidate_id": int(candidate_id),
            "geometry_node_id": int(geometry_node_id),
            "geometry_hash": geometry_key,
            "semantic_evidence_node_id": int(semantic_evidence_node_id),
            "semantic_evidence_node_key": f"{scene}:{semantic_evidence_node_id}",
            "frame_contract": frame_contract,
            "class_prompt_count": int(len(prompts)),
            **extra,
        }
        records.append(record)
        return record

    for candidate_id in range(masks.shape[1]):
        points = np.flatnonzero(masks[:, candidate_id]).astype(np.int64)
        append_record(
            "native", candidate_id, points, all_frame_ids, None,
            {
                "native_class_index": int(native_classes[candidate_id]),
                "native_score": float(native_scores[candidate_id]),
                "inference_semantic_source": "native_current_class",
            },
        )
    track_binding_by_id = {}
    for track in tracks:
        track_id = int(track["track_id"])
        points = _points(Path(track["points_path"]), point_count)
        frame_ids = [str(value) for value in track.get("frame_ids", [])]
        binding = append_record(
            "track", track_id, points, all_frame_ids, set(frame_ids),
            {
                "track_id": track_id,
                "track_support_view_count": int(track.get("support_view_count", len(frame_ids))),
                "track_quality": float(track.get("mean_node_quality", 0.0)),
                "inference_semantic_source": "track_yoloworld_vote",
            },
        )
        evidence = evidence_records[int(binding["semantic_evidence_node_id"])]
        frozen_vote = evidence["frozen_support_vote"]
        binding["current_voted_class_index"] = int(frozen_vote["top1_class_index"])
        binding["current_vote_probability"] = float(frozen_vote["top1_probability"])
        track_binding_by_id[track_id] = binding
    for row in union_rows:
        selected_track_id = int(row["selected_track_id"])
        if selected_track_id not in track_by_id:
            continue
        points = _points(Path(row["points_path"]), point_count)
        track_frame_ids = [str(value) for value in track_by_id[selected_track_id].get("frame_ids", [])]
        selected_track_binding = track_binding_by_id[selected_track_id]
        append_record(
            "pair_union", int(row["candidate_id"]), points, all_frame_ids, set(track_frame_ids),
            {
                "selected_track_id": selected_track_id,
                "selected_track_semantic_evidence_node_id": int(
                    selected_track_binding["semantic_evidence_node_id"]
                ),
                "selected_track_semantic_evidence_node_key": str(
                    selected_track_binding["semantic_evidence_node_key"]
                ),
                "inherited_voted_class_index": int(
                    selected_track_binding["current_voted_class_index"]
                ),
                "pair_union_new_score": float(row["new_score"]),
                "inference_semantic_source": "selected_track_yoloworld_vote",
                "geometry_distribution_is_diagnostic_only": True,
            },
        )
    source_counts = Counter(row["candidate_source"] for row in records)
    evidence_by_id = {int(row["semantic_evidence_node_id"]): row for row in evidence_records}
    summary = {
        "scene_name": scene,
        "point_count": point_count,
        "prepared_frame_count": len(all_frame_ids),
        "yoloworld_prediction_frame_count": len(predictions),
        "candidate_counts": dict(sorted(source_counts.items())),
        "geometry_node_count": len(geometry_nodes),
        "semantic_evidence_node_count": len(evidence_records),
        "duplicate_geometry_candidate_count": len(records) - len(geometry_nodes),
        "candidate_bindings_with_selected_views": sum(
            evidence_by_id[int(row["semantic_evidence_node_id"])]["distribution"]["selected_view_count"] > 0
            for row in records
        ),
        "candidate_bindings_with_evidence_views": sum(
            evidence_by_id[int(row["semantic_evidence_node_id"])]["distribution"]["evidence_view_count"] > 0
            for row in records
        ),
    }
    del world, projection, visibility
    return records, evidence_records, summary


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = [str(value) for value in args.config["network2d"]["text_prompts"]]
    if len(prompts) < 190:
        raise ValueError(f"YOLO-World prompt list is unexpectedly short: {len(prompts)}")
    scene_summaries = []
    total_records = Counter()
    total_geometry_nodes = 0
    total_semantic_evidence_nodes = 0
    with (args.output_dir / "candidate_bindings.jsonl").open("w") as candidate_output, \
            (args.output_dir / "semantic_evidence_nodes.jsonl").open("w") as evidence_output:
        for index, scene in enumerate(scenes, 1):
            records, evidence_records, summary = _scene_records(scene, args, prompts)
            for record in records:
                candidate_output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            for record in evidence_records:
                evidence_output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            scene_summaries.append(summary)
            total_records.update(summary["candidate_counts"])
            total_geometry_nodes += int(summary["geometry_node_count"])
            total_semantic_evidence_nodes += int(summary["semantic_evidence_node_count"])
            print(
                f"[z1 ledger] {index}/{len(scenes)} {scene}: "
                f"{len(records)} candidates, {len(evidence_records)} evidence nodes",
                flush=True,
            )
    summary = {
        "diagnostic_type": "Z1 GT-free YOLO-World multi-view 200-class distribution ledger",
        "diagnostic_only": True,
        "ground_truth_usage": "none",
        "candidate_mutation": False,
        "scene_count": len(scenes),
        "class_prompt_count": len(prompts),
        "class_prompts": prompts,
        "class_space_size": len(prompts),
        "class_prompt_contract": "config.network2d.text_prompts; prediction indices 0..class_space_size-1; current ScanNet200 cache exposes 198 instance prompts",
        "evidence_contract": "per-frame per-class max(score * projected-point-inside-fraction), summed over selected views",
        "join_contract": "candidate_bindings.semantic_evidence_node_key -> semantic_evidence_nodes.semantic_evidence_node_key; key is globally unique scene_name:local_id",
        "view_contract": {
            "native": "all prepared frames, ranked by visible point count",
            "track": "all prepared frames; support and independent-review views are stratified before ranking",
            "pair_union": "all prepared frames with selected-track support/review stratification; geometry distribution diagnostic-only",
            "max_views": args.max_views,
            "min_visible_points": args.min_visible_points,
        },
        "candidate_counts": dict(sorted(total_records.items())),
        "geometry_node_count_sum_over_scenes": total_geometry_nodes,
        "semantic_evidence_node_count_sum_over_scenes": total_semantic_evidence_nodes,
        "scene_summaries": scene_summaries,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--prepared-dataset-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--max-views", type=int, default=40)
    parser.add_argument("--min-visible-points", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if args.max_views <= 0 or args.min_visible_points <= 0 or args.top_k <= 0:
        raise SystemExit("--max-views, --min-visible-points and --top-k must be positive")
    for name in (
        "scene_list", "stream_records_root", "prepared_dataset_root", "combined_plan_root",
        "output_dir", "config_path",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
