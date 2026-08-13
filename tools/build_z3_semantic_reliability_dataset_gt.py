#!/usr/bin/env python3
"""Build official-train supervision for class-agnostic semantic reliability."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402
from tools.diagnose_z0_open_vocab_oracle_gt import (  # noqa: E402
    _load_gt,
    _load_native,
    _load_track_score_overrides,
    _load_union_rows,
    _one_to_one_scores,
    _same_class_gt_scores,
)


SOURCE_TO_INDEX = {"native": 0, "track": 1, "pair_union": 2}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0]
    return float(-(positive * np.log(positive)).sum() / math.log(len(distribution))) if len(positive) else 0.0


def _js(left: np.ndarray, right: np.ndarray) -> float:
    if left.sum() <= 0 or right.sum() <= 0:
        return 0.0
    middle = 0.5 * (left + right)
    left_mask, right_mask = left > 0, right > 0
    value = 0.5 * float((left[left_mask] * np.log(left[left_mask] / middle[left_mask])).sum())
    value += 0.5 * float((right[right_mask] * np.log(right[right_mask] / middle[right_mask])).sum())
    return value / math.log(2.0)


def _stats(distribution: np.ndarray) -> tuple[float, float, float]:
    if distribution.sum() <= 0:
        return 0.0, 0.0, 0.0
    order = np.partition(distribution, -2)
    top, second = float(order[-1]), float(order[-2])
    return top, top - second, _entropy(distribution)


def _points(path: Path, point_count: int) -> np.ndarray:
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points) or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"invalid candidate points: {path}")
    return points


def _feature_names() -> list[str]:
    return [
        "source_native", "source_track", "source_pair_union", "original_score",
        "log1p_point_count", "log1p_bound_candidate_count", "track_quality",
        "log1p_track_support_views", "geometry_yolo_available", "inherited_yolo_available",
        "geometry_yolo_class_probability", "inherited_yolo_class_probability",
        "geometry_yolo_top_probability", "geometry_yolo_margin", "geometry_yolo_entropy",
        "inherited_yolo_top_probability", "inherited_yolo_margin", "inherited_yolo_entropy",
        "geometry_inherited_yolo_js", "class_is_geometry_yolo_top1", "class_is_inherited_yolo_top1",
        "geometry_alpha_available", "inherited_alpha_available",
        "geometry_alpha_class_probability", "inherited_alpha_class_probability",
        "geometry_alpha_top_probability", "geometry_alpha_margin", "geometry_alpha_entropy",
        "inherited_alpha_top_probability", "inherited_alpha_margin", "inherited_alpha_entropy",
        "geometry_yolo_alpha_js", "inherited_yolo_alpha_js", "geometry_inherited_alpha_js",
        "class_is_geometry_alpha_top1", "class_is_inherited_alpha_top1",
        "fused_inherited_class_probability", "fused_inherited_top_probability",
        "fused_inherited_margin", "fused_inherited_entropy", "class_is_fused_inherited_top1",
    ]


def _feature_row(
    source: str, class_index: int, original_score: float, point_count: int,
    bound_candidate_count: int, track_quality: float, support_views: int,
    geometry_yolo: np.ndarray, inherited_yolo: np.ndarray,
    geometry_alpha: np.ndarray, inherited_alpha: np.ndarray,
    geometry_alpha_available: bool, inherited_alpha_available: bool,
) -> list[float]:
    geometry_yolo_available = bool(geometry_yolo.sum() > 0)
    inherited_yolo_available = bool(inherited_yolo.sum() > 0)
    gy_top, gy_margin, gy_entropy = _stats(geometry_yolo)
    iy_top, iy_margin, iy_entropy = _stats(inherited_yolo)
    ga_top, ga_margin, ga_entropy = _stats(geometry_alpha)
    ia_top, ia_margin, ia_entropy = _stats(inherited_alpha)
    fused = inherited_yolo.copy()
    if inherited_alpha_available:
        fused = 0.5 * inherited_yolo + 0.5 * inherited_alpha if inherited_yolo_available else inherited_alpha.copy()
    if fused.sum() > 0:
        fused /= fused.sum()
    fused_top, fused_margin, fused_entropy = _stats(fused)
    source_flags = [float(source == name) for name in ("native", "track", "pair_union")]
    return source_flags + [
        float(original_score), math.log1p(point_count), math.log1p(bound_candidate_count),
        float(track_quality), math.log1p(support_views),
        float(geometry_yolo_available), float(inherited_yolo_available),
        float(geometry_yolo[class_index]), float(inherited_yolo[class_index]),
        gy_top, gy_margin, gy_entropy, iy_top, iy_margin, iy_entropy,
        _js(geometry_yolo, inherited_yolo),
        float(geometry_yolo_available and class_index == int(np.argmax(geometry_yolo))),
        float(inherited_yolo_available and class_index == int(np.argmax(inherited_yolo))),
        float(geometry_alpha_available), float(inherited_alpha_available),
        float(geometry_alpha[class_index]), float(inherited_alpha[class_index]),
        ga_top, ga_margin, ga_entropy, ia_top, ia_margin, ia_entropy,
        _js(geometry_yolo, geometry_alpha), _js(inherited_yolo, inherited_alpha),
        _js(geometry_alpha, inherited_alpha),
        float(geometry_alpha_available and class_index == int(np.argmax(geometry_alpha))),
        float(inherited_alpha_available and class_index == int(np.argmax(inherited_alpha))),
        float(fused[class_index]), fused_top, fused_margin, fused_entropy,
        float(fused.sum() > 0 and class_index == int(np.argmax(fused))),
    ]


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    bindings = _read_jsonl(args.z1_root / "candidate_bindings.jsonl")
    by_scene_source: dict[tuple[str, str], dict[int, dict]] = {}
    for scene in scenes:
        for source in SOURCE_TO_INDEX:
            rows = [row for row in bindings if str(row["scene_name"]) == scene and str(row["candidate_source"]) == source]
            mapped = {int(row["candidate_id"]): row for row in rows}
            if len(mapped) != len(rows):
                raise ValueError(f"{scene}:{source}: duplicate candidate ID")
            by_scene_source[(scene, source)] = mapped

    node_rows = _read_jsonl(args.unified_ledger_root / "nodes.jsonl")
    node_by_key = {str(row["semantic_evidence_node_key"]): row for row in node_rows}
    if len(node_by_key) != len(node_rows):
        raise ValueError("duplicate unified-ledger node key")
    with np.load(args.unified_ledger_root / "semantic_distributions.npz") as payload:
        distributions = {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files}
    if any(array.shape != (len(node_rows), args.class_count) for array in distributions.values()):
        raise ValueError("unified-ledger distribution dimensions disagree")

    union_rows = _read_jsonl(args.combined_plan_root / "pair_union_append_candidates.jsonl")
    union_by_scene = {
        scene: {int(row["candidate_id"]): row for row in union_rows if str(row["scene_name"]) == scene}
        for scene in scenes
    }
    metadata, features, labels_iou, labels_one_to_one_iou = [], [], [], []
    thresholds = np.arange(0.50, 0.951, 0.05, dtype=np.float32)
    source_counts = {source: 0 for source in SOURCE_TO_INDEX}
    omitted_invalid_class = {source: 0 for source in SOURCE_TO_INDEX}

    for scene_index, scene in enumerate(scenes, 1):
        native = _load_native(Path(), scene, args.stream_records_root)
        point_count = native["pred_masks"].shape[0]
        gt_rows = _load_gt(args.gt_instance_dir / f"{scene}.txt", args.min_region_size)[1]
        overrides = _load_track_score_overrides(args.combined_plan_root, scene)
        track_path = args.stream_records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
        tracks = {int(row["track_id"]): row for row in json.loads(track_path.read_text()).get("tracks", [])}

        scene_masks, scene_classes, scene_rows = [], [], []
        for candidate_id, binding in sorted(by_scene_source[(scene, "native")].items()):
            if candidate_id >= native["pred_masks"].shape[1]:
                raise ValueError(f"{scene}: native candidate ID outside cache: {candidate_id}")
            class_index = int(binding["native_class_index"])
            if int(instance_eval.PRED_ID_TO_ID.get(class_index, -1)) < 0:
                omitted_invalid_class["native"] += 1
                continue
            scene_masks.append(native["pred_masks"][:, candidate_id])
            scene_classes.append(class_index)
            scene_rows.append(("native", candidate_id, binding, float(binding["native_score"]), None))

        for candidate_id, binding in sorted(by_scene_source[(scene, "track")].items()):
            node = node_by_key[str(binding["semantic_evidence_node_key"])]
            index = int(node["node_index"])
            inherited_yolo = distributions["inherited_yolo"][index]
            inherited_alpha = distributions["inherited_alpha"][index]
            fused = inherited_yolo.copy()
            if bool(node["inherited_alpha_available"]):
                fused = 0.5 * inherited_yolo + 0.5 * inherited_alpha if fused.sum() > 0 else inherited_alpha.copy()
            class_index = int(np.argmax(fused)) if fused.sum() > 0 else -1
            if int(instance_eval.PRED_ID_TO_ID.get(class_index, -1)) < 0:
                omitted_invalid_class["track"] += 1
                continue
            track = tracks[candidate_id]
            mask = np.zeros(point_count, dtype=bool)
            mask[_points(Path(track["points_path"]), point_count)] = True
            scene_masks.append(mask)
            scene_classes.append(class_index)
            score = overrides.get(candidate_id, max(0.0, float(track.get("mean_node_quality", 0.0))))
            scene_rows.append(("track", candidate_id, binding, score, track))

        for candidate_id, binding in sorted(by_scene_source[(scene, "pair_union")].items()):
            node = node_by_key[str(binding["semantic_evidence_node_key"])]
            index = int(node["node_index"])
            inherited_yolo = distributions["inherited_yolo"][index]
            inherited_alpha = distributions["inherited_alpha"][index]
            fused = inherited_yolo.copy()
            if bool(node["inherited_alpha_available"]):
                fused = 0.5 * inherited_yolo + 0.5 * inherited_alpha if fused.sum() > 0 else inherited_alpha.copy()
            class_index = int(np.argmax(fused)) if fused.sum() > 0 else -1
            if int(instance_eval.PRED_ID_TO_ID.get(class_index, -1)) < 0:
                omitted_invalid_class["pair_union"] += 1
                continue
            union = union_by_scene[scene][candidate_id]
            mask = np.zeros(point_count, dtype=bool)
            mask[_points(Path(union["points_path"]), point_count)] = True
            scene_masks.append(mask)
            scene_classes.append(class_index)
            selected_track = tracks[int(binding["selected_track_id"])]
            scene_rows.append(("pair_union", candidate_id, binding, float(binding["pair_union_new_score"]), selected_track))

        masks = np.stack(scene_masks, axis=1) if scene_masks else np.zeros((point_count, 0), dtype=bool)
        semantic_classes = np.asarray([
            int(instance_eval.PRED_ID_TO_ID.get(int(value), -1)) for value in scene_classes
        ], dtype=np.int64)
        if np.any(semantic_classes < 0):
            raise AssertionError(f"{scene}: invalid semantic class escaped prefilter")
        same_class_iou = _same_class_gt_scores(masks, semantic_classes, gt_rows)
        one_to_one_iou = _one_to_one_scores(masks, semantic_classes, gt_rows)

        for local_index, (source, candidate_id, binding, score, track) in enumerate(scene_rows):
            key = str(binding["semantic_evidence_node_key"])
            node = node_by_key[key]
            index = int(node["node_index"])
            class_index = int(scene_classes[local_index])
            selected_binding = binding
            if source == "pair_union":
                selected_binding = by_scene_source[(scene, "track")][int(binding["selected_track_id"])]
            track_quality = float(selected_binding.get("track_quality", 0.0)) if source != "native" else 0.0
            support_views = int(selected_binding.get("track_support_view_count", 0)) if source != "native" else 0
            feature = _feature_row(
                source, class_index, score, int(node["point_count"]), int(node["bound_candidate_count"]),
                track_quality, support_views, distributions["geometry_yolo"][index],
                distributions["inherited_yolo"][index], distributions["geometry_alpha"][index],
                distributions["inherited_alpha"][index], bool(node["geometry_alpha_available"]),
                bool(node["inherited_alpha_available"]),
            )
            if len(feature) != len(_feature_names()) or not np.all(np.isfinite(feature)):
                raise ValueError(f"{scene}:{source}:{candidate_id}: invalid feature row")
            iou = float(same_class_iou[local_index])
            one_to_one = float(one_to_one_iou[local_index])
            metadata.append({
                "row_index": len(metadata), "scene_name": scene, "candidate_source": source,
                "candidate_id": candidate_id, "semantic_evidence_node_key": key,
                "class_index": class_index, "semantic_class_id": int(semantic_classes[local_index]),
                "original_score": float(score), "label_same_class_iou": iou,
                "label_ap_quality": float((iou >= thresholds).mean()),
                "label_tp50": int(iou >= 0.5), "label_tp25": int(iou >= 0.25),
                "label_one_to_one_iou": one_to_one,
                "label_one_to_one_ap_quality": float((one_to_one >= thresholds).mean()),
                "label_one_to_one_tp50": int(one_to_one >= 0.5),
            })
            features.append(feature)
            labels_iou.append(iou)
            labels_one_to_one_iou.append(one_to_one)
            source_counts[source] += 1
        print(f"[Z3 dataset] {scene_index}/{len(scenes)} {scene}: rows={len(scene_rows)}", flush=True)

    matrix = np.asarray(features, dtype=np.float32)
    labels_iou_array = np.asarray(labels_iou, dtype=np.float32)
    labels_one_to_one_iou_array = np.asarray(labels_one_to_one_iou, dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_dir / "dataset.npz", features=matrix, label_same_class_iou=labels_iou_array,
        label_ap_quality=np.asarray([(value >= thresholds).mean() for value in labels_iou_array], dtype=np.float32),
        label_tp50=(labels_iou_array >= 0.5).astype(np.int8),
        label_tp25=(labels_iou_array >= 0.25).astype(np.int8),
        label_one_to_one_iou=labels_one_to_one_iou_array,
        label_one_to_one_ap_quality=np.asarray(
            [(value >= thresholds).mean() for value in labels_one_to_one_iou_array], dtype=np.float32
        ),
        label_one_to_one_tp50=(labels_one_to_one_iou_array >= 0.5).astype(np.int8),
    )
    with (args.output_dir / "rows.jsonl").open("w") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "feature_schema.json").write_text(
        json.dumps({"feature_names": _feature_names(), "class_id_is_feature": False}, indent=2) + "\n"
    )
    summary = {
        "diagnostic_type": "official-train GT semantic reliability supervision dataset",
        "ground_truth_usage": "explicit_official_train_supervision", "candidate_mutation": False,
        "scene_count": len(scenes), "row_count": len(metadata), "feature_count": matrix.shape[1],
        "source_row_counts": source_counts, "omitted_invalid_class_counts": omitted_invalid_class,
        "positive_counts": {
            "tp25": int((labels_iou_array >= 0.25).sum()),
            "tp50": int((labels_iou_array >= 0.50).sum()),
            "ap_quality_nonzero": int((labels_iou_array >= 0.50).sum()),
            "one_to_one_tp50": int((labels_one_to_one_iou_array >= 0.50).sum()),
        },
        "class_selection_contract": {
            "native": "retain every existing native candidate class hypothesis",
            "track": "frozen-support YOLO plus limited-context Alpha equal-weight top1",
            "pair_union": "selected-track frozen-support YOLO plus limited-context Alpha equal-weight top1",
        },
        "target_contract": {
            "independent": "mean TP eligibility over IoU thresholds 0.50:0.05:0.95",
            "one_to_one": "same thresholds after per-scene, per-class Hungarian candidate-GT assignment",
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--unified-ledger-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--allow-gt-supervision", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_supervision:
        raise SystemExit("dataset construction requires --allow-gt-supervision")
    for name in (
        "scene_list", "z1_root", "unified_ledger_root", "stream_records_root",
        "combined_plan_root", "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
