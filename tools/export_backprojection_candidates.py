import argparse
import gc
import json
import os
import os.path as osp
import sys
from collections import defaultdict

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200
from run_evaluation import load_yaml
from utils import OpenYolo3D


DEFAULT_SMALL_OBJECT_CLASSES = (
    "book",
    "bottle",
    "bowl",
    "camera",
    "candle",
    "clock",
    "plate",
    "sculpture",
    "switch",
    "tablet",
    "tissue-paper",
    "vase",
    "vent",
    "wall-plug",
)
DEFAULT_LARGE_SURFACE_CLASSES = (
    "bed",
    "blanket",
    "blinds",
    "cabinet",
    "comforter",
    "cushion",
    "desk",
    "door",
    "panel",
    "picture",
    "rug",
    "shelf",
    "sofa",
    "table",
    "tv-screen",
    "tv-stand",
    "window",
)
DEFAULT_SAM_REFINE_CLASSES = DEFAULT_LARGE_SURFACE_CLASSES
DEFAULT_MLLM_REFINE_CLASSES = (
    "book",
    "bottle",
    "box",
    "bowl",
    "camera",
    "desk-organizer",
    "plant-stand",
    "sculpture",
    "switch",
    "tablet",
    "tissue-paper",
    "vent",
    "wall-plug",
)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _safe_label(labels, label_id):
    if label_id < 0 or label_id >= len(labels):
        return "unknown"
    return labels[label_id]


def _parse_class_names(value, default=()):
    if value is None:
        return set(default)
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _frame_key(image_path):
    return osp.basename(image_path).split(".")[0]


def _clamp_box(box, width, height, min_size=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + min_size, min(width, x2))
    y2 = max(y1 + min_size, min(height, y2))
    return x1, y1, x2, y2


def _prepare_image(image):
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8, copy=True)


def _draw_rectangle(image, box, color=(0, 255, 0), thickness=3):
    x1, y1, x2, y2 = box
    for offset in range(thickness):
        image[max(0, y1 - offset):min(image.shape[0], y1 + offset + 1), x1:x2] = color
        image[max(0, y2 - 1 - offset):min(image.shape[0], y2 + offset), x1:x2] = color
        image[y1:y2, max(0, x1 - offset):min(image.shape[1], x1 + offset + 1)] = color
        image[y1:y2, max(0, x2 - 1 - offset):min(image.shape[1], x2 + offset)] = color


def _save_evidence_images(openyolo3d, frame_idx, box_color, seed_indices, output_prefix):
    image = _prepare_image(imageio.imread(openyolo3d.world2cam.color_paths[frame_idx]))
    height, width = image.shape[:2]
    box_color = _clamp_box(box_color, width, height)

    projections, _ = openyolo3d.mesh_projections
    coords_depth = _to_numpy(projections)[frame_idx, seed_indices].astype(np.float32)
    xs = np.round(coords_depth[:, 0] / openyolo3d.scaling_params[1]).astype(np.int64)
    ys = np.round(coords_depth[:, 1] / openyolo3d.scaling_params[0]).astype(np.int64)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    overlay = image.copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    if valid.any():
        current = overlay[ys[valid], xs[valid]].astype(np.float32)
        overlay[ys[valid], xs[valid]] = (0.35 * current + 0.65 * red).astype(np.uint8)
    _draw_rectangle(overlay, box_color)

    x1, y1, x2, y2 = box_color
    crop = image[y1:y2, x1:x2]
    overlay_crop = overlay[y1:y2, x1:x2]

    context_path = f"{output_prefix}_context.jpg"
    crop_path = f"{output_prefix}_crop.jpg"
    overlay_path = f"{output_prefix}_overlay.jpg"
    imageio.imwrite(context_path, overlay)
    imageio.imwrite(crop_path, crop)
    imageio.imwrite(overlay_path, overlay_crop)
    return {
        "color_path": openyolo3d.world2cam.color_paths[frame_idx],
        "bbox_xyxy": [int(v) for v in box_color],
        "context_path": context_path,
        "crop_path": crop_path,
        "overlay_path": overlay_path,
    }


def _existing_mask_metrics(existing_masks, seed_indices):
    seed_count = int(len(seed_indices))
    if seed_count == 0 or existing_masks.size == 0:
        return {
            "seed_in_existing_mask_ratio": 0.0,
            "best_existing_mask_id": None,
            "best_existing_seed_coverage": 0.0,
            "best_existing_iou": 0.0,
        }

    seed_rows = existing_masks[seed_indices]
    seed_in_existing = seed_rows.any(axis=1)
    intersections = seed_rows.sum(axis=0).astype(np.float64)
    mask_sizes = existing_masks.sum(axis=0).astype(np.float64)
    unions = mask_sizes + seed_count - intersections
    ious = np.divide(intersections, np.maximum(unions, 1.0))
    coverages = intersections / max(1, seed_count)

    best_iou_id = int(np.argmax(ious)) if len(ious) > 0 else None
    best_cov_id = int(np.argmax(coverages)) if len(coverages) > 0 else None
    best_id = best_iou_id if best_iou_id is not None else best_cov_id
    return {
        "seed_in_existing_mask_ratio": float(seed_in_existing.sum() / max(1, seed_count)),
        "best_existing_mask_id": best_id,
        "best_existing_seed_coverage": float(coverages[best_cov_id]) if best_cov_id is not None else 0.0,
        "best_existing_iou": float(ious[best_iou_id]) if best_iou_id is not None else 0.0,
    }


def _seed_iou(left_indices, right_indices):
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0.0
    intersection = np.intersect1d(left_indices, right_indices, assume_unique=False).size
    union = len(left_indices) + len(right_indices) - intersection
    return float(intersection / max(1, union))


def _box_iou_np(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + boxes_area - inter, 1.0)


def _select_2d_nms_indices(boxes, scores, class_ids, iou_threshold=0.0, same_class_only=True):
    if iou_threshold is None or float(iou_threshold) <= 0.0 or len(boxes) == 0:
        return np.arange(len(boxes), dtype=np.int64)
    order = np.argsort(-scores)
    selected = []
    for det_id in order:
        det_id = int(det_id)
        suppress = False
        for kept_id in selected:
            if same_class_only and int(class_ids[det_id]) != int(class_ids[kept_id]):
                continue
            iou = float(_box_iou_np(boxes[det_id], boxes[np.asarray([kept_id], dtype=np.int64)])[0])
            if iou >= float(iou_threshold):
                suppress = True
                break
        if not suppress:
            selected.append(det_id)
    return np.asarray(selected, dtype=np.int64)


def _resolve_scene_names(scene_names, scene_list=None, max_scenes=None):
    if scene_list is not None:
        raw = str(scene_list).strip()
        if osp.exists(raw):
            with open(raw) as f:
                requested = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        else:
            requested = [item.strip() for item in raw.split(",") if item.strip()]
        allowed = set(scene_names)
        scene_names = [scene for scene in requested if scene in allowed]
    if max_scenes is not None:
        scene_names = scene_names[: int(max_scenes)]
    return scene_names


def _support_metrics(
    openyolo3d,
    seed_indices,
    class_id,
    projections_np,
    keep_visible_np,
    support_iou_th,
    min_support_visible_points,
):
    support_views = []
    label_evidence = defaultdict(float)
    label_evidence_view_count = 0
    label_consensus_view_count = 0
    label_conflict_view_count = 0
    best_iou = 0.0
    best_score = 0.0
    for frame_idx, image_path in enumerate(openyolo3d.world2cam.color_paths):
        visible_seed = seed_indices[keep_visible_np[frame_idx, seed_indices]]
        if len(visible_seed) < min_support_visible_points:
            continue
        coords = projections_np[frame_idx, visible_seed].astype(np.float32)
        box_depth = np.array(
            [
                coords[:, 0].min() / openyolo3d.scaling_params[1],
                coords[:, 1].min() / openyolo3d.scaling_params[0],
                (coords[:, 0].max() + 1.0) / openyolo3d.scaling_params[1],
                (coords[:, 1].max() + 1.0) / openyolo3d.scaling_params[0],
            ],
            dtype=np.float32,
        )
        frame_id = _frame_key(image_path)
        frame_pred = openyolo3d.preds_2d.get(frame_id)
        if frame_pred is None:
            continue
        labels = _to_numpy(frame_pred["labels"]).astype(np.int64)
        boxes = _to_numpy(frame_pred["bbox"]).astype(np.float32)
        scores = _to_numpy(frame_pred["scores"]).astype(np.float32)
        ious_all = _box_iou_np(box_depth, boxes)
        matched = ious_all >= float(support_iou_th)
        if matched.any():
            evidence = ious_all[matched] * scores[matched]
            matched_labels = labels[matched]
            if len(evidence) > 0:
                label_evidence_view_count += 1
                top_idx = int(np.argmax(evidence))
                if int(matched_labels[top_idx]) == int(class_id):
                    label_consensus_view_count += 1
                else:
                    label_conflict_view_count += 1
                for label, value in zip(matched_labels, evidence):
                    label_evidence[int(label)] += float(value)

        same_class = labels == int(class_id)
        if not same_class.any():
            continue
        same_boxes = boxes[same_class]
        same_scores = scores[same_class]
        ious = _box_iou_np(box_depth, same_boxes)
        local_best = int(np.argmax(ious))
        local_iou = float(ious[local_best])
        local_score = float(same_scores[local_best])
        best_iou = max(best_iou, local_iou)
        best_score = max(best_score, local_score)
        if local_iou >= support_iou_th:
            local_box = same_boxes[local_best]
            support_views.append(
                {
                    "frame_id": frame_id,
                    "frame_index": int(frame_idx),
                    "visible_seed_points": int(len(visible_seed)),
                    "iou": local_iou,
                    "score": local_score,
                    "bbox_xyxy": [float(v) for v in local_box.tolist()],
                }
            )

    if support_views:
        mean_iou = float(np.mean([item["iou"] for item in support_views]))
        mean_score = float(np.mean([item["score"] for item in support_views]))
    else:
        mean_iou = 0.0
        mean_score = 0.0

    total_evidence = float(sum(label_evidence.values()))
    target_evidence = float(label_evidence.get(int(class_id), 0.0))
    top_conflicting_class_id = None
    top_conflicting_evidence = 0.0
    for label, value in label_evidence.items():
        if int(label) == int(class_id):
            continue
        if float(value) > top_conflicting_evidence:
            top_conflicting_class_id = int(label)
            top_conflicting_evidence = float(value)
    if total_evidence > 0.0:
        probabilities = np.asarray(list(label_evidence.values()), dtype=np.float64) / total_evidence
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
        entropy /= max(np.log(max(2, len(probabilities))), 1e-12)
        label_consensus_score = float(target_evidence / total_evidence)
        label_conflict_score = float(max(0.0, 1.0 - label_consensus_score))
        label_margin = float((target_evidence - top_conflicting_evidence) / total_evidence)
    else:
        entropy = 0.0
        label_consensus_score = 1.0
        label_conflict_score = 0.0
        label_margin = 0.0

    return {
        "support_view_count": int(len(support_views)),
        "support_mean_iou": mean_iou,
        "support_mean_score": mean_score,
        "support_best_iou": best_iou,
        "support_best_score": best_score,
        "support_views": support_views,
        "label_consensus_score": label_consensus_score,
        "label_conflict_score": label_conflict_score,
        "label_margin": label_margin,
        "label_entropy": entropy,
        "label_consensus_view_count": int(label_consensus_view_count),
        "label_conflict_view_count": int(label_conflict_view_count),
        "label_evidence_view_count": int(label_evidence_view_count),
        "label_target_evidence": target_evidence,
        "label_total_evidence": total_evidence,
        "top_conflicting_class_id": top_conflicting_class_id,
        "top_conflicting_evidence": top_conflicting_evidence,
    }


def _refinement_routing(
    class_name,
    score,
    box_area_ratio,
    num_seed_points,
    support,
    metrics,
    sam_refine_classes,
    mllm_refine_classes,
):
    route = []
    reasons = []

    if class_name in sam_refine_classes:
        route.append("sam")
        reasons.append("surface_or_boundary_sensitive_class")
    if box_area_ratio >= 0.12 and num_seed_points >= 500:
        route.append("sam")
        reasons.append("large_2d_extent")
    if support["support_view_count"] >= 2 and support["support_mean_iou"] < 0.35:
        route.append("sam")
        reasons.append("multiview_bbox_disagreement")
    if metrics["seed_in_existing_mask_ratio"] > 0.45:
        route.append("sam")
        reasons.append("partial_overlap_with_existing_3d_masks")

    if class_name in mllm_refine_classes:
        route.append("mllm")
        reasons.append("semantic_ambiguous_or_long_tail_class")
    if float(score) < 0.55:
        route.append("mllm")
        reasons.append("low_2d_semantic_confidence")

    route = sorted(set(route), key=("sam", "mllm").index)
    if not route:
        route = ["fast"]
        reasons = ["high_confidence_geometry_and_semantics"]

    return {
        "route": route,
        "needs_sam": "sam" in route,
        "needs_mllm": "mllm" in route,
        "reasons": reasons,
    }


def _select_seed_nms(candidates, seed_nms_iou, max_candidates, max_candidates_per_class):
    selected = []
    per_class_counts = {}
    for candidate in candidates:
        class_id = candidate["class_id"]
        if (
            max_candidates_per_class is not None
            and per_class_counts.get(class_id, 0) >= max_candidates_per_class
        ):
            continue

        duplicate = False
        for kept in selected:
            if class_id != kept["class_id"]:
                continue
            if _seed_iou(candidate["_seed_indices"], kept["_seed_indices"]) > seed_nms_iou:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
            per_class_counts[class_id] = per_class_counts.get(class_id, 0) + 1
        if max_candidates is not None and len(selected) >= max_candidates:
            break
    return selected


def export_backprojection_candidates(
    openyolo3d,
    scene_name,
    output_dir,
    detection_score_th=0.35,
    min_seed_points=80,
    max_box_area_ratio=0.35,
    max_existing_iou=0.30,
    max_seed_in_existing_mask_ratio=0.70,
    seed_nms_iou=0.50,
    max_candidates=None,
    max_candidates_per_class=3,
    min_support_views=1,
    support_iou_th=0.25,
    min_support_visible_points=30,
    small_object_classes=None,
    small_min_seed_points=30,
    large_surface_classes=None,
    large_min_seed_points=120,
    large_max_box_area_ratio=0.25,
    sam_refine_classes=None,
    mllm_refine_classes=None,
    include_filtered=False,
    box_nms_iou=0.0,
    box_nms_same_class_only=True,
):
    os.makedirs(output_dir, exist_ok=True)
    scene_dir = osp.join(output_dir, scene_name)
    image_dir = osp.join(scene_dir, "images")
    seed_dir = osp.join(scene_dir, "seed_points")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(seed_dir, exist_ok=True)

    labels = openyolo3d.openyolo3d_config["network2d"]["text_prompts"]
    small_object_classes = _parse_class_names(small_object_classes, DEFAULT_SMALL_OBJECT_CLASSES)
    large_surface_classes = _parse_class_names(large_surface_classes, DEFAULT_LARGE_SURFACE_CLASSES)
    sam_refine_classes = _parse_class_names(sam_refine_classes, DEFAULT_SAM_REFINE_CLASSES)
    mllm_refine_classes = _parse_class_names(mllm_refine_classes, DEFAULT_MLLM_REFINE_CLASSES)
    projections, keep_visible_points = openyolo3d.mesh_projections
    projections_np = _to_numpy(projections).astype(np.int64)
    keep_visible_np = _to_numpy(keep_visible_points).astype(bool)
    existing_masks = _to_numpy(openyolo3d.preds_3d[0]).astype(bool)
    if existing_masks.shape[0] != projections_np.shape[1]:
        existing_masks = existing_masks.T
    if existing_masks.shape[0] != projections_np.shape[1]:
        raise ValueError(
            f"Unexpected 3D mask shape {existing_masks.shape}; "
            f"expected first dim to match {projections_np.shape[1]} points."
        )

    image_height, image_width = openyolo3d.world2cam.image_resolution
    pending = []
    for frame_idx, image_path in enumerate(openyolo3d.world2cam.color_paths):
        frame_id = _frame_key(image_path)
        if frame_id not in openyolo3d.preds_2d:
            continue
        frame_pred = openyolo3d.preds_2d[frame_id]
        boxes = _to_numpy(frame_pred["bbox"]).astype(np.float32)
        class_ids = _to_numpy(frame_pred["labels"]).astype(np.int64)
        scores = _to_numpy(frame_pred["scores"]).astype(np.float32)
        det_indices = _select_2d_nms_indices(
            boxes,
            scores,
            class_ids,
            iou_threshold=box_nms_iou,
            same_class_only=box_nms_same_class_only,
        )

        for det_id in det_indices:
            box, class_id, score = boxes[det_id], class_ids[det_id], scores[det_id]
            if int(class_id) >= len(labels):
                continue
            class_name = _safe_label(labels, int(class_id))
            class_min_seed_points = min_seed_points
            class_max_box_area_ratio = max_box_area_ratio
            if class_name in small_object_classes:
                class_min_seed_points = min(class_min_seed_points, small_min_seed_points)
            if class_name in large_surface_classes:
                class_min_seed_points = max(class_min_seed_points, large_min_seed_points)
                class_max_box_area_ratio = min(class_max_box_area_ratio, large_max_box_area_ratio)

            box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
            box_area_ratio = box_area / max(1, image_width * image_height)

            filter_reasons = []
            if float(score) < detection_score_th:
                filter_reasons.append("low_2d_score")
            if box_area_ratio > class_max_box_area_ratio:
                filter_reasons.append("large_2d_box")

            x1 = box[0] * openyolo3d.scaling_params[1]
            y1 = box[1] * openyolo3d.scaling_params[0]
            x2 = box[2] * openyolo3d.scaling_params[1]
            y2 = box[3] * openyolo3d.scaling_params[0]
            coords = projections_np[frame_idx]
            seed_mask = (
                keep_visible_np[frame_idx]
                & (coords[:, 0] >= x1)
                & (coords[:, 0] <= x2)
                & (coords[:, 1] >= y1)
                & (coords[:, 1] <= y2)
            )
            seed_indices = np.flatnonzero(seed_mask).astype(np.int64)
            if len(seed_indices) < class_min_seed_points:
                filter_reasons.append("few_backprojected_points")

            support = _support_metrics(
                openyolo3d,
                seed_indices,
                int(class_id),
                projections_np,
                keep_visible_np,
                support_iou_th,
                min_support_visible_points,
            )
            if support["support_view_count"] < min_support_views:
                filter_reasons.append("low_multiview_support")

            metrics = _existing_mask_metrics(existing_masks, seed_indices)
            if metrics["best_existing_iou"] > max_existing_iou:
                filter_reasons.append("matched_existing_3d_mask")
            if metrics["seed_in_existing_mask_ratio"] > max_seed_in_existing_mask_ratio:
                filter_reasons.append("mostly_covered_by_existing_masks")
            if filter_reasons and not include_filtered:
                continue

            priority = float(score)
            priority *= 1.0 - min(1.0, metrics["best_existing_seed_coverage"])
            priority *= 1.0 - min(1.0, metrics["best_existing_iou"])
            priority *= 1.0 + min(1.0, support["support_view_count"] / 4.0)
            priority *= 0.75 + 0.25 * max(support["support_mean_iou"], support["support_best_iou"])
            fusion_score = float(score)
            fusion_score *= 0.5 + 0.5 * min(1.0, support["support_view_count"] / 3.0)
            fusion_score *= 1.0 - 0.5 * min(1.0, metrics["seed_in_existing_mask_ratio"])
            routing = _refinement_routing(
                class_name,
                float(score),
                box_area_ratio,
                len(seed_indices),
                support,
                metrics,
                sam_refine_classes,
                mllm_refine_classes,
            )

            pending.append(
                {
                    "scene_name": scene_name,
                    "frame_id": frame_id,
                    "frame_index": int(frame_idx),
                    "detection_id": int(det_id),
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "score": float(score),
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "box_area_ratio": float(box_area_ratio),
                    "num_seed_points": int(len(seed_indices)),
                    "class_min_seed_points": int(class_min_seed_points),
                    "class_max_box_area_ratio": float(class_max_box_area_ratio),
                    "filter_reasons": filter_reasons,
                    "proposal_priority": float(priority),
                    "fusion_score": float(max(0.0, min(1.0, fusion_score))),
                    "refinement": routing,
                    **support,
                    **metrics,
                    "_seed_indices": seed_indices,
                }
            )

    pending = sorted(
        pending,
        key=lambda item: (
            len(item["filter_reasons"]) > 0,
            -item["proposal_priority"],
            item["best_existing_iou"],
            -item["score"],
            -item["num_seed_points"],
        ),
    )
    selected = _select_seed_nms(pending, seed_nms_iou, max_candidates, max_candidates_per_class)

    candidates = []
    for idx, item in enumerate(selected):
        seed_indices = item.pop("_seed_indices")
        prefix = osp.join(image_dir, f"candidate{idx:04d}_frame{item['frame_id']}_det{item['detection_id']:03d}")
        evidence = _save_evidence_images(
            openyolo3d,
            item["frame_index"],
            item["bbox_xyxy"],
            seed_indices,
            prefix,
        )
        seed_path = osp.join(seed_dir, f"candidate{idx:04d}_points.npz")
        np.savez_compressed(seed_path, point_indices=seed_indices)
        item["candidate_id"] = int(idx)
        item["evidence"] = evidence
        item["seed_points_path"] = seed_path
        candidates.append(item)

    json_path = osp.join(scene_dir, "backprojection_candidates.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "num_candidates": len(candidates),
                "num_pending_before_seed_nms": len(pending),
                "filters": {
                    "detection_score_th": detection_score_th,
                    "min_seed_points": min_seed_points,
                    "max_box_area_ratio": max_box_area_ratio,
                    "max_existing_iou": max_existing_iou,
                    "max_seed_in_existing_mask_ratio": max_seed_in_existing_mask_ratio,
                    "seed_nms_iou": seed_nms_iou,
                    "max_candidates": max_candidates,
                    "max_candidates_per_class": max_candidates_per_class,
                    "min_support_views": min_support_views,
                    "support_iou_th": support_iou_th,
                    "min_support_visible_points": min_support_visible_points,
                    "small_object_classes": sorted(small_object_classes),
                    "small_min_seed_points": small_min_seed_points,
                    "large_surface_classes": sorted(large_surface_classes),
                    "large_min_seed_points": large_min_seed_points,
                    "large_max_box_area_ratio": large_max_box_area_ratio,
                    "sam_refine_classes": sorted(sam_refine_classes),
                    "mllm_refine_classes": sorted(mllm_refine_classes),
                    "include_filtered": include_filtered,
                    "box_nms_iou": box_nms_iou,
                    "box_nms_same_class_only": box_nms_same_class_only,
                },
                "candidates": candidates,
            },
            f,
            indent=2,
        )
    return json_path, candidates


def export_dataset_backprojection_candidates(
    dataset_name,
    path_to_3d_masks,
    output_dir,
    scene_name=None,
    detection_score_th=0.35,
    min_seed_points=80,
    max_box_area_ratio=0.35,
    max_existing_iou=0.30,
    max_seed_in_existing_mask_ratio=0.70,
    seed_nms_iou=0.50,
    max_candidates_per_scene=None,
    max_candidates_per_class=3,
    min_support_views=1,
    support_iou_th=0.25,
    min_support_visible_points=30,
    small_object_classes=None,
    small_min_seed_points=30,
    large_surface_classes=None,
    large_min_seed_points=120,
    large_max_box_area_ratio=0.25,
    sam_refine_classes=None,
    mllm_refine_classes=None,
    include_filtered=False,
    box_nms_iou=0.0,
    box_nms_same_class_only=True,
    path_to_2d_preds=None,
    save_2d_preds=False,
    reuse_2d_preds=True,
    scene_list=None,
    max_scenes=None,
):
    config = load_yaml(osp.join(f"./pretrained/config_{dataset_name}.yaml"))
    path_2_dataset = osp.join("./data", dataset_name)
    depth_scale = config["openyolo3d"]["depth_scale"]

    if dataset_name == "replica":
        scene_names = SCENE_NAMES_REPLICA
        datatype = "point cloud"
    elif dataset_name == "scannet200":
        scene_names = SCENE_NAMES_SCANNET200
        datatype = "mesh"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if scene_name is not None:
        scene_names = [scene_name]
    scene_names = _resolve_scene_names(scene_names, scene_list=scene_list, max_scenes=max_scenes)

    openyolo3d = OpenYolo3D(f"./pretrained/config_{dataset_name}.yaml")
    summaries = []
    for current_scene in tqdm(scene_names):
        scene_id = current_scene.replace("scene", "")
        processed_file = (
            osp.join(path_2_dataset, current_scene, f"{scene_id}.npy")
            if dataset_name == "scannet200"
            else None
        )

        openyolo3d.predict(
            path_2_scene_data=osp.join(path_2_dataset, current_scene),
            depth_scale=depth_scale,
            datatype=datatype,
            processed_scene=processed_file,
            path_to_3d_masks=path_to_3d_masks,
            is_gt=False,
            path_to_2d_preds=path_to_2d_preds,
            save_2d_preds=save_2d_preds,
            reuse_2d_preds=reuse_2d_preds,
        )

        json_path, candidates = export_backprojection_candidates(
            openyolo3d=openyolo3d,
            scene_name=current_scene,
            output_dir=output_dir,
            detection_score_th=detection_score_th,
            min_seed_points=min_seed_points,
            max_box_area_ratio=max_box_area_ratio,
            max_existing_iou=max_existing_iou,
            max_seed_in_existing_mask_ratio=max_seed_in_existing_mask_ratio,
            seed_nms_iou=seed_nms_iou,
            max_candidates=max_candidates_per_scene,
            max_candidates_per_class=max_candidates_per_class,
            min_support_views=min_support_views,
            support_iou_th=support_iou_th,
            min_support_visible_points=min_support_visible_points,
            small_object_classes=small_object_classes,
            small_min_seed_points=small_min_seed_points,
            large_surface_classes=large_surface_classes,
            large_min_seed_points=large_min_seed_points,
            large_max_box_area_ratio=large_max_box_area_ratio,
            sam_refine_classes=sam_refine_classes,
            mllm_refine_classes=mllm_refine_classes,
            include_filtered=include_filtered,
            box_nms_iou=box_nms_iou,
            box_nms_same_class_only=box_nms_same_class_only,
        )
        summaries.append(
            {
                "scene_name": current_scene,
                "num_candidates": len(candidates),
                "json_path": json_path,
            }
        )

        for attr in (
            "world2cam",
            "mesh_projections",
            "preds_3d",
            "preds_2d",
            "predicted_masks",
            "predicated_scores",
            "predicated_classes",
        ):
            if hasattr(openyolo3d, attr):
                setattr(openyolo3d, attr, None)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = osp.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"dataset_name": dataset_name, "scenes": summaries}, f, indent=2)

    print(f"Saved backprojection candidate summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--output_dir", default="./output/backprojection_candidates")
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--detection_score_th", default=0.35, type=float)
    parser.add_argument("--min_seed_points", default=80, type=int)
    parser.add_argument("--max_box_area_ratio", default=0.35, type=float)
    parser.add_argument("--max_existing_iou", default=0.30, type=float)
    parser.add_argument("--max_seed_in_existing_mask_ratio", default=0.70, type=float)
    parser.add_argument("--seed_nms_iou", default=0.50, type=float)
    parser.add_argument("--max_candidates_per_scene", default=None, type=int)
    parser.add_argument("--max_candidates_per_class", default=3, type=int)
    parser.add_argument("--min_support_views", default=1, type=int)
    parser.add_argument("--support_iou_th", default=0.25, type=float)
    parser.add_argument("--min_support_visible_points", default=30, type=int)
    parser.add_argument("--small_object_classes", default=None, help="Comma-separated classes that can use --small_min_seed_points")
    parser.add_argument("--small_min_seed_points", default=30, type=int)
    parser.add_argument("--large_surface_classes", default=None, help="Comma-separated classes that use stricter seed/area thresholds")
    parser.add_argument("--large_min_seed_points", default=120, type=int)
    parser.add_argument("--large_max_box_area_ratio", default=0.25, type=float)
    parser.add_argument("--sam_refine_classes", default=None, help="Comma-separated classes routed to optional SAM refinement metadata")
    parser.add_argument("--mllm_refine_classes", default=None, help="Comma-separated classes routed to optional MLLM semantic refinement metadata")
    parser.add_argument("--include_filtered", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--box_nms_iou", default=0.0, type=float, help="Optional 2D box NMS IoU before BPR candidate export; 0 disables")
    parser.add_argument("--box_nms_same_class_only", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None, help="Optional directory or .pt file for cached YOLO-World 2D detections")
    parser.add_argument("--save_2d_preds", default=False, action=argparse.BooleanOptionalAction, help="Save YOLO-World 2D detections to --path_to_2d_preds after inference")
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction, help="Reuse cached YOLO-World 2D detections when available")
    parser.add_argument("--scene_list", default=None, help="Optional comma-separated scene names or file with one scene per line")
    parser.add_argument("--max_scenes", default=None, type=int)
    args = parser.parse_args()

    export_dataset_backprojection_candidates(
        dataset_name=args.dataset_name,
        path_to_3d_masks=args.path_to_3d_masks,
        output_dir=args.output_dir,
        scene_name=args.scene_name,
        detection_score_th=args.detection_score_th,
        min_seed_points=args.min_seed_points,
        max_box_area_ratio=args.max_box_area_ratio,
        max_existing_iou=args.max_existing_iou,
        max_seed_in_existing_mask_ratio=args.max_seed_in_existing_mask_ratio,
        seed_nms_iou=args.seed_nms_iou,
        max_candidates_per_scene=args.max_candidates_per_scene,
        max_candidates_per_class=args.max_candidates_per_class,
        min_support_views=args.min_support_views,
        support_iou_th=args.support_iou_th,
        min_support_visible_points=args.min_support_visible_points,
        small_object_classes=args.small_object_classes,
        small_min_seed_points=args.small_min_seed_points,
        large_surface_classes=args.large_surface_classes,
        large_min_seed_points=args.large_min_seed_points,
        large_max_box_area_ratio=args.large_max_box_area_ratio,
        sam_refine_classes=args.sam_refine_classes,
        mllm_refine_classes=args.mllm_refine_classes,
        include_filtered=args.include_filtered,
        box_nms_iou=args.box_nms_iou,
        box_nms_same_class_only=args.box_nms_same_class_only,
        path_to_2d_preds=args.path_to_2d_preds,
        save_2d_preds=args.save_2d_preds,
        reuse_2d_preds=args.reuse_2d_preds,
        scene_list=args.scene_list,
        max_scenes=args.max_scenes,
    )


if __name__ == "__main__":
    main()
