import json
import os
import os.path as osp

import imageio.v2 as imageio
import numpy as np
import torch

from utils import compute_iou, get_visibility_mat


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _safe_label(labels, label_id):
    if label_id < 0 or label_id >= len(labels):
        return "unknown"
    return labels[label_id]


def _clamp_box(box, width, height, min_size=2):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(x1 + min_size, min(width, x2)))
    y2 = int(max(y1 + min_size, min(height, y2)))
    return x1, y1, x2, y2


def _save_context_images(image, coords_depth, scaling_params, bbox_color, output_prefix):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clamp_box(bbox_color, width, height)

    crop = image[y1:y2, x1:x2]
    mask_full = np.zeros((height, width), dtype=np.uint8)
    xs = np.round(coords_depth[:, 0] / scaling_params[1]).astype(np.int64)
    ys = np.round(coords_depth[:, 1] / scaling_params[0]).astype(np.int64)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    mask_full[ys[valid], xs[valid]] = 255
    mask_crop = mask_full[y1:y2, x1:x2]

    overlay = crop.copy()
    if overlay.ndim == 2:
        overlay = np.repeat(overlay[..., None], 3, axis=-1)
    if overlay.shape[-1] == 4:
        overlay = overlay[..., :3]
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    alpha = (mask_crop > 0)[..., None]
    overlay = np.where(alpha, (0.55 * overlay + 0.45 * red).astype(np.uint8), overlay)

    crop_path = f"{output_prefix}_crop.jpg"
    mask_path = f"{output_prefix}_mask.png"
    overlay_path = f"{output_prefix}_overlay.jpg"
    imageio.imwrite(crop_path, crop)
    imageio.imwrite(mask_path, mask_crop)
    imageio.imwrite(overlay_path, overlay)
    return crop_path, mask_path, overlay_path, (x1, y1, x2, y2)


def _label_distribution_for_mask(label_maps, projections, keep_visible_points, mask, frame_ids):
    labels_distribution = []
    for frame_id in frame_ids:
        visible_points = (keep_visible_points[frame_id].squeeze() * mask).astype(bool)
        if not visible_points.any():
            continue
        coords = projections[frame_id][visible_points].astype(np.int64)
        selected = label_maps[frame_id, coords[:, 1], coords[:, 0]]
        labels_distribution.append(selected[selected != -1])

    if not labels_distribution:
        return {}, 0.0

    labels_distribution = np.concatenate(labels_distribution)
    if len(labels_distribution) == 0:
        return {}, 0.0

    unique, counts = np.unique(labels_distribution, return_counts=True)
    distribution = {int(label): int(count) for label, count in zip(unique, counts)}
    sorted_counts = sorted(counts.tolist(), reverse=True)
    top = sorted_counts[0] / max(1, int(counts.sum()))
    second = sorted_counts[1] / max(1, int(counts.sum())) if len(sorted_counts) > 1 else 0.0
    return distribution, float(top - second)


def _distribution_stats(distribution):
    total = int(sum(distribution.values()))
    if total == 0:
        return {
            "label_vote_count": 0,
            "num_label_candidates": 0,
            "label_entropy": 0.0,
            "top_label_ratio": 0.0,
            "second_label_ratio": 0.0,
        }

    counts = np.asarray(sorted(distribution.values(), reverse=True), dtype=np.float64)
    probs = counts / total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    if len(probs) > 1:
        entropy = entropy / np.log(len(probs))
    else:
        entropy = 0.0

    return {
        "label_vote_count": total,
        "num_label_candidates": int(len(counts)),
        "label_entropy": entropy,
        "top_label_ratio": float(probs[0]),
        "second_label_ratio": float(probs[1]) if len(probs) > 1 else 0.0,
    }


def _view_quality(mask, frame_id, projections, keep_visible_points, scaling_params, image_resolution):
    visible_points = (keep_visible_points[frame_id].squeeze() * mask).astype(bool)
    visible_count = int(visible_points.sum())
    if visible_count == 0:
        return {
            "frame_id": int(frame_id),
            "visible_points": 0,
            "bbox_area_ratio": 1.0,
        }

    coords = projections[frame_id][visible_points].astype(np.int64)
    x_l, x_r = coords[:, 0].min(), coords[:, 0].max() + 1
    y_t, y_b = coords[:, 1].min(), coords[:, 1].max() + 1
    width = int(image_resolution[1])
    height = int(image_resolution[0])
    bbox_w = max(1.0, (x_r - x_l) / scaling_params[1])
    bbox_h = max(1.0, (y_b - y_t) / scaling_params[0])
    bbox_area_ratio = float((bbox_w * bbox_h) / max(1, width * height))
    return {
        "frame_id": int(frame_id),
        "visible_points": visible_count,
        "bbox_area_ratio": bbox_area_ratio,
    }


def _candidate_quality_reasons(
    mask_point_ratio,
    best_view_quality,
    label_stats,
    max_mask_point_ratio,
    max_bbox_area_ratio,
    min_visible_points,
    min_label_votes,
):
    reasons = []
    if max_mask_point_ratio is not None and mask_point_ratio > max_mask_point_ratio:
        reasons.append("large_3d_mask")
    if max_bbox_area_ratio is not None and best_view_quality["bbox_area_ratio"] > max_bbox_area_ratio:
        reasons.append("large_view_bbox")
    if min_visible_points is not None and best_view_quality["visible_points"] < min_visible_points:
        reasons.append("low_visibility")
    if min_label_votes is not None and label_stats["label_vote_count"] < min_label_votes:
        reasons.append("low_label_evidence")
    return reasons


def _frame_evidence(openyolo3d, mask, frame_id, image_dir, prefix):
    projections, keep_visible_points = openyolo3d.mesh_projections
    projections = _to_numpy(projections)
    keep_visible_points = _to_numpy(keep_visible_points)

    visible_points = (keep_visible_points[frame_id].squeeze() * mask).astype(bool)
    visible_count = int(visible_points.sum())
    if visible_count == 0:
        return None

    coords = projections[frame_id][visible_points].astype(np.int64)
    x_l, x_r = coords[:, 0].min(), coords[:, 0].max() + 1
    y_t, y_b = coords[:, 1].min(), coords[:, 1].max() + 1
    bbox_color = (
        x_l / openyolo3d.scaling_params[1],
        y_t / openyolo3d.scaling_params[0],
        x_r / openyolo3d.scaling_params[1],
        y_b / openyolo3d.scaling_params[0],
    )

    max_iou = 0.0
    boxes = list(openyolo3d.preds_2d.values())[frame_id]["bbox"].long()
    if len(boxes) > 0 and len(coords) > 10:
        box = torch.tensor(bbox_color)
        max_iou = float(compute_iou(box, boxes).max().item())

    image = imageio.imread(openyolo3d.world2cam.color_paths[frame_id])
    os.makedirs(image_dir, exist_ok=True)
    crop_path, mask_path, overlay_path, clamped_box = _save_context_images(
        image,
        coords,
        openyolo3d.scaling_params,
        bbox_color,
        osp.join(image_dir, prefix),
    )

    return {
        "frame_id": int(frame_id),
        "color_path": openyolo3d.world2cam.color_paths[frame_id],
        "visible_points": visible_count,
        "bbox_xyxy": [int(v) for v in clamped_box],
        "max_2d_iou": max_iou,
        "crop_path": crop_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
    }


def export_context_candidates(
    openyolo3d,
    scene_name,
    output_dir,
    top_views=3,
    uncertain_score_th=0.35,
    uncertain_margin_th=0.15,
    longtail_classes=None,
    max_candidates=None,
    max_mask_point_ratio=0.20,
    max_bbox_area_ratio=0.65,
    min_visible_points=50,
    min_label_votes=10,
    include_bad_quality=False,
):
    """Export OV3D-CG-style offline context packets for uncertain 3D instances."""

    os.makedirs(output_dir, exist_ok=True)
    scene_dir = osp.join(output_dir, scene_name)
    image_dir = osp.join(scene_dir, "images")
    os.makedirs(scene_dir, exist_ok=True)

    labels = openyolo3d.openyolo3d_config["network2d"]["text_prompts"]
    longtail_classes = set(longtail_classes or [])
    pred_masks = _to_numpy(openyolo3d.predicted_masks).astype(bool)
    pred_classes = _to_numpy(openyolo3d.predicated_classes).astype(np.int64)
    pred_scores = _to_numpy(openyolo3d.predicated_scores).astype(np.float32)

    projections, keep_visible_points = openyolo3d.mesh_projections
    projections_np = _to_numpy(projections)
    keep_visible_points_np = _to_numpy(keep_visible_points)
    label_maps = _to_numpy(openyolo3d.construct_label_maps(openyolo3d.preds_2d))

    visibility = get_visibility_mat(
        torch.from_numpy(pred_masks.T.astype(np.float32)),
        torch.from_numpy(keep_visible_points_np.astype(np.float32)),
        topk=top_views,
    ).numpy()

    pending_candidates = []
    for mask_id in range(pred_masks.shape[1]):
        frame_ids = np.where(visibility[mask_id])[0]
        distribution, margin = _label_distribution_for_mask(
            label_maps,
            projections_np,
            keep_visible_points_np,
            pred_masks[:, mask_id],
            frame_ids,
        )
        label_name = _safe_label(labels, int(pred_classes[mask_id]))
        reasons = []
        if float(pred_scores[mask_id]) < uncertain_score_th:
            reasons.append("low_score")
        if margin < uncertain_margin_th:
            reasons.append("low_label_margin")
        if label_name in longtail_classes:
            reasons.append("longtail_class")
        if not reasons:
            continue

        ranked_frames_with_quality = []
        for frame_id in frame_ids:
            view_quality = _view_quality(
                pred_masks[:, mask_id],
                int(frame_id),
                projections_np,
                keep_visible_points_np,
                openyolo3d.scaling_params,
                openyolo3d.world2cam.image_resolution,
            )
            ranked_frames_with_quality.append((view_quality["visible_points"], int(frame_id), view_quality))
        ranked_frames_with_quality = sorted(ranked_frames_with_quality, reverse=True)
        if not ranked_frames_with_quality:
            continue
        ranked_frames = [frame_id for _, frame_id, _ in ranked_frames_with_quality[:top_views]]
        best_view_quality = ranked_frames_with_quality[0][2]
        label_stats = _distribution_stats(distribution)
        mask_point_ratio = float(pred_masks[:, mask_id].sum() / max(1, pred_masks.shape[0]))
        quality_reasons = _candidate_quality_reasons(
            mask_point_ratio,
            best_view_quality,
            label_stats,
            max_mask_point_ratio,
            max_bbox_area_ratio,
            min_visible_points,
            min_label_votes,
        )
        if quality_reasons and not include_bad_quality:
            continue

        pending_candidates.append(
            {
                "scene_name": scene_name,
                "mask_id": int(mask_id),
                "predicted_class_id": int(pred_classes[mask_id]),
                "predicted_class_name": label_name,
                "predicted_score": float(pred_scores[mask_id]),
                "label_margin": margin,
                "selection_reasons": reasons,
                "quality_reasons": quality_reasons,
                "quality": {
                    "mask_point_ratio": mask_point_ratio,
                    "best_view_visible_points": best_view_quality["visible_points"],
                    "best_view_bbox_area_ratio": best_view_quality["bbox_area_ratio"],
                    **label_stats,
                },
                "label_distribution": {
                    _safe_label(labels, label_id): count for label_id, count in distribution.items()
                },
                "views": [],
                "mllm_prompt": (
                    "Identify the object instance highlighted by the mask. "
                    "Use the surrounding scene context, not only the cropped pixels. "
                    "Return change, keep, unknown, or bad_mask. "
                    "Use bad_mask for walls, ceilings, floors, or masks that do not isolate one object."
                ),
                "_mask": pred_masks[:, mask_id],
                "_ranked_frames": ranked_frames,
            }
        )

    pending_candidates = sorted(
        pending_candidates,
        key=lambda item: (
            len(item["quality_reasons"]) > 0,
            item["predicted_score"],
            -item["quality"]["label_entropy"],
            item["label_margin"],
        ),
    )
    if max_candidates is not None:
        pending_candidates = pending_candidates[:max_candidates]

    candidates = []
    for item in pending_candidates:
        mask = item.pop("_mask")
        ranked_frames = item.pop("_ranked_frames")
        views = []
        for rank, frame_id in enumerate(ranked_frames):
            evidence = _frame_evidence(
                openyolo3d,
                mask,
                frame_id,
                image_dir,
                f"mask{item['mask_id']:04d}_view{rank:02d}",
            )
            if evidence is not None:
                views.append(evidence)

        if not views:
            continue

        item["views"] = views
        candidates.append(item)

    json_path = osp.join(scene_dir, "context_candidates.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "num_candidates": len(candidates),
                "top_views": top_views,
                "uncertain_score_th": uncertain_score_th,
                "uncertain_margin_th": uncertain_margin_th,
                "quality_filter": {
                    "max_mask_point_ratio": max_mask_point_ratio,
                    "max_bbox_area_ratio": max_bbox_area_ratio,
                    "min_visible_points": min_visible_points,
                    "min_label_votes": min_label_votes,
                    "include_bad_quality": include_bad_quality,
                },
                "candidates": candidates,
            },
            f,
            indent=2,
        )

    return json_path, candidates
