import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch

from utils.backprojection_fusion import _load_seed_indices


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _parse_class_filter(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parsed = {str(item).strip() for item in value if str(item).strip()}
    else:
        parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def _iter_feature_json_paths(path):
    if path is None:
        return
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "clip_object_features.json":
                yield osp.join(root, filename)


def load_clip_object_features(path):
    grouped = defaultdict(list)
    if path is None:
        return {}, {"files": [], "loaded": 0}
    if not osp.exists(path):
        raise FileNotFoundError(f"CLIP object feature path does not exist: {path}")

    summary = {"files": [], "loaded": 0}
    for json_path in _iter_feature_json_paths(path):
        summary["files"].append(json_path)
        with open(json_path) as f:
            payload = json.load(f)
        scene_name = payload.get("scene_name")
        for item in payload.get("features", []):
            record = dict(item)
            record.setdefault("scene_name", scene_name)
            record["_source_json"] = record.get("candidate_source_json") or json_path
            grouped[str(record["scene_name"])].append(record)
            summary["loaded"] += 1

    return {scene: items for scene, items in grouped.items()}, summary


def _noisy_or_update(old_value, contribution):
    old_value = float(np.clip(old_value, 0.0, 1.0))
    contribution = float(np.clip(contribution, 0.0, 1.0))
    return 1.0 - (1.0 - old_value) * (1.0 - contribution)


def _support_weight(record, support_scale):
    if support_scale <= 0:
        return 1.0
    support = max(1.0, float(record.get("support_view_count", 1.0)))
    return min(1.0, support / float(support_scale))


def rescore_with_clip_object_features(
    scene_name,
    pred_masks,
    pred_classes,
    pred_scores,
    features_by_scene,
    labels=None,
    min_seed_points=80,
    min_seed_overlap=0.45,
    min_support_views=2,
    support_scale=8.0,
    topk_classes=5,
    min_clip_prob=0.10,
    min_evidence=0.35,
    min_margin=0.12,
    max_base_score=1.01,
    score_alpha=0.50,
    allowed_classes=None,
    blocked_classes=None,
):
    """Rescore existing masks with cached CLIP crop-text similarities."""

    masks_np = _to_numpy(pred_masks).astype(bool)
    classes_np = _to_numpy(pred_classes).astype(np.int64).copy()
    scores_np = _to_numpy(pred_scores).astype(np.float32).copy()

    num_points, num_instances = masks_np.shape
    scene_features = features_by_scene.get(scene_name, [])
    report = {
        "loaded": len(scene_features),
        "used_features": 0,
        "matched_features": 0,
        "applied": [],
        "skipped_features": [],
        "skipped_instances": [],
    }
    if not scene_features or num_instances == 0:
        return classes_np, scores_np, report

    labels = labels or []
    allowed_classes = _parse_class_filter(allowed_classes)
    blocked_classes = _parse_class_filter(blocked_classes)
    blocked_ids = {
        idx for idx, name in enumerate(labels) if blocked_classes is not None and name in blocked_classes
    }
    allowed_ids = None
    if allowed_classes is not None:
        allowed_ids = {idx for idx, name in enumerate(labels) if name in allowed_classes}

    evidence = [defaultdict(float) for _ in range(num_instances)]
    evidence_meta = [defaultdict(list) for _ in range(num_instances)]

    for record in scene_features:
        candidate_id = record.get("candidate_id")
        if int(record.get("num_seed_points", 0)) < min_seed_points:
            report["skipped_features"].append({"candidate_id": candidate_id, "reason": "few_seed_points"})
            continue
        if int(record.get("support_view_count", 1)) < min_support_views:
            report["skipped_features"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_multiview_support",
                    "support_view_count": int(record.get("support_view_count", 1)),
                }
            )
            continue

        seed_indices = _load_seed_indices(record, num_points)
        if seed_indices is None or len(seed_indices) < min_seed_points:
            report["skipped_features"].append(
                {"candidate_id": candidate_id, "reason": "missing_or_small_seed_file"}
            )
            continue

        clip_probs = np.asarray(record.get("clip_probs", []), dtype=np.float32)
        if clip_probs.size == 0:
            report["skipped_features"].append({"candidate_id": candidate_id, "reason": "missing_clip_probs"})
            continue
        if len(labels) > 0:
            clip_probs = clip_probs[: len(labels)]
        if allowed_ids is not None:
            keep = np.zeros_like(clip_probs, dtype=bool)
            for idx in allowed_ids:
                if idx < len(clip_probs):
                    keep[idx] = True
            clip_probs = np.where(keep, clip_probs, 0.0)
        for idx in blocked_ids:
            if idx < len(clip_probs):
                clip_probs[idx] = 0.0

        if clip_probs.max(initial=0.0) < min_clip_prob:
            report["skipped_features"].append(
                {"candidate_id": candidate_id, "reason": "weak_clip_prob", "max_prob": float(clip_probs.max())}
            )
            continue

        topk = min(int(topk_classes), len(clip_probs))
        class_ids = np.argpartition(-clip_probs, topk - 1)[:topk]
        class_ids = class_ids[np.argsort(-clip_probs[class_ids])]
        class_ids = [int(class_id) for class_id in class_ids if float(clip_probs[class_id]) >= min_clip_prob]
        if len(class_ids) == 0:
            report["skipped_features"].append({"candidate_id": candidate_id, "reason": "no_top_class"})
            continue

        report["used_features"] += 1
        seed_inside_counts = masks_np[seed_indices, :].sum(axis=0).astype(np.float32)
        seed_overlap = seed_inside_counts / max(1.0, float(len(seed_indices)))
        matched_instances = np.flatnonzero(seed_overlap >= min_seed_overlap)
        if len(matched_instances) == 0:
            continue

        report["matched_features"] += 1
        support = _support_weight(record, support_scale)
        for inst_id in matched_instances:
            overlap = float(seed_overlap[inst_id])
            for class_id in class_ids:
                clip_prob = float(clip_probs[class_id])
                contribution = clip_prob * support * overlap
                evidence[inst_id][class_id] = _noisy_or_update(evidence[inst_id][class_id], contribution)
                evidence_meta[inst_id][class_id].append(
                    {
                        "candidate_id": int(candidate_id) if candidate_id is not None else None,
                        "clip_prob": clip_prob,
                        "seed_overlap": overlap,
                        "support_view_count": int(record.get("support_view_count", 1)),
                        "contribution": float(contribution),
                        "crop_path": record.get("crop_path"),
                    }
                )

    for inst_id, class_evidence in enumerate(evidence):
        if not class_evidence:
            continue
        current_class = int(classes_np[inst_id])
        current_score = float(scores_np[inst_id])
        ranked = sorted(class_evidence.items(), key=lambda item: item[1], reverse=True)
        best_class, best_score = int(ranked[0][0]), float(ranked[0][1])
        second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        current_evidence = float(class_evidence.get(current_class, 0.0))
        decision_margin = best_score - max(second_score, current_evidence)

        if best_class == current_class:
            scores_np[inst_id] = max(
                current_score,
                np.float32(score_alpha * current_score + (1.0 - score_alpha) * best_score),
            )
            continue
        if current_score > max_base_score:
            report["skipped_instances"].append(
                {"mask_id": int(inst_id), "reason": "high_base_score", "base_score": current_score}
            )
            continue
        if best_score < min_evidence:
            report["skipped_instances"].append(
                {
                    "mask_id": int(inst_id),
                    "reason": "weak_evidence",
                    "best_evidence": best_score,
                    "best_class_id": best_class,
                }
            )
            continue
        if decision_margin < min_margin:
            report["skipped_instances"].append(
                {
                    "mask_id": int(inst_id),
                    "reason": "small_margin",
                    "best_evidence": best_score,
                    "current_evidence": current_evidence,
                    "second_evidence": second_score,
                }
            )
            continue

        classes_np[inst_id] = best_class
        scores_np[inst_id] = max(
            current_score,
            np.float32(score_alpha * current_score + (1.0 - score_alpha) * best_score),
        )
        report["applied"].append(
            {
                "mask_id": int(inst_id),
                "old_class_id": current_class,
                "old_class_name": labels[current_class] if 0 <= current_class < len(labels) else str(current_class),
                "new_class_id": best_class,
                "new_class_name": labels[best_class] if 0 <= best_class < len(labels) else str(best_class),
                "base_score": current_score,
                "new_score": float(scores_np[inst_id]),
                "best_evidence": best_score,
                "current_evidence": current_evidence,
                "second_evidence": second_score,
                "decision_margin": float(decision_margin),
                "evidence": evidence_meta[inst_id][best_class],
            }
        )

    return classes_np, scores_np, report


def save_clip_object_report(path, summary, scene_reports, params):
    report_dir = osp.dirname(path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "feature_summary": summary,
                "scene_reports": scene_reports,
                "params": params,
            },
            f,
            indent=2,
        )
