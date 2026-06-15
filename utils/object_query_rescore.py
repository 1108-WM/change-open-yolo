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


def _candidate_score(candidate, use_fusion_score):
    if use_fusion_score and candidate.get("fusion_score") is not None:
        return float(candidate.get("fusion_score", 0.0))
    return float(candidate.get("score", 0.0))


def _support_weight(candidate, support_scale):
    if support_scale <= 0:
        return 1.0
    support = max(1.0, float(candidate.get("support_view_count", 1.0)))
    return min(1.0, support / float(support_scale))


def _noisy_or_update(old_value, contribution):
    old_value = float(np.clip(old_value, 0.0, 1.0))
    contribution = float(np.clip(contribution, 0.0, 1.0))
    return 1.0 - (1.0 - old_value) * (1.0 - contribution)


def rescore_with_object_queries(
    scene_name,
    pred_masks,
    pred_classes,
    pred_scores,
    candidates_by_scene,
    labels=None,
    min_candidate_score=0.30,
    min_seed_points=80,
    min_seed_overlap=0.35,
    min_support_views=2,
    support_scale=8.0,
    overlap_power=1.0,
    min_evidence=0.45,
    min_margin=0.10,
    max_base_score=1.01,
    score_alpha=0.50,
    use_candidate_fusion_score=True,
    allowed_classes=None,
    blocked_classes=None,
):
    """Rescore existing 3D instances with spatially gated 2D object evidence.

    This is a lightweight, evaluation-time analogue of SegDINO3D's 2D object
    query cross-attention: candidate 2D detections are treated as object queries,
    and only candidates whose back-projected seed points overlap a 3D mask can
    provide semantic evidence for that mask.
    """

    masks_np = _to_numpy(pred_masks).astype(bool)
    classes_np = _to_numpy(pred_classes).astype(np.int64).copy()
    scores_np = _to_numpy(pred_scores).astype(np.float32).copy()

    num_points, num_instances = masks_np.shape
    scene_candidates = candidates_by_scene.get(scene_name, [])
    report = {
        "loaded": len(scene_candidates),
        "used_candidates": 0,
        "matched_candidates": 0,
        "applied": [],
        "skipped_candidates": [],
        "skipped_instances": [],
    }
    if not scene_candidates or num_instances == 0:
        return classes_np, scores_np, report

    allowed_classes = _parse_class_filter(allowed_classes)
    blocked_classes = _parse_class_filter(blocked_classes)
    labels = labels or []

    evidence = [defaultdict(float) for _ in range(num_instances)]
    evidence_meta = [defaultdict(list) for _ in range(num_instances)]

    for candidate in scene_candidates:
        candidate_id = candidate.get("candidate_id")
        class_name = candidate.get("class_name")
        class_id = int(candidate.get("class_id", -1))
        score = _candidate_score(candidate, use_candidate_fusion_score)
        raw_score = float(candidate.get("score", score))

        if class_id < 0:
            report["skipped_candidates"].append({"candidate_id": candidate_id, "reason": "missing_class_id"})
            continue
        if allowed_classes is not None and class_name not in allowed_classes:
            report["skipped_candidates"].append(
                {"candidate_id": candidate_id, "reason": "class_not_allowed", "class_name": class_name}
            )
            continue
        if blocked_classes is not None and class_name in blocked_classes:
            report["skipped_candidates"].append(
                {"candidate_id": candidate_id, "reason": "class_blocked", "class_name": class_name}
            )
            continue
        if score < min_candidate_score:
            report["skipped_candidates"].append(
                {"candidate_id": candidate_id, "reason": "low_score", "score": score}
            )
            continue
        if int(candidate.get("num_seed_points", 0)) < min_seed_points:
            report["skipped_candidates"].append({"candidate_id": candidate_id, "reason": "few_seed_points"})
            continue
        if int(candidate.get("support_view_count", 1)) < min_support_views:
            report["skipped_candidates"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_multiview_support",
                    "support_view_count": int(candidate.get("support_view_count", 1)),
                }
            )
            continue

        seed_indices = _load_seed_indices(candidate, num_points)
        if seed_indices is None or len(seed_indices) < min_seed_points:
            report["skipped_candidates"].append({"candidate_id": candidate_id, "reason": "missing_or_small_seed_file"})
            continue

        report["used_candidates"] += 1
        seed_inside_counts = masks_np[seed_indices, :].sum(axis=0).astype(np.float32)
        seed_overlap = seed_inside_counts / max(1.0, float(len(seed_indices)))
        matched_instances = np.flatnonzero(seed_overlap >= min_seed_overlap)
        if len(matched_instances) == 0:
            continue

        report["matched_candidates"] += 1
        support = _support_weight(candidate, support_scale)
        for inst_id in matched_instances:
            overlap = float(seed_overlap[inst_id])
            contribution = score * support * (overlap ** float(overlap_power))
            evidence[inst_id][class_id] = _noisy_or_update(evidence[inst_id][class_id], contribution)
            evidence_meta[inst_id][class_id].append(
                {
                    "candidate_id": int(candidate_id) if candidate_id is not None else None,
                    "class_name": class_name,
                    "score": raw_score,
                    "fusion_score": score,
                    "seed_overlap": overlap,
                    "support_view_count": int(candidate.get("support_view_count", 1)),
                    "contribution": float(contribution),
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
            scores_np[inst_id] = max(current_score, np.float32(score_alpha * current_score + (1.0 - score_alpha) * best_score))
            continue
        if current_score > max_base_score:
            report["skipped_instances"].append(
                {
                    "mask_id": int(inst_id),
                    "reason": "high_base_score",
                    "base_score": current_score,
                    "best_evidence": best_score,
                }
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


def save_object_query_report(path, summary, scene_reports, params):
    report_dir = osp.dirname(path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "candidate_summary": summary,
                "scene_reports": scene_reports,
                "params": params,
            },
            f,
            indent=2,
        )
