import json
import os
import os.path as osp
from collections import defaultdict

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial import cKDTree


def _parse_class_filter(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parsed = {str(item).strip() for item in value if str(item).strip()}
    else:
        parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def _parse_decision_filter(value):
    parsed = _parse_class_filter(value)
    if parsed is None:
        return None
    return {item.lower() for item in parsed}


def _parse_source_filter(value):
    parsed = _parse_class_filter(value)
    if parsed is None:
        return None
    return {item.lower() for item in parsed}


def _parse_source_rules(value, value_type=float):
    if value is None:
        return None
    rules = {}
    if isinstance(value, dict):
        items = value.items()
    else:
        items = []
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                key, raw_value = item.split("=", 1)
            elif ":" in item:
                key, raw_value = item.split(":", 1)
            else:
                continue
            items.append((key, raw_value))

    for key, raw_value in items:
        key = str(key).strip().lower()
        if not key:
            continue
        try:
            rules[key] = value_type(raw_value)
        except (TypeError, ValueError):
            continue
    return rules or None


def _candidate_source_name(candidate):
    source_name = candidate.get("_source_name")
    if source_name:
        return str(source_name)
    source_json = candidate.get("_source_json")
    if source_json:
        return osp.basename(osp.dirname(osp.dirname(source_json))) or osp.basename(osp.dirname(source_json))
    return "unknown"


def _candidate_source_kind(candidate):
    candidate_kind = candidate.get("source_kind") or candidate.get("candidate_source_kind")
    if candidate_kind:
        return str(candidate_kind).strip().lower()
    source_name = _candidate_source_name(candidate).lower()
    if "mask_graph_multi_view" in source_name or "mask-graph-multi-view" in source_name:
        return "mask_graph_multi_view"
    if "mask_graph_single_view" in source_name or "mask-graph-single-view" in source_name:
        return "mask_graph_single_view"
    if "mask_graph" in source_name or "mask-graph" in source_name:
        return "mask_graph"
    if "sam_fused" in source_name or "sam-fused" in source_name:
        return "sam_fused"
    if "backprojection" in source_name or source_name.startswith("bpr"):
        return "bpr"
    return source_name


def _lookup_source_rule(candidate, rules, default):
    if not rules:
        return default
    source_name = _candidate_source_name(candidate).lower()
    source_kind = _candidate_source_kind(candidate).lower()
    for key in (source_name, source_kind):
        if key in rules:
            return rules[key]
    for key, value in rules.items():
        if key and key in source_name:
            return value
    return default


def _candidate_matches_source_filter(candidate, source_filter):
    if source_filter is None:
        return True
    source_name = _candidate_source_name(candidate).lower()
    source_kind = _candidate_source_kind(candidate).lower()
    if source_name in source_filter or source_kind in source_filter:
        return True
    return any(key and key in source_name for key in source_filter)


def _is_mask_graph_source(source_kind):
    return str(source_kind or "").strip().lower().startswith("mask_graph")


def _lookup_class_rule(class_name, rules, default):
    if not rules:
        return default
    class_name = str(class_name or "").strip().lower()
    if class_name in rules:
        return rules[class_name]
    return default


def _annotate_candidate_quality_stats(candidates):
    groups = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[_candidate_source_kind(candidate)].append(index)

    for indices in groups.values():
        values = np.asarray([_candidate_quality_score(candidates[index]) for index in indices], dtype=np.float32)
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) if len(values) else 0.0
        for index in indices:
            quality_score = _candidate_quality_score(candidates[index])
            candidates[index]["_quality_scene_source_z"] = 0.0 if std < 1e-6 else float((quality_score - mean) / std)
    return candidates


def _candidate_novelty_score(candidate):
    seed_covered = min(1.0, max(0.0, float(candidate.get("seed_in_existing_mask_ratio", 0.0))))
    existing_iou = min(1.0, max(0.0, float(candidate.get("best_existing_iou", 0.0))))
    existing_seed_coverage = min(1.0, max(0.0, float(candidate.get("best_existing_seed_coverage", 0.0))))
    novelty = (1.0 - seed_covered) * ((1.0 - existing_iou) ** 0.5) * ((1.0 - existing_seed_coverage) ** 0.5)
    return float(min(1.0, max(0.0, novelty)))


def _candidate_score_calibration(
    candidate,
    quality_weight=0.0,
    novelty_weight=0.0,
    label_consensus_weight=0.0,
    min_factor=0.2,
    max_factor=1.2,
):
    quality_weight = float(quality_weight or 0.0)
    novelty_weight = float(novelty_weight or 0.0)
    label_consensus_weight = float(label_consensus_weight or 0.0)
    factor = 1.0
    if quality_weight > 0.0:
        quality_z = float(candidate.get("_quality_scene_source_z", 0.0))
        factor *= 1.0 + quality_weight * float(np.tanh(quality_z / 2.0))
    if novelty_weight > 0.0:
        novelty = _candidate_novelty_score(candidate)
        factor *= (1.0 - novelty_weight) + novelty_weight * novelty
    if label_consensus_weight > 0.0 and "label_consensus_score" in candidate:
        consensus = min(1.0, max(0.0, float(candidate.get("label_consensus_score", 1.0))))
        conflict = min(1.0, max(0.0, float(candidate.get("label_conflict_score", 0.0))))
        label_factor = consensus * (1.0 - 0.5 * conflict)
        factor *= (1.0 - label_consensus_weight) + label_consensus_weight * label_factor
    return float(min(float(max_factor), max(float(min_factor), factor)))


def _candidate_selection_score(candidate, quality_weight=0.0, novelty_weight=0.0, label_consensus_weight=0.0):
    return _candidate_quality_score(candidate) * _candidate_score_calibration(
        candidate,
        quality_weight=quality_weight,
        novelty_weight=novelty_weight,
        label_consensus_weight=label_consensus_weight,
    )


def _frame_key(image_path):
    return osp.basename(image_path).split(".")[0]


def _candidate_label_consensus_score(candidate):
    if "label_consensus_score" not in candidate:
        return 1.0
    consensus = min(1.0, max(0.0, float(candidate.get("label_consensus_score", 1.0))))
    conflict = min(1.0, max(0.0, float(candidate.get("label_conflict_score", 0.0))))
    return float(consensus * (1.0 - 0.5 * conflict))


def _projected_label_consensus_metrics(
    candidate,
    seed_indices,
    projections,
    point_visibility,
    preds_2d,
    color_paths,
    scaling_params,
    iou_threshold=0.25,
    min_visible_points=30,
    frame_mode="support",
):
    if (
        seed_indices is None
        or len(seed_indices) == 0
        or projections is None
        or point_visibility is None
        or preds_2d is None
        or color_paths is None
    ):
        return None

    projections_np = _to_numpy(projections)
    visibility_np = _to_numpy(point_visibility).astype(bool)
    if projections_np.ndim != 3 or visibility_np.ndim != 2:
        return None
    if projections_np.shape[1] <= int(np.max(seed_indices)) or visibility_np.shape[1] <= int(np.max(seed_indices)):
        return None

    if str(frame_mode or "support").lower() == "all":
        frame_indices = range(len(color_paths))
    else:
        selected = set()
        if candidate.get("frame_index") is not None:
            try:
                selected.add(int(candidate.get("frame_index")))
            except (TypeError, ValueError):
                pass
        for view in candidate.get("support_views") or ():
            if not isinstance(view, dict) or view.get("frame_index") is None:
                continue
            try:
                selected.add(int(view.get("frame_index")))
            except (TypeError, ValueError):
                continue
        frame_indices = sorted(selected)

    if not frame_indices:
        return None

    class_id = int(candidate.get("class_id", -1))
    evidence_by_label = defaultdict(float)
    view_count = 0
    consensus_view_count = 0
    conflict_view_count = 0
    target_evidence = 0.0
    total_evidence = 0.0
    best_conflict_label = None
    best_conflict_evidence = 0.0

    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(color_paths) or frame_idx >= visibility_np.shape[0]:
            continue
        visible_seed = seed_indices[visibility_np[frame_idx, seed_indices]]
        if len(visible_seed) < int(min_visible_points):
            continue
        coords = projections_np[frame_idx, visible_seed].astype(np.float32)
        box_depth = np.array(
            [
                coords[:, 0].min() / scaling_params[1],
                coords[:, 1].min() / scaling_params[0],
                (coords[:, 0].max() + 1.0) / scaling_params[1],
                (coords[:, 1].max() + 1.0) / scaling_params[0],
            ],
            dtype=np.float32,
        )
        frame_pred = preds_2d.get(_frame_key(color_paths[frame_idx]))
        if frame_pred is None:
            continue
        frame_labels = _to_numpy(frame_pred["labels"]).astype(np.int64)
        frame_boxes = _to_numpy(frame_pred["bbox"]).astype(np.float32)
        frame_scores = _to_numpy(frame_pred["scores"]).astype(np.float32)
        ious = _box_iou_np(box_depth, frame_boxes)
        eligible = ious >= float(iou_threshold)
        if not eligible.any():
            continue

        evidence = ious[eligible] * frame_scores[eligible]
        labels = frame_labels[eligible]
        if len(evidence) == 0:
            continue
        view_count += 1
        top_index = int(np.argmax(evidence))
        if int(labels[top_index]) == class_id:
            consensus_view_count += 1
        else:
            conflict_view_count += 1

        for label, value in zip(labels, evidence):
            label = int(label)
            value = float(value)
            evidence_by_label[label] += value
            total_evidence += value
            if label == class_id:
                target_evidence += value

    if total_evidence <= 0.0:
        return None

    for label, value in evidence_by_label.items():
        if int(label) == class_id:
            continue
        if float(value) > best_conflict_evidence:
            best_conflict_label = int(label)
            best_conflict_evidence = float(value)

    probabilities = np.asarray(list(evidence_by_label.values()), dtype=np.float64) / max(total_evidence, 1e-12)
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    normalized_entropy = entropy / max(np.log(max(2, len(probabilities))), 1e-12)
    consensus_score = float(target_evidence / max(total_evidence, 1e-12))
    conflict_score = float(max(0.0, 1.0 - consensus_score))
    margin = float((target_evidence - best_conflict_evidence) / max(total_evidence, 1e-12))
    return {
        "label_consensus_score": consensus_score,
        "label_conflict_score": conflict_score,
        "label_margin": margin,
        "label_entropy": normalized_entropy,
        "label_consensus_view_count": int(consensus_view_count),
        "label_conflict_view_count": int(conflict_view_count),
        "label_evidence_view_count": int(view_count),
        "label_target_evidence": float(target_evidence),
        "label_total_evidence": float(total_evidence),
        "top_conflicting_class_id": best_conflict_label,
        "top_conflicting_evidence": float(best_conflict_evidence),
    }


def _projected_box_consistency_metrics(
    proposal_mask,
    candidate,
    projections,
    point_visibility,
    preds_2d,
    color_paths,
    scaling_params,
    min_visible_points=30,
    frame_mode="support",
    box_padding_ratio=0.05,
    same_class_only=True,
):
    if (
        proposal_mask is None
        or projections is None
        or point_visibility is None
        or preds_2d is None
        or color_paths is None
        or scaling_params is None
    ):
        return {"enabled": False, "reason": "missing_context"}

    proposal_indices = np.flatnonzero(proposal_mask)
    if len(proposal_indices) == 0:
        return {"enabled": True, "reason": "empty_mask", "usable_view_count": 0}

    projections_np = _to_numpy(projections)
    visibility_np = _to_numpy(point_visibility).astype(bool)
    if projections_np.ndim != 3 or visibility_np.ndim != 2:
        return {"enabled": False, "reason": "invalid_projection_shape"}
    if projections_np.shape[1] <= int(proposal_indices.max()) or visibility_np.shape[1] <= int(proposal_indices.max()):
        return {"enabled": False, "reason": "point_count_mismatch"}

    if str(frame_mode or "support").lower() == "all":
        frame_indices = range(len(color_paths))
    else:
        selected = set()
        if candidate.get("frame_index") is not None:
            try:
                selected.add(int(candidate.get("frame_index")))
            except (TypeError, ValueError):
                pass
        for view in candidate.get("support_views") or ():
            if not isinstance(view, dict) or view.get("frame_index") is None:
                continue
            try:
                selected.add(int(view.get("frame_index")))
            except (TypeError, ValueError):
                continue
        frame_indices = sorted(selected)

    if not frame_indices:
        return {"enabled": True, "reason": "no_frames", "usable_view_count": 0}

    class_id = int(candidate.get("class_id", -1))
    box_ious = []
    point_ratios = []
    visible_counts = []
    matched_views = 0
    padding_ratio = max(0.0, float(box_padding_ratio or 0.0))

    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(color_paths) or frame_idx >= visibility_np.shape[0]:
            continue
        visible_indices = proposal_indices[visibility_np[frame_idx, proposal_indices]]
        if len(visible_indices) < int(min_visible_points):
            continue
        frame_pred = preds_2d.get(_frame_key(color_paths[frame_idx]))
        if frame_pred is None:
            continue
        frame_labels = _to_numpy(frame_pred["labels"]).astype(np.int64)
        frame_boxes = _to_numpy(frame_pred["bbox"]).astype(np.float32)
        if same_class_only:
            eligible = frame_labels == class_id
            frame_boxes = frame_boxes[eligible]
        if len(frame_boxes) == 0:
            continue

        coords = projections_np[frame_idx, visible_indices].astype(np.float32)
        x = coords[:, 0] / scaling_params[1]
        y = coords[:, 1] / scaling_params[0]
        projected_box = np.asarray(
            [x.min(), y.min(), x.max() + 1.0 / scaling_params[1], y.max() + 1.0 / scaling_params[0]],
            dtype=np.float32,
        )
        ious = _box_iou_np(projected_box, frame_boxes)
        if len(ious) == 0:
            continue
        best_index = int(np.argmax(ious))
        best_box = frame_boxes[best_index].astype(np.float32)
        best_iou = float(ious[best_index])

        width = max(1.0, float(best_box[2] - best_box[0]))
        height = max(1.0, float(best_box[3] - best_box[1]))
        pad_x = width * padding_ratio
        pad_y = height * padding_ratio
        inside = (
            (x >= best_box[0] - pad_x)
            & (x <= best_box[2] + pad_x)
            & (y >= best_box[1] - pad_y)
            & (y <= best_box[3] + pad_y)
        )
        box_ious.append(best_iou)
        point_ratios.append(float(inside.mean()))
        visible_counts.append(int(len(visible_indices)))
        matched_views += int(best_iou > 0.0)

    if not box_ious:
        return {"enabled": True, "reason": "no_usable_views", "usable_view_count": 0}

    return {
        "enabled": True,
        "reason": "ok",
        "usable_view_count": int(len(box_ious)),
        "matched_view_count": int(matched_views),
        "mean_box_iou": float(np.mean(box_ious)),
        "min_box_iou": float(np.min(box_ious)),
        "max_box_iou": float(np.max(box_ious)),
        "mean_point_in_box_ratio": float(np.mean(point_ratios)),
        "min_point_in_box_ratio": float(np.min(point_ratios)),
        "max_point_in_box_ratio": float(np.max(point_ratios)),
        "mean_visible_points": float(np.mean(visible_counts)),
        "min_visible_points": int(np.min(visible_counts)),
        "box_padding_ratio": float(padding_ratio),
    }


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


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _iter_candidate_json_paths(path):
    if path is None:
        return
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "backprojection_candidates.json":
                yield osp.join(root, filename)


def load_backprojection_candidates(path):
    """Load ESAM-style 2D-to-3D proposal candidates grouped by scene."""

    grouped = defaultdict(list)
    if path is None:
        return {}, {"files": [], "loaded": 0}

    summary = {"files": [], "loaded": 0}
    paths = [item.strip() for item in str(path).split(",") if item.strip()]
    for candidate_path in paths:
        if not osp.exists(candidate_path):
            raise FileNotFoundError(f"Back-projection candidate path does not exist: {candidate_path}")
        for json_path in _iter_candidate_json_paths(candidate_path):
            summary["files"].append(json_path)
            with open(json_path) as f:
                payload = json.load(f)
            scene_name = payload.get("scene_name")
            source_name = osp.basename(osp.dirname(osp.dirname(json_path))) or osp.basename(osp.dirname(json_path))
            for candidate in payload.get("candidates", []):
                record = dict(candidate)
                record.setdefault("scene_name", scene_name)
                record["_source_json"] = json_path
                record["_source_name"] = source_name
                source_dir = osp.dirname(json_path)
                evidence = record.get("evidence")
                if isinstance(evidence, dict):
                    mask_path = evidence.get("sam_mask_path")
                    if mask_path and not osp.exists(mask_path):
                        evidence["sam_mask_path"] = osp.join(source_dir, mask_path)
                for view in record.get("support_views", []) or []:
                    if not isinstance(view, dict):
                        continue
                    mask_path = view.get("sam_mask_path")
                    if mask_path and not osp.exists(mask_path):
                        view["sam_mask_path"] = osp.join(source_dir, mask_path)
                grouped[str(record["scene_name"])].append(record)
                summary["loaded"] += 1

    return {scene: items for scene, items in grouped.items()}, summary


def _iter_json_paths(path):
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename.endswith((".json", ".jsonl")):
                yield osp.join(root, filename)


def _read_json_or_jsonl(path):
    with open(path) as f:
        text = f.read().strip()
    if not text:
        return None
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _as_verification_records(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    for key in ("verifications", "results", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]

    if isinstance(payload.get("candidates"), list):
        scene_name = payload.get("scene_name")
        records = []
        for candidate in payload["candidates"]:
            verifier = candidate.get("verification") or candidate.get("mllm_verification")
            if isinstance(verifier, dict):
                record = dict(verifier)
                record.setdefault("scene_name", candidate.get("scene_name", scene_name))
                record.setdefault("candidate_id", candidate.get("candidate_id"))
                records.append(record)
        return records

    return [payload]


def load_backprojection_verifications(path, min_confidence=0.0, strict=False):
    """Load offline MLLM/VLM verifier decisions keyed by scene and BPR candidate id."""

    grouped = defaultdict(dict)
    if path is None:
        return {}, {"files": [], "loaded": 0, "used": 0, "skipped": 0}
    if not osp.exists(path):
        raise FileNotFoundError(f"Back-projection verification path does not exist: {path}")

    summary = {"files": [], "loaded": 0, "used": 0, "skipped": 0}
    for json_path in _iter_json_paths(path):
        summary["files"].append(json_path)
        payload = _read_json_or_jsonl(json_path)
        for record in _as_verification_records(payload):
            if not isinstance(record, dict):
                summary["skipped"] += 1
                continue
            summary["loaded"] += 1
            scene_name = record.get("scene_name")
            candidate_id = record.get("candidate_id")
            if scene_name is None or candidate_id is None:
                summary["skipped"] += 1
                if strict:
                    raise ValueError(f"Missing scene_name/candidate_id in verifier record: {record}")
                continue
            try:
                candidate_id = int(candidate_id)
            except (TypeError, ValueError):
                summary["skipped"] += 1
                if strict:
                    raise ValueError(f"Invalid candidate_id in verifier record: {record}")
                continue

            confidence = record.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = None
            if confidence is not None and confidence < float(min_confidence):
                summary["skipped"] += 1
                continue

            grouped[str(scene_name)][candidate_id] = {
                "decision": str(record.get("decision", "")).strip().lower(),
                "confidence": confidence,
                "reason": str(record.get("reason", "")),
                "source_path": json_path,
            }
            summary["used"] += 1

    return {scene: dict(items) for scene, items in grouped.items()}, summary


def _resolve_seed_path(candidate):
    seed_path = candidate.get("refined_seed_points_path") or candidate.get("seed_points_path")
    if seed_path is None:
        return None
    if osp.exists(seed_path):
        return seed_path
    source_json = candidate.get("_source_json")
    if source_json is not None:
        local_path = osp.join(osp.dirname(source_json), seed_path)
        if osp.exists(local_path):
            return local_path
    return seed_path


def _load_seed_indices(candidate, num_points):
    seed_path = _resolve_seed_path(candidate)
    if seed_path is None or not osp.exists(seed_path):
        return None
    payload = np.load(seed_path)
    seed_indices = payload["point_indices"].astype(np.int64)
    seed_indices = seed_indices[(seed_indices >= 0) & (seed_indices < num_points)]
    if len(seed_indices) == 0:
        return None
    return np.unique(seed_indices)


def _annotate_mask_graph_evidence(
    candidates,
    num_points,
    min_overlap=0.25,
    min_iou=0.03,
    priority_weight=0.0,
    same_class_only=True,
):
    graph_items = []
    seed_cache = {}

    def seeds_for(index, candidate):
        if index not in seed_cache:
            seed_cache[index] = _load_seed_indices(candidate, num_points)
        return seed_cache[index]

    for index, candidate in enumerate(candidates):
        if not _is_mask_graph_source(_candidate_source_kind(candidate)):
            continue
        seed = seeds_for(index, candidate)
        if seed is None or len(seed) == 0:
            continue
        graph_quality = float(
            0.35 * min(1.0, float(candidate.get("graph_consensus_score", 0.0) or 0.0))
            + 0.25 * min(1.0, float(candidate.get("graph_edge_mean_score", 0.0) or 0.0))
            + 0.20 * min(1.0, float(candidate.get("support_mean_iou", 0.0) or 0.0))
            + 0.20 * min(1.0, float(candidate.get("selected_view_count", candidate.get("support_view_count", 0)) or 0) / 4.0)
        )
        conflict_edges = int(candidate.get("conflict_edge_count", 0) or 0)
        same_edges = int(candidate.get("same_object_edge_count", candidate.get("graph_edge_count", 0)) or 0)
        conflict_ratio = float(conflict_edges / max(1, same_edges))
        conflict_factor = max(0.0, 1.0 - min(1.0, conflict_ratio / 3.0))
        graph_items.append(
            {
                "index": index,
                "candidate": candidate,
                "seed": seed,
                "quality": graph_quality * conflict_factor,
            }
        )

    if not graph_items:
        for candidate in candidates:
            candidate["_mask_graph_evidence_score"] = 0.0
            candidate["_mask_graph_evidence_count"] = 0
        return candidates

    for index, candidate in enumerate(candidates):
        if _is_mask_graph_source(_candidate_source_kind(candidate)):
            candidate["_mask_graph_evidence_score"] = 0.0
            candidate["_mask_graph_evidence_count"] = 0
            continue
        seed = seeds_for(index, candidate)
        if seed is None or len(seed) == 0:
            candidate["_mask_graph_evidence_score"] = 0.0
            candidate["_mask_graph_evidence_count"] = 0
            continue

        best_score = 0.0
        support_count = 0
        best_overlap = 0.0
        best_iou = 0.0
        class_id = int(candidate.get("class_id", -1))
        for item in graph_items:
            graph_candidate = item["candidate"]
            if same_class_only and int(graph_candidate.get("class_id", -2)) != class_id:
                continue
            graph_seed = item["seed"]
            intersection = int(np.intersect1d(seed, graph_seed, assume_unique=False).size)
            if intersection <= 0:
                continue
            union = int(len(seed) + len(graph_seed) - intersection)
            iou = float(intersection / max(1, union))
            overlap = float(intersection / max(1, min(len(seed), len(graph_seed))))
            if overlap < float(min_overlap) and iou < float(min_iou):
                continue
            support_count += 1
            evidence = float(max(overlap, min(1.0, iou / max(1e-6, float(min_iou)))) * item["quality"])
            if evidence > best_score:
                best_score = evidence
                best_overlap = overlap
                best_iou = iou

        candidate["_mask_graph_evidence_score"] = float(best_score)
        candidate["_mask_graph_evidence_count"] = int(support_count)
        candidate["_mask_graph_evidence_best_overlap"] = float(best_overlap)
        candidate["_mask_graph_evidence_best_iou"] = float(best_iou)
        if best_score > 0.0 and float(priority_weight or 0.0) > 0.0:
            base_priority = float(candidate.get("proposal_priority", candidate.get("score", 0.0)) or 0.0)
            candidate["proposal_priority"] = float(base_priority * (1.0 + float(priority_weight) * best_score))

    return candidates


def _mask_iou(mask, masks):
    if masks.size == 0:
        return np.zeros((0,), dtype=np.float32)
    intersection = np.logical_and(masks, mask[:, None]).sum(axis=0).astype(np.float32)
    union = np.logical_or(masks, mask[:, None]).sum(axis=0).astype(np.float32)
    return intersection / np.maximum(union, 1.0)


def _support_frame_indices(support_views):
    frame_indices = []
    for view in support_views or ():
        if not isinstance(view, dict):
            continue
        frame_index = view.get("frame_index")
        if frame_index is None:
            continue
        try:
            frame_indices.append(int(frame_index))
        except (TypeError, ValueError):
            continue
    return sorted(set(frame_indices))


def _support_views_with_boxes(support_views):
    views = []
    for view in support_views or ():
        if not isinstance(view, dict):
            continue
        frame_index = view.get("frame_index")
        box = view.get("bbox_xyxy")
        if frame_index is None or box is None:
            continue
        try:
            box = [float(value) for value in box]
            if len(box) != 4:
                continue
            views.append({"frame_index": int(frame_index), "bbox_xyxy": box})
        except (TypeError, ValueError):
            continue
    return views


def _support_views_with_masks(support_views):
    views = []
    for view in support_views or ():
        if not isinstance(view, dict):
            continue
        frame_index = view.get("frame_index")
        mask_path = view.get("sam_mask_path")
        if frame_index is None or not mask_path or not osp.exists(mask_path):
            continue
        try:
            mask = imageio.imread(mask_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            views.append(
                {
                    "frame_index": int(frame_index),
                    "sam_mask_path": mask_path,
                    "mask": mask.astype(np.uint8) > 0,
                }
            )
        except (OSError, ValueError, TypeError):
            continue
    return views


def _filter_superpoint_segments_by_box_support(
    selected_segments,
    segments,
    proposal_mask,
    point_visibility,
    support_views,
    projections,
    scaling_params,
    min_positive_ratio=0.0,
    max_negative_ratio=1.0,
    min_visible_points=5,
    min_views=1,
    box_padding_ratio=0.05,
):
    info = {
        "enabled": (
            point_visibility is not None
            and projections is not None
            and support_views is not None
            and float(min_positive_ratio or 0.0) > 0.0
        ),
        "input_segments": int(len(selected_segments)),
        "output_segments": int(len(selected_segments)),
        "filtered_segments": 0,
        "usable_view_count": 0,
        "fallback": None,
    }
    if not info["enabled"]:
        return selected_segments, info

    support_mask_views = _support_views_with_masks(support_views)
    support_box_views = [] if support_mask_views else _support_views_with_boxes(support_views)
    if not support_mask_views and not support_box_views:
        info["fallback"] = "no_support_boxes"
        return selected_segments, info

    visibility = _to_numpy(point_visibility).astype(bool)
    projections_np = _to_numpy(projections).astype(np.float32)
    if visibility.ndim != 2 or visibility.shape[1] != proposal_mask.shape[0]:
        info["fallback"] = "visibility_shape_mismatch"
        return selected_segments, info
    if projections_np.ndim != 3 or projections_np.shape[:2] != visibility.shape:
        info["fallback"] = "projection_shape_mismatch"
        return selected_segments, info
    if scaling_params is None or len(scaling_params) < 2:
        info["fallback"] = "missing_scaling_params"
        return selected_segments, info

    selected_set = set(int(item) for item in selected_segments)
    segment_indices = {
        int(segment_id): np.flatnonzero(segments == segment_id)
        for segment_id in selected_set
    }
    kept = []
    segment_scores = {}
    min_visible_points = int(max(1, min_visible_points))
    min_views = int(max(1, min_views))
    for segment_id, indices in segment_indices.items():
        total_visible = 0
        total_inside = 0
        usable_views = 0
        for view in support_mask_views:
            frame_index = int(view["frame_index"])
            if frame_index < 0 or frame_index >= visibility.shape[0]:
                continue
            visible_indices = indices[visibility[frame_index, indices]]
            if len(visible_indices) <= 0:
                continue
            coords = projections_np[frame_index, visible_indices]
            xs = np.round(coords[:, 0] / float(scaling_params[1])).astype(np.int64)
            ys = np.round(coords[:, 1] / float(scaling_params[0])).astype(np.int64)
            mask = view["mask"]
            valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
            if int(np.count_nonzero(valid)) < min_visible_points:
                continue
            inside = mask[ys[valid], xs[valid]]
            total_visible += int(np.count_nonzero(valid))
            total_inside += int(np.count_nonzero(inside))
            usable_views += 1
        for view in support_box_views:
            frame_index = int(view["frame_index"])
            if frame_index < 0 or frame_index >= visibility.shape[0]:
                continue
            visible_indices = indices[visibility[frame_index, indices]]
            if len(visible_indices) < min_visible_points:
                continue
            box = view["bbox_xyxy"]
            width = max(1.0, box[2] - box[0])
            height = max(1.0, box[3] - box[1])
            pad_x = width * float(box_padding_ratio or 0.0)
            pad_y = height * float(box_padding_ratio or 0.0)
            x1, y1, x2, y2 = box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y
            coords = projections_np[frame_index, visible_indices]
            xs = coords[:, 0] / float(scaling_params[1])
            ys = coords[:, 1] / float(scaling_params[0])
            inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
            total_visible += int(len(visible_indices))
            total_inside += int(np.count_nonzero(inside))
            usable_views += 1
        if usable_views < min_views or total_visible <= 0:
            kept.append(segment_id)
            segment_scores[segment_id] = {
                "usable_views": int(usable_views),
                "positive_ratio": 1.0,
                "negative_ratio": 0.0,
                "fallback": "few_usable_views",
            }
            continue
        positive_ratio = float(total_inside / max(1, total_visible))
        negative_ratio = float(1.0 - positive_ratio)
        segment_scores[segment_id] = {
            "usable_views": int(usable_views),
            "visible_points": int(total_visible),
            "inside_points": int(total_inside),
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
        }
        if positive_ratio >= float(min_positive_ratio) and negative_ratio <= float(max_negative_ratio):
            kept.append(segment_id)

    info["usable_view_count"] = int(len(support_mask_views) or len(support_box_views))
    info["support_mode"] = "sam_mask" if support_mask_views else "box"
    info["output_segments"] = int(len(kept))
    info["filtered_segments"] = int(max(0, len(selected_segments) - len(kept)))
    info["min_positive_ratio"] = float(min_positive_ratio)
    info["max_negative_ratio"] = float(max_negative_ratio)
    info["min_visible_points"] = int(min_visible_points)
    info["min_views"] = int(min_views)
    if not kept:
        info["fallback"] = "no_segments_above_box_support"
        info["output_segments"] = int(len(selected_segments))
        return selected_segments, info
    ratios = [value["positive_ratio"] for value in segment_scores.values() if "positive_ratio" in value]
    if ratios:
        info["mean_positive_ratio"] = float(np.mean(ratios))
    return np.asarray(kept, dtype=selected_segments.dtype), info


def _superpoint_view_siou_metrics(
    proposal_mask,
    point_segments,
    point_visibility,
    support_views,
    min_visible_points=1,
):
    info = {
        "enabled": point_segments is not None and point_visibility is not None,
        "support_view_count": 0,
        "usable_view_count": 0,
        "mean_pairwise_siou": 1.0,
        "min_pairwise_siou": 1.0,
        "max_pairwise_siou": 1.0,
        "mean_visible_segments": 0.0,
        "reason": None,
    }
    if point_segments is None or point_visibility is None:
        info["reason"] = "missing_superpoint_or_visibility"
        return info

    support_frame_indices = _support_frame_indices(support_views)
    info["support_view_count"] = int(len(support_frame_indices))
    if len(support_frame_indices) < 2:
        info["reason"] = "few_support_views"
        return info

    segments = np.asarray(point_segments)
    visibility = _to_numpy(point_visibility).astype(bool)
    if segments.shape[0] != proposal_mask.shape[0]:
        info["reason"] = "segment_length_mismatch"
        return info
    if visibility.ndim != 2 or visibility.shape[1] != proposal_mask.shape[0]:
        info["reason"] = "visibility_shape_mismatch"
        return info

    proposal_indices = np.flatnonzero(proposal_mask)
    view_segment_sets = []
    for frame_index in support_frame_indices:
        if frame_index < 0 or frame_index >= visibility.shape[0]:
            continue
        visible_indices = proposal_indices[visibility[frame_index, proposal_indices]]
        if len(visible_indices) < int(min_visible_points):
            continue
        view_segments = np.unique(segments[visible_indices].astype(np.int64))
        if len(view_segments) == 0:
            continue
        view_segment_sets.append(set(int(item) for item in view_segments))

    info["usable_view_count"] = int(len(view_segment_sets))
    if len(view_segment_sets) < 2:
        info["reason"] = "few_usable_views"
        return info

    pairwise = []
    for left_idx, left in enumerate(view_segment_sets):
        for right in view_segment_sets[left_idx + 1 :]:
            intersection = len(left & right)
            union = len(left | right)
            pairwise.append(intersection / max(1, union))
    if not pairwise:
        info["reason"] = "no_pairwise_views"
        return info

    info["mean_pairwise_siou"] = float(np.mean(pairwise))
    info["min_pairwise_siou"] = float(np.min(pairwise))
    info["max_pairwise_siou"] = float(np.max(pairwise))
    info["mean_visible_segments"] = float(np.mean([len(item) for item in view_segment_sets]))
    info["reason"] = "ok"
    return info


def _postprocess_appended_proposals(
    masks_np,
    classes_np,
    scores_np,
    original_num_masks,
    merge_iou=0.0,
    inclusion_threshold=0.0,
    same_class_only=True,
    appended_metadata=None,
    point_segments=None,
    containment_action="none",
    containment_threshold=0.85,
    containment_min_area_ratio=1.5,
    containment_score_ratio=0.75,
    containment_quality_margin=0.0,
    containment_score_factor=0.5,
    containment_min_points=50,
    hierarchy_substitution_action="none",
    hierarchy_substitution_min_child_coverage=0.80,
    hierarchy_substitution_max_parent_exclusive_ratio=0.20,
    hierarchy_substitution_min_area_ratio=1.2,
    hierarchy_substitution_min_children=1,
):
    appended_count = int(masks_np.shape[1] - original_num_masks)
    containment_action = str(containment_action or "none").strip().lower()
    containment_enabled = containment_action not in ("", "none")
    hierarchy_substitution_action = str(hierarchy_substitution_action or "none").strip().lower()
    hierarchy_substitution_enabled = hierarchy_substitution_action not in ("", "none")
    summary = {
        "enabled": bool(
            (merge_iou and merge_iou > 0)
            or (inclusion_threshold and inclusion_threshold > 0)
            or containment_enabled
            or hierarchy_substitution_enabled
        ),
        "input_appended": appended_count,
        "merged": 0,
        "removed_included": 0,
        "containment_action": containment_action,
        "containment_events": [],
        "downweighted_containing": 0,
        "carved_containing": 0,
        "removed_containing": 0,
        "carved_points": 0,
        "hierarchy_substitution_action": hierarchy_substitution_action,
        "hierarchy_substitution_events": [],
        "hierarchy_removed_parents": 0,
        "output_appended": appended_count,
    }
    if not summary["enabled"] or appended_count <= 0:
        return masks_np, classes_np, scores_np, summary

    base_masks = masks_np[:, :original_num_masks]
    base_classes = classes_np[:original_num_masks]
    base_scores = scores_np[:original_num_masks]
    appended_masks = [masks_np[:, original_num_masks + idx].copy() for idx in range(appended_count)]
    appended_classes = [int(classes_np[original_num_masks + idx]) for idx in range(appended_count)]
    appended_scores = [float(scores_np[original_num_masks + idx]) for idx in range(appended_count)]
    appended_metadata = list(appended_metadata or [])
    if len(appended_metadata) < appended_count:
        appended_metadata.extend({} for _ in range(appended_count - len(appended_metadata)))
    keep = [True] * appended_count

    if merge_iou is not None and float(merge_iou) > 0:
        changed = True
        while changed:
            changed = False
            active = [idx for idx, is_kept in enumerate(keep) if is_kept]
            for pos, left in enumerate(active):
                if not keep[left]:
                    continue
                left_area = int(appended_masks[left].sum())
                if left_area == 0:
                    keep[left] = False
                    changed = True
                    continue
                for right in active[pos + 1 :]:
                    if not keep[right]:
                        continue
                    if same_class_only and appended_classes[left] != appended_classes[right]:
                        continue
                    right_area = int(appended_masks[right].sum())
                    if right_area == 0:
                        keep[right] = False
                        changed = True
                        continue
                    intersection = int(np.logical_and(appended_masks[left], appended_masks[right]).sum())
                    union = left_area + right_area - intersection
                    iou = intersection / max(1, union)
                    if iou < float(merge_iou):
                        continue
                    target, source = (left, right)
                    if appended_scores[right] > appended_scores[left]:
                        target, source = (right, left)
                    appended_masks[target] = np.logical_or(appended_masks[target], appended_masks[source])
                    appended_scores[target] = max(appended_scores[target], appended_scores[source])
                    keep[source] = False
                    summary["merged"] += 1
                    changed = True
                    break
                if changed:
                    break

    if inclusion_threshold is not None and float(inclusion_threshold) > 0:
        active = [idx for idx, is_kept in enumerate(keep) if is_kept]
        areas = {idx: int(appended_masks[idx].sum()) for idx in active}
        for small in sorted(active, key=lambda idx: (areas[idx], appended_scores[idx])):
            if not keep[small] or areas[small] == 0:
                continue
            for large in active:
                if small == large or not keep[large] or areas[large] < areas[small]:
                    continue
                if same_class_only and appended_classes[small] != appended_classes[large]:
                    continue
                intersection = int(np.logical_and(appended_masks[small], appended_masks[large]).sum())
                inclusion = intersection / max(1, areas[small])
                if inclusion >= float(inclusion_threshold):
                    keep[small] = False
                    summary["removed_included"] += 1
                    break

    if containment_enabled:
        valid_actions = {"downweight", "carve", "remove_large"}
        if containment_action not in valid_actions:
            summary["invalid_containment_action"] = containment_action
        else:
            threshold = float(containment_threshold)
            min_area_ratio = float(containment_min_area_ratio)
            score_ratio = float(containment_score_ratio)
            quality_margin = float(containment_quality_margin)
            score_factor = float(containment_score_factor)
            min_points = int(max(1, containment_min_points))
            downweighted = set()

            def small_item(global_index):
                if global_index < original_num_masks:
                    return {
                        "kind": "base",
                        "mask": base_masks[:, global_index],
                        "class_id": int(base_classes[global_index]),
                        "score": float(base_scores[global_index]),
                        "area": int(base_masks[:, global_index].sum()),
                        "quality": None,
                        "index": int(global_index),
                    }
                appended_index = int(global_index - original_num_masks)
                metadata = appended_metadata[appended_index] if appended_index < len(appended_metadata) else {}
                return {
                    "kind": "appended",
                    "mask": appended_masks[appended_index],
                    "class_id": int(appended_classes[appended_index]),
                    "score": float(appended_scores[appended_index]),
                    "area": int(appended_masks[appended_index].sum()),
                    "quality": metadata.get("quality_score"),
                    "candidate_id": metadata.get("candidate_id"),
                    "component_id": metadata.get("component_id"),
                    "index": appended_index,
                }

            def protects_small(large_index, small):
                large_score = float(appended_scores[large_index])
                small_score = float(small["score"])
                if small_score < large_score * score_ratio:
                    return False
                large_quality = appended_metadata[large_index].get("quality_score")
                small_quality = small.get("quality")
                if large_quality is not None and small_quality is not None:
                    try:
                        if float(small_quality) + quality_margin < float(large_quality):
                            return False
                    except (TypeError, ValueError):
                        pass
                return True

            for large in sorted(range(appended_count), key=lambda idx: int(appended_masks[idx].sum()), reverse=True):
                if not keep[large]:
                    continue
                large_area = int(appended_masks[large].sum())
                blockers = []
                global_indices = list(range(original_num_masks)) + [
                    original_num_masks + idx
                    for idx, is_kept in enumerate(keep)
                    if is_kept and idx != large
                ]
                for global_index in global_indices:
                    small = small_item(global_index)
                    small_area = int(small["area"])
                    if small_area <= 0 or large_area < small_area * min_area_ratio:
                        continue
                    if same_class_only and int(appended_classes[large]) != int(small["class_id"]):
                        continue
                    intersection = int(np.logical_and(appended_masks[large], small["mask"]).sum())
                    if intersection <= 0:
                        continue
                    small_coverage = intersection / max(1, small_area)
                    if small_coverage < threshold:
                        continue
                    if not protects_small(large, small):
                        continue
                    blockers.append(
                        {
                            "small_kind": small["kind"],
                            "small_index": int(small["index"]),
                            "small_candidate_id": small.get("candidate_id"),
                            "small_component_id": small.get("component_id"),
                            "small_area": small_area,
                            "small_score": float(small["score"]),
                            "intersection": intersection,
                            "small_coverage": float(small_coverage),
                            "large_area": large_area,
                            "large_score": float(appended_scores[large]),
                            "large_candidate_id": appended_metadata[large].get("candidate_id"),
                            "large_component_id": appended_metadata[large].get("component_id"),
                        }
                    )

                if not blockers:
                    continue

                event = {
                    "action": containment_action,
                    "large_index": int(large),
                    "large_candidate_id": appended_metadata[large].get("candidate_id"),
                    "large_component_id": appended_metadata[large].get("component_id"),
                    "large_area": int(large_area),
                    "large_score_before": float(appended_scores[large]),
                    "blockers": blockers,
                }
                if containment_action == "remove_large":
                    keep[large] = False
                    summary["removed_containing"] += 1
                    event["removed"] = True
                elif containment_action == "downweight":
                    if large not in downweighted:
                        appended_scores[large] *= max(0.0, min(1.0, score_factor))
                        downweighted.add(large)
                        summary["downweighted_containing"] += 1
                    event["large_score_after"] = float(appended_scores[large])
                elif containment_action == "carve":
                    carve_mask = np.zeros_like(appended_masks[large])
                    for blocker in blockers:
                        if blocker["small_kind"] == "base":
                            carve_mask |= base_masks[:, blocker["small_index"]]
                        else:
                            carve_mask |= appended_masks[blocker["small_index"]]
                    before = int(appended_masks[large].sum())
                    appended_masks[large] = np.logical_and(appended_masks[large], ~carve_mask)
                    after = int(appended_masks[large].sum())
                    removed_points = max(0, before - after)
                    summary["carved_points"] += int(removed_points)
                    if after < min_points:
                        keep[large] = False
                        summary["removed_containing"] += 1
                        event["removed"] = True
                    else:
                        summary["carved_containing"] += 1
                    event["large_area_after"] = int(after)
                    event["removed_points"] = int(removed_points)
                summary["containment_events"].append(event)

    if hierarchy_substitution_enabled:
        valid_actions = {"remove_parent"}
        if hierarchy_substitution_action not in valid_actions:
            summary["invalid_hierarchy_substitution_action"] = hierarchy_substitution_action
        else:
            min_child_coverage = float(hierarchy_substitution_min_child_coverage)
            max_parent_exclusive_ratio = float(hierarchy_substitution_max_parent_exclusive_ratio)
            min_area_ratio = float(hierarchy_substitution_min_area_ratio)
            min_children = int(max(1, hierarchy_substitution_min_children))
            active = [idx for idx, is_kept in enumerate(keep) if is_kept]
            areas = {idx: int(appended_masks[idx].sum()) for idx in active}
            segment_inverse = None
            segment_sizes = None
            occupancies = {}
            masses = {}
            if point_segments is not None:
                segments = np.asarray(point_segments)
                if segments.shape[0] == masks_np.shape[0]:
                    _, segment_inverse = np.unique(segments.astype(np.int64, copy=False), return_inverse=True)
                    segment_sizes = np.maximum(np.bincount(segment_inverse).astype(np.float32), 1.0)
                    for idx in active:
                        counts = np.bincount(
                            segment_inverse[appended_masks[idx]],
                            minlength=len(segment_sizes),
                        ).astype(np.float32)
                        occupancy = counts / segment_sizes
                        occupancies[idx] = occupancy
                        masses[idx] = float(np.sum(occupancy * segment_sizes))

            use_superpoint_occupancy = segment_inverse is not None and bool(occupancies)
            for parent in sorted(active, key=lambda idx: areas[idx], reverse=True):
                if not keep[parent] or areas[parent] <= 0:
                    continue
                child_items = []
                child_union = (
                    np.zeros_like(occupancies[parent], dtype=np.float32)
                    if use_superpoint_occupancy
                    else np.zeros_like(appended_masks[parent])
                )
                parent_mass = masses.get(parent, float(areas[parent]))
                for child in active:
                    if child == parent or not keep[child] or areas.get(child, 0) <= 0:
                        continue
                    if same_class_only and appended_classes[parent] != appended_classes[child]:
                        continue
                    child_mass = masses.get(child, float(areas[child]))
                    if parent_mass < child_mass * min_area_ratio:
                        continue
                    if use_superpoint_occupancy:
                        overlap = float(np.sum(np.minimum(occupancies[parent], occupancies[child]) * segment_sizes))
                        if overlap <= 0.0:
                            continue
                        child_coverage = overlap / max(child_mass, 1.0)
                        child_union = np.maximum(child_union, occupancies[child])
                        intersection = int(round(overlap))
                    else:
                        intersection = int(np.logical_and(appended_masks[parent], appended_masks[child]).sum())
                        if intersection <= 0:
                            continue
                        child_coverage = intersection / max(1, areas[child])
                        child_union |= appended_masks[child]
                    if child_coverage < min_child_coverage:
                        continue
                    child_items.append(
                        {
                            "child_index": int(child),
                            "child_candidate_id": appended_metadata[child].get("candidate_id"),
                            "child_component_id": appended_metadata[child].get("component_id"),
                            "child_area": int(areas[child]),
                            "child_mass": float(child_mass),
                            "child_score": float(appended_scores[child]),
                            "intersection": int(intersection),
                            "child_coverage": float(child_coverage),
                        }
                    )

                if len(child_items) < min_children:
                    continue
                if use_superpoint_occupancy:
                    covered_parent = float(np.sum(np.minimum(occupancies[parent], child_union) * segment_sizes))
                    parent_area = max(1.0, parent_mass)
                else:
                    covered_parent = float(np.logical_and(appended_masks[parent], child_union).sum())
                    parent_area = max(1.0, float(areas[parent]))
                exclusive_ratio = float(max(0.0, parent_area - covered_parent) / parent_area)
                child_union_coverage = float(covered_parent / parent_area)
                if exclusive_ratio > max_parent_exclusive_ratio:
                    continue
                keep[parent] = False
                summary["hierarchy_removed_parents"] += 1
                summary["hierarchy_substitution_events"].append(
                    {
                        "action": hierarchy_substitution_action,
                        "parent_index": int(parent),
                        "parent_candidate_id": appended_metadata[parent].get("candidate_id"),
                        "parent_component_id": appended_metadata[parent].get("component_id"),
                        "parent_area": int(areas[parent]),
                        "parent_mass": float(parent_mass),
                        "parent_score": float(appended_scores[parent]),
                        "child_count": int(len(child_items)),
                        "child_union_coverage": child_union_coverage,
                        "parent_exclusive_ratio": exclusive_ratio,
                        "use_superpoint_occupancy": bool(use_superpoint_occupancy),
                        "children": child_items,
                    }
                )

    kept_indices = [idx for idx, is_kept in enumerate(keep) if is_kept]
    if kept_indices:
        kept_masks = np.stack([appended_masks[idx] for idx in kept_indices], axis=1)
        kept_classes = np.asarray([appended_classes[idx] for idx in kept_indices], dtype=np.int64)
        kept_scores = np.asarray([appended_scores[idx] for idx in kept_indices], dtype=np.float32)
        masks_np = np.concatenate([base_masks, kept_masks], axis=1)
        classes_np = np.concatenate([base_classes, kept_classes])
        scores_np = np.concatenate([base_scores, kept_scores])
    else:
        masks_np = base_masks
        classes_np = base_classes
        scores_np = base_scores
    summary["output_appended"] = int(masks_np.shape[1] - original_num_masks)
    return masks_np, classes_np, scores_np, summary


def _refine_mask_with_superpoints(
    proposal_mask,
    point_segments,
    point_visibility=None,
    support_views=None,
    projections=None,
    scaling_params=None,
    min_coverage=0.30,
    max_expansion_ratio=2.0,
    max_segment_ratio=None,
    large_segment_min_coverage=None,
    min_seed_retention=0.0,
    min_support_views=0,
    min_support_ratio=0.0,
    min_box_positive_ratio=0.0,
    max_box_negative_ratio=1.0,
    box_support_min_visible_points=5,
    box_support_min_views=1,
    box_support_padding_ratio=0.05,
    min_output_points=1,
):
    info = {
        "enabled": point_segments is not None,
        "input_points": int(proposal_mask.sum()),
        "output_points": int(proposal_mask.sum()),
        "selected_segments": 0,
        "touched_segments": 0,
        "size_filtered_segments": 0,
        "support_filtered_segments": 0,
        "box_filtered_segments": 0,
        "required_support_views": 0,
        "fallback": None,
    }
    if point_segments is None:
        return proposal_mask, info
    input_points = int(proposal_mask.sum())
    if input_points <= 0:
        info["fallback"] = "empty_input"
        return proposal_mask, info

    segments = np.asarray(point_segments)
    if segments.shape[0] != proposal_mask.shape[0]:
        info["fallback"] = "segment_length_mismatch"
        return proposal_mask, info

    touched_segments, seed_counts = np.unique(segments[proposal_mask], return_counts=True)
    info["touched_segments"] = int(len(touched_segments))
    if len(touched_segments) == 0:
        info["fallback"] = "no_touched_segments"
        return proposal_mask, info

    segment_sizes = np.bincount(segments.astype(np.int64), minlength=int(segments.max()) + 1)
    touched_sizes = segment_sizes[touched_segments.astype(np.int64)]
    coverage = seed_counts / np.maximum(touched_sizes, 1)
    coverage_thresholds = np.full_like(coverage, float(min_coverage), dtype=np.float64)
    if max_segment_ratio is not None and float(max_segment_ratio) > 0:
        large_segment_mask = touched_sizes > max(1.0, input_points * float(max_segment_ratio))
        if large_segment_min_coverage is not None:
            coverage_thresholds[large_segment_mask] = np.maximum(
                coverage_thresholds[large_segment_mask],
                float(large_segment_min_coverage),
            )
        else:
            coverage_thresholds[large_segment_mask] = 1.01
        info["size_filtered_segments"] = int(np.count_nonzero(large_segment_mask & (coverage < coverage_thresholds)))
        info["max_segment_ratio"] = float(max_segment_ratio)
        info["large_segment_min_coverage"] = (
            None if large_segment_min_coverage is None else float(large_segment_min_coverage)
        )
    selected_segments = touched_segments[coverage >= coverage_thresholds]
    if len(selected_segments) == 0:
        info["fallback"] = "no_segments_above_coverage"
        return proposal_mask, info

    support_frame_indices = _support_frame_indices(support_views)
    if (
        point_visibility is not None
        and support_frame_indices
        and (float(min_support_ratio) > 0.0 or int(min_support_views) > 0)
    ):
        visibility = _to_numpy(point_visibility).astype(bool)
        if visibility.ndim == 2 and visibility.shape[1] == proposal_mask.shape[0]:
            proposal_indices = np.flatnonzero(proposal_mask)
            segment_support = {}
            for frame_index in support_frame_indices:
                if frame_index < 0 or frame_index >= visibility.shape[0]:
                    continue
                visible_seed = proposal_indices[visibility[frame_index, proposal_indices]]
                if len(visible_seed) == 0:
                    continue
                for segment_id in np.unique(segments[visible_seed]):
                    segment_support[int(segment_id)] = segment_support.get(int(segment_id), 0) + 1
            valid_view_count = max(1, len(support_frame_indices))
            required_views = min(
                valid_view_count,
                max(
                    1,
                    int(min_support_views),
                    int(np.ceil(float(min_support_ratio) * valid_view_count)),
                ),
            )
            if int(min_support_views) <= 0 and float(min_support_ratio) <= 0.0:
                required_views = 0
            info["required_support_views"] = int(required_views)
            if required_views > 0:
                selected_segments = np.asarray(
                    [
                        segment_id
                        for segment_id in selected_segments
                        if segment_support.get(int(segment_id), 0) >= required_views
                    ],
                    dtype=selected_segments.dtype,
                )
                info["support_filtered_segments"] = int(len(selected_segments))
                if len(selected_segments) == 0:
                    info["fallback"] = "no_segments_above_support"
                    return proposal_mask, info

    if float(min_box_positive_ratio or 0.0) > 0.0:
        selected_segments, box_info = _filter_superpoint_segments_by_box_support(
            selected_segments,
            segments,
            proposal_mask,
            point_visibility,
            support_views,
            projections,
            scaling_params,
            min_positive_ratio=min_box_positive_ratio,
            max_negative_ratio=max_box_negative_ratio,
            min_visible_points=box_support_min_visible_points,
            min_views=box_support_min_views,
            box_padding_ratio=box_support_padding_ratio,
        )
        info["box_support_filter"] = box_info
        info["box_filtered_segments"] = int(box_info.get("filtered_segments", 0))

    refined_mask = np.isin(segments, selected_segments)
    output_points = int(refined_mask.sum())
    retained_seed_points = int(np.logical_and(refined_mask, proposal_mask).sum())
    seed_retention = retained_seed_points / max(1, input_points)
    info["selected_segments"] = int(len(selected_segments))
    info["output_points"] = output_points
    info["retained_seed_points"] = retained_seed_points
    info["seed_retention"] = float(seed_retention)
    if float(min_seed_retention or 0.0) > 0.0 and seed_retention < float(min_seed_retention):
        info["fallback"] = "low_seed_retention"
        info["output_points"] = input_points
        return proposal_mask, info
    if output_points < int(min_output_points):
        info["fallback"] = "small_output"
        info["output_points"] = input_points
        return proposal_mask, info
    if max_expansion_ratio is not None and output_points > input_points * float(max_expansion_ratio):
        info["fallback"] = "too_much_expansion"
        info["output_points"] = input_points
        return proposal_mask, info
    return refined_mask, info


def _superpoint_hierarchy_score_factor(
    proposal_mask,
    point_segments,
    low_occupancy_threshold=0.25,
    score_weight=0.0,
    min_score_factor=0.5,
):
    info = {
        "enabled": point_segments is not None and float(score_weight or 0.0) > 0.0,
        "factor": 1.0,
        "superpoint_count": 0,
        "mean_occupancy": 0.0,
        "min_occupancy": 0.0,
        "max_occupancy": 0.0,
        "low_occupancy_mass_ratio": 0.0,
        "low_occupancy_threshold": float(low_occupancy_threshold),
        "score_weight": float(score_weight or 0.0),
        "min_score_factor": float(min_score_factor),
        "fallback": None,
    }
    if not info["enabled"]:
        return 1.0, info
    if point_segments is None:
        info["fallback"] = "missing_segments"
        return 1.0, info
    if int(proposal_mask.sum()) <= 0:
        info["fallback"] = "empty_mask"
        return 1.0, info

    segments = np.asarray(point_segments)
    if segments.shape[0] != proposal_mask.shape[0]:
        info["fallback"] = "segment_length_mismatch"
        return 1.0, info

    _, inverse = np.unique(segments.astype(np.int64, copy=False), return_inverse=True)
    segment_sizes = np.maximum(np.bincount(inverse).astype(np.float32), 1.0)
    proposal_counts = np.bincount(inverse[proposal_mask], minlength=len(segment_sizes)).astype(np.float32)
    occupancy = proposal_counts / segment_sizes
    touched = occupancy > 0.0
    if not touched.any():
        info["fallback"] = "no_touched_segments"
        return 1.0, info

    touched_occupancy = occupancy[touched]
    mass = float(np.sum(proposal_counts))
    low_threshold = min(1.0, max(0.0, float(low_occupancy_threshold)))
    low_mask = (occupancy > 0.0) & (occupancy < low_threshold)
    low_mass = float(np.sum(proposal_counts[low_mask]))
    low_ratio = float(low_mass / max(mass, 1.0))
    weight = min(1.0, max(0.0, float(score_weight)))
    min_factor = min(1.0, max(0.0, float(min_score_factor)))
    factor = 1.0 - weight * low_ratio
    factor = float(min(1.0, max(min_factor, factor)))

    info.update(
        {
            "factor": factor,
            "superpoint_count": int(touched.sum()),
            "mean_occupancy": float(np.mean(touched_occupancy)),
            "min_occupancy": float(np.min(touched_occupancy)),
            "max_occupancy": float(np.max(touched_occupancy)),
            "low_occupancy_mass_ratio": low_ratio,
        }
    )
    return factor, info


class _LocalUnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.uint8)
        self.component_size = np.ones(size, dtype=np.int32)
        self.internal = np.zeros(size, dtype=np.float32)

    def find(self, index):
        parent = self.parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(self, left, right, edge_weight):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] += self.component_size[right_root]
        self.internal[left_root] = float(edge_weight)
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root


def _local_felzenszwalb_labels(points, knn=10, merge_k=0.25, min_size=10):
    num_points = int(points.shape[0])
    if num_points <= 1:
        return np.zeros(num_points, dtype=np.int32), {"edges": 0}

    knn = int(max(1, min(int(knn), num_points - 1)))
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=knn + 1, workers=-1)
    neighbors = np.asarray(neighbors)[:, 1:]
    distances = np.asarray(distances)[:, 1:]
    left = np.repeat(np.arange(num_points, dtype=np.int32), knn)
    right = neighbors.reshape(-1).astype(np.int32)
    dist = distances.reshape(-1).astype(np.float32)
    valid = (right >= 0) & (right < num_points) & (left < right)
    left = left[valid]
    right = right[valid]
    dist = dist[valid]
    if len(dist) == 0:
        return np.arange(num_points, dtype=np.int32), {"edges": 0}

    scale = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
    weights = (dist / max(scale, 1e-6)).astype(np.float32)
    order = np.argsort(weights, kind="mergesort")
    uf = _LocalUnionFind(num_points)
    merge_k = float(merge_k)
    for edge_index in order:
        lidx = int(left[edge_index])
        ridx = int(right[edge_index])
        weight = float(weights[edge_index])
        lroot = uf.find(lidx)
        rroot = uf.find(ridx)
        if lroot == rroot:
            continue
        left_threshold = float(uf.internal[lroot]) + merge_k / float(uf.component_size[lroot])
        right_threshold = float(uf.internal[rroot]) + merge_k / float(uf.component_size[rroot])
        if weight <= min(left_threshold, right_threshold):
            uf.union(lroot, rroot, weight)

    min_size = int(max(1, min_size))
    if min_size > 1:
        for edge_index in order:
            lroot = uf.find(int(left[edge_index]))
            rroot = uf.find(int(right[edge_index]))
            if lroot == rroot:
                continue
            if uf.component_size[lroot] < min_size or uf.component_size[rroot] < min_size:
                uf.union(lroot, rroot, float(weights[edge_index]))

    roots = np.asarray([uf.find(index) for index in range(num_points)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32), {"edges": int(len(weights))}


def _refine_mask_with_local_superpoints(
    proposal_mask,
    seed_reference_mask,
    points_xyz,
    knn=10,
    merge_k=0.25,
    min_size=10,
    min_coverage=0.25,
    max_expansion_ratio=1.0,
    min_seed_retention=0.80,
    min_output_points=1,
    max_points=30000,
):
    info = {
        "enabled": points_xyz is not None,
        "input_points": int(proposal_mask.sum()),
        "output_points": int(proposal_mask.sum()),
        "seed_reference_points": int(np.logical_and(proposal_mask, seed_reference_mask).sum())
        if seed_reference_mask is not None
        else 0,
        "local_segments": 0,
        "selected_segments": 0,
        "fallback": None,
    }
    if points_xyz is None:
        return proposal_mask, info
    input_points = int(proposal_mask.sum())
    if input_points <= 0:
        info["fallback"] = "empty_input"
        return proposal_mask, info
    if seed_reference_mask is None or seed_reference_mask.shape[0] != proposal_mask.shape[0]:
        info["fallback"] = "missing_seed_reference"
        return proposal_mask, info
    if points_xyz.shape[0] != proposal_mask.shape[0]:
        info["fallback"] = "point_count_mismatch"
        return proposal_mask, info
    if max_points is not None and input_points > int(max_points):
        info["fallback"] = "too_many_points"
        return proposal_mask, info

    local_indices = np.flatnonzero(proposal_mask)
    local_seed = seed_reference_mask[local_indices].astype(bool)
    seed_points = int(local_seed.sum())
    info["seed_reference_points"] = seed_points
    if seed_points <= 0:
        info["fallback"] = "empty_seed_reference"
        return proposal_mask, info

    local_points = np.asarray(points_xyz[local_indices], dtype=np.float32)
    labels, label_info = _local_felzenszwalb_labels(
        local_points,
        knn=knn,
        merge_k=merge_k,
        min_size=min_size,
    )
    segment_ids, segment_sizes = np.unique(labels, return_counts=True)
    info["local_segments"] = int(len(segment_ids))
    info["edges"] = int(label_info.get("edges", 0))
    if len(segment_ids) <= 1:
        info["fallback"] = "single_local_segment"
        return proposal_mask, info

    seed_counts = np.bincount(labels[local_seed], minlength=int(labels.max()) + 1)
    all_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    coverage = seed_counts / np.maximum(all_sizes, 1)
    selected_segments = np.flatnonzero(coverage >= float(min_coverage))
    info["selected_segments"] = int(len(selected_segments))
    info["max_segment_coverage"] = float(coverage.max(initial=0.0))
    if len(selected_segments) == 0:
        info["fallback"] = "no_segments_above_coverage"
        return proposal_mask, info

    keep_local = np.isin(labels, selected_segments)
    refined_indices = local_indices[keep_local]
    refined_mask = np.zeros_like(proposal_mask)
    refined_mask[refined_indices] = True
    output_points = int(refined_mask.sum())
    retained_seed_points = int(np.logical_and(refined_mask, seed_reference_mask).sum())
    seed_retention = retained_seed_points / max(1, seed_points)
    info["output_points"] = output_points
    info["retained_seed_points"] = retained_seed_points
    info["seed_retention"] = float(seed_retention)
    if seed_retention < float(min_seed_retention):
        info["fallback"] = "low_seed_retention"
        info["output_points"] = input_points
        return proposal_mask, info
    if output_points < int(min_output_points):
        info["fallback"] = "small_output"
        info["output_points"] = input_points
        return proposal_mask, info
    if max_expansion_ratio is not None and output_points > input_points * float(max_expansion_ratio):
        info["fallback"] = "too_much_expansion"
        info["output_points"] = input_points
        return proposal_mask, info
    return refined_mask, info


def _grow_seed_mask(seed_indices, points_xyz, num_points, grow_radius, max_growth_ratio, max_grow_points):
    mask = np.zeros(num_points, dtype=bool)
    mask[seed_indices] = True
    if points_xyz is None or grow_radius <= 0:
        return mask, 1.0

    seed_points = points_xyz[seed_indices]
    lower = seed_points.min(axis=0) - float(grow_radius)
    upper = seed_points.max(axis=0) + float(grow_radius)
    grown = np.all((points_xyz >= lower) & (points_xyz <= upper), axis=1)
    grown_count = int(grown.sum())
    seed_count = max(1, int(len(seed_indices)))
    growth_ratio = grown_count / seed_count

    if grown_count == 0:
        return mask, 1.0
    if max_grow_points is not None and grown_count > max_grow_points:
        return mask, 1.0
    if max_growth_ratio is not None and growth_ratio > max_growth_ratio:
        return mask, 1.0
    return grown, float(growth_ratio)


def _connected_component_cleanup(
    proposal_mask,
    points_xyz,
    radius,
    min_component_points,
    keep_topk,
    max_points,
):
    if points_xyz is None or radius <= 0:
        return proposal_mask, {"enabled": False}

    proposal_indices = np.flatnonzero(proposal_mask)
    if len(proposal_indices) == 0:
        return proposal_mask, {"enabled": True, "reason": "empty_mask"}
    if max_points is not None and len(proposal_indices) > max_points:
        return proposal_mask, {
            "enabled": True,
            "reason": "too_many_points",
            "input_points": int(len(proposal_indices)),
        }

    local_points = points_xyz[proposal_indices]
    tree = cKDTree(local_points)
    neighbors = tree.query_ball_point(local_points, r=float(radius))
    num_points = len(proposal_indices)
    parent = np.arange(num_points, dtype=np.int32)
    rank = np.zeros(num_points, dtype=np.uint8)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    for index, local_neighbors in enumerate(neighbors):
        for neighbor in local_neighbors:
            if neighbor > index:
                union(index, int(neighbor))

    roots = np.asarray([find(index) for index in range(num_points)], dtype=np.int32)
    unique_roots, counts = np.unique(roots, return_counts=True)
    order = np.argsort(-counts)
    min_component_points = int(max(1, min_component_points))
    keep_topk = int(max(1, keep_topk))
    kept_roots = []
    for order_idx in order:
        if len(kept_roots) >= keep_topk:
            break
        if int(counts[order_idx]) < min_component_points:
            continue
        kept_roots.append(unique_roots[order_idx])

    if not kept_roots:
        kept_roots = [unique_roots[order[0]]]

    keep_local = np.isin(roots, np.asarray(kept_roots, dtype=np.int32))
    cleaned_indices = proposal_indices[keep_local]
    cleaned_mask = np.zeros_like(proposal_mask)
    cleaned_mask[cleaned_indices] = True
    return cleaned_mask, {
        "enabled": True,
        "input_points": int(len(proposal_indices)),
        "output_points": int(len(cleaned_indices)),
        "num_components": int(len(unique_roots)),
        "largest_component_points": int(counts[order[0]]),
        "kept_components": int(len(kept_roots)),
        "radius": float(radius),
    }


def _connected_component_split(
    proposal_mask,
    points_xyz,
    radius,
    min_component_points,
    keep_topk,
    max_points,
):
    if points_xyz is None or radius <= 0:
        return [(proposal_mask, {"enabled": False})]

    proposal_indices = np.flatnonzero(proposal_mask)
    if len(proposal_indices) == 0:
        return []
    if max_points is not None and len(proposal_indices) > max_points:
        return [
            (
                proposal_mask,
                {
                    "enabled": True,
                    "split_components": True,
                    "reason": "too_many_points",
                    "input_points": int(len(proposal_indices)),
                },
            )
        ]

    local_points = points_xyz[proposal_indices]
    tree = cKDTree(local_points)
    neighbors = tree.query_ball_point(local_points, r=float(radius))
    num_points = len(proposal_indices)
    parent = np.arange(num_points, dtype=np.int32)
    rank = np.zeros(num_points, dtype=np.uint8)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    for index, local_neighbors in enumerate(neighbors):
        for neighbor in local_neighbors:
            if neighbor > index:
                union(index, int(neighbor))

    roots = np.asarray([find(index) for index in range(num_points)], dtype=np.int32)
    unique_roots, counts = np.unique(roots, return_counts=True)
    order = np.argsort(-counts)
    min_component_points = int(max(1, min_component_points))
    keep_topk = int(max(1, keep_topk))
    outputs = []
    for component_rank, order_idx in enumerate(order):
        if len(outputs) >= keep_topk:
            break
        component_points = int(counts[order_idx])
        if component_points < min_component_points:
            continue
        component_indices = proposal_indices[roots == unique_roots[order_idx]]
        component_mask = np.zeros_like(proposal_mask)
        component_mask[component_indices] = True
        outputs.append(
            (
                component_mask,
                {
                    "enabled": True,
                    "split_components": True,
                    "input_points": int(len(proposal_indices)),
                    "output_points": int(len(component_indices)),
                    "num_components": int(len(unique_roots)),
                    "largest_component_points": int(counts[order[0]]),
                    "kept_components": int(min(keep_topk, len(order))),
                    "component_rank": int(component_rank),
                    "radius": float(radius),
                },
            )
        )

    if not outputs:
        component_indices = proposal_indices[roots == unique_roots[order[0]]]
        component_mask = np.zeros_like(proposal_mask)
        component_mask[component_indices] = True
        outputs.append(
            (
                component_mask,
                {
                    "enabled": True,
                    "split_components": True,
                    "input_points": int(len(proposal_indices)),
                    "output_points": int(len(component_indices)),
                    "num_components": int(len(unique_roots)),
                    "largest_component_points": int(counts[order[0]]),
                    "kept_components": 1,
                    "component_rank": 0,
                    "radius": float(radius),
                },
            )
        )
    return outputs


def _candidate_quality_score(candidate):
    detector_score = float(candidate.get("score", 0.0))
    fusion_score = float(candidate.get("fusion_score", detector_score))
    support_views = float(candidate.get("support_view_count", 0.0))
    support_mean_iou = float(candidate.get("support_mean_iou", 0.0))
    support_best_iou = float(candidate.get("support_best_iou", 0.0))
    seed_covered = float(candidate.get("seed_in_existing_mask_ratio", 0.0))
    existing_iou = float(candidate.get("best_existing_iou", 0.0))
    box_area_ratio = float(candidate.get("box_area_ratio", 0.0))

    support_term = 0.45 + 0.55 * min(1.0, support_views / 50.0)
    consistency_term = 0.35 + 0.45 * min(1.0, support_mean_iou) + 0.20 * min(1.0, support_best_iou)
    label_term = _candidate_label_consensus_score(candidate)
    novelty_term = 1.0 - 0.65 * min(1.0, seed_covered)
    separation_term = 1.0 - 0.35 * min(1.0, existing_iou)
    extent_term = 1.0 - 0.25 * min(1.0, box_area_ratio / 0.25)
    return float(
        max(0.0, fusion_score)
        * max(0.0, detector_score)
        * support_term
        * consistency_term
        * label_term
        * novelty_term
        * separation_term
        * extent_term
    )


def append_backprojection_proposals(
    scene_name,
    pred_masks,
    pred_classes,
    pred_scores,
    candidates_by_scene,
    points_xyz=None,
    point_segments=None,
    point_visibility=None,
    min_score=0.35,
    min_seed_points=80,
    max_existing_iou=0.30,
    max_seed_in_existing_mask_ratio=0.70,
    max_proposal_iou=0.50,
    max_candidates=None,
    score_scale=0.50,
    use_candidate_fusion_score=True,
    allowed_classes=None,
    blocked_classes=None,
    min_support_views=0,
    min_support_mean_iou=0.0,
    min_support_best_iou=0.0,
    min_fusion_score=0.0,
    max_box_area_ratio=None,
    min_quality_score=0.0,
    min_scene_source_quality_z=None,
    quality_sort=False,
    verifications_by_scene=None,
    verifier_suppress_decisions=None,
    grow_radius=0.0,
    max_growth_ratio=4.0,
    max_grow_points=50000,
    seed_cc_cleanup=False,
    seed_cc_radius=0.03,
    seed_cc_min_component_points=30,
    seed_cc_keep_topk=1,
    seed_cc_max_points=30000,
    seed_cc_min_keep_ratio=0.0,
    cc_cleanup=False,
    cc_radius=0.03,
    cc_min_component_points=50,
    cc_keep_topk=1,
    cc_max_points=30000,
    cc_split_components=False,
    cc_source_kinds=None,
    cc_min_keep_ratio=0.0,
    cc_keep_ratio_score_weight=0.0,
    source_priorities=None,
    source_max_candidates=None,
    source_score_scales=None,
    source_min_scores=None,
    max_candidates_per_class=None,
    class_max_candidates=None,
    quality_calibration_weight=0.0,
    novelty_calibration_weight=0.0,
    label_consensus_calibration_weight=0.0,
    score_calibration_min=0.2,
    score_calibration_max=1.2,
    max_proposal_score=None,
    min_label_consensus_score=0.0,
    max_label_conflict_score=1.0,
    label_consensus_context=None,
    label_consensus_iou_threshold=0.25,
    label_consensus_min_visible_points=30,
    label_consensus_frame_mode="support",
    projection_consistency_context=None,
    projection_consistency_min_box_iou=0.0,
    projection_consistency_min_point_ratio=0.0,
    projection_consistency_min_views=1,
    projection_consistency_min_visible_points=30,
    projection_consistency_frame_mode="support",
    projection_consistency_box_padding_ratio=0.05,
    projection_consistency_score_weight=0.0,
    superpoint_refine=False,
    superpoint_min_coverage=0.30,
    superpoint_max_expansion_ratio=2.0,
    superpoint_max_segment_ratio=None,
    superpoint_large_segment_min_coverage=None,
    superpoint_min_seed_retention=0.0,
    superpoint_min_support_views=0,
    superpoint_min_support_ratio=0.0,
    superpoint_min_view_siou=0.0,
    superpoint_view_siou_min_views=2,
    superpoint_view_siou_min_visible_points=1,
    superpoint_box_context=None,
    superpoint_min_box_positive_ratio=0.0,
    superpoint_max_box_negative_ratio=1.0,
    superpoint_box_min_visible_points=5,
    superpoint_box_min_views=1,
    superpoint_box_padding_ratio=0.05,
    local_superpoint_refine=False,
    local_superpoint_knn=10,
    local_superpoint_merge_k=0.25,
    local_superpoint_min_size=10,
    local_superpoint_min_coverage=0.25,
    local_superpoint_max_expansion_ratio=1.0,
    local_superpoint_min_seed_retention=0.80,
    local_superpoint_max_points=30000,
    hierarchy_score_weight=0.0,
    hierarchy_low_occupancy_threshold=0.25,
    hierarchy_min_score_factor=0.5,
    mask_graph_min_cluster_observations=0,
    mask_graph_min_selected_views=0,
    mask_graph_min_same_object_edges=0,
    mask_graph_min_edge_mean_score=0.0,
    mask_graph_min_consensus_score=0.0,
    mask_graph_min_depth_consistency=0.0,
    mask_graph_max_conflict_edges=None,
    mask_graph_max_conflict_ratio=None,
    mask_graph_evidence_rescore=False,
    mask_graph_evidence_min_overlap=0.25,
    mask_graph_evidence_min_iou=0.03,
    mask_graph_evidence_priority_weight=0.0,
    mask_graph_evidence_same_class_only=True,
    merge_iou=0.0,
    inclusion_threshold=0.0,
    postprocess_same_class_only=True,
    containment_action="none",
    containment_threshold=0.85,
    containment_min_area_ratio=1.5,
    containment_score_ratio=0.75,
    containment_quality_margin=0.0,
    containment_score_factor=0.5,
    containment_min_points=50,
    hierarchy_substitution_action="none",
    hierarchy_substitution_min_child_coverage=0.80,
    hierarchy_substitution_max_parent_exclusive_ratio=0.20,
    hierarchy_substitution_min_area_ratio=1.2,
    hierarchy_substitution_min_children=1,
):
    """Append conservative 2D-to-3D proposal masks to one scene prediction."""

    masks_np = _to_numpy(pred_masks).astype(bool)
    classes_np = _to_numpy(pred_classes).astype(np.int64)
    scores_np = _to_numpy(pred_scores).astype(np.float32)
    original_num_masks = int(masks_np.shape[1])
    num_points = masks_np.shape[0]
    scene_candidates = candidates_by_scene.get(scene_name, [])
    scene_verifications = (verifications_by_scene or {}).get(scene_name, {})
    verifier_suppress_decisions = _parse_decision_filter(
        verifier_suppress_decisions or "suppress,bad_mask,invalid,reject"
    )
    report = {"loaded": len(scene_candidates), "applied": [], "skipped": []}
    if not scene_candidates:
        return masks_np, classes_np, scores_np, report
    allowed_classes = _parse_class_filter(allowed_classes)
    blocked_classes = _parse_class_filter(blocked_classes)
    source_priorities = _parse_source_rules(source_priorities, float)
    source_max_candidates = _parse_source_rules(source_max_candidates, int)
    source_score_scales = _parse_source_rules(source_score_scales, float)
    source_min_scores = _parse_source_rules(source_min_scores, float)
    class_max_candidates = _parse_source_rules(class_max_candidates, int)
    cc_source_filter = _parse_source_filter(cc_source_kinds)
    scene_candidates = _annotate_candidate_quality_stats([dict(item) for item in scene_candidates])
    if mask_graph_evidence_rescore:
        scene_candidates = _annotate_mask_graph_evidence(
            scene_candidates,
            num_points,
            min_overlap=mask_graph_evidence_min_overlap,
            min_iou=mask_graph_evidence_min_iou,
            priority_weight=mask_graph_evidence_priority_weight,
            same_class_only=mask_graph_evidence_same_class_only,
        )

    scene_candidates = sorted(
        scene_candidates,
        key=lambda item: (
            -float(_lookup_source_rule(item, source_priorities, 1.0)),
            -_candidate_selection_score(
                item,
                quality_calibration_weight,
                novelty_calibration_weight,
                label_consensus_calibration_weight,
            )
            if quality_sort
            else 0.0,
            -float(item.get("proposal_priority", item.get("score", 0.0))),
            float(item.get("best_existing_iou", 1.0)),
            -float(item.get("score", 0.0)),
        ),
    )

    appended_masks = []
    source_counts = defaultdict(int)
    class_counts = defaultdict(int)
    for candidate in scene_candidates:
        candidate_id = candidate.get("candidate_id")
        class_name = candidate.get("class_name")
        source_name = _candidate_source_name(candidate)
        source_kind = _candidate_source_kind(candidate)
        source_limit = _lookup_source_rule(candidate, source_max_candidates, None)
        if source_limit is not None and source_counts[source_kind] >= int(source_limit):
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "source_limit",
                    "source_name": source_name,
                    "source_kind": source_kind,
                    "source_limit": int(source_limit),
                }
            )
            continue
        class_limit = _lookup_class_rule(class_name, class_max_candidates, max_candidates_per_class)
        if class_limit is not None and class_counts[str(class_name).strip().lower()] >= int(class_limit):
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "class_limit",
                    "class_name": class_name,
                    "class_limit": int(class_limit),
                }
            )
            continue
        verification = None
        if candidate_id is not None:
            try:
                verification = scene_verifications.get(int(candidate_id))
            except (TypeError, ValueError):
                verification = None
        if verification is not None and verification.get("decision") in verifier_suppress_decisions:
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "mllm_verifier_suppressed",
                    "decision": verification.get("decision"),
                    "confidence": verification.get("confidence"),
                    "verifier_reason": verification.get("reason"),
                }
            )
            continue
        if allowed_classes is not None and class_name not in allowed_classes:
            report["skipped"].append(
                {"candidate_id": candidate_id, "reason": "class_not_allowed", "class_name": class_name}
            )
            continue
        if blocked_classes is not None and class_name in blocked_classes:
            report["skipped"].append(
                {"candidate_id": candidate_id, "reason": "class_blocked", "class_name": class_name}
            )
            continue

        score = float(candidate.get("score", 0.0))
        effective_min_score = float(_lookup_source_rule(candidate, source_min_scores, min_score))
        if score < effective_min_score:
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_score",
                    "score": score,
                    "min_score": effective_min_score,
                    "source_name": source_name,
                    "source_kind": source_kind,
                }
            )
            continue
        if int(candidate.get("num_seed_points", 0)) < min_seed_points:
            report["skipped"].append({"candidate_id": candidate_id, "reason": "few_seed_points"})
            continue
        if float(candidate.get("best_existing_iou", 0.0)) > max_existing_iou:
            report["skipped"].append({"candidate_id": candidate_id, "reason": "matched_existing_3d_mask"})
            continue
        if float(candidate.get("seed_in_existing_mask_ratio", 0.0)) > max_seed_in_existing_mask_ratio:
            report["skipped"].append({"candidate_id": candidate_id, "reason": "mostly_covered_by_existing_masks"})
            continue
        if int(candidate.get("support_view_count", 0)) < min_support_views:
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_multiview_support",
                    "support_view_count": int(candidate.get("support_view_count", 0)),
                }
            )
            continue
        if float(candidate.get("support_mean_iou", 0.0)) < min_support_mean_iou:
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_multiview_consistency",
                    "support_mean_iou": float(candidate.get("support_mean_iou", 0.0)),
                }
            )
            continue
        if float(candidate.get("support_best_iou", 0.0)) < min_support_best_iou:
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "weak_best_view_alignment",
                    "support_best_iou": float(candidate.get("support_best_iou", 0.0)),
                }
            )
            continue
        if float(candidate.get("fusion_score", score)) < min_fusion_score:
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_fusion_score",
                    "fusion_score": float(candidate.get("fusion_score", score)),
                }
            )
            continue

        if _is_mask_graph_source(source_kind):
            cluster_observations = int(
                candidate.get(
                    "cluster_observation_count",
                    candidate.get("merged_observations", candidate.get("support_view_count", 0)),
                )
                or 0
            )
            selected_views = int(
                candidate.get(
                    "selected_view_count",
                    candidate.get("selected_seed_view_count", candidate.get("support_view_count", 0)),
                )
                or 0
            )
            same_object_edges = int(candidate.get("same_object_edge_count", candidate.get("graph_edge_count", 0)) or 0)
            graph_edge_mean_score = float(candidate.get("graph_edge_mean_score", 0.0) or 0.0)
            graph_consensus_score = float(candidate.get("graph_consensus_score", 0.0) or 0.0)
            depth_consistency_score = float(candidate.get("depth_consistency_score", 0.0) or 0.0)
            conflict_edges = int(candidate.get("conflict_edge_count", 0) or 0)
            conflict_ratio = float(conflict_edges / max(1, same_object_edges))

            if cluster_observations < int(mask_graph_min_cluster_observations or 0):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "low_mask_graph_observations",
                        "cluster_observation_count": cluster_observations,
                        "min_cluster_observations": int(mask_graph_min_cluster_observations or 0),
                        "source_name": source_name,
                        "source_kind": source_kind,
                    }
                )
                continue
            if selected_views < int(mask_graph_min_selected_views or 0):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "low_mask_graph_selected_views",
                        "selected_view_count": selected_views,
                        "min_selected_views": int(mask_graph_min_selected_views or 0),
                        "source_name": source_name,
                        "source_kind": source_kind,
                    }
                )
                continue
            if same_object_edges < int(mask_graph_min_same_object_edges or 0):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "low_mask_graph_same_object_edges",
                        "same_object_edge_count": same_object_edges,
                        "min_same_object_edges": int(mask_graph_min_same_object_edges or 0),
                        "source_name": source_name,
                        "source_kind": source_kind,
                    }
                )
                continue
            if graph_edge_mean_score < float(mask_graph_min_edge_mean_score or 0.0):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "low_mask_graph_edge_score",
                        "graph_edge_mean_score": graph_edge_mean_score,
                        "min_edge_mean_score": float(mask_graph_min_edge_mean_score or 0.0),
                        "source_name": source_name,
                        "source_kind": source_kind,
                    }
                )
                continue
            if graph_consensus_score < float(mask_graph_min_consensus_score or 0.0):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "low_mask_graph_consensus",
                        "graph_consensus_score": graph_consensus_score,
                        "min_consensus_score": float(mask_graph_min_consensus_score or 0.0),
                        "source_name": source_name,
                        "source_kind": source_kind,
                    }
                )
                continue
            if depth_consistency_score < float(mask_graph_min_depth_consistency or 0.0):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "low_mask_graph_depth_consistency",
                        "depth_consistency_score": depth_consistency_score,
                        "min_depth_consistency": float(mask_graph_min_depth_consistency or 0.0),
                        "source_name": source_name,
                        "source_kind": source_kind,
                    }
                )
                continue
            if mask_graph_max_conflict_edges is not None and conflict_edges > int(mask_graph_max_conflict_edges):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "high_mask_graph_conflict_edges",
                        "conflict_edge_count": conflict_edges,
                        "max_conflict_edges": int(mask_graph_max_conflict_edges),
                        "source_name": source_name,
                        "source_kind": source_kind,
                    }
                )
                continue
            if mask_graph_max_conflict_ratio is not None and conflict_ratio > float(mask_graph_max_conflict_ratio):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "high_mask_graph_conflict_ratio",
                        "conflict_ratio": conflict_ratio,
                        "conflict_edge_count": conflict_edges,
                        "same_object_edge_count": same_object_edges,
                        "max_conflict_ratio": float(mask_graph_max_conflict_ratio),
                        "source_name": source_name,
                        "source_kind": source_kind,
                    }
                )
                continue

        if max_box_area_ratio is not None and float(candidate.get("box_area_ratio", 0.0)) > max_box_area_ratio:
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "large_2d_box",
                    "box_area_ratio": float(candidate.get("box_area_ratio", 0.0)),
                }
            )
            continue
        seed_indices = _load_seed_indices(candidate, num_points)
        if seed_indices is None or len(seed_indices) < min_seed_points:
            report["skipped"].append({"candidate_id": candidate_id, "reason": "missing_or_small_seed_file"})
            continue

        if label_consensus_context is not None and "label_consensus_score" not in candidate:
            consensus = _projected_label_consensus_metrics(
                candidate,
                seed_indices,
                label_consensus_context.get("projections"),
                label_consensus_context.get("point_visibility"),
                label_consensus_context.get("preds_2d"),
                label_consensus_context.get("color_paths"),
                label_consensus_context.get("scaling_params"),
                iou_threshold=label_consensus_iou_threshold,
                min_visible_points=label_consensus_min_visible_points,
                frame_mode=label_consensus_frame_mode,
            )
            if consensus is not None:
                candidate.update(consensus)

        if (
            "label_consensus_score" in candidate
            and float(candidate.get("label_consensus_score", 1.0)) < float(min_label_consensus_score)
        ):
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_label_consensus",
                    "label_consensus_score": float(candidate.get("label_consensus_score", 1.0)),
                    "label_conflict_score": float(candidate.get("label_conflict_score", 0.0)),
                }
            )
            continue
        if (
            "label_conflict_score" in candidate
            and float(candidate.get("label_conflict_score", 0.0)) > float(max_label_conflict_score)
        ):
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "high_label_conflict",
                    "label_consensus_score": float(candidate.get("label_consensus_score", 1.0)),
                    "label_conflict_score": float(candidate.get("label_conflict_score", 0.0)),
                }
            )
            continue

        quality_score = _candidate_quality_score(candidate)
        if quality_score < min_quality_score:
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_candidate_quality_score",
                    "quality_score": quality_score,
                }
            )
            continue
        scene_source_quality_z = float(candidate.get("_quality_scene_source_z", 0.0))
        if min_scene_source_quality_z is not None and scene_source_quality_z < float(min_scene_source_quality_z):
            report["skipped"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "low_scene_source_quality_z",
                    "scene_source_quality_z": scene_source_quality_z,
                    "min_scene_source_quality_z": float(min_scene_source_quality_z),
                    "quality_score": quality_score,
                    "source_name": source_name,
                    "source_kind": source_kind,
                }
            )
            continue

        proposal_mask, growth_ratio = _grow_seed_mask(
            seed_indices,
            points_xyz,
            num_points,
            grow_radius,
            max_growth_ratio,
            max_grow_points,
        )
        if int(proposal_mask.sum()) < min_seed_points:
            report["skipped"].append({"candidate_id": candidate_id, "reason": "small_grown_mask"})
            continue

        seed_cc_info = {"enabled": False}
        if seed_cc_cleanup:
            proposal_mask, seed_cc_info = _connected_component_cleanup(
                proposal_mask,
                points_xyz,
                seed_cc_radius,
                seed_cc_min_component_points,
                seed_cc_keep_topk,
                seed_cc_max_points,
            )
            if seed_cc_info.get("enabled") and float(seed_cc_min_keep_ratio or 0.0) > 0.0:
                input_points = int(seed_cc_info.get("input_points", 0) or 0)
                output_points = int(seed_cc_info.get("output_points", int(proposal_mask.sum())) or 0)
                keep_ratio = float(output_points / max(1, input_points))
                if keep_ratio < float(seed_cc_min_keep_ratio):
                    report["skipped"].append(
                        {
                            "candidate_id": candidate_id,
                            "reason": "low_seed_connected_component_keep_ratio",
                            "seed_cc_keep_ratio": keep_ratio,
                            "seed_cc_min_keep_ratio": float(seed_cc_min_keep_ratio),
                            "seed_cc_cleanup": seed_cc_info,
                        }
                    )
                    continue
            if int(proposal_mask.sum()) < min_seed_points:
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "small_seed_connected_component_mask",
                        "seed_cc_cleanup": seed_cc_info,
                    }
                )
                continue

        seed_reference_mask = proposal_mask.copy()

        view_siou_info = {"enabled": False}
        if superpoint_min_view_siou is not None and float(superpoint_min_view_siou) > 0.0:
            view_siou_info = _superpoint_view_siou_metrics(
                proposal_mask,
                point_segments,
                point_visibility,
                candidate.get("support_views"),
                min_visible_points=superpoint_view_siou_min_visible_points,
            )
            if int(view_siou_info.get("usable_view_count", 0)) >= int(superpoint_view_siou_min_views):
                if float(view_siou_info.get("mean_pairwise_siou", 1.0)) < float(superpoint_min_view_siou):
                    report["skipped"].append(
                        {
                            "candidate_id": candidate_id,
                            "reason": "low_superpoint_view_siou",
                            "superpoint_view_siou": view_siou_info,
                        }
                    )
                    continue

        superpoint_info = {"enabled": False}
        if superpoint_refine:
            superpoint_box_context = superpoint_box_context or {}
            proposal_mask, superpoint_info = _refine_mask_with_superpoints(
                proposal_mask,
                point_segments,
                point_visibility=point_visibility,
                support_views=candidate.get("support_views"),
                projections=superpoint_box_context.get("projections"),
                scaling_params=superpoint_box_context.get("scaling_params"),
                min_coverage=superpoint_min_coverage,
                max_expansion_ratio=superpoint_max_expansion_ratio,
                max_segment_ratio=superpoint_max_segment_ratio,
                large_segment_min_coverage=superpoint_large_segment_min_coverage,
                min_seed_retention=superpoint_min_seed_retention,
                min_support_views=superpoint_min_support_views,
                min_support_ratio=superpoint_min_support_ratio,
                min_box_positive_ratio=superpoint_min_box_positive_ratio,
                max_box_negative_ratio=superpoint_max_box_negative_ratio,
                box_support_min_visible_points=superpoint_box_min_visible_points,
                box_support_min_views=superpoint_box_min_views,
                box_support_padding_ratio=superpoint_box_padding_ratio,
                min_output_points=min_seed_points,
            )
            if int(proposal_mask.sum()) < min_seed_points:
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "small_superpoint_refined_mask",
                        "superpoint_refine": superpoint_info,
                    }
                )
                continue

        local_superpoint_info = {"enabled": False}
        if local_superpoint_refine:
            proposal_mask, local_superpoint_info = _refine_mask_with_local_superpoints(
                proposal_mask,
                seed_reference_mask,
                points_xyz,
                knn=local_superpoint_knn,
                merge_k=local_superpoint_merge_k,
                min_size=local_superpoint_min_size,
                min_coverage=local_superpoint_min_coverage,
                max_expansion_ratio=local_superpoint_max_expansion_ratio,
                min_seed_retention=local_superpoint_min_seed_retention,
                min_output_points=min_seed_points,
                max_points=local_superpoint_max_points,
            )
            if int(proposal_mask.sum()) < min_seed_points:
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "small_local_superpoint_refined_mask",
                        "local_superpoint_refine": local_superpoint_info,
                    }
                )
                continue

        proposal_items = [(proposal_mask, {"enabled": False})]
        use_cc_cleanup = cc_cleanup and _candidate_matches_source_filter(candidate, cc_source_filter)
        if use_cc_cleanup and cc_split_components:
            proposal_items = _connected_component_split(
                proposal_mask,
                points_xyz,
                cc_radius,
                cc_min_component_points,
                cc_keep_topk,
                cc_max_points,
            )
        elif use_cc_cleanup:
            proposal_mask, cc_info = _connected_component_cleanup(
                proposal_mask,
                points_xyz,
                cc_radius,
                cc_min_component_points,
                cc_keep_topk,
                cc_max_points,
            )
            if int(proposal_mask.sum()) < min_seed_points:
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "small_connected_component_mask",
                        "cc_cleanup": cc_info,
                    }
                )
                continue
            proposal_items = [(proposal_mask, cc_info)]

        appended_from_candidate = 0
        for component_id, (proposal_mask, cc_info) in enumerate(proposal_items):
            if (
                cc_info.get("enabled")
                and not cc_info.get("split_components")
                and float(cc_min_keep_ratio or 0.0) > 0.0
            ):
                input_points = int(cc_info.get("input_points", 0) or 0)
                output_points = int(cc_info.get("output_points", int(proposal_mask.sum())) or 0)
                keep_ratio = float(output_points / max(1, input_points))
                if keep_ratio < float(cc_min_keep_ratio):
                    report["skipped"].append(
                        {
                            "candidate_id": candidate_id,
                            "component_id": int(component_id),
                            "reason": "low_connected_component_keep_ratio",
                            "cc_keep_ratio": keep_ratio,
                            "cc_min_keep_ratio": float(cc_min_keep_ratio),
                            "cc_cleanup": cc_info,
                        }
                    )
                    continue
            if source_limit is not None and source_counts[source_kind] >= int(source_limit):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "component_id": int(component_id),
                        "reason": "source_limit",
                        "source_name": source_name,
                        "source_kind": source_kind,
                        "source_limit": int(source_limit),
                    }
                )
                continue
            if class_limit is not None and class_counts[str(class_name).strip().lower()] >= int(class_limit):
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "component_id": int(component_id),
                        "reason": "class_limit",
                        "class_name": class_name,
                        "class_limit": int(class_limit),
                    }
                )
                continue
            if int(proposal_mask.sum()) < min_seed_points:
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "component_id": int(component_id),
                        "reason": "small_connected_component_mask",
                        "cc_cleanup": cc_info,
                    }
                )
                continue

            projection_info = {"enabled": False}
            if projection_consistency_context is not None:
                projection_info = _projected_box_consistency_metrics(
                    proposal_mask,
                    candidate,
                    projection_consistency_context.get("projections"),
                    projection_consistency_context.get("point_visibility"),
                    projection_consistency_context.get("preds_2d"),
                    projection_consistency_context.get("color_paths"),
                    projection_consistency_context.get("scaling_params"),
                    min_visible_points=projection_consistency_min_visible_points,
                    frame_mode=projection_consistency_frame_mode,
                    box_padding_ratio=projection_consistency_box_padding_ratio,
                    same_class_only=True,
                )
                if int(projection_info.get("usable_view_count", 0)) >= int(projection_consistency_min_views):
                    if (
                        float(projection_consistency_min_box_iou or 0.0) > 0.0
                        and float(projection_info.get("mean_box_iou", 0.0))
                        < float(projection_consistency_min_box_iou)
                    ):
                        report["skipped"].append(
                            {
                                "candidate_id": candidate_id,
                                "component_id": int(component_id),
                                "reason": "low_projected_box_iou",
                                "projected_box_consistency": projection_info,
                            }
                        )
                        continue
                    if (
                        float(projection_consistency_min_point_ratio or 0.0) > 0.0
                        and float(projection_info.get("mean_point_in_box_ratio", 0.0))
                        < float(projection_consistency_min_point_ratio)
                    ):
                        report["skipped"].append(
                            {
                                "candidate_id": candidate_id,
                                "component_id": int(component_id),
                                "reason": "low_projected_box_point_ratio",
                                "projected_box_consistency": projection_info,
                            }
                        )
                        continue

            existing_iou = _mask_iou(proposal_mask, masks_np).max(initial=0.0)
            if existing_iou > max_existing_iou:
                report["skipped"].append(
                    {
                        "candidate_id": candidate_id,
                        "component_id": int(component_id),
                        "reason": "grown_mask_matches_existing",
                        "iou": float(existing_iou),
                    }
                )
                continue

            if appended_masks:
                proposal_iou = _mask_iou(proposal_mask, np.stack(appended_masks, axis=1)).max(initial=0.0)
                if proposal_iou > max_proposal_iou:
                    report["skipped"].append(
                        {
                            "candidate_id": candidate_id,
                            "component_id": int(component_id),
                            "reason": "duplicate_new_proposal",
                            "iou": float(proposal_iou),
                        }
                    )
                    continue

            class_id = int(candidate["class_id"])
            base_score = float(candidate.get("fusion_score", score)) if use_candidate_fusion_score else score
            component_scale = 1.0 if component_id == 0 else 0.85
            cc_score_factor = 1.0
            if (
                cc_info.get("enabled")
                and not cc_info.get("split_components")
                and float(cc_keep_ratio_score_weight or 0.0) > 0.0
            ):
                input_points = int(cc_info.get("input_points", 0) or 0)
                output_points = int(cc_info.get("output_points", int(proposal_mask.sum())) or 0)
                keep_ratio = min(1.0, max(0.0, float(output_points / max(1, input_points))))
                weight = min(1.0, max(0.0, float(cc_keep_ratio_score_weight)))
                cc_score_factor = (1.0 - weight) + weight * keep_ratio
            projection_score_factor = 1.0
            if projection_info.get("enabled") and float(projection_consistency_score_weight or 0.0) > 0.0:
                projection_score = 0.5 * min(1.0, max(0.0, float(projection_info.get("mean_box_iou", 0.0)))) + 0.5 * min(
                    1.0,
                    max(0.0, float(projection_info.get("mean_point_in_box_ratio", 0.0))),
                )
                weight = min(1.0, max(0.0, float(projection_consistency_score_weight)))
                projection_score_factor = (1.0 - weight) + weight * projection_score
            hierarchy_score_factor, hierarchy_score_info = _superpoint_hierarchy_score_factor(
                proposal_mask,
                point_segments,
                low_occupancy_threshold=hierarchy_low_occupancy_threshold,
                score_weight=hierarchy_score_weight,
                min_score_factor=hierarchy_min_score_factor,
            )
            source_score_scale = float(_lookup_source_rule(candidate, source_score_scales, 1.0))
            score_calibration = _candidate_score_calibration(
                candidate,
                quality_weight=quality_calibration_weight,
                novelty_weight=novelty_calibration_weight,
                label_consensus_weight=label_consensus_calibration_weight,
                min_factor=score_calibration_min,
                max_factor=score_calibration_max,
            )
            proposal_score = max(
                0.0,
                min(
                    1.0,
                    base_score
                    * float(score_scale)
                    * source_score_scale
                    * component_scale
                    * score_calibration
                    * cc_score_factor
                    * projection_score_factor
                    * hierarchy_score_factor,
                ),
            )
            if max_proposal_score is not None:
                proposal_score = min(float(max_proposal_score), proposal_score)
            masks_np = np.concatenate([masks_np, proposal_mask[:, None]], axis=1)
            classes_np = np.concatenate([classes_np, np.asarray([class_id], dtype=np.int64)])
            scores_np = np.concatenate([scores_np, np.asarray([proposal_score], dtype=np.float32)])
            appended_masks.append(proposal_mask)
            appended_from_candidate += 1
            source_counts[source_kind] += 1
            class_counts[str(class_name).strip().lower()] += 1
            report["applied"].append(
                {
                    "candidate_id": int(candidate_id) if candidate_id is not None else None,
                    "component_id": int(component_id),
                    "class_id": class_id,
                    "class_name": candidate.get("class_name"),
                    "score": score,
                    "fusion_score": float(candidate.get("fusion_score", score)),
                    "quality_score": quality_score,
                    "quality_scene_source_z": float(candidate.get("_quality_scene_source_z", 0.0)),
                    "mask_graph_evidence_score": float(candidate.get("_mask_graph_evidence_score", 0.0)),
                    "mask_graph_evidence_count": int(candidate.get("_mask_graph_evidence_count", 0)),
                    "mask_graph_evidence_best_overlap": float(candidate.get("_mask_graph_evidence_best_overlap", 0.0)),
                    "mask_graph_evidence_best_iou": float(candidate.get("_mask_graph_evidence_best_iou", 0.0)),
                    "novelty_score": _candidate_novelty_score(candidate),
                    "label_consensus_score": float(candidate.get("label_consensus_score", 1.0)),
                    "label_conflict_score": float(candidate.get("label_conflict_score", 0.0)),
                    "label_margin": float(candidate.get("label_margin", 0.0)),
                    "score_calibration": score_calibration,
                    "cc_score_factor": cc_score_factor,
                    "projection_score_factor": projection_score_factor,
                    "hierarchy_score_factor": hierarchy_score_factor,
                    "support_view_count": int(candidate.get("support_view_count", 0)),
                    "support_mean_iou": float(candidate.get("support_mean_iou", 0.0)),
                    "support_best_iou": float(candidate.get("support_best_iou", 0.0)),
                    "cluster_observation_count": int(
                        candidate.get(
                            "cluster_observation_count",
                            candidate.get("merged_observations", candidate.get("support_view_count", 0)),
                        )
                        or 0
                    ),
                    "selected_view_count": int(
                        candidate.get(
                            "selected_view_count",
                            candidate.get("selected_seed_view_count", candidate.get("support_view_count", 0)),
                        )
                        or 0
                    ),
                    "graph_edge_count": int(candidate.get("graph_edge_count", 0) or 0),
                    "same_object_edge_count": int(candidate.get("same_object_edge_count", 0) or 0),
                    "weak_edge_count": int(candidate.get("weak_edge_count", 0) or 0),
                    "conflict_edge_count": int(candidate.get("conflict_edge_count", 0) or 0),
                    "graph_edge_mean_score": float(candidate.get("graph_edge_mean_score", 0.0) or 0.0),
                    "graph_consensus_score": float(candidate.get("graph_consensus_score", 0.0) or 0.0),
                    "depth_consistency_score": float(candidate.get("depth_consistency_score", 0.0) or 0.0),
                    "seed_in_existing_mask_ratio": float(candidate.get("seed_in_existing_mask_ratio", 0.0)),
                    "best_existing_iou": float(candidate.get("best_existing_iou", 0.0)),
                    "box_area_ratio": float(candidate.get("box_area_ratio", 0.0)),
                    "verification": verification,
                    "proposal_score": proposal_score,
                    "source_score_scale": source_score_scale,
                    "num_seed_points": int(len(seed_indices)),
                    "num_mask_points": int(proposal_mask.sum()),
                    "growth_ratio": growth_ratio,
                    "seed_cc_cleanup": seed_cc_info,
                    "superpoint_refine": superpoint_info,
                    "local_superpoint_refine": local_superpoint_info,
                    "superpoint_view_siou": view_siou_info,
                    "hierarchy_score": hierarchy_score_info,
                    "cc_cleanup": cc_info,
                    "projected_box_consistency": projection_info,
                    "source_json": candidate.get("_source_json"),
                    "source_name": source_name,
                    "source_kind": source_kind,
                }
            )

            if max_candidates is not None and len(report["applied"]) >= max_candidates:
                break

        if max_candidates is not None and len(report["applied"]) >= max_candidates:
            break

    masks_np, classes_np, scores_np, postprocess_summary = _postprocess_appended_proposals(
        masks_np,
        classes_np,
        scores_np,
        original_num_masks,
        merge_iou=merge_iou,
        inclusion_threshold=inclusion_threshold,
        same_class_only=postprocess_same_class_only,
        appended_metadata=report["applied"],
        point_segments=point_segments,
        containment_action=containment_action,
        containment_threshold=containment_threshold,
        containment_min_area_ratio=containment_min_area_ratio,
        containment_score_ratio=containment_score_ratio,
        containment_quality_margin=containment_quality_margin,
        containment_score_factor=containment_score_factor,
        containment_min_points=containment_min_points,
        hierarchy_substitution_action=hierarchy_substitution_action,
        hierarchy_substitution_min_child_coverage=hierarchy_substitution_min_child_coverage,
        hierarchy_substitution_max_parent_exclusive_ratio=hierarchy_substitution_max_parent_exclusive_ratio,
        hierarchy_substitution_min_area_ratio=hierarchy_substitution_min_area_ratio,
        hierarchy_substitution_min_children=hierarchy_substitution_min_children,
    )
    report["postprocess"] = postprocess_summary
    return masks_np, classes_np, scores_np, report
