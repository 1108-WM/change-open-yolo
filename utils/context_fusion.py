import json
import os
import os.path as osp
import re
from collections import defaultdict

import numpy as np
import torch


_CLASS_FIELD_CANDIDATES = (
    "corrected_class_name",
    "corrected_label",
    "class_name",
    "label",
    "mllm_class_name",
    "vlm_class_name",
    "object_name",
    "answer",
)
_CLASS_ID_FIELD_CANDIDATES = (
    "corrected_class_id",
    "class_id",
    "label_id",
    "mllm_class_id",
    "vlm_class_id",
)
_CONFIDENCE_FIELD_CANDIDATES = (
    "confidence",
    "corrected_score",
    "score",
    "mllm_confidence",
    "vlm_confidence",
)
_NESTED_RESULT_FIELDS = (
    "correction",
    "semantic_correction",
    "mllm_result",
    "vlm_result",
    "result",
)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _normalize_label(value):
    value = str(value).lower().strip()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _label_lookup(labels):
    lookup = {}
    for class_id, label in enumerate(labels):
        lookup[_normalize_label(label)] = int(class_id)
    return lookup


def _read_json_or_jsonl(path):
    with open(path) as f:
        text = f.read().strip()
    if not text:
        return None

    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _iter_json_paths(path):
    if osp.isfile(path):
        yield path
        return

    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename.endswith((".json", ".jsonl")):
                yield osp.join(root, filename)


def _as_records(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    for key in ("corrections", "results", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]

    if isinstance(payload.get("candidates"), list):
        scene_name = payload.get("scene_name")
        records = []
        for candidate in payload["candidates"]:
            record = dict(candidate)
            if scene_name is not None:
                record.setdefault("scene_name", scene_name)
            records.append(record)
        return records

    return [payload]


def _find_first(record, keys):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _merged_result(record):
    merged = dict(record)
    for field in _NESTED_RESULT_FIELDS:
        nested = record.get(field)
        if isinstance(nested, dict):
            merged.update(nested)
    return merged


def _parse_confidence(value):
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(confidence):
        return None
    return confidence


def _parse_class_filter(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parsed = {str(item).strip() for item in value if str(item).strip()}
    else:
        parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def _parse_class_id(record, labels, lookup):
    class_id = _find_first(record, _CLASS_ID_FIELD_CANDIDATES)
    if class_id is not None:
        try:
            class_id = int(class_id)
        except (TypeError, ValueError):
            class_id = None
        if class_id is not None and 0 <= class_id < len(labels):
            return class_id, labels[class_id]

    class_name = _find_first(record, _CLASS_FIELD_CANDIDATES)
    if class_name is None:
        return None, None

    normalized = _normalize_label(class_name)
    class_id = lookup.get(normalized)
    if class_id is None:
        padded = f" {normalized} "
        for label, candidate_id in sorted(lookup.items(), key=lambda item: len(item[0]), reverse=True):
            if f" {label} " in padded:
                return candidate_id, labels[candidate_id]
        return None, str(class_name).strip()
    return class_id, labels[class_id]


def load_context_corrections(path, labels, min_confidence=0.0, strict=False):
    """Load offline MLLM/VLM corrections keyed by scene and prediction mask id.

    Supported inputs:
    - a JSON/JSONL list of records with scene_name, mask_id, corrected_class_name
    - a dict with corrections/results/items
    - context_candidates.json augmented with mllm_result/vlm_result/correction fields
    - a directory containing any of the above files
    """

    if path is None:
        return {}, {"loaded": 0, "used": 0, "skipped": 0, "files": []}
    if not osp.exists(path):
        raise FileNotFoundError(f"Context correction path does not exist: {path}")

    lookup = _label_lookup(labels)
    corrections = defaultdict(dict)
    summary = {"loaded": 0, "used": 0, "skipped": 0, "files": []}

    for json_path in _iter_json_paths(path):
        summary["files"].append(json_path)
        payload = _read_json_or_jsonl(json_path)
        for raw_record in _as_records(payload):
            if not isinstance(raw_record, dict):
                summary["skipped"] += 1
                continue

            record = _merged_result(raw_record)
            summary["loaded"] += 1
            scene_name = record.get("scene_name")
            mask_id = record.get("mask_id")
            if scene_name is None or mask_id is None:
                summary["skipped"] += 1
                if strict:
                    raise ValueError(f"Missing scene_name/mask_id in correction record: {raw_record}")
                continue

            confidence = _parse_confidence(_find_first(record, _CONFIDENCE_FIELD_CANDIDATES))
            if confidence is not None and confidence < min_confidence:
                summary["skipped"] += 1
                continue

            class_id, class_name = _parse_class_id(record, labels, lookup)
            if class_id is None:
                summary["skipped"] += 1
                if strict:
                    raise ValueError(f"Unknown correction label {class_name!r} in record: {raw_record}")
                continue

            try:
                mask_id = int(mask_id)
            except (TypeError, ValueError):
                summary["skipped"] += 1
                if strict:
                    raise ValueError(f"Invalid mask_id in correction record: {raw_record}")
                continue

            corrections[str(scene_name)][mask_id] = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "decision": str(record.get("decision", "")).strip().lower(),
                "source_path": json_path,
            }
            summary["used"] += 1

    return {scene: dict(items) for scene, items in corrections.items()}, summary


def apply_context_corrections(
    scene_name,
    pred_classes,
    pred_scores,
    corrections,
    score_policy="keep",
    score_blend_alpha=0.5,
    apply_decisions=None,
    apply_min_confidence=None,
    apply_min_score=None,
    apply_max_score=None,
    bad_mask_policy="skip",
    bad_mask_score=0.0,
    score_boost=0.0,
    allowed_classes=None,
    blocked_classes=None,
):
    """Apply loaded semantic corrections to one scene prediction."""

    pred_classes = _to_numpy(pred_classes).copy()
    pred_scores = _to_numpy(pred_scores).copy()
    scene_corrections = corrections.get(scene_name, {})
    if apply_decisions is None:
        apply_decisions = {"change", "keep"}
    else:
        apply_decisions = {str(item).strip().lower() for item in apply_decisions}
    allowed_classes = _parse_class_filter(allowed_classes)
    blocked_classes = _parse_class_filter(blocked_classes)
    applied = []
    skipped = []

    for mask_id, correction in scene_corrections.items():
        if mask_id < 0 or mask_id >= len(pred_classes):
            skipped.append({"mask_id": int(mask_id), "reason": "mask_id_out_of_range"})
            continue
        old_class = int(pred_classes[mask_id])
        old_score = float(pred_scores[mask_id])
        decision = correction.get("decision")
        confidence = correction.get("confidence")

        if decision == "bad_mask":
            if bad_mask_policy == "suppress":
                new_score = min(old_score, float(bad_mask_score))
                pred_scores[mask_id] = new_score
                applied.append(
                    {
                        "mask_id": int(mask_id),
                        "old_class_id": old_class,
                        "new_class_id": old_class,
                        "new_class_name": correction["class_name"],
                        "old_score": old_score,
                        "new_score": float(new_score),
                        "confidence": confidence,
                        "decision": decision,
                        "action": "suppress",
                    }
                )
            else:
                skipped.append({"mask_id": int(mask_id), "reason": "bad_mask_decision"})
            continue
        if decision == "unknown":
            skipped.append({"mask_id": int(mask_id), "reason": "unknown_decision"})
            continue
        if decision not in apply_decisions:
            skipped.append({"mask_id": int(mask_id), "reason": f"{decision}_not_enabled"})
            continue
        new_class_name = correction["class_name"]
        if allowed_classes is not None and new_class_name not in allowed_classes:
            skipped.append(
                {
                    "mask_id": int(mask_id),
                    "reason": "class_not_allowed",
                    "new_class_name": new_class_name,
                }
            )
            continue
        if blocked_classes is not None and new_class_name in blocked_classes:
            skipped.append(
                {
                    "mask_id": int(mask_id),
                    "reason": "class_blocked",
                    "new_class_name": new_class_name,
                }
            )
            continue
        if apply_min_confidence is not None:
            if confidence is None or confidence < float(apply_min_confidence):
                skipped.append(
                    {
                        "mask_id": int(mask_id),
                        "reason": "low_apply_confidence",
                        "confidence": confidence,
                    }
                )
                continue
        if apply_min_score is not None and old_score < float(apply_min_score):
            skipped.append(
                {
                    "mask_id": int(mask_id),
                    "reason": "below_apply_score_range",
                    "old_score": old_score,
                }
            )
            continue
        if apply_max_score is not None and old_score > float(apply_max_score):
            skipped.append(
                {
                    "mask_id": int(mask_id),
                    "reason": "above_apply_score_range",
                    "old_score": old_score,
                }
            )
            continue

        new_score = old_score

        if confidence is not None:
            if score_policy == "replace":
                new_score = float(confidence)
            elif score_policy == "max":
                new_score = max(old_score, float(confidence))
            elif score_policy == "blend":
                alpha = float(score_blend_alpha)
                new_score = alpha * old_score + (1.0 - alpha) * float(confidence)
            elif score_policy == "boost":
                new_score = max(old_score, min(float(confidence), float(score_boost)))
            elif score_policy != "keep":
                raise ValueError(f"Unsupported correction score policy: {score_policy}")

        pred_classes[mask_id] = int(correction["class_id"])
        pred_scores[mask_id] = new_score
        applied.append(
            {
                "mask_id": int(mask_id),
                "old_class_id": old_class,
                "new_class_id": int(correction["class_id"]),
                "new_class_name": correction["class_name"],
                "old_score": old_score,
                "new_score": float(new_score),
                "confidence": confidence,
                "decision": decision,
                "action": "apply",
            }
        )

    return pred_classes, pred_scores, {"applied": applied, "skipped": skipped}
