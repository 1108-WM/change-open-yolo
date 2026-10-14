"""Shared instance matching/AP formulas for Full training and evaluation.

GT helpers belong to supervision or post-prediction evaluation; inference
callers do not invoke them. No oracle selection CLI is retained here.
"""
import numpy as np
from evaluate.scannet200 import eval_semantic_instance as instance_eval


UNIFIED_PREDICTED_CLASS = 0

UNIFIED_GT_CLASS = int(instance_eval.PRED_ID_TO_ID[UNIFIED_PREDICTED_CLASS])

VALID_GT_CLASSES = frozenset(
    int(class_id) for class_id in instance_eval.PRED_ID_TO_ID.values() if int(class_id) >= 0
)

def _class_agnostic_gt_ids(ids):
    """保留实例边界，移除语义标签；无效 GT 继续作为 evaluator 的 void。"""
    ids = np.asarray(ids, dtype=np.int64)
    mapped = np.zeros_like(ids)
    valid_instance_ids = [
        int(instance_id)
        for instance_id in np.unique(ids)
        if instance_id > 0 and int(instance_id) // 1000 in VALID_GT_CLASSES
    ]
    if len(valid_instance_ids) >= 1000:
        raise ValueError("单场景有效 GT 实例超过类别无关编码上限")
    for new_instance_index, instance_id in enumerate(valid_instance_ids, start=1):
        mapped[ids == instance_id] = UNIFIED_GT_CLASS * 1000 + new_instance_index
    return mapped

def _configure_scannet200_instance_eval():
    valid_ids = tuple(
        int(instance_eval.PRED_ID_TO_ID[index])
        for index in range(len(instance_eval.VALID_CLASS_IDS_200_INST))
    )
    all_labels = {
        int(class_id): label
        for class_id, label in zip(
            instance_eval.VALID_CLASS_IDS_200,
            instance_eval.CLASS_LABELS_200,
        )
    }
    labels = tuple(all_labels[class_id] for class_id in valid_ids)
    instance_eval.DATASET_NAME = "scannet200"
    instance_eval.VALID_CLASS_IDS = np.asarray(valid_ids, dtype=np.int64)
    instance_eval.CLASS_LABELS = labels
    instance_eval.ID_TO_LABEL = dict(zip(valid_ids, labels))
    instance_eval.LABEL_TO_ID = dict(zip(labels, valid_ids))
    instance_eval.HEAD_CATS_SCANNET_200 = set(instance_eval.HEAD_CATS_SCANNET_200)
    instance_eval.COMMON_CATS_SCANNET_200 = set(instance_eval.COMMON_CATS_SCANNET_200)
    instance_eval.TAIL_CATS_SCANNET_200 = set(instance_eval.TAIL_CATS_SCANNET_200)

def _merge_scan_matches(native_gt, native_pred, track_gt, track_pred):
    if set(native_gt) != set(track_gt) or set(native_pred) != set(track_pred):
        raise ValueError("native and track match label domains differ")
    gt_identity_keys = (
        "instance_id", "label_id", "vert_count", "med_dist", "dist_conf"
    )
    for label_name in native_gt:
        native_rows = native_gt[label_name]
        track_rows = track_gt[label_name]
        if len(native_rows) != len(track_rows):
            raise ValueError(f"GT instance count differs for {label_name}")
        for native_row, track_row in zip(native_rows, track_rows):
            if any(native_row[key] != track_row[key] for key in gt_identity_keys):
                raise ValueError(f"GT instance identity differs for {label_name}")
            native_row["matched_pred"].extend(track_row["matched_pred"])
        native_pred[label_name].extend(track_pred[label_name])
    return native_gt, native_pred

def _scene_ap_records(gt_by_label, pred_by_label, overlap_th):
    """The evaluator's chair-only matching, reduced to sufficient AP records."""
    label_name = "chair"
    gt_instances = [
        gt for gt in gt_by_label[label_name]
        if gt["instance_id"] >= 1000 and gt["vert_count"] >= 100
    ]
    pred_instances = pred_by_label[label_name]
    pred_visited = {pred["uuid"]: False for pred in pred_instances}
    true = np.ones(len(gt_instances), dtype=np.float64)
    score = np.full(len(gt_instances), -float("inf"), dtype=np.float64)
    matched = np.zeros(len(gt_instances), dtype=bool)
    hard_false_negatives = 0
    for index, gt in enumerate(gt_instances):
        found_match = False
        for pred in gt["matched_pred"]:
            if pred_visited[pred["uuid"]]:
                continue
            overlap = float(pred["intersection"]) / (
                gt["vert_count"] + pred["vert_count"] - pred["intersection"]
            )
            if overlap > overlap_th:
                confidence = pred["confidence"]
                if matched[index]:
                    maximum = max(score[index], confidence)
                    minimum = min(score[index], confidence)
                    score[index] = maximum
                    true = np.append(true, 0.0)
                    score = np.append(score, minimum)
                    matched = np.append(matched, True)
                else:
                    found_match = True
                    matched[index] = True
                    score[index] = confidence
                    pred_visited[pred["uuid"]] = True
        if not found_match:
            hard_false_negatives += 1
    true = true[matched]
    score = score[matched]
    for pred in pred_instances:
        found_gt = False
        for gt in pred["matched_gt"]:
            overlap = float(gt["intersection"]) / (
                gt["vert_count"] + pred["vert_count"] - gt["intersection"]
            )
            if overlap > overlap_th:
                found_gt = True
                break
        if not found_gt:
            ignored = pred["void_intersection"]
            for gt in pred["matched_gt"]:
                if gt["instance_id"] < 1000 or gt["vert_count"] < 100:
                    ignored += gt["intersection"]
            if float(ignored) / pred["vert_count"] <= overlap_th:
                true = np.append(true, 0.0)
                score = np.append(score, pred["confidence"])
    return true, score, hard_false_negatives, bool(gt_instances), bool(pred_instances)

def _average_precision(true_parts, score_parts, hard_false_negatives, has_gt, has_pred):
    if not has_gt:
        return float("nan")
    if not has_pred:
        return 0.0
    y_true = np.concatenate(true_parts) if true_parts else np.empty(0)
    y_score = np.concatenate(score_parts) if score_parts else np.empty(0)
    if not len(y_score):
        return 0.0
    order = np.argsort(y_score)
    scores = y_score[order]
    truth = y_true[order]
    cumulative = np.cumsum(truth)
    _, unique_indices = np.unique(scores, return_index=True)
    num_points = len(unique_indices) + 1
    num_true_examples = cumulative[-1] if len(cumulative) else 0.0
    cumulative = np.append(cumulative, 0)
    precision = np.zeros(num_points)
    recall = np.zeros(num_points)
    for index, score_index in enumerate(unique_indices):
        true_positive = num_true_examples - cumulative[score_index - 1]
        false_positive = len(scores) - score_index - true_positive
        false_negative = cumulative[score_index - 1] + hard_false_negatives
        precision[index] = float(true_positive) / (true_positive + false_positive)
        recall[index] = float(true_positive) / (true_positive + false_negative)
    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_conv = np.append(recall[0], recall)
    recall_for_conv = np.append(recall_for_conv, 0.0)
    return float(np.dot(precision, np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")))

def _empty_record():
    return {"true": [], "score": [], "fn": 0, "has_gt": False, "has_pred": False}

def _append(record, values):
    true, score, fn, has_gt, has_pred = values
    record["true"].append(true)
    record["score"].append(score)
    record["fn"] += int(fn)
    record["has_gt"] = record["has_gt"] or bool(has_gt)
    record["has_pred"] = record["has_pred"] or bool(has_pred)

AP25=.25

OFFICIAL=tuple(sorted(round(float(x),2) for x in instance_eval.opt['overlaps'] if float(x)>=.5))

DIAGNOSTIC=(AP25,*OFFICIAL,.95)

def _record_from_matches(gt, pred, t):
    return _scene_ap_records(gt, pred, t)
