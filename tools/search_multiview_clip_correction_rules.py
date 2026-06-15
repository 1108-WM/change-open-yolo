import argparse
import csv
import glob
import json
import os
import os.path as osp
from collections import Counter, defaultdict

import numpy as np


def _read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_multiview_features(path):
    features = {}
    labels = None
    for json_path in glob.glob(osp.join(path, "*", "multiview_object_clip_features.json")):
        with open(json_path) as f:
            payload = json.load(f)
        labels = labels or payload.get("labels")
        for record in payload.get("features", []):
            features[(record["scene_name"], int(record["prediction_id"]))] = record
    return features, labels or []


def _parse_float_grid(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _split_records(records, split_name):
    if split_name == "all":
        return records
    if split_name == "val":
        return [record for record in records if str(record["scene_name"]).startswith("room")]
    if split_name == "train":
        return [record for record in records if str(record["scene_name"]).startswith("office")]
    raise ValueError(f"Unknown split: {split_name}")


def _top2(probs):
    order = np.argsort(-probs)
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else -1
    second_score = float(probs[second]) if second >= 0 else 0.0
    return top, float(probs[top]), second, second_score


def _candidate_events(records, feature_map, min_iou):
    events = []
    for record in records:
        if float(record.get("best_iou", 0.0)) < min_iou:
            continue
        gt_class = int(record.get("gt_pred_class_id", -1))
        if gt_class < 0:
            continue
        feature = feature_map.get((record["scene_name"], int(record["prediction_id"])))
        if not feature or not feature.get("clip_probs"):
            continue
        probs = np.asarray(feature["clip_probs"], dtype=np.float32)
        current = int(record.get("pred_class_id", -1))
        top, top_score, second, second_score = _top2(probs)
        current_score = float(probs[current]) if 0 <= current < len(probs) else 0.0
        events.append(
            {
                "scene_name": record["scene_name"],
                "prediction_id": int(record["prediction_id"]),
                "source_kind": record.get("source_kind"),
                "current": current,
                "top": top,
                "gt": gt_class,
                "base_score": float(record.get("features", {}).get("base_score", 0.0)),
                "top_score": top_score,
                "second_score": second_score,
                "margin": top_score - second_score,
                "gain": top_score - current_score,
                "base_correct": current == gt_class,
                "top_correct": top == gt_class,
            }
        )
    return events


def _score_rule(events, old_class, new_class, confidence, margin, gain, max_base_score):
    applied = []
    for event in events:
        if event["current"] != old_class or event["top"] != new_class:
            continue
        if event["top_score"] < confidence:
            continue
        if event["margin"] < margin:
            continue
        if event["gain"] < gain:
            continue
        if event["base_score"] > max_base_score:
            continue
        applied.append(event)

    helpful = sum((not event["base_correct"]) and event["top_correct"] for event in applied)
    harmful = sum(event["base_correct"] and (not event["top_correct"]) for event in applied)
    neutral = len(applied) - helpful - harmful
    base_correct = sum(event["base_correct"] for event in events)
    corrected = base_correct + helpful - harmful
    return {
        "applied": len(applied),
        "helpful": helpful,
        "harmful": harmful,
        "neutral": neutral,
        "base_accuracy": base_correct / max(1, len(events)),
        "corrected_accuracy": corrected / max(1, len(events)),
        "accuracy_delta": (helpful - harmful) / max(1, len(events)),
        "precision": helpful / max(1, helpful + harmful),
    }


def search_rules(args):
    records = _read_jsonl(args.dataset_jsonl)
    feature_map, labels = _load_multiview_features(args.multiview_clip_features)
    records = _split_records(records, args.split)
    events = _candidate_events(records, feature_map, args.min_iou)
    if not events:
        raise RuntimeError("No matched events found. Check dataset/features paths.")

    pair_counts = Counter((event["current"], event["top"]) for event in events if event["current"] != event["top"])
    rows = []
    for old_class, new_class in sorted(pair_counts):
        if pair_counts[(old_class, new_class)] < args.min_pair_count:
            continue
        for confidence in args.confidences:
            for margin in args.margins:
                for gain in args.gains:
                    for max_base_score in args.max_base_scores:
                        metrics = _score_rule(events, old_class, new_class, confidence, margin, gain, max_base_score)
                        if metrics["applied"] < args.min_applied:
                            continue
                        if metrics["helpful"] < args.min_helpful:
                            continue
                        if metrics["harmful"] > args.max_harmful:
                            continue
                        rows.append(
                            {
                                "old_class_id": old_class,
                                "old_class_name": labels[old_class] if 0 <= old_class < len(labels) else str(old_class),
                                "new_class_id": new_class,
                                "new_class_name": labels[new_class] if 0 <= new_class < len(labels) else str(new_class),
                                "confidence": confidence,
                                "margin": margin,
                                "gain": gain,
                                "max_base_score": max_base_score,
                                **metrics,
                            }
                        )

    rows.sort(
        key=lambda row: (
            -row["accuracy_delta"],
            -row["precision"],
            -row["helpful"],
            row["harmful"],
            row["applied"],
        )
    )

    output_dir = osp.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "old_class_id",
        "old_class_name",
        "new_class_id",
        "new_class_name",
        "confidence",
        "margin",
        "gain",
        "max_base_score",
        "applied",
        "helpful",
        "harmful",
        "neutral",
        "precision",
        "base_accuracy",
        "corrected_accuracy",
        "accuracy_delta",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_jsonl": args.dataset_jsonl,
        "multiview_clip_features": args.multiview_clip_features,
        "split": args.split,
        "min_iou": args.min_iou,
        "num_records": len(records),
        "num_events": len(events),
        "base_accuracy": sum(event["base_correct"] for event in events) / max(1, len(events)),
        "clip_top1_accuracy": sum(event["top_correct"] for event in events) / max(1, len(events)),
        "num_rules": len(rows),
        "top_rules": rows[:20],
    }
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Events: {len(events)} base_acc={summary['base_accuracy']:.3f} clip_acc={summary['clip_top1_accuracy']:.3f}")
    print(f"Saved {len(rows)} candidate rules to {args.output_csv}")
    for row in rows[: min(10, len(rows))]:
        print(
            f"{row['old_class_name']} -> {row['new_class_name']} "
            f"conf={row['confidence']} margin={row['margin']} gain={row['gain']} maxbase={row['max_base_score']} "
            f"applied={row['applied']} helpful={row['helpful']} harmful={row['harmful']} "
            f"delta={row['accuracy_delta']:.3f}"
        )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", default="./output/semantic_fusion_dataset_replica_s5_m30_bpr/features.jsonl")
    parser.add_argument("--multiview_clip_features", default="./output/multiview_object_clip_replica_s5_m30_bpr")
    parser.add_argument("--output_csv", default="./output/multiview_clip_correction_eval/rule_search_val.csv")
    parser.add_argument("--output_json", default="./output/multiview_clip_correction_eval/rule_search_val.json")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--min_iou", default=0.25, type=float)
    parser.add_argument("--confidences", default="0.20,0.30,0.40,0.50,0.60", type=_parse_float_grid)
    parser.add_argument("--margins", default="0.05,0.10,0.20,0.30", type=_parse_float_grid)
    parser.add_argument("--gains", default="0.05,0.10,0.20,0.30", type=_parse_float_grid)
    parser.add_argument("--max_base_scores", default="0.25,0.35,0.50,1.10", type=_parse_float_grid)
    parser.add_argument("--min_pair_count", default=1, type=int)
    parser.add_argument("--min_applied", default=1, type=int)
    parser.add_argument("--min_helpful", default=1, type=int)
    parser.add_argument("--max_harmful", default=0, type=int)
    return parser


if __name__ == "__main__":
    search_rules(build_parser().parse_args())
