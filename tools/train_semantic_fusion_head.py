import argparse
import json
import os
import os.path as osp
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn


EXCLUDED_NUMERIC_FEATURES = {
    "candidate_id",
    "component_id",
    "object_evidence_top1_class_id",
    "object_evidence_top2_class_id",
    "sam_fused_evidence_top1_class_id",
    "bpr_evidence_top1_class_id",
}


def _read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _as_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    positives = float(labels.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels), dtype=np.float32) + 1.0)
    return float((precision * sorted_labels).sum() / positives)


def _binary_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    pred = scores >= 0.5
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "ap": _average_precision(labels, scores),
        "accuracy_at_0_5": float((pred == labels).mean()) if len(labels) else 0.0,
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positives": int(labels.sum()),
        "total": int(len(labels)),
    }


def _collect_numeric_feature_names(records):
    names = set()
    for record in records:
        for key, value in record.get("features", {}).items():
            if key in EXCLUDED_NUMERIC_FEATURES:
                continue
            if isinstance(value, (int, float, bool)) or value is None:
                names.add(key)
    return sorted(names)


def _source_values(records):
    values = sorted({str(record.get("source_kind", "unknown")) for record in records})
    preferred = ["mask3d", "sam_fused", "bpr"]
    ordered = [value for value in preferred if value in values]
    ordered.extend(value for value in values if value not in ordered)
    return ordered


def _build_feature_matrix(records, numeric_feature_names, source_values, num_classes=49):
    source_to_idx = {name: idx for idx, name in enumerate(source_values)}
    rows = []
    for record in records:
        features = record.get("features", {})
        row = [_as_float(features.get(name)) for name in numeric_feature_names]

        pred_class_id = int(record.get("pred_class_id", -1))
        class_onehot = [0.0] * num_classes
        if 0 <= pred_class_id < num_classes:
            class_onehot[pred_class_id] = 1.0
        row.extend(class_onehot)

        source_onehot = [0.0] * len(source_values)
        source_idx = source_to_idx.get(str(record.get("source_kind", "unknown")))
        if source_idx is not None:
            source_onehot[source_idx] = 1.0
        row.extend(source_onehot)

        object_top1 = int(features.get("object_evidence_top1_class_id", -1) or -1)
        sam_top1 = int(features.get("sam_fused_evidence_top1_class_id", -1) or -1)
        bpr_top1 = int(features.get("bpr_evidence_top1_class_id", -1) or -1)
        row.extend(
            [
                1.0 if object_top1 == pred_class_id else 0.0,
                1.0 if sam_top1 == pred_class_id else 0.0,
                1.0 if bpr_top1 == pred_class_id else 0.0,
            ]
        )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _feature_names(numeric_feature_names, source_values, num_classes=49):
    names = list(numeric_feature_names)
    names.extend([f"pred_class_{idx}" for idx in range(num_classes)])
    names.extend([f"source_{name}" for name in source_values])
    names.extend(["object_top1_matches_pred", "sam_top1_matches_pred", "bpr_top1_matches_pred"])
    return names


class FusionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=0, dropout=0.0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _split_records(records, val_scenes):
    val_scenes = {item.strip() for item in (val_scenes or "").split(",") if item.strip()}
    if not val_scenes:
        scenes = sorted({record["scene_name"] for record in records})
        val_scenes = {scene for scene in scenes if scene.startswith("room")}
    train = [record for record in records if record["scene_name"] not in val_scenes]
    val = [record for record in records if record["scene_name"] in val_scenes]
    if not train or not val:
        raise ValueError(f"Invalid split: train={len(train)} val={len(val)} val_scenes={sorted(val_scenes)}")
    return train, val, sorted(val_scenes)


def _standardize(train_x, val_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def _train_model(train_x, train_y, val_x, val_y, args):
    device = torch.device(args.device)
    model = FusionHead(train_x.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    train_x_t = torch.from_numpy(train_x).to(device)
    train_y_t = torch.from_numpy(train_y.astype(np.float32)).to(device)
    val_x_t = torch.from_numpy(val_x).to(device)

    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_ap = -1.0
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x_t)
        loss = loss_fn(logits, train_y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                train_scores = torch.sigmoid(model(train_x_t)).detach().cpu().numpy()
                val_scores = torch.sigmoid(model(val_x_t)).detach().cpu().numpy()
            train_ap = _average_precision(train_y, train_scores)
            val_ap = _average_precision(val_y, val_scores)
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "loss": float(loss.detach().cpu().item()),
                    "train_ap": float(train_ap),
                    "val_ap": float(val_ap),
                }
            )
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(model(train_x_t)).detach().cpu().numpy()
        val_scores = torch.sigmoid(model(val_x_t)).detach().cpu().numpy()
    return model.cpu(), train_scores, val_scores, history, best_epoch


def _source_breakdown(records, labels, scores):
    by_source = defaultdict(list)
    for idx, record in enumerate(records):
        by_source[str(record.get("source_kind", "unknown"))].append(idx)
    report = {}
    for source, indices in sorted(by_source.items()):
        report[source] = _binary_metrics(labels[indices], scores[indices])
    return report


def train(args):
    records = _read_jsonl(args.dataset_jsonl)
    if args.only_kept_after_threshold:
        records = [record for record in records if record.get("features", {}).get("keep_after_score_threshold", True)]
    if not records:
        raise ValueError("No records loaded for training.")

    train_records, val_records, val_scenes = _split_records(records, args.val_scenes)
    numeric_feature_names = _collect_numeric_feature_names(train_records)
    source_values = _source_values(records)
    feature_names = _feature_names(numeric_feature_names, source_values, num_classes=args.num_classes)

    train_x_raw = _build_feature_matrix(train_records, numeric_feature_names, source_values, args.num_classes)
    val_x_raw = _build_feature_matrix(val_records, numeric_feature_names, source_values, args.num_classes)
    train_x, val_x, mean, std = _standardize(train_x_raw, val_x_raw)

    train_y = np.asarray([1 if record.get(args.label_key) else 0 for record in train_records], dtype=np.int64)
    val_y = np.asarray([1 if record.get(args.label_key) else 0 for record in val_records], dtype=np.int64)
    train_base = np.asarray([_as_float(record.get("features", {}).get("base_score")) for record in train_records])
    val_base = np.asarray([_as_float(record.get("features", {}).get("base_score")) for record in val_records])

    model, train_scores, val_scores, history, best_epoch = _train_model(train_x, train_y, val_x, val_y, args)
    blended_train = args.blend_alpha * train_base + (1.0 - args.blend_alpha) * train_scores
    blended_val = args.blend_alpha * val_base + (1.0 - args.blend_alpha) * val_scores

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = osp.join(args.output_dir, args.model_name)
    report_path = osp.join(args.output_dir, args.report_name)
    pred_path = osp.join(args.output_dir, args.predictions_name)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "numeric_feature_names": numeric_feature_names,
            "source_values": source_values,
            "num_classes": args.num_classes,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "label_key": args.label_key,
        },
        model_path,
    )

    report = {
        "dataset_jsonl": args.dataset_jsonl,
        "label_key": args.label_key,
        "val_scenes": val_scenes,
        "num_records": len(records),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "train_label_counts": dict(Counter(train_y.tolist())),
        "val_label_counts": dict(Counter(val_y.tolist())),
        "num_features": len(feature_names),
        "hidden_dim": args.hidden_dim,
        "best_epoch": best_epoch,
        "history": history,
        "metrics": {
            "train_base_score": _binary_metrics(train_y, train_base),
            "train_model_score": _binary_metrics(train_y, train_scores),
            "train_blended_score": _binary_metrics(train_y, blended_train),
            "val_base_score": _binary_metrics(val_y, val_base),
            "val_model_score": _binary_metrics(val_y, val_scores),
            "val_blended_score": _binary_metrics(val_y, blended_val),
        },
        "source_breakdown": {
            "val_base_score": _source_breakdown(val_records, val_y, val_base),
            "val_model_score": _source_breakdown(val_records, val_y, val_scores),
            "val_blended_score": _source_breakdown(val_records, val_y, blended_val),
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(pred_path, "w") as f:
        for split_name, split_records, labels, base_scores, model_scores, blended_scores in (
            ("train", train_records, train_y, train_base, train_scores, blended_train),
            ("val", val_records, val_y, val_base, val_scores, blended_val),
        ):
            for record, label, base_score, model_score, blended_score in zip(
                split_records, labels, base_scores, model_scores, blended_scores
            ):
                f.write(
                    json.dumps(
                        {
                            "split": split_name,
                            "scene_name": record["scene_name"],
                            "prediction_id": record["prediction_id"],
                            "source_kind": record.get("source_kind"),
                            "pred_class_name": record.get("pred_class_name"),
                            "gt_label_name": record.get("gt_label_name"),
                            "best_iou": record.get("best_iou"),
                            "label": int(label),
                            "base_score": float(base_score),
                            "model_score": float(model_score),
                            "blended_score": float(blended_score),
                        }
                    )
                    + "\n"
                )

    print(f"Saved model to {model_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved predictions to {pred_path}")
    print(
        "Validation AP: "
        f"base={report['metrics']['val_base_score']['ap']:.3f} "
        f"model={report['metrics']['val_model_score']['ap']:.3f} "
        f"blend={report['metrics']['val_blended_score']['ap']:.3f}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", default="./output/semantic_fusion_dataset_replica_s5_m30_bpr/features.jsonl")
    parser.add_argument("--output_dir", default="./output/semantic_fusion_head_replica_s5_m30_bpr")
    parser.add_argument("--model_name", default="fusion_head.pt")
    parser.add_argument("--report_name", default="report.json")
    parser.add_argument("--predictions_name", default="predictions.jsonl")
    parser.add_argument("--label_key", default="class_correct_25")
    parser.add_argument("--val_scenes", default="room0,room1,room2")
    parser.add_argument("--only_kept_after_threshold", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_classes", default=49, type=int)
    parser.add_argument("--hidden_dim", default=0, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument("--eval_every", default=20, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--blend_alpha", default=0.35, type=float)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
