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


def _onehot(index, size):
    values = [0.0] * size
    if 0 <= int(index) < size:
        values[int(index)] = 1.0
    return values


def build_feature_matrix(records, numeric_feature_names, source_values, num_classes=49):
    source_to_idx = {name: idx for idx, name in enumerate(source_values)}
    rows = []
    for record in records:
        features = record.get("features", {})
        row = [_as_float(features.get(name)) for name in numeric_feature_names]

        pred_class_id = int(record.get("pred_class_id", -1))
        row.extend(_onehot(pred_class_id, num_classes))

        for key in (
            "object_evidence_top1_class_id",
            "object_evidence_top2_class_id",
            "sam_fused_evidence_top1_class_id",
            "bpr_evidence_top1_class_id",
        ):
            row.extend(_onehot(int(features.get(key, -1) or -1), num_classes))

        source_onehot = [0.0] * len(source_values)
        source_idx = source_to_idx.get(str(record.get("source_kind", "unknown")))
        if source_idx is not None:
            source_onehot[source_idx] = 1.0
        row.extend(source_onehot)

        row.extend(
            [
                1.0 if int(features.get("object_evidence_top1_class_id", -1) or -1) == pred_class_id else 0.0,
                1.0 if int(features.get("sam_fused_evidence_top1_class_id", -1) or -1) == pred_class_id else 0.0,
                1.0 if int(features.get("bpr_evidence_top1_class_id", -1) or -1) == pred_class_id else 0.0,
            ]
        )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def feature_names(numeric_feature_names, source_values, num_classes=49):
    names = list(numeric_feature_names)
    names.extend([f"pred_class_{idx}" for idx in range(num_classes)])
    for prefix in ("object_top1", "object_top2", "sam_top1", "bpr_top1"):
        names.extend([f"{prefix}_class_{idx}" for idx in range(num_classes)])
    names.extend([f"source_{name}" for name in source_values])
    names.extend(["object_top1_matches_pred", "sam_top1_matches_pred", "bpr_top1_matches_pred"])
    return names


class SemanticCorrectionHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, dropout=0.15):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)


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


def _filter_supervised(records, min_iou):
    filtered = []
    for record in records:
        if float(record.get("best_iou", 0.0)) < float(min_iou):
            continue
        if int(record.get("gt_pred_class_id", -1)) < 0:
            continue
        filtered.append(record)
    return filtered


def _standardize(train_x, val_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def _accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def _topk_accuracy(y_true, probs, k=3):
    if len(y_true) == 0:
        return 0.0
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.asarray([label in row for label, row in zip(y_true, topk)]).mean())


def _correction_metrics(records, y_true, probs, min_confidence, min_margin):
    current = np.asarray([int(record.get("pred_class_id", -1)) for record in records], dtype=np.int64)
    top2 = np.argsort(-probs, axis=1)[:, :2]
    top1_class = top2[:, 0]
    top1_score = probs[np.arange(len(probs)), top1_class]
    top2_score = probs[np.arange(len(probs)), top2[:, 1]] if probs.shape[1] > 1 else np.zeros_like(top1_score)
    margin = top1_score - top2_score
    apply = (top1_class != current) & (top1_score >= min_confidence) & (margin >= min_margin)
    corrected = current.copy()
    corrected[apply] = top1_class[apply]
    return {
        "base_accuracy": _accuracy(y_true, current),
        "model_top1_accuracy": _accuracy(y_true, top1_class),
        "model_top3_accuracy": _topk_accuracy(y_true, probs, k=3),
        "corrected_accuracy": _accuracy(y_true, corrected),
        "num_applied": int(apply.sum()),
        "num_helpful": int(((current != y_true) & (corrected == y_true)).sum()),
        "num_harmful": int(((current == y_true) & (corrected != y_true)).sum()),
        "num_changed_wrong_to_wrong": int(((current != y_true) & (corrected != y_true) & apply).sum()),
    }


def _class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    if present.any():
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights[~present] = 0.0
    return torch.from_numpy(weights).to(device)


def train(args):
    all_records = _read_jsonl(args.dataset_jsonl)
    records = _filter_supervised(all_records, args.min_iou)
    if not records:
        raise ValueError("No supervised positive records after filtering.")

    train_records, val_records, val_scenes = _split_records(records, args.val_scenes)
    numeric_feature_names = _collect_numeric_feature_names(train_records)
    source_values = _source_values(records)
    names = feature_names(numeric_feature_names, source_values, args.num_classes)

    train_x_raw = build_feature_matrix(train_records, numeric_feature_names, source_values, args.num_classes)
    val_x_raw = build_feature_matrix(val_records, numeric_feature_names, source_values, args.num_classes)
    train_x, val_x, mean, std = _standardize(train_x_raw, val_x_raw)
    train_y = np.asarray([int(record["gt_pred_class_id"]) for record in train_records], dtype=np.int64)
    val_y = np.asarray([int(record["gt_pred_class_id"]) for record in val_records], dtype=np.int64)

    device = torch.device(args.device)
    model = SemanticCorrectionHead(
        input_dim=train_x.shape[1],
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    train_x_t = torch.from_numpy(train_x).to(device)
    train_y_t = torch.from_numpy(train_y).to(device)
    val_x_t = torch.from_numpy(val_x).to(device)
    class_weights = _class_weights(train_y, args.num_classes, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_metric = -1.0
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
                train_probs = torch.softmax(model(train_x_t), dim=1).detach().cpu().numpy()
                val_probs = torch.softmax(model(val_x_t), dim=1).detach().cpu().numpy()
            train_top1 = train_probs.argmax(axis=1)
            val_top1 = val_probs.argmax(axis=1)
            val_acc = _accuracy(val_y, val_top1)
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "loss": float(loss.detach().cpu().item()),
                    "train_top1_accuracy": _accuracy(train_y, train_top1),
                    "val_top1_accuracy": val_acc,
                    "val_base_accuracy": _accuracy(
                        val_y, [int(record.get("pred_class_id", -1)) for record in val_records]
                    ),
                    "val_top3_accuracy": _topk_accuracy(val_y, val_probs, k=3),
                }
            )
            if val_acc > best_metric:
                best_metric = val_acc
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_probs = torch.softmax(model(train_x_t), dim=1).detach().cpu().numpy()
        val_probs = torch.softmax(model(val_x_t), dim=1).detach().cpu().numpy()

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = osp.join(args.output_dir, args.model_name)
    report_path = osp.join(args.output_dir, args.report_name)
    pred_path = osp.join(args.output_dir, args.predictions_name)

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "feature_names": names,
            "numeric_feature_names": numeric_feature_names,
            "source_values": source_values,
            "num_classes": args.num_classes,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "min_iou": args.min_iou,
        },
        model_path,
    )

    thresholds = []
    for confidence in args.report_confidences:
        for margin in args.report_margins:
            thresholds.append(
                {
                    "min_confidence": confidence,
                    "min_margin": margin,
                    "train": _correction_metrics(train_records, train_y, train_probs, confidence, margin),
                    "val": _correction_metrics(val_records, val_y, val_probs, confidence, margin),
                }
            )

    report = {
        "dataset_jsonl": args.dataset_jsonl,
        "min_iou": args.min_iou,
        "val_scenes": val_scenes,
        "num_all_records": len(all_records),
        "num_supervised_records": len(records),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "train_label_counts": dict(Counter(train_y.tolist())),
        "val_label_counts": dict(Counter(val_y.tolist())),
        "num_features": len(names),
        "hidden_dim": args.hidden_dim,
        "best_epoch": best_epoch,
        "history": history,
        "threshold_metrics": thresholds,
        "source_counts": dict(Counter(record.get("source_kind", "unknown") for record in records)),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(pred_path, "w") as f:
        for split_name, split_records, labels, probs in (
            ("train", train_records, train_y, train_probs),
            ("val", val_records, val_y, val_probs),
        ):
            top2 = np.argsort(-probs, axis=1)[:, :2]
            for record, label, top_pair, prob in zip(split_records, labels, top2, probs):
                f.write(
                    json.dumps(
                        {
                            "split": split_name,
                            "scene_name": record["scene_name"],
                            "prediction_id": record["prediction_id"],
                            "source_kind": record.get("source_kind"),
                            "pred_class_id": int(record.get("pred_class_id", -1)),
                            "gt_pred_class_id": int(label),
                            "top1_class_id": int(top_pair[0]),
                            "top1_prob": float(prob[top_pair[0]]),
                            "top2_class_id": int(top_pair[1]),
                            "top2_prob": float(prob[top_pair[1]]),
                            "margin": float(prob[top_pair[0]] - prob[top_pair[1]]),
                        }
                    )
                    + "\n"
                )

    val_base = _accuracy(val_y, [int(record.get("pred_class_id", -1)) for record in val_records])
    val_model = _accuracy(val_y, val_probs.argmax(axis=1))
    print(f"Saved model to {model_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved predictions to {pred_path}")
    print(f"Validation class accuracy: base={val_base:.3f} model={val_model:.3f}")


def _float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", default="./output/semantic_fusion_dataset_replica_s5_m30_bpr/features.jsonl")
    parser.add_argument("--output_dir", default="./output/semantic_correction_head_replica_s5_m30_bpr")
    parser.add_argument("--model_name", default="semantic_correction_head.pt")
    parser.add_argument("--report_name", default="report.json")
    parser.add_argument("--predictions_name", default="predictions.jsonl")
    parser.add_argument("--min_iou", default=0.25, type=float)
    parser.add_argument("--val_scenes", default="room0,room1,room2")
    parser.add_argument("--num_classes", default=49, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--epochs", default=1200, type=int)
    parser.add_argument("--eval_every", default=20, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--weight_decay", default=0.005, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report_confidences", default="0.45,0.55,0.65,0.75", type=_float_list)
    parser.add_argument("--report_margins", default="0.05,0.10,0.20,0.30", type=_float_list)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
