#!/usr/bin/env python3
"""仅离线 GT 账本：对比自动轨迹的 Alpha-CLIP 与冻结 YOLO-World 语义。"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL, PRED_ID_TO_ID


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_alpha_rows(root):
    rows = []
    for path in sorted(root.glob("*/automatic_track_alphaclip_semantics.json")):
        rows.extend(json.loads(path.read_text()))
    return rows


def _join_rows(alpha_rows, yolo_rows):
    yolo = {(row["scene_name"], int(row["track_id"])): row for row in yolo_rows}
    joined = []
    for alpha in alpha_rows:
        key = (alpha["scene_name"], int(alpha["track_id"]))
        row = yolo.get(key)
        if row is None:
            raise ValueError(f"Alpha-CLIP 轨迹未在冻结 YOLO-World GT 账本中找到：{key}")
        alpha_index = int(alpha["alphaclip_class_index"])
        alpha_class_id = int(PRED_ID_TO_ID.get(alpha_index, -1))
        gt_class_id = int(row["best_gt_instance_id"]) // 1000 if int(row["best_gt_instance_id"]) > 0 else -1
        joined.append({
            "scene_name": key[0],
            "track_id": key[1],
            "best_gt_instance_id": int(row["best_gt_instance_id"]),
            "best_gt_iou": float(row["best_gt_iou"]),
            "matched_gt_residual_type": row["matched_gt_residual_type"],
            "gt_class": row["best_gt_class"],
            "yoloworld_class": row["voted_class"],
            "yoloworld_class_id": int(row["voted_class_id"]),
            "yoloworld_correct": row["semantic_correct"].lower() == "true",
            "alphaclip_class": ID_TO_LABEL.get(alpha_class_id, "无语义结果"),
            "alphaclip_class_id": alpha_class_id,
            "alphaclip_correct": bool(gt_class_id > 0 and alpha_class_id == gt_class_id),
            "alphaclip_logit_margin": float(alpha["alphaclip_logit_margin"]),
            "same_prediction": bool(alpha_class_id >= 0 and alpha_class_id == int(row["voted_class_id"])),
        })
    return joined


def _accuracy(rows, field):
    return float(sum(row[field] for row in rows) / max(1, len(rows)))


def _summary(rows):
    geometric = [row for row in rows if row["best_gt_iou"] >= 0.25]
    strict = [row for row in rows if row["best_gt_iou"] >= 0.50]
    recovered = defaultdict(set)
    for row in geometric:
        if row["alphaclip_correct"] and row["best_gt_instance_id"] > 0:
            recovered[row["matched_gt_residual_type"]].add((row["scene_name"], row["best_gt_instance_id"]))
    cases = {
        "两者都正确": lambda row: row["alphaclip_correct"] and row["yoloworld_correct"],
        "仅 Alpha-CLIP 正确": lambda row: row["alphaclip_correct"] and not row["yoloworld_correct"],
        "仅 YOLO-World 正确": lambda row: not row["alphaclip_correct"] and row["yoloworld_correct"],
        "两者都错误": lambda row: not row["alphaclip_correct"] and not row["yoloworld_correct"],
    }
    return {
        "全部轨迹数": len(rows),
        "几何 IoU 不低于 25% 的轨迹数": len(geometric),
        "几何 IoU 不低于 25% 的 Alpha-CLIP 语义正确率": _accuracy(geometric, "alphaclip_correct"),
        "几何 IoU 不低于 25% 的 YOLO-World 语义正确率": _accuracy(geometric, "yoloworld_correct"),
        "几何 IoU 不低于 50% 的 Alpha-CLIP 语义正确率": _accuracy(strict, "alphaclip_correct"),
        "几何 IoU 不低于 50% 的 YOLO-World 语义正确率": _accuracy(strict, "yoloworld_correct"),
        "几何 IoU 不低于 25% 且 Alpha-CLIP 语义正确的独立实例数": {
            key: len(value) for key, value in sorted(recovered.items())
        },
        "几何 IoU 不低于 25% 的互补情况": {
            name: sum(predicate(row) for row in geometric) for name, predicate in cases.items()
        },
        "说明": "GT 仅用于离线对照，不能据此选择 Alpha-CLIP 阈值或切换规则。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha_root", type=Path, required=True)
    parser.add_argument("--yoloworld_gt_ledger", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线语义对照。")
    for name in ("alpha_root", "yoloworld_gt_ledger", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = _join_rows(_read_alpha_rows(args.alpha_root), _read_csv(args.yoloworld_gt_ledger))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "automatic_track_alphaclip_semantics_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线比较两个冻结语义源；绝不进入自动 mask、轨迹、候选、融合、评分或阈值。",
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
