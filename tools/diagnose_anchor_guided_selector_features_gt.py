#!/usr/bin/env python3
"""仅离线 GT 账本：审计锚点扩展区域的多源选择器特征。"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value):
    return str(value).lower() == "true"


def _join(expanded_rows, original_alpha_rows, yolo_rows):
    original = {(row["scene_name"], int(row["track_id"])): row for row in original_alpha_rows}
    yolo = {(row["scene_name"], int(row["track_id"])): row for row in yolo_rows}
    joined = []
    for row in expanded_rows:
        key = (row["scene_name"], int(row["track_id"]))
        alpha = original.get(key)
        vote = yolo.get(key)
        if alpha is None or vote is None:
            raise ValueError(f"扩展区域缺少原始语义记录：{key}")
        joined.append({
            **row,
            "original_alphaclip_class": alpha["alphaclip_class"],
            "original_alphaclip_margin": float(alpha["alphaclip_logit_margin"]),
            "original_alphaclip_correct": _truth(alpha["alphaclip_correct"]),
            "yoloworld_class": vote["voted_class"],
            "yoloworld_correct": _truth(vote["semantic_correct"]),
            "expanded_matches_original_alpha": row["expanded_alpha_class"] == alpha["alphaclip_class"],
            "expanded_matches_yoloworld": row["expanded_alpha_class"] == vote["voted_class"],
        })
    return joined


def _summary(rows):
    targets = [
        row for row in rows
        if float(row["anchor_iou"]) < 0.25 <= float(row["expanded_same_instance_iou"])
        and row["matched_gt_residual_type"] in {"无合格三维候选", "边界不足"}
    ]
    groups = {
        "扩展语义与原 Alpha-CLIP 一致": lambda row: row["expanded_matches_original_alpha"],
        "扩展语义与原 Alpha-CLIP 不一致": lambda row: not row["expanded_matches_original_alpha"],
        "扩展语义与 YOLO-World 一致": lambda row: row["expanded_matches_yoloworld"],
        "扩展语义与 YOLO-World 不一致": lambda row: not row["expanded_matches_yoloworld"],
        "原 Alpha-CLIP 与 YOLO-World 同时正确": lambda row: row["original_alphaclip_correct"] and row["yoloworld_correct"],
    }
    result = {"首次几何合格的目标扩展区域数": len(targets)}
    for name, predicate in groups.items():
        subset = [row for row in targets if predicate(row)]
        result[name] = {
            "区域数": len(subset),
            "扩展 Alpha-CLIP 正确区域数": sum(_truth(row["expanded_alpha_correct"]) for row in subset),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded_gt_ledger", type=Path, required=True)
    parser.add_argument("--original_alphaclip_gt_ledger", type=Path, required=True)
    parser.add_argument("--yoloworld_gt_ledger", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线选择器特征审计。")
    for name in ("expanded_gt_ledger", "original_alphaclip_gt_ledger", "yoloworld_gt_ledger", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = _join(_read_csv(args.expanded_gt_ledger), _read_csv(args.original_alphaclip_gt_ledger), _read_csv(args.yoloworld_gt_ledger))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "anchor_guided_selector_features_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于检查固定多源特征的可分性；不用于选择规则或阈值。",
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
