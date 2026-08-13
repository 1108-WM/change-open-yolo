#!/usr/bin/env python3
"""仅离线 GT 账本：判断锚点扩展后的 Alpha-CLIP 是否使几何增益可接入。"""

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


def _read_alpha(root):
    records = []
    for path in sorted(root.glob("*/anchor_guided_expansion_alphaclip_semantics.json")):
        records.extend(json.loads(path.read_text()))
    return records


def _join(expansion_rows, alpha_records):
    expansion = {(row["scene_name"], int(row["track_id"])): row for row in expansion_rows}
    rows = []
    for alpha in alpha_records:
        key = (alpha["scene_name"], int(alpha["track_id"]))
        source = expansion.get(key)
        if source is None:
            raise ValueError(f"Alpha-CLIP 扩展区域未在几何 GT 账本找到：{key}")
        alpha_id = int(PRED_ID_TO_ID.get(int(alpha["alphaclip_class_index"]), -1))
        gt_instance = int(source["anchor_gt_instance_id"])
        gt_id = gt_instance // 1000 if gt_instance > 0 else -1
        rows.append({
            "scene_name": key[0],
            "track_id": key[1],
            "anchor_gt_instance_id": gt_instance,
            "anchor_gt_class": source["anchor_gt_class"],
            "matched_gt_residual_type": source["matched_gt_residual_type"],
            "anchor_iou": float(source["anchor_iou"]),
            "expanded_same_instance_iou": float(source["expanded_same_instance_iou"]),
            "expanded_alpha_class": ID_TO_LABEL.get(alpha_id, "无语义结果"),
            "expanded_alpha_class_id": alpha_id,
            "expanded_alpha_correct": bool(gt_id > 0 and alpha_id == gt_id),
            "alphaclip_logit_margin": float(alpha["alphaclip_logit_margin"]),
        })
    return rows


def _summary(rows):
    first_geometric = [row for row in rows if row["anchor_iou"] < 0.25 <= row["expanded_same_instance_iou"]]
    result = {"首次达到几何 IoU 25% 的扩展区域数": len(first_geometric)}
    for residual in ("无合格三维候选", "边界不足"):
        subset = [row for row in first_geometric if row["matched_gt_residual_type"] == residual]
        correct = [row for row in subset if row["expanded_alpha_correct"]]
        result[residual] = {
            "首次几何合格区域数": len(subset),
            "Alpha-CLIP 类别正确区域数": len(correct),
            "几何和语义同时合格的独立实例数": len({(row["scene_name"], row["anchor_gt_instance_id"]) for row in correct}),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion_gt_ledger", type=Path, required=True)
    parser.add_argument("--alpha_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线几何与语义联合审计。")
    for name in ("expansion_gt_ledger", "alpha_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = _join(_read_csv(args.expansion_gt_ledger), _read_alpha(args.alpha_root))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "anchor_guided_expansion_alphaclip_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线检查固定扩展区域的几何和语义联合质量；不反向修改扩展或语义规则。",
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
