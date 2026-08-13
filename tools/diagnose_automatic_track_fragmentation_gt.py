#!/usr/bin/env python3
"""仅离线 GT 账本：归因自动 mask 观测上限到三维轨迹之间的覆盖损失。"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_RESIDUALS = {"无合格三维候选", "边界不足"}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _classify(auto_union_iou, best_track_iou, same_target_track_count, union_track_iou):
    if best_track_iou >= 0.25:
        return "已有几何合格轨迹"
    if same_target_track_count == 0:
        return "三维关联未形成主属轨迹"
    if union_track_iou >= 0.25:
        return "同目标轨迹碎裂且并集可恢复"
    if auto_union_iou >= 0.25:
        return "主属轨迹存在但并集仍不足"
    return "自动观测不足"


def _scene_rows(scene_name, targets, tracks, ledger, gt_ids):
    tracks_by_id = {int(track["track_id"]): track for track in tracks}
    by_gt = defaultdict(list)
    for row in ledger:
        if row["scene_name"] == scene_name and int(row["best_gt_instance_id"]) > 0:
            by_gt[int(row["best_gt_instance_id"])].append(row)
    rows = []
    for target in targets:
        instance_id = int(target["gt_instance_id"])
        gt_points = np.flatnonzero(gt_ids == instance_id)
        matched = by_gt.get(instance_id, [])
        unions = []
        for record in matched:
            track = tracks_by_id.get(int(record["track_id"]))
            if track is not None:
                unions.append(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        union_points = np.unique(np.concatenate(unions)) if unions else np.empty(0, dtype=np.int64)
        intersection = int(np.intersect1d(union_points, gt_points, assume_unique=True).size)
        union_iou = float(intersection / max(1, len(union_points) + len(gt_points) - intersection))
        best_track_iou = max((float(record["best_gt_iou"]) for record in matched), default=0.0)
        rows.append({
            "scene_name": scene_name,
            "gt_instance_id": instance_id,
            "residual_type": target["residual_type"],
            "auto_observation_union_iou": float(target["auto_union_iou"]),
            "same_target_track_count": len(matched),
            "best_same_target_track_iou": best_track_iou,
            "same_target_track_union_iou": union_iou,
            "loss_attribution": _classify(float(target["auto_union_iou"]), best_track_iou, len(matched), union_iou),
        })
    return rows


def _summary(rows):
    by_residual = {}
    for residual in sorted(TARGET_RESIDUALS):
        subset = [row for row in rows if row["residual_type"] == residual]
        by_residual[residual] = dict(sorted(Counter(row["loss_attribution"] for row in subset).items()))
    return {
        "观测并集达到 GT IoU 25% 的目标残差实例数": len(rows),
        "按残差类型归因": by_residual,
        "解释": {
            "同目标轨迹碎裂且并集可恢复": "自动观测已形成多个主属轨迹，存在改进无 GT 关联的明确空间。",
            "主属轨迹存在但并集仍不足": "仅合并当前轨迹不够，需检查观测选择、轨迹增长或新二维对象源。",
            "三维关联未形成主属轨迹": "自动观测虽然有 GT 上限，但当前图没有形成以该对象为主的轨迹，需检查图召回与种子-验证-扩展。",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage_gt_ledger", type=Path, required=True)
    parser.add_argument("--track_gt_ledger", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线覆盖损失归因。")
    for name in ("coverage_gt_ledger", "track_gt_ledger", "track_root", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    coverage = [
        row for row in _read_csv(args.coverage_gt_ledger)
        if row["residual_type"] in TARGET_RESIDUALS and float(row["auto_union_iou"]) >= 0.25
    ]
    ledger = _read_csv(args.track_gt_ledger)
    targets_by_scene = defaultdict(list)
    for row in coverage:
        targets_by_scene[row["scene_name"]].append(row)
    rows = []
    for scene_name, targets in sorted(targets_by_scene.items()):
        tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
        gt_ids = np.loadtxt(args.gt_instance_dir / f"{scene_name}.txt", dtype=np.int64)
        rows.extend(_scene_rows(scene_name, targets, tracks, ledger, gt_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "automatic_track_fragmentation_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线归因自动观测到自动轨迹的损失；不反向修改关联阈值，绝不进入自动 mask、轨迹、候选、融合、评分或评测。",
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
