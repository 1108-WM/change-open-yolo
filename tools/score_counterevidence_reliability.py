#!/usr/bin/env python3
"""以固定、无 GT 的物理证据组合计算 superpoint 正反证据连续分数。

每个场景内按特征的稳健中位尺度归一化：可信反证据要求负证据、相互独立的
多视角、低深度残差、远离二维 mask 边界且不深度包含于 native 候选；正支持
要求正证据、稳定二维覆盖和 mask 内部支持。脚本不定义阈值、不删点、不产生
候选、类别、融合或 AP。
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_scene_rows(root):
    result = {}
    for path in sorted(root.glob("*/counterevidence_superpoint_reliability.jsonl")):
        with path.open() as handle:
            result[path.parent.name] = [json.loads(line) for line in handle if line.strip()]
    return result


def _median_scale(values):
    values = np.asarray(values, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    return float(np.median(positive)) if len(positive) else 1.0


def _bounded_ratio(value, scale):
    return float(max(0.0, value) / (max(0.0, value) + max(scale, 1e-8)))


def _geometric_mean(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-8)))))


def _score_scene(rows, visibility_depth_threshold_m):
    baseline_scale = _median_scale([row["counter_view_max_camera_baseline_m"] for row in rows])
    angle_scale = _median_scale([row["counter_view_max_view_angle_deg"] for row in rows])
    scored = []
    for row in rows:
        positive = float(row["positive_weight"])
        negative = float(row["negative_weight"])
        polarity_total = max(positive + negative, 1e-8)
        positive_ratio = positive / polarity_total
        negative_ratio = negative / polarity_total
        independent_view = _geometric_mean([
            float(row["counterevidence_eligible_view_rate"]),
            _bounded_ratio(float(row["counter_view_max_camera_baseline_m"]), baseline_scale),
            _bounded_ratio(float(row["counter_view_max_view_angle_deg"]), angle_scale),
        ])
        depth_reliability = math.exp(-max(0.0, float(row["depth_residual_mean_m"])) / max(visibility_depth_threshold_m, 1e-8))
        away_from_boundary = 1.0 - min(1.0, max(0.0, float(row["uncovered_near_2px_mask_boundary_ratio"])))
        outside_native = 1.0 - min(1.0, max(0.0, float(row["sp_inside_top_native_ratio"])))
        counter_score = _geometric_mean([
            negative_ratio, independent_view, depth_reliability, away_from_boundary, outside_native,
        ])
        coverage_stability = 1.0 - min(1.0, max(0.0, float(row["mask_coverage_std"])))
        positive_score = _geometric_mean([
            positive_ratio,
            min(1.0, max(0.0, float(row["mask_coverage_mean"]))),
            min(1.0, max(0.0, float(row["covered_mask_2px_interior_ratio"]))),
            coverage_stability,
            depth_reliability,
        ])
        evidence_polarity = float((positive_score - counter_score) / max(positive_score + counter_score, 1e-8))
        scored.append({
            **row,
            "score_normalization": "场景内正值中位尺度；不使用 GT。",
            "counter_negative_ratio": negative_ratio,
            "counter_independent_view_score": independent_view,
            "counter_depth_reliability_score": depth_reliability,
            "counter_away_from_mask_boundary_score": away_from_boundary,
            "counter_outside_native_score": outside_native,
            "counterevidence_reliability_score": counter_score,
            "positiveevidence_reliability_score": positive_score,
            "evidence_polarity_score": evidence_polarity,
        })
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reliability_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--visibility_depth_threshold_m", type=float, default=0.05)
    args = parser.parse_args()
    for name in ("reliability_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    if args.visibility_depth_threshold_m <= 0.0:
        raise SystemExit("--visibility_depth_threshold_m 必须为正数。")
    scenes = _load_scene_rows(args.reliability_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for scene_name, rows in scenes.items():
        scored = _score_scene(rows, args.visibility_depth_threshold_m)
        all_rows.extend(scored)
        scene_root = args.output_root / scene_name
        scene_root.mkdir()
        with (scene_root / "counterevidence_reliability_scores.jsonl").open("w") as handle:
            for row in scored:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    payload = {
        "gt_usage": "不读取 GT；不定义阈值、不删点、不产生候选、类别、分数融合或 AP。",
        "decision_state": "仅固定正支持、可信反证据及证据极性连续分数，供独立 GT-only 排序审计。",
        "formula": "CER=几何均值(负证据比例,独立视角,深度可靠,远离二维边界,非native内部)；PES=几何均值(正证据比例,mask覆盖,mask内部支持,覆盖稳定,深度可靠)。",
        "scene_count": len(scenes),
        "record_count": len(all_rows),
        "negative_margin_record_count": sum(row["is_negative_margin"] for row in all_rows),
        "params": vars(args),
    }
    (args.output_root / "counterevidence_reliability_score_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
