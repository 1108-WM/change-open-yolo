#!/usr/bin/env python3
"""汇总自动 SAM 候选的多视图 native 增量证据，不产生候选决定。"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复项")
    return scenes


def _jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(values)), "mean": float(values.mean()), "p10": float(np.quantile(values, .10)),
        "p50": float(np.quantile(values, .50)), "p90": float(np.quantile(values, .90)), "max": float(values.max()),
    }


def candidate_increment_features(row):
    """将逐视图集合计数汇聚成连续候选特征，不使用阈值作筛选。"""
    views = row["view_records"]
    observed = sum(int(item["candidate_observation_point_count"]) for item in views)
    independent = sum(int(item["candidate_independent_point_count"]) for item in views)
    same_class = sum(int(item["candidate_same_class_native_explained_point_count"]) for item in views)
    any_class = sum(int(item["candidate_any_native_explained_point_count"]) for item in views)
    return {
        "track_observation_count": int(row["track_observation_count"]),
        "candidate_supported_observation_count": int(row["candidate_supported_observation_count"]),
        "candidate_independent_observation_count": int(row["candidate_independent_observation_count"]),
        "weighted_candidate_independent_ratio": float(independent / max(1, observed)),
        "weighted_candidate_same_class_native_explained_ratio": float(same_class / max(1, observed)),
        "weighted_candidate_any_native_explained_ratio": float(any_class / max(1, observed)),
    }


def summarize(rows):
    features = [candidate_increment_features(row) for row in rows]
    independent_counts = Counter(item["candidate_independent_observation_count"] for item in features)
    return {
        "candidate_count": len(features),
        "independent_observation_count_histogram": {str(key): int(value) for key, value in sorted(independent_counts.items())},
        "track_observation_count": _summary([item["track_observation_count"] for item in features]),
        "candidate_independent_observation_count": _summary([item["candidate_independent_observation_count"] for item in features]),
        "weighted_candidate_independent_ratio": _summary([item["weighted_candidate_independent_ratio"] for item in features]),
        "weighted_candidate_same_class_native_explained_ratio": _summary([item["weighted_candidate_same_class_native_explained_ratio"] for item in features]),
        "weighted_candidate_any_native_explained_ratio": _summary([item["weighted_candidate_any_native_explained_ratio"] for item in features]),
        "at_least_two_independent_observations_count": sum(item["candidate_independent_observation_count"] >= 2 for item in features),
        "all_track_observations_independent_count": sum(
            item["track_observation_count"] > 0
            and item["candidate_independent_observation_count"] == item["track_observation_count"]
            for item in features
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "ledger_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows = []
    for scene in _scenes(args.scene_list):
        rows.extend(_jsonl(args.ledger_root / scene / "automatic_sam_candidate_multiview_increment_ledger.jsonl"))
    payload = {
        "purpose": "汇总自动候选在自身轨迹观测中相对 native 的连续增量证据。",
        "gt_usage": "none",
        "decision_state": "只读汇总；不产生接受、抑制、合并、删除或预测决定。",
        **summarize(rows), "params": vars(args),
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
