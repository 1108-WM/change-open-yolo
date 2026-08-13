#!/usr/bin/env python3
"""建立自动 SAM 基础闭包、一跳变体与 native 的成对竞争证据账本。

该工具不选择胜者、不形成综合分数、不修改 native。它仅将同一轨迹的基础闭包与
每个一跳变体的类别无关几何质量和多视图语义变化并列记录。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _distribution_with_other(record):
    values = {
        int(item["class_index"]): max(0.0, float(item["probability"]))
        for item in record.get("semantic_class_distribution", [])
    }
    values[-1] = max(0.0, 1.0 - sum(values.values()))
    total = sum(values.values())
    return {key: value / max(total, 1e-12) for key, value in values.items()}


def semantic_js_divergence(left, right):
    """比较保存的 top-k 分布，并将未保存尾部统一放入 other 桶。"""
    left_values = _distribution_with_other(left)
    right_values = _distribution_with_other(right)
    keys = sorted(set(left_values) | set(right_values))
    left_array = np.asarray([left_values.get(key, 0.0) for key in keys], dtype=np.float64)
    right_array = np.asarray([right_values.get(key, 0.0) for key in keys], dtype=np.float64)
    midpoint = 0.5 * (left_array + right_array)
    left_positive = left_array > 0.0
    right_positive = right_array > 0.0
    kl_left = float(np.sum(left_array[left_positive] * np.log(left_array[left_positive] / midpoint[left_positive])))
    kl_right = float(np.sum(right_array[right_positive] * np.log(right_array[right_positive] / midpoint[right_positive])))
    return float(0.5 * (kl_left + kl_right) / math.log(2.0))


def _pair_record(base_quality, variant_quality, base_semantic, variant_semantic):
    base_top = int(base_semantic.get("semantic_evidence_top_class_index", -1))
    variant_top = int(variant_semantic.get("semantic_evidence_top_class_index", -1))
    return {
        "scene_name": variant_quality["scene_name"],
        "source_track_id": int(variant_quality["source_track_id"]),
        "base_variant_id": base_quality["variant_id"],
        "variant_id": variant_quality["variant_id"],
        "variant_type": variant_quality["variant_type"],
        "geometry_delta": {
            "point_count": int(variant_quality["variant_point_count"]) - int(base_quality["variant_point_count"]),
            "gvc_score": float(variant_quality["gvc_score"]) - float(base_quality["gvc_score"]),
            "native_top_iou": float(variant_quality["native_top_iou"]) - float(base_quality["native_top_iou"]),
            "inside_top_native_ratio": float(variant_quality["variant_inside_top_native_ratio"]) - float(base_quality["variant_inside_top_native_ratio"]),
            "gvc_selected_match_ratio": float(variant_quality["gvc_selected_match_ratio"]) - float(base_quality["gvc_selected_match_ratio"]),
        },
        "semantic_delta": {
            "top_class_changed": bool(base_top >= 0 and variant_top >= 0 and base_top != variant_top),
            "vote_margin": float(variant_semantic["semantic_vote_margin"]) - float(base_semantic["semantic_vote_margin"]),
            "normalized_entropy": float(variant_semantic["semantic_normalized_entropy"]) - float(base_semantic["semantic_normalized_entropy"]),
            "top_class_view_ratio": float(variant_semantic["semantic_top_class_view_ratio"]) - float(base_semantic["semantic_top_class_view_ratio"]),
            "distribution_js_divergence": semantic_js_divergence(base_semantic, variant_semantic),
        },
        "native_relation": {
            "base_top_candidate_id": int(base_quality["native_top_candidate_id"]),
            "variant_top_candidate_id": int(variant_quality["native_top_candidate_id"]),
            "same_top_candidate": int(base_quality["native_top_candidate_id"]) == int(variant_quality["native_top_candidate_id"]),
        },
        "decision_state": "成对连续证据；不定义胜者、不输出候选或最终分数。",
    }


def _build_scene(scene_name, args):
    quality_rows = json.loads((args.quality_ledger_root / scene_name / "automatic_sam_variant_quality_ledger.json").read_text())
    semantic_rows = json.loads((args.semantic_ledger_root / scene_name / "automatic_sam_variant_semantic_ledger.json").read_text())
    quality_by_id = {str(row["variant_id"]): row for row in quality_rows}
    semantic_by_id = {str(row["variant_id"]): row for row in semantic_rows}
    base_by_track = {
        int(row["source_track_id"]): row
        for row in quality_rows if row["variant_type"] == "seed_superpoint_closure"
    }
    pairs = []
    skipped = []
    for variant in quality_rows:
        if variant["variant_type"] != "one_hop_positive_evidence_addition":
            continue
        track_id = int(variant["source_track_id"])
        base = base_by_track.get(track_id)
        variant_semantic = semantic_by_id.get(str(variant["variant_id"]))
        base_semantic = semantic_by_id.get(str(base["variant_id"])) if base is not None else None
        if base is None or base_semantic is None or variant_semantic is None:
            skipped.append({"variant_id": variant["variant_id"], "reason": "missing_base_or_semantic_evidence"})
            continue
        pairs.append(_pair_record(base, variant, base_semantic, variant_semantic))
    scene_root = args.output_root / scene_name
    scene_root.mkdir(parents=True, exist_ok=False)
    (scene_root / "automatic_sam_variant_competition_ledger.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "scene_name": scene_name,
        "quality_variant_count": len(quality_rows),
        "semantic_variant_count": len(semantic_rows),
        "pair_count": len(pairs),
        "skipped_pair_count": len(skipped),
        "decision_state": "不读取 GT；不定义胜者、不输出候选或最终分数。",
    }
    (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--quality_ledger_root", type=Path, required=True)
    parser.add_argument("--semantic_ledger_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "quality_ledger_root", "semantic_ledger_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {summary['pair_count']} 条成对证据", flush=True)
    payload = {
        "purpose": "为自动 SAM 一跳变体与基础闭包、native 的竞争保留连续成对证据。",
        "gt_usage": "不读取 GT；不定义胜者、不输出候选或最终分数。",
        "scene_count": len(summaries),
        "pair_count": sum(item["pair_count"] for item in summaries),
        "skipped_pair_count": sum(item["skipped_pair_count"] for item in summaries),
        "params": vars(args),
    }
    (args.output_root / "automatic_sam_variant_competition_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
