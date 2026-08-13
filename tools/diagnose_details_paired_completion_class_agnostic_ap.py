#!/usr/bin/env python3
"""GT-only：比较原始/局部补全 Mask3D 在追加同一全部共识 v0 后的 AP。"""

import argparse
import gc
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_details_core_prompt_native_relation_ledger import (
    _native_cache_contract,
    _read_scenes,
)
from diagnose_gvc_class_agnostic_ap import _evaluate_variant


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _refined_cache_contract(root, source_root, scene_count):
    path = root / "paired_mask3d_completion_manifest.json"
    if not path.is_file():
        raise ValueError(f"补全缓存缺少 manifest：{path}")
    manifest = json.loads(path.read_text())
    mode = manifest.get("candidate_inputs", {}).get("mode")
    if mode != "mask3d_yoloworld_paired_completion":
        raise ValueError(f"补全缓存模式不符：{mode}")
    params = manifest.get("params", {})
    if Path(params.get("native_prediction_cache", "")) != source_root:
        raise ValueError("补全缓存的 source native 路径与本次对照不一致")
    if int(manifest.get("scene_count", -1)) != int(scene_count):
        raise ValueError("补全缓存的场景数与本次对照不一致")
    integrity = manifest.get("integrity", {})
    required_integrity = {
        "candidate_count_unchanged",
        "classes_unchanged",
        "scores_unchanged",
        "masks_only_grow",
        "unpaired_masks_unchanged",
        "consensus_v0_not_read_or_modified",
    }
    if set(integrity) != required_integrity or not all(integrity.values()):
        raise ValueError("补全缓存未通过完整性合同")
    return manifest


def _delta(right, left):
    return {key: float(right[key] - left[key]) for key in left}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--refined-prediction-cache", type=Path, required=True)
    parser.add_argument("--consensus-v0-track-root", type=Path, required=True)
    parser.add_argument("--track-score-field", default="mean_node_quality")
    parser.add_argument(
        "--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics；本工具只能用于事后诊断。")
    for name in (
        "scene_list",
        "native_prediction_cache",
        "refined_prediction_cache",
        "consensus_v0_track_root",
        "gt_instance_dir",
        "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"输出目录已存在，为避免覆盖已拒绝执行：{args.output_dir}")
    scenes = _read_scenes(args.scene_list)
    _native_cache_contract(args.native_prediction_cache, "mask3d_yoloworld_only")
    refined_manifest = _refined_cache_contract(
        args.refined_prediction_cache, args.native_prediction_cache, len(scenes)
    )
    for root in (
        args.native_prediction_cache,
        args.refined_prediction_cache,
        args.consensus_v0_track_root,
        args.gt_instance_dir,
    ):
        if not root.is_dir():
            raise ValueError(f"缺少输入目录：{root}")
    args.output_dir.mkdir(parents=True)

    baseline = _evaluate_variant(
        "mask3d_plus_all_consensus_v0",
        args.native_prediction_cache,
        scenes,
        args.gt_instance_dir,
        args.output_dir,
        track_root=args.consensus_v0_track_root,
        score_field=args.track_score_field,
    )
    gc.collect()
    refined = _evaluate_variant(
        "paired_completed_mask3d_plus_all_consensus_v0",
        args.refined_prediction_cache,
        scenes,
        args.gt_instance_dir,
        args.output_dir,
        track_root=args.consensus_v0_track_root,
        score_field=args.track_score_field,
    )
    payload = {
        "diagnostic_type": "GT-only 类别无关实例 AP；不是开放词汇主结果。",
        "decision_constraint": (
            "只比较预先物化的固定缓存；GT 不得回流至匹配、mask、类别、分数或阈值。"
        ),
        "scene_count": len(scenes),
        "consensus_v0_contract": "两侧追加同一 track root 的全部轨迹，分数不变。",
        "paired_completion_counts": {
            key: refined_manifest[key]
            for key in (
                "core_prompt_changed_track_count",
                "mutual_best_positive_pair_count",
                "effective_refined_candidate_count",
                "actual_added_point_count",
            )
        },
        "mask3d_plus_all_consensus_v0": baseline,
        "paired_completed_mask3d_plus_all_consensus_v0": refined,
        "delta_paired_completion_minus_baseline": _delta(refined, baseline),
        "params": {key: value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
