#!/usr/bin/env python3
"""核验 official100 冻结后“基线分组原始分数 + 轨迹质量分数”的 safety60 AP。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE, read_scene_list  # noqa: E402
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    _merge_scan_matches,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    EXPECTED_COMBINED,
    EXPECTED_NATIVE,
    _evaluate,
    _load_track_points,
    _prediction,
    _set_match_scores,
    _sha256,
    _uuid_score_map,
    assert_baseline,
    fit_quality_model,
    load_safety_scene,
)
from tools.evaluate_official100_geometry_group_ranking_oof_ap import (  # noqa: E402
    build_group_aware_scores,
    geometry_groups_and_audit,
)
from tools.train_candidate_quality_head_oof import canonicalize_predictions, feature_matrix  # noqa: E402


FEATURE_GROUP = "D_plus_gvc"
PROTOCOL_NAME = "official100_v2"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def run(args: argparse.Namespace) -> dict:
    training_scenes = read_scene_list(args.official_scene_list)
    safety_scenes = read_scene_list(args.safety_scene_list)
    overlap = sorted(set(training_scenes) & set(safety_scenes))
    if len(training_scenes) != 100 or len(safety_scenes) != 60 or overlap:
        raise ValueError("本实验要求零重叠的 official100 与 safety60")
    official_oof_summary = json.loads(args.official_oof_summary.read_text())
    official_selected = official_oof_summary["AP结果"]["combined_baseline_original_track_quality"]
    official_delta = official_oof_summary["变化"]["组合_仅轨迹使用质量_减_双方原始分数"]
    if not all(official_delta[key] > 0.0 for key in ("ap", "ap50", "ap25")):
        raise ValueError("official100 折外三项未同时为正，拒绝进入 safety60")

    model, feature_names, model_contract = fit_quality_model(
        args.official_scene_list, args.records_root, args.random_seed
    )
    _configure_scannet200_instance_eval()
    baseline_matches, combined_matches = {}, {}
    score_maps = {
        "baseline_original": {},
        "combined_original": {},
        "combined_group_original": {},
        "combined_group_original_track_quality": {},
    }
    identity_rows, group_rows_all, score_rows = [], [], []
    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        for scene_index, scene in enumerate(safety_scenes, start=1):
            masks, baseline_original, baseline_classes, rows, tracks, audit = load_safety_scene(
                scene, args.native_cache, args.safety_ledger_root, args.track_root
            )
            matrix, names = feature_matrix(rows, FEATURE_GROUP, PROTOCOL_NAME)
            if names != feature_names:
                raise ValueError("safety60 特征顺序与 official100 模型不一致")
            predicted_quality = canonicalize_predictions("q", model.predict(matrix))
            baseline_count = masks.shape[1]
            baseline_rows = rows[:baseline_count]
            track_rows = rows[baseline_count:]
            baseline_quality = predicted_quality[:baseline_count]
            track_quality = predicted_quality[baseline_count:]

            groups, group_audit = geometry_groups_and_audit(
                masks,
                [int(row["point_count"]) for row in baseline_rows],
                None,
            )
            group_original, _, scene_group_rows = build_group_aware_scores(
                baseline_original, baseline_quality, groups
            )
            for row in scene_group_rows:
                row["scene_name"] = scene
            group_rows_all.extend(scene_group_rows)

            track_original = np.asarray(
                [float(track["mean_node_quality"]) for track in tracks], dtype=np.float64
            )
            track_masks = np.zeros((masks.shape[0], len(tracks)), dtype=bool)
            for column_index, track in enumerate(tracks):
                points, _ = _load_track_points(track, masks.shape[0])
                track_masks[points, column_index] = True

            gt_file = str(args.gt_dir / f"{scene}.txt")
            baseline_gt, baseline_pred = instance_eval.assign_instances_for_scan(
                _prediction(masks, baseline_original, baseline_count), gt_file
            )
            baseline_point_counts = [int(row["point_count"]) for row in baseline_rows]
            scene_baseline_original = _uuid_score_map(
                baseline_pred, baseline_original, baseline_point_counts
            )
            scene_baseline_group_original = _uuid_score_map(
                baseline_pred, group_original, baseline_point_counts
            )
            score_maps["baseline_original"].update(scene_baseline_original)
            baseline_matches[os.path.abspath(gt_file)] = {
                "gt": copy.deepcopy(baseline_gt), "pred": copy.deepcopy(baseline_pred)
            }

            track_gt, track_pred = instance_eval.assign_instances_for_scan(
                _prediction(track_masks, track_original, len(tracks)), gt_file
            )
            track_point_counts = [int(row["point_count"]) for row in track_rows]
            scene_track_original = _uuid_score_map(track_pred, track_original, track_point_counts)
            scene_track_quality = _uuid_score_map(track_pred, track_quality, track_point_counts)
            merged_gt, merged_pred = _merge_scan_matches(
                baseline_gt, baseline_pred, track_gt, track_pred
            )
            combined_matches[os.path.abspath(gt_file)] = {"gt": merged_gt, "pred": merged_pred}
            score_maps["combined_original"].update(scene_baseline_original)
            score_maps["combined_original"].update(scene_track_original)
            score_maps["combined_group_original"].update(scene_baseline_group_original)
            score_maps["combined_group_original"].update(scene_track_original)
            score_maps["combined_group_original_track_quality"].update(
                scene_baseline_group_original
            )
            score_maps["combined_group_original_track_quality"].update(scene_track_quality)

            identity_rows.append({**audit, **group_audit})
            representative_ids = {
                row["representative_candidate_id"] for row in scene_group_rows
            }
            for candidate_id in range(baseline_count):
                score_rows.append({
                    "scene_name": scene,
                    "candidate_source": "基线候选",
                    "candidate_id": candidate_id,
                    "is_geometry_group_representative": candidate_id in representative_ids,
                    "original_score": float(baseline_original[candidate_id]),
                    "selected_score": float(group_original[candidate_id]),
                })
            for index, row in enumerate(track_rows):
                score_rows.append({
                    "scene_name": scene,
                    "candidate_source": "轨迹候选",
                    "candidate_id": int(row["candidate_id"]),
                    "is_geometry_group_representative": True,
                    "original_score": float(track_original[index]),
                    "selected_score": float(track_quality[index]),
                })
            print(f"[冻结核验] {scene_index}/60 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    _set_match_scores(baseline_matches, score_maps["baseline_original"])
    baseline = _evaluate("baseline_original", baseline_matches, args.output_dir)
    assert_baseline(baseline, EXPECTED_NATIVE, args.baseline_tolerance)
    _set_match_scores(combined_matches, score_maps["combined_original"])
    combined_original = _evaluate("combined_original", combined_matches, args.output_dir)
    assert_baseline(combined_original, EXPECTED_COMBINED, args.baseline_tolerance)
    _set_match_scores(combined_matches, score_maps["combined_group_original"])
    combined_group_original = _evaluate(
        "combined_group_original", combined_matches, args.output_dir
    )
    _set_match_scores(
        combined_matches, score_maps["combined_group_original_track_quality"]
    )
    selected = _evaluate(
        "combined_group_original_track_quality", combined_matches, args.output_dir
    )

    _write_jsonl(args.output_dir / "geometry_groups.jsonl", group_rows_all)
    _write_jsonl(args.output_dir / "candidate_scores.jsonl", score_rows)
    _write_jsonl(args.output_dir / "scene_identity_audit.jsonl", identity_rows)
    summary = {
        "实验类型": "official100 冻结后的 safety60 一次性类别无关 AP 核验",
        "固定版本": "基线候选按完全相同掩码组保留原始分数代表；轨迹候选使用 official100 全量训练质量分数",
        "约束": "不扫描权重、阈值或其他分数；不删除候选，不修改掩码、类别、数量或几何",
        "official100折外入选结果": official_selected,
        "official100折外入选变化": official_delta,
        "训练与评测场景重叠": overlap,
        "模型合同": model_contract,
        "AP结果": {
            "基线候选_原始分数": baseline,
            "基线加轨迹_双方原始分数": combined_original,
            "基线分组原始分数加轨迹原始分数": combined_group_original,
            "基线分组原始分数加轨迹质量分数": selected,
        },
        "最终变化": {
            key: selected[key] - combined_original[key] for key in combined_original
        },
        "相对分组原始分数变化": {
            key: selected[key] - combined_group_original[key]
            for key in combined_group_original
        },
        "输入摘要": {
            "official100场景清单_SHA256": _sha256(args.official_scene_list),
            "safety60场景清单_SHA256": _sha256(args.safety_scene_list),
            "official100折外汇总_SHA256": _sha256(args.official_oof_summary),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-scene-list", type=Path, required=True)
    parser.add_argument("--official-oof-summary", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--safety-scene-list", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--safety-ledger-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260808)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-12)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics")
    for name in (
        "official_scene_list", "official_oof_summary", "records_root",
        "safety_scene_list", "native_cache", "safety_ledger_root",
        "track_root", "gt_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
