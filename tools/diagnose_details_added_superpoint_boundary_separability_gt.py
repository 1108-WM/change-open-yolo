#!/usr/bin/env python3
"""GT-only：诊断新增 superpoint 边界证据能否区分改善与过扩张。

输入是冻结的无 GT 原子账本和固定 Mask3D 补全归因。工具按
``(scene_name, track_id)`` 将原子聚合回候选，只比较目标一致改善与目标一致
过扩张两组的特征方向和分布。它不扫描阈值、不生成规则、不修改候选，也不运行 AP。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENT = "aligned_improvement"
OVEREXPANSION = {
    "aligned_overexpansion_original_iou_ge50",
    "aligned_overexpansion_original_iou_lt50",
}

# 固定候选级特征。AUC 只描述高值是否更常出现在改善组，不选择阈值。
FEATURES = (
    "common_core_native_iou",
    "expansion_ratio",
    "added_point_count",
    "added_superpoint_count",
    "already_inside_native_point_ratio",
    "independent_joint_visible_atom_ratio",
    "reliable_anchor_atom_ratio",
    "positive_anchor_atom_ratio",
    "exclusion_anchor_atom_ratio",
    "independent_joint_visible_point_ratio",
    "reliable_anchor_point_ratio",
    "positive_anchor_point_ratio",
    "exclusion_anchor_point_ratio",
    "delta_mean_best_siou_point_weighted_all",
    "delta_mean_best_siou_point_weighted_evidenced",
    "delta_support_frame_rate_point_weighted_all",
    "delta_support_frame_rate_point_weighted_evidenced",
    "positive_anchor_frame_rate",
    "exclusion_anchor_frame_rate",
    "direct_core_contact_point_ratio",
    "direct_core_contact_density_point_weighted",
    "boundary_distance_point_weighted",
    "normal_difference_point_weighted",
    "color_difference_point_weighted",
    "reachable_from_core_point_ratio",
    "graph_hops_point_weighted_reachable",
    "prompt_support_frame_count_point_weighted",
    "other_score_one_native_coverage_point_weighted",
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _as_float(row, key):
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _weighted_mean(rows, field, weight_field="actual_added_point_count", predicate=None):
    selected = []
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        value = _as_float(row, field)
        if value is None:
            continue
        weight = float(row[weight_field])
        if weight > 0:
            selected.append((value, weight))
    if not selected:
        return None
    denominator = sum(weight for _, weight in selected)
    return float(sum(value * weight for value, weight in selected) / denominator)


def _weighted_ratio(rows, predicate, weight_field="actual_added_point_count"):
    denominator = sum(float(row[weight_field]) for row in rows)
    if denominator <= 0:
        return None
    numerator = sum(
        float(row[weight_field]) for row in rows if predicate(row)
    )
    return float(numerator / denominator)


def _atom_ratio(rows, predicate):
    return float(sum(predicate(row) for row in rows) / len(rows)) if rows else None


def _outcome_label(category):
    if category == IMPROVEMENT:
        return "improvement"
    if category in OVEREXPANSION:
        return "overexpansion"
    return None


def prepare_atoms(atoms, candidate_id):
    """核对 native 所有权，并计算每个 superpoint 对该候选的真实新增点数。"""
    output = []
    for source in atoms:
        row = dict(source)
        owner = int(row["core_best_native_candidate_id"])
        if owner != int(candidate_id):
            raise ValueError(
                f"{row['scene_name']} track {row['track_id']} 的核心 owner {owner} "
                f"与固定补全候选 {candidate_id} 不一致"
            )
        point_count = int(row["added_point_count"])
        inside_ratio = float(row["added_inside_core_best_native_ratio"])
        inside_count = int(round(point_count * inside_ratio))
        if not 0 <= inside_count <= point_count:
            raise ValueError("新增 superpoint 的 native 内部点数非法")
        row["already_inside_native_point_count"] = inside_count
        row["actual_added_point_count"] = point_count - inside_count
        output.append(row)
    return output


def aggregate_candidate(gt_row, atoms):
    candidate_id = int(gt_row["native_candidate_id"])
    atoms = prepare_atoms(atoms, candidate_id)
    expected_added = int(gt_row["added_point_count"])
    actual_added = sum(int(row["actual_added_point_count"]) for row in atoms)
    if actual_added != expected_added:
        raise ValueError(
            f"{gt_row['scene_name']} track {gt_row['track_id']} 的原子新增点 "
            f"{actual_added} 与固定候选 {expected_added} 不一致"
        )
    declared_points = sum(int(row["added_point_count"]) for row in atoms)
    effective_atoms = [row for row in atoms if int(row["actual_added_point_count"]) > 0]
    total_reliable_frames = sum(
        int(row["reliable_anchor_frame_count"]) for row in effective_atoms
    )
    total_positive_frames = sum(
        int(row["positive_anchor_frame_count"]) for row in effective_atoms
    )
    total_exclusion_frames = sum(
        int(row["exclusion_anchor_frame_count"]) for row in effective_atoms
    )
    evidenced = lambda row: int(row["joint_visible_independent_frame_count"]) > 0
    reliable = lambda row: int(row["reliable_anchor_frame_count"]) > 0
    positive = lambda row: int(row["positive_anchor_frame_count"]) > 0
    exclusion = lambda row: int(row["exclusion_anchor_frame_count"]) > 0
    direct_contact = lambda row: int(row["direct_core_neighbor_count"]) > 0
    reachable = lambda row: bool(row["reachable_from_core_via_prompt_added"])
    label = _outcome_label(gt_row["failure_category"])
    return {
        "scene_name": gt_row["scene_name"],
        "track_id": int(gt_row["track_id"]),
        "native_candidate_id": candidate_id,
        "outcome": label,
        "failure_category": gt_row["failure_category"],
        "refined_minus_original_target_iou": float(
            gt_row["refined_minus_original_target_iou"]
        ),
        "common_core_native_iou": float(gt_row["common_core_native_iou"]),
        "expansion_ratio": float(gt_row["expansion_ratio"]),
        "added_point_count": expected_added,
        "added_superpoint_count": len(effective_atoms),
        "declared_added_superpoint_count": len(atoms),
        "already_inside_native_point_ratio": float(
            (declared_points - actual_added) / max(1, declared_points)
        ),
        "independent_joint_visible_atom_ratio": _atom_ratio(effective_atoms, evidenced),
        "reliable_anchor_atom_ratio": _atom_ratio(effective_atoms, reliable),
        "positive_anchor_atom_ratio": _atom_ratio(effective_atoms, positive),
        "exclusion_anchor_atom_ratio": _atom_ratio(effective_atoms, exclusion),
        "independent_joint_visible_point_ratio": _weighted_ratio(atoms, evidenced),
        "reliable_anchor_point_ratio": _weighted_ratio(atoms, reliable),
        "positive_anchor_point_ratio": _weighted_ratio(atoms, positive),
        "exclusion_anchor_point_ratio": _weighted_ratio(atoms, exclusion),
        "delta_mean_best_siou_point_weighted_all": _weighted_mean(
            atoms, "delta_mean_best_siou"
        ),
        "delta_mean_best_siou_point_weighted_evidenced": _weighted_mean(
            atoms, "delta_mean_best_siou", predicate=evidenced
        ),
        "delta_support_frame_rate_point_weighted_all": _weighted_mean(
            atoms, "delta_support_frame_rate"
        ),
        "delta_support_frame_rate_point_weighted_evidenced": _weighted_mean(
            atoms, "delta_support_frame_rate", predicate=evidenced
        ),
        "positive_anchor_frame_rate": (
            float(total_positive_frames / total_reliable_frames)
            if total_reliable_frames else None
        ),
        "exclusion_anchor_frame_rate": (
            float(total_exclusion_frames / total_reliable_frames)
            if total_reliable_frames else None
        ),
        "direct_core_contact_point_ratio": _weighted_ratio(atoms, direct_contact),
        "direct_core_contact_density_point_weighted": _weighted_mean(
            atoms, "max_direct_core_contact_density", predicate=direct_contact
        ),
        "boundary_distance_point_weighted": _weighted_mean(
            atoms, "contact_weighted_boundary_distance"
        ),
        "normal_difference_point_weighted": _weighted_mean(
            atoms, "contact_weighted_normal_difference"
        ),
        "color_difference_point_weighted": _weighted_mean(
            atoms, "contact_weighted_color_difference"
        ),
        "reachable_from_core_point_ratio": _weighted_ratio(atoms, reachable),
        "graph_hops_point_weighted_reachable": _weighted_mean(
            atoms, "graph_hops_from_core", predicate=reachable
        ),
        "prompt_support_frame_count_point_weighted": _weighted_mean(
            atoms, "prompt_support_frame_count"
        ),
        "other_score_one_native_coverage_point_weighted": _weighted_mean(
            atoms, "best_other_score_one_native_coverage_ratio"
        ),
    }, atoms


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _auc_high_value(improvement_values, overexpansion_values):
    positive = np.asarray(improvement_values, dtype=np.float64)
    negative = np.asarray(overexpansion_values, dtype=np.float64)
    if not len(positive) or not len(negative):
        return None
    comparisons = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positive) * len(negative)))


def summarize_features(rows):
    payload = {}
    for feature in FEATURES:
        improvement = [
            float(row[feature]) for row in rows
            if row["outcome"] == "improvement" and row[feature] is not None
        ]
        overexpansion = [
            float(row[feature]) for row in rows
            if row["outcome"] == "overexpansion" and row[feature] is not None
        ]
        auc = _auc_high_value(improvement, overexpansion)
        payload[feature] = {
            "improvement": _stats(improvement),
            "overexpansion": _stats(overexpansion),
            "high_value_toward_improvement_auc": auc,
            "rank_effect_high_toward_improvement": (
                float(2.0 * auc - 1.0) if auc is not None else None
            ),
            "descriptive_direction": (
                None if auc is None else
                "high_toward_improvement" if auc > 0.5 else
                "low_toward_improvement" if auc < 0.5 else "no_monotonic_direction"
            ),
        }
    return payload


def summarize_fixed_scene_splits(rows):
    return {
        split: {
            "candidate_counts": {
                outcome: sum(
                    row["scene_split"] == split and row["outcome"] == outcome
                    for row in rows
                )
                for outcome in ("improvement", "overexpansion")
            },
            "feature_distributions": summarize_features([
                row for row in rows if row["scene_split"] == split
            ]),
        }
        for split in ("A", "B")
    }


def _load_atomic_map(root):
    output = {}
    for path in sorted(root.glob("scene*/added_superpoint_boundary_ledger.json")):
        rows = json.loads(path.read_text())
        for row in rows:
            key = (row["scene_name"], int(row["track_id"]))
            output.setdefault(key, []).append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-ledger-root", type=Path, required=True)
    parser.add_argument("--completion-attribution-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics；GT 只能用于离线可分性诊断。")
    for name in ("boundary_ledger_root", "completion_attribution_csv", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"输出目录已存在，为避免覆盖已拒绝执行：{args.output_dir}")
    summary_path = (
        args.boundary_ledger_root / "added_superpoint_boundary_ledger_summary.json"
    )
    if not summary_path.is_file() or not args.completion_attribution_csv.is_file():
        raise SystemExit("缺少完整无 GT 原子账本或固定 GT 归因 CSV")
    ledger_summary = json.loads(summary_path.read_text())
    scene_order = [row["scene_name"] for row in ledger_summary["scenes"]]
    if len(scene_order) != len(set(scene_order)):
        raise ValueError("原子账本 summary 含重复场景")
    scene_split = {
        scene_name: "A" if index % 2 == 0 else "B"
        for index, scene_name in enumerate(scene_order)
    }
    atomic_map = _load_atomic_map(args.boundary_ledger_root)
    with args.completion_attribution_csv.open(newline="") as handle:
        gt_rows = list(csv.DictReader(handle))

    candidates = []
    output_atoms = []
    mapped_all_candidate_count = 0
    for gt_row in gt_rows:
        key = (gt_row["scene_name"], int(gt_row["track_id"]))
        atoms = atomic_map.get(key)
        if not atoms:
            raise ValueError(f"固定补全候选 {key} 在原子账本中缺失")
        candidate, prepared_atoms = aggregate_candidate(gt_row, atoms)
        mapped_all_candidate_count += 1
        if candidate["outcome"] is None:
            continue
        candidate["scene_split"] = scene_split[candidate["scene_name"]]
        candidates.append(candidate)
        for atom in prepared_atoms:
            output_atoms.append({
                "scene_name": atom["scene_name"],
                "track_id": int(atom["track_id"]),
                "native_candidate_id": int(gt_row["native_candidate_id"]),
                "outcome": candidate["outcome"],
                "failure_category": gt_row["failure_category"],
                "added_superpoint_id": int(atom["added_superpoint_id"]),
                "added_point_count": int(atom["added_point_count"]),
                "actual_added_point_count": int(atom["actual_added_point_count"]),
                **{
                    key: atom[key] for key in atom
                    if key not in {
                        "scene_name", "track_id", "added_superpoint_id",
                        "added_point_count", "frames", "decision_state",
                    }
                },
            })

    counts = {
        label: sum(row["outcome"] == label for row in candidates)
        for label in ("improvement", "overexpansion")
    }
    if counts != {"improvement": 24, "overexpansion": 53}:
        raise ValueError(f"固定目标一致组数量改变：{counts}")
    payload = {
        "diagnostic_type": "GT-only 固定候选边界证据方向/分布诊断；不是阈值搜索或 AP 分解。",
        "decision_constraint": (
            "GT 标签不得回流至证据构造、候选、mask、分数、规则或阈值；"
            "AUC 只描述固定两组的候选级秩方向。"
        ),
        "mapped_fixed_candidate_count": mapped_all_candidate_count,
        "target_aligned_candidate_counts": counts,
        "target_aligned_atomic_row_count": len(output_atoms),
        "target_aligned_effective_atomic_row_count": sum(
            int(row["actual_added_point_count"]) > 0 for row in output_atoms
        ),
        "source_ledger_contract": {
            key: ledger_summary[key] for key in (
                "scene_count", "changed_track_count", "added_superpoint_count",
                "added_point_count", "with_independent_joint_visible_frame_count",
                "with_reliable_anchor_frame_count", "with_positive_anchor_count",
                "with_exclusion_anchor_count", "delta_gvc_positive_count",
                "delta_gvc_negative_count", "integrity",
            )
        },
        "feature_distributions": summarize_features(candidates),
        "fixed_alternating_scene_splits": summarize_fixed_scene_splits(candidates),
        "limitations": [
            "统计单位是 77 个候选；原子行只用于审计，不作为独立样本重复计数。",
            "缺失的独立视角或可靠锚定证据单独保留为覆盖率，不当作零支持。",
            "样本来自 dev30 的已失败固定补全，只能决定该证据分支是否值得预注册，不能证明 AP 提升。",
        ],
        "params": {key: str(value) for key, value in vars(args).items()},
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "candidate_boundary_features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    with (args.output_dir / "atomic_boundary_outcomes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_atoms[0]))
        writer.writeheader()
        writer.writerows(output_atoms)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "mapped_fixed_candidate_count": mapped_all_candidate_count,
        "target_aligned_candidate_counts": counts,
        "target_aligned_atomic_row_count": len(output_atoms),
        "target_aligned_effective_atomic_row_count": sum(
            int(row["actual_added_point_count"]) > 0 for row in output_atoms
        ),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
