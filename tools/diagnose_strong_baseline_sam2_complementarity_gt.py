#!/usr/bin/env python3
"""仅离线 GT 诊断：量化完整强基线对 SAM2 的真实互补性与融合去向。

强基线原始候选由 Mask3D、SAM-fused 和 BPR 组成。GT 仅用于事后比较每个
真实实例的最佳几何 IoU；候选来源、融合报告和拒绝原因均来自推理时已有元数据。
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _canonical_path(path):
    if not path:
        return ""
    return str(_resolve(path).resolve())


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _load_masks(root, scene_name, point_count):
    payload = torch.load(root / f"{scene_name}.pt", map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = _as_numpy(masks).astype(bool, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的 Mask3D mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的 Mask3D 点数不一致：{masks.shape[0]} 与 {point_count}")
    return masks


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        semantic_id = instance_id // 1000
        if instance_id <= 0 or semantic_id not in valid_classes:
            continue
        indices = np.flatnonzero(gt_ids == instance_id).astype(np.int32)
        if len(indices) < min_region_size:
            continue
        instances.append(
            {
                "gt_instance_id": instance_id,
                "gt_class": str(ID_TO_LABEL.get(semantic_id, semantic_id)),
                "indices": indices,
                "point_count": int(len(indices)),
            }
        )
    return gt_ids, instances


def _load_candidates(root, scene_name, source_kind, point_count):
    source_json = root / scene_name / "backprojection_candidates.json"
    if not source_json.exists():
        return []
    candidates = []
    for item in json.loads(source_json.read_text()).get("candidates", []):
        seed_path = _resolve(item["seed_points_path"])
        indices = np.unique(np.load(seed_path)["point_indices"].astype(np.int64))
        indices = indices[(indices >= 0) & (indices < point_count)].astype(np.int32, copy=False)
        if not len(indices):
            continue
        candidates.append(
            {
                "candidate_id": int(item.get("candidate_id", -1)),
                "class_name": str(item.get("class_name", "")),
                "score": float(item.get("score", 0.0)),
                "indices": indices,
                "point_count": int(len(indices)),
                "source_kind": source_kind,
                "source_json": str(source_json.resolve()),
            }
        )
    return candidates


def _best_mask_iou(masks, sizes, gt_indices, gt_size):
    if masks.shape[1] == 0:
        return {"iou": 0.0, "candidate_id": -1, "class_name": ""}
    intersections = masks[gt_indices].sum(axis=0, dtype=np.int64)
    values = intersections / np.maximum(1, sizes + gt_size - intersections)
    index = int(np.argmax(values))
    return {"iou": float(values[index]), "candidate_id": index, "class_name": ""}


def _best_candidate_iou(gt_ids, gt_instance_id, gt_size, candidates):
    best = {"iou": 0.0, "candidate_id": -1, "class_name": "", "source_json": "", "source_kind": ""}
    for candidate in candidates:
        intersection = int(np.count_nonzero(gt_ids[candidate["indices"]] == gt_instance_id))
        iou = intersection / max(1, candidate["point_count"] + gt_size - intersection)
        if iou > best["iou"]:
            best = {
                "iou": float(iou),
                "candidate_id": candidate["candidate_id"],
                "class_name": candidate["class_name"],
                "source_json": candidate["source_json"],
                "source_kind": candidate["source_kind"],
            }
    return best


def _report_entries(fusion_report, sam2_kind):
    data = json.loads(fusion_report.read_text())
    result = {}
    for scene_name, scene_report in data.get("scene_reports", {}).items():
        report = scene_report.get("backprojection", scene_report)
        for outcome in ("applied", "skipped"):
            for item in report.get(outcome, []):
                if str(item.get("source_kind", "")) != sam2_kind:
                    continue
                key = (
                    scene_name,
                    int(item.get("candidate_id", -1)),
                    _canonical_path(item.get("source_json")),
                )
                entry = {
                    "outcome": outcome,
                    "reason": str(item.get("reason", "applied" if outcome == "applied" else "unknown")),
                }
                previous = result.get(key)
                if previous is None or outcome == "applied":
                    result[key] = entry
    return result


def _rejection_stage(outcome, reason):
    if outcome == "applied":
        return "已接入"
    if reason in {"source_limit", "class_limit"}:
        return "来源或类别预算"
    if reason in {"matched_existing_3d_mask", "mostly_covered_by_existing_masks", "grown_mask_matches_existing"}:
        return "与已有实例重叠"
    if reason in {"duplicate_new_proposal"}:
        return "与新增候选重复"
    if reason in {"low_score", "low_fusion_score", "low_candidate_quality_score", "low_scene_source_quality_z"}:
        return "分数或质量门控"
    if "superpoint" in reason or "connected_component" in reason or reason in {"small_grown_mask", "few_seed_points", "missing_or_small_seed_file"}:
        return "几何或 superpoint 精炼"
    if "view" in reason or "consistency" in reason or "label" in reason or "box" in reason:
        return "多视角或二维一致性"
    return "其他或历史报告未记录"


def _coverage_group(strong_iou, sam2_iou, threshold):
    strong = strong_iou >= threshold
    sam2 = sam2_iou >= threshold
    if not strong and sam2:
        return "强基线漏检但SAM2覆盖"
    if strong and sam2:
        return "两者均覆盖"
    if strong:
        return "强基线覆盖但SAM2未覆盖"
    return "两者均未覆盖"


def _summary(rows, threshold, suffix):
    complement = [row for row in rows if row[f"coverage_{suffix}"] == "强基线漏检但SAM2覆盖"]
    source_groups = Counter(row[f"strong_best_source_{suffix}"] for row in rows)
    fusion_outcomes = Counter(row["sam2_best_fusion_outcome"] for row in complement)
    rejection_reasons = Counter(
        row["sam2_best_fusion_reason"] for row in complement if row["sam2_best_fusion_outcome"] == "skipped"
    )
    rejection_stages = Counter(
        row["sam2_best_fusion_stage"] for row in complement if row["sam2_best_fusion_outcome"] == "skipped"
    )
    return {
        "coverage_groups": dict(sorted(Counter(row[f"coverage_{suffix}"] for row in rows).items())),
        "strong_baseline_best_source_counts": dict(sorted(source_groups.items())),
        "sam2_genuine_complement_count": len(complement),
        "sam2_genuine_complement_rate": float(len(complement) / max(1, len(rows))),
        "sam2_complement_semantic_exact_accuracy": float(
            sum(row["sam2_semantic_exact"] for row in complement) / max(1, len(complement))
        ),
        "sam2_best_candidate_fusion_outcomes_for_complements": dict(sorted(fusion_outcomes.items())),
        "sam2_best_candidate_rejection_reasons_for_complements": dict(sorted(rejection_reasons.items())),
        "sam2_best_candidate_rejection_stages_for_complements": dict(sorted(rejection_stages.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, required=True)
    parser.add_argument("--sam_fused_root", type=Path, required=True)
    parser.add_argument("--bpr_root", type=Path, required=True)
    parser.add_argument("--sam2_root", type=Path, required=True)
    parser.add_argument("--fusion_report", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sam2_source_kind", default="sam2_details_mvpdist")
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    for name in ("baseline_masks_root", "sam_fused_root", "bpr_root", "sam2_root", "fusion_report", "gt_instance_dir", "output_dir"):
        value = getattr(args, name)
        setattr(args, name, _resolve(value))
    scenes = _read_scenes(_resolve(args.scene_list))
    fusion_entries = _report_entries(args.fusion_report, args.sam2_source_kind)
    rows = []
    source_candidate_counts = Counter()
    for scene_name in scenes:
        gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        masks = _load_masks(args.baseline_masks_root, scene_name, len(gt_ids))
        mask_sizes = masks.sum(axis=0, dtype=np.int64)
        sam_fused = _load_candidates(args.sam_fused_root, scene_name, "sam_fused", len(gt_ids))
        bpr = _load_candidates(args.bpr_root, scene_name, "bpr", len(gt_ids))
        sam2 = _load_candidates(args.sam2_root, scene_name, args.sam2_source_kind, len(gt_ids))
        source_candidate_counts.update({"mask3d": int(masks.shape[1]), "sam_fused": len(sam_fused), "bpr": len(bpr), "sam2": len(sam2)})
        for gt in instances:
            mask3d = _best_mask_iou(masks, mask_sizes, gt["indices"], gt["point_count"])
            fused = _best_candidate_iou(gt_ids, gt["gt_instance_id"], gt["point_count"], sam_fused)
            bpr_best = _best_candidate_iou(gt_ids, gt["gt_instance_id"], gt["point_count"], bpr)
            sam2_best = _best_candidate_iou(gt_ids, gt["gt_instance_id"], gt["point_count"], sam2)
            strong_sources = {"mask3d": mask3d, "sam_fused": fused, "bpr": bpr_best}
            strong_source, strong_best = max(strong_sources.items(), key=lambda item: item[1]["iou"])
            trace_key = (scene_name, sam2_best["candidate_id"], sam2_best["source_json"])
            trace = fusion_entries.get(trace_key)
            if trace is None and sam2_best["candidate_id"] >= 0:
                trace = {"outcome": "untraced", "reason": "历史报告未保留该候选来源"}
            outcome = trace["outcome"] if trace else "no_sam2_candidate"
            reason = trace["reason"] if trace else "无SAM2候选"
            row = {
                "scene_name": scene_name,
                "gt_instance_id": gt["gt_instance_id"],
                "gt_class": gt["gt_class"],
                "gt_point_count": gt["point_count"],
                "mask3d_best_iou": mask3d["iou"],
                "sam_fused_best_iou": fused["iou"],
                "bpr_best_iou": bpr_best["iou"],
                "strong_baseline_best_iou": strong_best["iou"],
                "strong_baseline_best_source": strong_source,
                "sam2_best_iou": sam2_best["iou"],
                "sam2_best_candidate_id": sam2_best["candidate_id"],
                "sam2_best_class": sam2_best["class_name"],
                "sam2_semantic_exact": bool(sam2_best["class_name"] == gt["gt_class"]),
                "sam2_best_fusion_outcome": outcome,
                "sam2_best_fusion_reason": reason,
                "sam2_best_fusion_stage": _rejection_stage(outcome, reason),
            }
            for threshold, suffix in ((0.25, "iou25"), (0.50, "iou50")):
                row[f"coverage_{suffix}"] = _coverage_group(strong_best["iou"], sam2_best["iou"], threshold)
                row[f"strong_best_source_{suffix}"] = strong_source if strong_best["iou"] >= threshold else "未覆盖"
            rows.append(row)
        print(f"[场景完成] {scene_name}: {len(instances)} 个有效 GT 实例", flush=True)
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、候选生成、融合、打分或阈值选择。",
        "purpose": "统计 Mask3D + SAM-fused + BPR 相对 SAM2 的实例级几何互补性，并追踪 SAM2 最佳候选的融合去向。",
        "scene_count": len(scenes),
        "num_gt_instances": len(rows),
        "raw_candidate_counts": dict(sorted(source_candidate_counts.items())),
        "fusion_report": str(args.fusion_report),
        "fusion_trace_note": "只有融合报告带有 source_kind/source_json 的候选可精确归因；untraced 表示历史报告缺少来源元数据。",
        "iou25": _summary(rows, 0.25, "iou25"),
        "iou50": _summary(rows, 0.50, "iou50"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "strong_baseline_sam2_complementarity.csv").open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
