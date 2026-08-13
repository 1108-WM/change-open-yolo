#!/usr/bin/env python3
"""导出每条自动 SAM 轨迹的一个 Pareto 几何候选。

此工具只读取无 GT 的变体计划、几何质量、冻结多视图语义和 Pareto 账本。每条
源轨迹最多导出一个候选：仅考虑语义类别有效的 Pareto 非支配变体，依次按较高
类别无关 GVC、较高语义间隔、较低语义熵和稳定 variant_id 选择。native 候选
完全不读取、更不修改；导出结果供 fusion 的独立 append-only 阶段使用。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_KIND = "automatic_sam_pareto_append_only"
SELECTION_POLICY = (
    "每个 source_track 只在语义类别有效的 Pareto 非支配变体中选择一个："
    "GVC 降序、语义间隔降序、归一化熵升序、variant_id 升序。"
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError("场景列表含重复场景")
    return scenes


def raw_superpoint_points(processed):
    raw_ids = np.asarray(processed[:, 9], dtype=np.int64)
    order = np.argsort(raw_ids, kind="mergesort")
    ids, starts = np.unique(raw_ids[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    return {
        int(superpoint_id): np.asarray(order[start:end], dtype=np.int64)
        for superpoint_id, start, end in zip(ids, starts, ends)
    }


def variant_points(variant, superpoint_atoms):
    ids = {int(item) for item in variant["base_superpoint_ids"]}
    ids.update(int(item) for item in variant.get("added_superpoint_ids", []))
    missing = sorted(item for item in ids if item not in superpoint_atoms)
    if missing:
        raise ValueError(f"变体 {variant['variant_id']} 引用了不存在的 superpoint: {missing[:8]}")
    chunks = [superpoint_atoms[item] for item in sorted(ids)]
    return np.concatenate(chunks).astype(np.int64, copy=False) if chunks else np.empty(0, dtype=np.int64)


def _selection_key(record):
    return (
        -float(record["gvc_score"]),
        -float(record["semantic_vote_margin"]),
        float(record["semantic_normalized_entropy"]),
        str(record["variant_id"]),
    )


def select_track_variants(variant_rows, quality_rows, semantic_rows, pareto_rows, labels):
    """返回每条轨迹一个确定性、无 GT 的最终几何/语义记录及跳过审计。"""
    variants = {str(row["variant_id"]): row for row in variant_rows}
    quality = {str(row["variant_id"]): row for row in quality_rows}
    semantic = {str(row["variant_id"]): row for row in semantic_rows}
    eligible_by_track = defaultdict(list)
    skipped = []
    for pareto in pareto_rows:
        variant_id = str(pareto["variant_id"])
        if not bool(pareto.get("pareto_non_dominated", False)):
            skipped.append({"variant_id": variant_id, "reason": "pareto_dominated"})
            continue
        variant = variants.get(variant_id)
        quality_row = quality.get(variant_id)
        semantic_row = semantic.get(variant_id)
        if variant is None or quality_row is None or semantic_row is None:
            skipped.append({"variant_id": variant_id, "reason": "missing_variant_quality_or_semantic"})
            continue
        track_id = int(pareto["source_track_id"])
        if int(variant["source_track_id"]) != track_id:
            skipped.append({"variant_id": variant_id, "reason": "source_track_id_mismatch"})
            continue
        class_id = int(semantic_row.get("semantic_evidence_top_class_index", -1))
        if class_id < 0 or class_id >= len(labels):
            skipped.append({"variant_id": variant_id, "reason": "invalid_semantic_evidence_class"})
            continue
        eligible_by_track[track_id].append({
            "variant": variant,
            "quality": quality_row,
            "semantic": semantic_row,
            "pareto": pareto,
            "variant_id": variant_id,
            "source_track_id": track_id,
            "class_id": class_id,
            "gvc_score": float(quality_row["gvc_score"]),
            "semantic_vote_margin": float(semantic_row["semantic_vote_margin"]),
            "semantic_normalized_entropy": float(semantic_row["semantic_normalized_entropy"]),
        })
    selected = []
    for track_id, records in sorted(eligible_by_track.items()):
        winner = min(records, key=_selection_key)
        winner["pareto_eligible_variant_count"] = len(records)
        winner["selected_from_variant_ids"] = sorted(item["variant_id"] for item in records)
        selected.append(winner)
    return selected, skipped


def _numeric_summary(candidates, field):
    values = np.asarray([float(item.get(field, 0.0) or 0.0) for item in candidates], dtype=np.float64)
    if len(values) == 0:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(values)), "min": float(values.min()), "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)), "p90": float(np.quantile(values, 0.90)), "max": float(values.max()),
    }


def _candidate(scene_name, record, points, labels, seed_path):
    quality, semantic, variant = record["quality"], record["semantic"], record["variant"]
    class_id = int(record["class_id"])
    return {
        "scene_name": scene_name,
        "candidate_id": int(record["source_track_id"]),
        "source_kind": SOURCE_KIND,
        "candidate_source": "automatic_sam_superpoint_pareto_variant",
        "class_id": class_id,
        "class_name": str(labels[class_id]),
        "score": float(record["gvc_score"]),
        "fusion_score": float(record["gvc_score"]),
        "proposal_priority": float(record["gvc_score"]),
        "seed_points_path": str(seed_path),
        "num_seed_points": int(len(points)),
        "support_view_count": int(semantic.get("semantic_evidence_frame_count", 0)),
        "selection_policy": SELECTION_POLICY,
        "source_track_id": int(record["source_track_id"]),
        "selected_variant_id": str(record["variant_id"]),
        "selected_variant_type": str(variant["variant_type"]),
        "pareto_eligible_variant_count": int(record["pareto_eligible_variant_count"]),
        "pareto_eligible_variant_ids": record["selected_from_variant_ids"],
        "base_superpoint_ids": [int(item) for item in variant["base_superpoint_ids"]],
        "added_superpoint_ids": [int(item) for item in variant.get("added_superpoint_ids", [])],
        "gvc_score": float(record["gvc_score"]),
        "gvc_selected_view_count": int(quality.get("gvc_selected_view_count", 0)),
        "gvc_matched_view_count": int(quality.get("gvc_matched_view_count", 0)),
        "semantic_evidence_top_class_index": class_id,
        "semantic_vote_margin": float(record["semantic_vote_margin"]),
        "semantic_normalized_entropy": float(record["semantic_normalized_entropy"]),
        "semantic_top_class_view_ratio": float(semantic.get("semantic_top_class_view_ratio", 0.0)),
        "native_relation_diagnostic": {
            key: quality.get(key) for key in (
                "native_top_candidate_id", "native_top_iou", "variant_inside_top_native_ratio", "native_overlap_count",
            )
        },
    }


def _export_scene(scene_name, args):
    plan_path = args.variant_plan_root / scene_name / "automatic_sam_growth_variant_plan.jsonl"
    variant_rows = [json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
    quality_rows = json.loads((args.quality_ledger_root / scene_name / "automatic_sam_variant_quality_ledger.json").read_text())
    semantic_rows = json.loads((args.semantic_ledger_root / scene_name / "automatic_sam_variant_semantic_ledger.json").read_text())
    pareto_rows = json.loads((args.pareto_ledger_root / scene_name / "automatic_sam_variant_pareto_ledger.json").read_text())
    selected, skipped = select_track_variants(variant_rows, quality_rows, semantic_rows, pareto_rows, args.labels)
    processed = np.load(args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 ScanNet superpoint 列")
    superpoint_atoms = raw_superpoint_points(processed)
    scene_root = args.output_root / scene_name
    seed_root = scene_root / "seed_points"
    seed_root.mkdir(parents=True, exist_ok=False)
    candidates = []
    for record in selected:
        points = variant_points(record["variant"], superpoint_atoms)
        if len(points) == 0:
            skipped.append({"variant_id": record["variant_id"], "reason": "empty_reconstructed_superpoint_points"})
            continue
        seed_path = Path("seed_points") / f"track{int(record['source_track_id']):05d}_{record['variant_id']}.npz"
        np.savez_compressed(scene_root / seed_path, point_indices=points.astype(np.int32))
        candidates.append(_candidate(scene_name, record, points, args.labels, seed_path))
    payload = {
        "scene_name": scene_name,
        "source_kind": SOURCE_KIND,
        "gt_usage": "none",
        "append_only_contract": {"native_candidates_mutated": False, "native_overlap_filtering": False},
        "selection_policy": SELECTION_POLICY,
        "score_policy": "候选分数固定为类别无关 GVC；不含类别相关加权或场景内排名。",
        "candidates": candidates,
    }
    (scene_root / "backprojection_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    audit = {
        "scene_name": scene_name, "input_variant_count": len(variant_rows),
        "pareto_record_count": len(pareto_rows), "exported_candidate_count": len(candidates),
        "selection_skip_counts": dict(sorted(Counter(item["reason"] for item in skipped).items())),
        "class_counts": dict(sorted(Counter(item["class_name"] for item in candidates).items())),
        "gvc_score": _numeric_summary(candidates, "gvc_score"),
        "num_seed_points": _numeric_summary(candidates, "num_seed_points"),
    }
    (scene_root / "export_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--variant-plan-root", required=True, type=Path)
    parser.add_argument("--quality-ledger-root", required=True, type=Path)
    parser.add_argument("--semantic-ledger-root", required=True, type=Path)
    parser.add_argument("--pareto-ledger-root", required=True, type=Path)
    parser.add_argument("--processed-scene-root", default=Path("data/scannet200"), type=Path)
    parser.add_argument("--config-path", default=Path("pretrained/config_scannet200.yaml"), type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "variant_plan_root", "quality_ledger_root", "semantic_ledger_root", "pareto_ledger_root", "processed_scene_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.labels = list(yaml.safe_load(handle)["network2d"]["text_prompts"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    audits = []
    for index, scene_name in enumerate(scenes, start=1):
        audit = _export_scene(scene_name, args)
        audits.append(audit)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: 导出 {audit['exported_candidate_count']} 条", flush=True)
    summary = {
        "purpose": "为自动 SAM 的 Pareto 非支配 superpoint 变体导出每轨迹一个 append-only 候选。",
        "gt_usage": "none", "selection_policy": SELECTION_POLICY, "scene_count": len(audits),
        "candidate_count": sum(item["exported_candidate_count"] for item in audits),
        "params": {key: value for key, value in vars(args).items() if key != "labels"},
    }
    (args.output_root / "automatic_sam_pareto_candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
