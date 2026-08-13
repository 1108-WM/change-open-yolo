#!/usr/bin/env python3
"""验证自动 SAM Pareto append-only 候选契约，不读取 GT 或运行评测。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_KIND = "automatic_sam_pareto_append_only"


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError("场景列表含重复场景")
    return scenes


def _superpoint_points(processed):
    raw_ids = np.asarray(processed[:, 9], dtype=np.int64)
    return {int(item): np.flatnonzero(raw_ids == item).astype(np.int64) for item in np.unique(raw_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--variant-plan-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--processed-scene-root", default=Path("data/scannet200"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-scenes", required=True, type=int)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "variant_plan_root", "candidate_root", "processed_scene_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    if len(scenes) != args.expected_scenes:
        raise SystemExit(f"场景数为 {len(scenes)}，期望为 {args.expected_scenes}。")
    errors, candidate_count = [], 0
    for scene_name in scenes:
        plan_path = args.variant_plan_root / scene_name / "automatic_sam_growth_variant_plan.jsonl"
        candidate_path = args.candidate_root / scene_name / "backprojection_candidates.json"
        for path in (plan_path, candidate_path):
            if not path.is_file():
                errors.append(f"缺少输入: {path}")
        if not plan_path.is_file() or not candidate_path.is_file():
            continue
        variants = {str(row["variant_id"]): row for row in (json.loads(line) for line in plan_path.read_text().splitlines() if line.strip())}
        payload = json.loads(candidate_path.read_text())
        contract = payload.get("append_only_contract", {})
        if payload.get("scene_name") != scene_name or payload.get("source_kind") != SOURCE_KIND:
            errors.append(f"{scene_name} 候选来源或场景名不匹配")
        if contract != {"native_candidates_mutated": False, "native_overlap_filtering": False}:
            errors.append(f"{scene_name} 不满足 append-only/native 保留契约")
        processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
        if not processed_path.is_file():
            errors.append(f"缺少输入: {processed_path}")
            continue
        atoms = _superpoint_points(np.load(processed_path, mmap_mode="r"))
        seen_track_ids = set()
        for candidate in payload.get("candidates", []):
            candidate_count += 1
            track_id = candidate.get("source_track_id")
            variant_id = str(candidate.get("selected_variant_id", ""))
            if track_id in seen_track_ids:
                errors.append(f"{scene_name} 同一轨迹导出多个候选: {track_id}")
            seen_track_ids.add(track_id)
            variant = variants.get(variant_id)
            required = ("candidate_id", "class_id", "class_name", "score", "fusion_score", "proposal_priority", "seed_points_path", "num_seed_points", "gvc_score", "source_track_id", "selected_variant_id")
            missing = [field for field in required if field not in candidate]
            if missing or variant is None or int(track_id) != int(candidate.get("candidate_id", -1)):
                errors.append(f"{scene_name} 候选 {track_id} 字段、轨迹或变体不匹配")
                continue
            values = np.asarray([candidate["score"], candidate["fusion_score"], candidate["proposal_priority"], candidate["gvc_score"]], dtype=np.float64)
            if not np.isfinite(values).all() or not np.allclose(values, candidate["gvc_score"]):
                errors.append(f"{scene_name} 候选 {track_id} 分数不符合固定 GVC 策略")
            seed_path = candidate_path.parent / candidate["seed_points_path"]
            if not seed_path.is_file():
                errors.append(f"{scene_name} 候选 {track_id} 缺少点索引文件")
                continue
            actual = np.unique(np.asarray(np.load(seed_path)["point_indices"], dtype=np.int64))
            ids = set(map(int, variant["base_superpoint_ids"])) | set(map(int, variant.get("added_superpoint_ids", [])))
            expected = np.unique(np.concatenate([atoms[item] for item in sorted(ids)])) if ids and all(item in atoms for item in ids) else np.empty(0, dtype=np.int64)
            if len(actual) != int(candidate["num_seed_points"]) or not np.array_equal(actual, expected):
                errors.append(f"{scene_name} 候选 {track_id} 未精确重建所选原始 superpoint 原子")
    if errors:
        raise SystemExit("\n".join(errors[:30]))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"gt_usage": "none", "scene_count": len(scenes), "candidate_count": candidate_count, "candidate_root": str(args.candidate_root), "append_only_contract": {"native_candidates_mutated": False, "native_overlap_filtering": False}}
    (args.output_dir / "automatic_sam_pareto_candidate_preflight_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
