#!/usr/bin/env python3
"""检查 GVC append-only 候选输入和 JSON 契约，不读取 GT 或运行评测。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError("场景列表含重复场景")
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--track-root", required=True, type=Path)
    parser.add_argument("--semantic-root", required=True, type=Path)
    parser.add_argument("--gvc-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-scenes", required=True, type=int)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "semantic_root", "gvc_root", "candidate_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    if len(scenes) != args.expected_scenes:
        raise SystemExit(f"场景数为 {len(scenes)}，期望为 {args.expected_scenes}。")
    errors = []
    candidate_count = 0
    for scene_name in scenes:
        required = (
            args.track_root / scene_name / "automatic_tracks.json",
            args.semantic_root / scene_name / "automatic_track_yoloworld_semantics.json",
            args.gvc_root / scene_name / "track_gvc_feature_ledger.json",
            args.candidate_root / scene_name / "backprojection_candidates.json",
        )
        for path in required:
            if not path.is_file():
                errors.append(f"缺少输入: {path}")
        candidate_path = required[-1]
        if not candidate_path.is_file():
            continue
        payload = json.loads(candidate_path.read_text())
        contract = payload.get("append_only_contract", {})
        if payload.get("scene_name") != scene_name or payload.get("source_kind") != "gvc_append_only":
            errors.append(f"{scene_name} 候选来源或场景名不匹配")
        if contract.get("native_candidates_mutated") is not False or contract.get("native_overlap_filtering") is not False:
            errors.append(f"{scene_name} 不满足 append-only/native 保留契约")
        seen_ids = set()
        for candidate in payload.get("candidates", []):
            candidate_count += 1
            candidate_id = candidate.get("candidate_id")
            if candidate_id in seen_ids:
                errors.append(f"{scene_name} 含重复 candidate_id={candidate_id}")
            seen_ids.add(candidate_id)
            required_fields = ("class_id", "class_name", "score", "fusion_score", "proposal_priority", "seed_points_path", "num_seed_points", "gvc_score")
            missing = [field for field in required_fields if field not in candidate]
            if missing:
                errors.append(f"{scene_name} candidate={candidate_id} 缺少字段: {missing}")
                continue
            score_values = (candidate["score"], candidate["fusion_score"], candidate["proposal_priority"], candidate["gvc_score"])
            if not np.isfinite(np.asarray(score_values, dtype=np.float64)).all():
                errors.append(f"{scene_name} candidate={candidate_id} 含非有限分数")
            seed_path = candidate_path.parent / candidate["seed_points_path"]
            if not seed_path.is_file():
                errors.append(f"{scene_name} candidate={candidate_id} 缺少点索引文件")
                continue
            points = np.asarray(np.load(seed_path)["point_indices"], dtype=np.int64)
            if len(points) != int(candidate["num_seed_points"]) or len(points) == 0:
                errors.append(f"{scene_name} candidate={candidate_id} 点索引数量不匹配或为空")
    if errors:
        raise SystemExit("\n".join(errors[:30]))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "gt_usage": "none",
        "decision_state": "仅检查 GVC append-only 候选输入与 JSON 契约；不读取 GT、不运行 AP。",
        "scene_count": len(scenes),
        "candidate_count": candidate_count,
        "candidate_root": str(args.candidate_root),
        "append_only_contract": {"native_candidates_mutated": False, "native_overlap_filtering": False},
    }
    (args.output_dir / "gvc_append_only_preflight_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
