#!/usr/bin/env python3
"""运行前检查冻结的 ScanNet200 native 强基线入口，不读取 GT。"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError("场景列表含重复场景")
    return scenes


def _candidate_file(root, scene_name):
    return root / scene_name / "backprojection_candidates.json"


def _source_kind(candidate, root):
    value = candidate.get("source_kind") or candidate.get("candidate_source_kind")
    if value:
        return str(value).strip().lower()
    value = candidate.get("source_name") or candidate.get("candidate_source_name") or root.name
    value = str(value).strip().lower()
    if "sam_fused" in value or "sam-fused" in value:
        return "sam_fused"
    if "backprojection" in value or "bpr" in value:
        return "bpr"
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--bbox-root", required=True, type=Path)
    parser.add_argument("--candidate-roots", required=True, help="逗号分隔的候选根目录")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-mode", required=True, choices=["native", "uniform", "calibrated"])
    parser.add_argument("--expected-scenes", default=48, type=int)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()

    args.scene_list = _resolve(args.scene_list)
    args.mask_root = _resolve(args.mask_root)
    args.bbox_root = _resolve(args.bbox_root)
    args.output_dir = _resolve(args.output_dir)
    candidate_roots = [_resolve(item.strip()) for item in args.candidate_roots.split(",") if item.strip()]

    if args.score_mode != "native":
        raise SystemExit("当前正式实验只允许 native 评分入口；uniform 仅用于历史结果解释。")
    if not candidate_roots:
        raise SystemExit("至少需要一个候选根目录。")
    scenes = _read_scenes(args.scene_list)
    if len(scenes) != args.expected_scenes:
        raise SystemExit(f"场景数为 {len(scenes)}，期望为 {args.expected_scenes}。")

    missing = []
    for scene_name in scenes:
        if not (args.mask_root / f"{scene_name}.pt").is_file():
            missing.append(f"缺少 Mask3D: {scene_name}")
        if not (args.bbox_root / f"{scene_name}.pt").is_file():
            missing.append(f"缺少 YOLO-World 二维缓存: {scene_name}")
        for root in candidate_roots:
            if not _candidate_file(root, scene_name).is_file():
                missing.append(f"缺少候选文件: {root.name}/{scene_name}")
    if missing:
        raise SystemExit("\n".join(missing[:20]))

    root_names = [str(root).lower() for root in candidate_roots]
    forbidden_root_tokens = ("sam2", "details", "ibsp", "mask_graph")
    bad_roots = [str(root) for root, name in zip(candidate_roots, root_names) if any(token in name for token in forbidden_root_tokens)]
    if bad_roots:
        raise SystemExit(f"候选根目录含已冻结来源: {bad_roots}")

    source_kinds = set()
    for root in candidate_roots:
        for scene_name in scenes:
            payload = json.loads(_candidate_file(root, scene_name).read_text())
            for candidate in payload.get("candidates", []):
                source_kinds.add(_source_kind(candidate, root))
    allowed_sources = {"sam_fused", "bpr"}
    unexpected_sources = sorted(source_kinds.difference(allowed_sources))
    if unexpected_sources:
        raise SystemExit(f"候选文件含非冻结强基线来源: {unexpected_sources}")

    existing_results = list(args.output_dir.glob("*.csv")) + list(args.output_dir.glob("prediction_cache/*"))
    if existing_results and not args.allow_existing_output:
        raise SystemExit(f"输出目录已有结果，拒绝覆盖或混用: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "gt_usage": "none",
        "score_mode": args.score_mode,
        "scene_count": len(scenes),
        "mask_root": str(args.mask_root),
        "bbox_root": str(args.bbox_root),
        "candidate_roots": [str(root) for root in candidate_roots],
        "candidate_source_kinds": sorted(source_kinds),
        "processed_scene_root": None,
        "frozen_modules": ["sam2", "details_matter_postprocess", "ibsp", "mask_graph"],
    }
    (args.output_dir / "entry_preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
