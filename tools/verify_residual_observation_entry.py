#!/usr/bin/env python3
"""检查多视角 SAM 残差观测缓存的无 GT 运行入口。"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _require_file(path, label):
    if not path.is_file():
        raise SystemExit(f"缺少{label}：{path}")


def _require_empty_output(path, label):
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{label}已存在且非空，为避免复用或覆盖已拒绝执行：{path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument("--bbox_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--sam_checkpoint", type=Path, required=True)
    parser.add_argument("--sam_source", type=Path, required=True)
    parser.add_argument("--observation_output_root", type=Path, required=True)
    parser.add_argument("--residual_output_root", type=Path, required=True)
    parser.add_argument("--expected_scene_count", type=int, default=48)
    args = parser.parse_args()
    for name in vars(args):
        if name.endswith("_root") or name in {"scene_list", "prediction_cache_dir", "sam_checkpoint", "sam_source"}:
            setattr(args, name, _resolve(getattr(args, name)))

    _require_file(args.scene_list, "场景列表")
    scenes = _read_scenes(args.scene_list)
    if len(scenes) != args.expected_scene_count or len(set(scenes)) != len(scenes):
        raise SystemExit(f"场景列表必须含 {args.expected_scene_count} 个互异场景，当前为 {len(scenes)} 个。")
    _require_file(args.sam_checkpoint, "SAM 权重")
    if not args.sam_source.is_dir():
        raise SystemExit(f"缺少 SAM 源码目录：{args.sam_source}")
    for scene_name in scenes:
        _require_file(args.mask_root / f"{scene_name}.pt", f"{scene_name} 的 Mask3D mask")
        _require_file(args.bbox_root / f"{scene_name}.pt", f"{scene_name} 的 YOLO-World 缓存")
        for suffix in ("pred_masks.npy", "pred_scores.npy", "pred_classes.npy"):
            _require_file(args.prediction_cache_dir / f"{scene_name}_{suffix}", f"{scene_name} 的 native 最终缓存")
    _require_empty_output(args.observation_output_root, "SAM 观测输出目录")
    _require_empty_output(args.residual_output_root, "残差归因输出目录")

    manifest = {
        "purpose": "仅生成无 GT 的 YOLO-World + SAM 帧级观测及其相对 native 最终候选的三维残差归因。",
        "gt_usage": "不读取 GT；不生成最终候选、不融合、不评测。",
        "scene_count": len(scenes),
        "scene_list": str(args.scene_list),
        "mask_root": str(args.mask_root),
        "bbox_root": str(args.bbox_root),
        "prediction_cache_dir": str(args.prediction_cache_dir),
        "sam_checkpoint": str(args.sam_checkpoint),
        "sam_source": str(args.sam_source),
        "observation_output_root": str(args.observation_output_root),
        "residual_output_root": str(args.residual_output_root),
    }
    args.observation_output_root.mkdir(parents=True, exist_ok=True)
    (args.observation_output_root / "entry_preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
