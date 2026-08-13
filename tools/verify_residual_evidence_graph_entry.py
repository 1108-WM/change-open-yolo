#!/usr/bin/env python3
"""检查残差专属多视角互证图的无 GT 运行入口。"""

import argparse
import json
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


def _require_empty_output(path):
    if not path.exists():
        return
    existing_names = {child.name for child in path.iterdir()}
    if existing_names - {"entry_preflight_manifest.json"}:
        raise SystemExit(f"输出目录已存在真实结果，为避免覆盖已拒绝执行：{path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--residual_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--residual_mode", choices=("after_any", "after_same_class"), default="after_any")
    parser.add_argument("--expected_scene_count", type=int, default=48)
    args = parser.parse_args()
    for name in vars(args):
        if name.endswith("_root") or name in {"scene_list", "config_path"}:
            setattr(args, name, _resolve(getattr(args, name)))

    _require_file(args.scene_list, "场景列表")
    _require_file(args.config_path, "配置文件")
    scenes = _read_scenes(args.scene_list)
    if len(scenes) != args.expected_scene_count or len(set(scenes)) != len(scenes):
        raise SystemExit(f"场景列表必须含 {args.expected_scene_count} 个互异场景，当前为 {len(scenes)} 个。")

    for scene_name in scenes:
        _require_file(args.residual_root / scene_name / "residual_observations.jsonl", f"{scene_name} 的残差观测")
        scene_id = scene_name.replace("scene", "")
        _require_file(args.processed_scene_root / scene_name / f"{scene_id}.npy", f"{scene_name} 的处理点云")
        if not (args.dataset_root / scene_name).is_dir():
            raise SystemExit(f"缺少{scene_name} 的 RGB-D 数据目录：{args.dataset_root / scene_name}")
    _require_empty_output(args.output_root)

    manifest = {
        "purpose": "仅从冻结的逐帧残差观测构建残差专属多视角互证图与轨迹。",
        "gt_usage": "不读取 GT；不生成最终候选、不融合、不评分、不评测。",
        "scene_count": len(scenes),
        "scene_list": str(args.scene_list),
        "residual_root": str(args.residual_root),
        "dataset_root": str(args.dataset_root),
        "processed_scene_root": str(args.processed_scene_root),
        "config_path": str(args.config_path),
        "output_root": str(args.output_root),
        "residual_mode": args.residual_mode,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "entry_preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
