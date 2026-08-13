#!/usr/bin/env python3
"""检查自动 SAM 全局证据图的无 GT 运行入口。"""

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
        raise SystemExit(f"输出目录已有真实结果，为避免覆盖已拒绝执行：{path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--track_root", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--with_visibility", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "automatic_root", "processed_scene_root", "dataset_root", "config_path", "output_root"
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.track_root is not None:
        args.track_root = _resolve(args.track_root)

    _require_file(args.scene_list, "场景列表")
    _require_file(args.config_path, "配置文件")
    scenes = _read_scenes(args.scene_list)
    if not scenes or len(set(scenes)) != len(scenes):
        raise SystemExit("场景列表必须非空且场景名称不能重复。")
    for scene_name in scenes:
        _require_file(
            args.automatic_root / scene_name / "automatic_observations.jsonl",
            f"{scene_name} 的自动 SAM 观测",
        )
        scene_id = scene_name.replace("scene", "")
        _require_file(
            args.processed_scene_root / scene_name / f"{scene_id}.npy",
            f"{scene_name} 的处理点云",
        )
        if args.with_visibility and not (args.dataset_root / scene_name).is_dir():
            raise SystemExit(f"缺少{scene_name} 的 RGB-D 数据目录：{args.dataset_root / scene_name}")
        if args.track_root is not None:
            _require_file(
                args.track_root / scene_name / "automatic_tracks.json",
                f"{scene_name} 的既有自动轨迹",
            )
    _require_empty_output(args.output_root)
    manifest = {
        "purpose": "只从自动 SAM 单帧观测构建无 GT 全局证据图。",
        "gt_usage": "不读取 GT；不生成候选、不赋类别、不融合、不评分、不评测。",
        "scene_count": len(scenes),
        "scene_list": str(args.scene_list),
        "automatic_root": str(args.automatic_root),
        "processed_scene_root": str(args.processed_scene_root),
        "dataset_root": str(args.dataset_root),
        "config_path": str(args.config_path),
        "track_root": str(args.track_root) if args.track_root is not None else None,
        "output_root": str(args.output_root),
        "with_visibility": bool(args.with_visibility),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "entry_preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
