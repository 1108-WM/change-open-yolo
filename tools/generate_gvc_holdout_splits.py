#!/usr/bin/env python3
"""从未使用的 ScanNet200 场景生成固定的 GVC safety/test 划分，不读取 GT。"""

import argparse
import json
import random
from pathlib import Path


def _read_scenes(path):
    scenes = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"场景文件含重复项: {path}")
    return set(scenes)


def generate(scene_root, even96_path, odd96_path, output_dir, seed):
    all_scenes = {path.name for path in Path(scene_root).iterdir() if path.is_dir() and path.name.startswith("scene")}
    even96 = _read_scenes(even96_path)
    odd96 = _read_scenes(odd96_path)
    if even96 & odd96:
        raise ValueError("even96 与 odd96 必须不相交")
    used = even96 | odd96
    unknown = used - all_scenes
    if unknown:
        raise ValueError(f"拆分引用了不存在的场景: {sorted(unknown)[:5]}")
    remaining = sorted(all_scenes - used)
    if len(remaining) != 120:
        raise ValueError(f"排除 even96/odd96 后必须剩余 120 场景，当前为 {len(remaining)}")
    shuffled = list(remaining)
    random.Random(int(seed)).shuffle(shuffled)
    safety, test = sorted(shuffled[:60]), sorted(shuffled[60:])
    if set(safety) & set(test) or set(safety) & used or set(test) & used:
        raise RuntimeError("GVC 保留划分出现重叠")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gvc_safety60.txt").write_text("\n".join(safety) + "\n")
    (output_dir / "gvc_test60.txt").write_text("\n".join(test) + "\n")
    manifest = {
        "gt_usage": "none",
        "purpose": "GVC append-only 规则冻结后的独立安全审计与一次测试对照",
        "seed": int(seed),
        "total_scenes": len(all_scenes),
        "excluded": {"even96": len(even96), "odd96": len(odd96)},
        "remaining_count": len(remaining),
        "files": {
            "gvc_safety60.txt": {"count": len(safety), "role": "GT-only safety audit; no AP"},
            "gvc_test60.txt": {"count": len(test), "role": "one native AP comparison only after safety rule remains unchanged"},
        },
        "disjoint": {
            "safety_test": True,
            "safety_even96_odd96": set(safety).isdisjoint(used),
            "test_even96_odd96": set(test).isdisjoint(used),
        },
    }
    (output_dir / "gvc_holdout_splits_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--even96", type=Path, default=Path("output/scannet200/scene_splits/even96.txt"))
    parser.add_argument("--odd96", type=Path, default=Path("output/scannet200/scene_splits/odd96.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    print(json.dumps(generate(args.scene_root, args.even96, args.odd96, args.output_dir, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
