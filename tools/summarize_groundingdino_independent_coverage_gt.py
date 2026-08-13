#!/usr/bin/env python3
"""汇总分块的 GroundingDINO GT-only 覆盖账本，并核验 even48 场景恰好一次。"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--chunk_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.scene_list = _resolve(args.scene_list)
    args.chunk_root = _resolve(args.chunk_root)
    args.output_dir = _resolve(args.output_dir)
    expected_scenes = _read_scenes(args.scene_list)
    chunks = sorted(args.chunk_root.glob("chunk*/groundingdino_independent_coverage_gt.csv"))
    if not chunks:
        raise SystemExit("未找到任何分块 GT-only CSV。")
    rows = []
    for path in chunks:
        rows.extend(_load_rows(path))
    seen = Counter(row["scene_name"] for row in rows)
    missing = [scene for scene in expected_scenes if scene not in seen]
    duplicate = sorted(scene for scene, count in seen.items() if count and scene not in expected_scenes)
    # 同一场景会有多条残差，需依据各块 summary 的 scenes 而非 CSV 行数核验。
    chunk_scenes = []
    for path in chunks:
        summary = json.loads((path.parent / "summary.json").read_text())
        chunk_scenes.extend(summary["scenes"])
    scene_count = Counter(chunk_scenes)
    repeated = sorted(scene for scene, count in scene_count.items() if count != 1)
    missing_by_chunk = [scene for scene in expected_scenes if scene_count[scene] != 1]
    unexpected = sorted(scene for scene in scene_count if scene not in expected_scenes)
    if repeated or missing_by_chunk or unexpected or duplicate:
        raise SystemExit(
            f"分块场景不完整或重复：重复={repeated}，缺失={missing_by_chunk}，额外={unexpected or duplicate}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scene_name", "gt_instance_id", "gt_class"]
    with (args.output_dir / "groundingdino_independent_coverage_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = Counter(row["comparison"] for row in rows)
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、候选生成、融合、打分或 AP 评测。",
        "decision_rule": "仅当 GroundingDINO 的独有可靠二维证据有明确增益时，才进入 GD-SAM 的 mask 几何审计；否则保持 YOLO-World 强基线不变。",
        "scene_count": len(expected_scenes),
        "scenes": expected_scenes,
        "native_residual_instance_count": len(rows),
        "comparison_counts": dict(sorted(groups.items())),
        "source_chunks": [str(path.parent) for path in chunks],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
