#!/usr/bin/env python3
"""合并分块 GT-only YOLOE 二维覆盖账本，并验证 even48 场景完整性。"""

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


def _load_chunk(path):
    summary = json.loads((path / "summary.json").read_text())
    with (path / "yoloe_independent_coverage_gt.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return summary, rows


def _validate_scene_coverage(expected, summaries):
    observed = [scene for summary in summaries for scene in summary.get("scenes", [])]
    counts = Counter(observed)
    missing = [scene for scene in expected if counts[scene] == 0]
    duplicate = sorted(scene for scene, count in counts.items() if count > 1)
    unexpected = sorted(scene for scene in counts if scene not in set(expected))
    if missing or duplicate or unexpected:
        raise ValueError(
            f"场景分块不完整：缺少 {missing}，重复 {duplicate}，额外 {unexpected}。"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--chunk_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    expected = _read_scenes(_resolve(args.scene_list))
    chunks = [_resolve(path) for path in args.chunk_dirs]
    summaries, rows = [], []
    for path in chunks:
        summary, chunk_rows = _load_chunk(path)
        summaries.append(summary)
        rows.extend(chunk_rows)
    _validate_scene_coverage(expected, summaries)
    args.output_dir = _resolve(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scene_name", "gt_instance_id", "gt_class"]
    with (args.output_dir / "yoloe_independent_coverage_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = Counter(row["comparison"] for row in rows)
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、候选生成、融合、打分或 AP 评测。",
        "decision_rule": "仅当‘仅 YOLOE 有可靠二维证据’数量明确增加时，才考虑将 YOLOE 作为并行候选源；否则保持 YOLO-World 强基线不变。",
        "scene_count": len(expected),
        "scenes": expected,
        "native_residual_instance_count": len(rows),
        "comparison_counts": dict(sorted(groups.items())),
        "chunk_dirs": [str(path) for path in chunks],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
