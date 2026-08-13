#!/usr/bin/env python3
"""逐观测验证自动 SAM RLE 重导出未改变既有类别无关观测。"""

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = (
    "observation_id", "scene_name", "frame_id", "frame_index", "area",
    "bbox_xywh", "crop_box_xywh", "predicted_iou", "stability_score",
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或包含重复场景")
    return scenes


def _jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _equal(left, right):
    if isinstance(left, float) or isinstance(right, float):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=0.0))
    return left == right


def verify_scene(previous_root, reexport_root, scene):
    """比较一场景观测元数据、回投点和新增 RLE/真实关系的完整性。"""
    previous_scene = previous_root / scene
    reexport_scene = reexport_root / scene
    old_records = _jsonl(previous_scene / "automatic_observations.jsonl")
    new_records = _jsonl(reexport_scene / "automatic_observations.jsonl")
    if len(old_records) != len(new_records):
        raise ValueError(f"{scene}: 观测数不一致，旧 {len(old_records)}，新 {len(new_records)}")
    relation_path = reexport_scene / "same_frame_mask_relations.jsonl"
    if not relation_path.is_file():
        raise FileNotFoundError(f"{scene}: 缺少真实同帧关系文件 {relation_path}")
    relation_count = len(_jsonl(relation_path))
    for position, (old, new) in enumerate(zip(old_records, new_records)):
        for field in IDENTITY_FIELDS:
            if field not in old or field not in new or not _equal(old[field], new[field]):
                raise ValueError(f"{scene}: 第 {position} 条观测的 {field} 不一致")
        if "mask_rle" not in new:
            raise ValueError(f"{scene}: 第 {position} 条重导观测缺少 mask_rle")
        old_points = np.load(old["point_indices_path"])["point_indices"]
        new_points = np.load(new["point_indices_path"])["point_indices"]
        if not np.array_equal(old_points, new_points):
            raise ValueError(f"{scene}: 第 {position} 条观测的回投 point_indices 不一致")
    return {"scene_name": scene, "observation_count": len(new_records), "same_frame_relation_count": relation_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--reexport-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "previous_root", "reexport_root", "output"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output.exists():
        raise SystemExit(f"输出已存在，拒绝覆盖：{args.output}")
    scene_rows = []
    for index, scene in enumerate(_scenes(args.scene_list), start=1):
        row = verify_scene(args.previous_root, args.reexport_root, scene)
        scene_rows.append(row)
        print(f"[场景通过] {index}: {scene}，{row['observation_count']} 条观测，{row['same_frame_relation_count']} 条真实关系", flush=True)
    payload = {
        "purpose": "验证 RLE/真实二维关系重导出保留既有自动 SAM 观测和三维回投。",
        "gt_usage": "none",
        "decision_state": "仅验证缓存一致性；不生成候选、不聚合、不删除、不评测。",
        "scene_count": len(scene_rows),
        "observation_count": sum(row["observation_count"] for row in scene_rows),
        "same_frame_relation_count": sum(row["same_frame_relation_count"] for row in scene_rows),
        "scenes": scene_rows,
        "params": {key: str(value) for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("scene_count", "observation_count", "same_frame_relation_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
