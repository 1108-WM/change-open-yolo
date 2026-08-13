#!/usr/bin/env python3
"""GT-only：比较 GroundingDINO 与冻结 YOLO-World 的 native 残差二维覆盖。

GroundingDINO 框来自未读 GT 的固定 f30 导出。此工具只在离线账本中使用 GT；
输出不得回流到推理、候选生成、融合、打分或 AP 评测。
"""

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _coverage_module():
    path = PROJECT_ROOT / "tools" / "diagnose_yoloe_independent_coverage_gt.py"
    spec = importlib.util.spec_from_file_location("_yoloe_coverage_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_groundingdino(root, scene_name):
    payload = json.loads((root / scene_name / "groundingdino_boxes.json").read_text())
    result = {}
    for item in payload["frames"]:
        result[str(item["frame_id"])] = {
            "boxes_xyxy": np.asarray(item["boxes_xyxy"], dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(item["labels"], dtype=np.int64),
            "scores": np.asarray(item["scores"], dtype=np.float32),
        }
    return result


def _comparison_label(yoloworld, groundingdino):
    if groundingdino["reliable"] and not yoloworld["reliable"]:
        return "仅 GroundingDINO 有可靠二维证据"
    if yoloworld["reliable"] and not groundingdino["reliable"]:
        return "仅 YOLO-World 有可靠二维证据"
    if groundingdino["reliable"]:
        return "两者均有可靠二维证据"
    return "两者均无可靠二维证据"


def _rename_row(row):
    renamed = {}
    for key, value in row.items():
        renamed[key.replace("yoloe_", "groundingdino_")] = value
    return renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--groundingdino_root", type=Path, required=True)
    parser.add_argument("--yoloworld_root", type=Path, default=Path("output/scannet200/bboxes_2d"))
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--scene_offset", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--native_iou_threshold", type=float, default=0.50)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--min_visible_points", type=int, default=30)
    parser.add_argument("--min_box_point_coverage", type=float, default=0.50)
    parser.add_argument("--min_matched_frames", type=int, default=2)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    for name in (
        "scene_list", "groundingdino_root", "yoloworld_root", "prediction_cache_dir",
        "dataset_root", "gt_instance_dir", "config_path", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    prompt_to_id = {str(name): index for index, name in enumerate(args.config["network2d"]["text_prompts"])}
    base = _coverage_module()
    # 复用已审计的几何覆盖规则；仅替换第二个二维来源和输出字段名。
    base._load_yoloe = _load_groundingdino
    base._comparison_label = _comparison_label
    # 基础工具内部的参数名保留为 yoloe_root；此别名只服务于该只读诊断复用。
    args.yoloe_root = args.groundingdino_root
    scenes = base._read_scenes(args.scene_list)
    scenes = scenes[max(0, args.scene_offset):]
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    rows = []
    for scene_name in scenes:
        rows.extend(_rename_row(row) for row in base._scene_rows(scene_name, args, prompt_to_id))
        print(f"[场景完成] {scene_name}", flush=True)
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
        "native_iou_threshold": args.native_iou_threshold,
        "scene_count": len(scenes),
        "scenes": scenes,
        "native_residual_instance_count": len(rows),
        "comparison_counts": dict(sorted(groups.items())),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
