#!/usr/bin/env python3
"""在固定 f30 RGB 帧导出 YOLOE 的二维框，不读取 GT。

该工具仅构建新的二维对象观测账本。输出不得直接作为三维候选、融合输入、
评分依据或 AP 评测输入；是否值得接入由独立的 GT-only 诊断工具事后判断。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _frame_sort_key(path):
    """按 ScanNet 数字帧号排序，同时兼容非数字文件名。"""
    try:
        return (0, int(path.stem), path.name)
    except ValueError:
        return (1, 0, path.name)


def _sample_color_paths(scene_root, frame_stride, source_frame_frequency, max_frames):
    paths = sorted((scene_root / "color").glob("*"), key=_frame_sort_key)
    paths = [path for path in paths if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    stride = max(1, int(frame_stride)) * max(1, int(source_frame_frequency))
    return paths[::stride][:int(max_frames)]


def _result_record(frame_path, result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        xyxy = np.empty((0, 4), dtype=np.float32)
        labels = np.empty(0, dtype=np.int64)
        scores = np.empty(0, dtype=np.float32)
    else:
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        labels = boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "frame_id": frame_path.stem,
        "boxes_xyxy": xyxy.tolist(),
        "labels": labels.tolist(),
        "scores": scores.tolist(),
    }


def _load_model(args, labels):
    source = str(args.yoloe_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    from ultralytics import YOLOE

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("当前 CUDA 不可用；拒绝让 YOLOE 回退到 CPU。")
    model = YOLOE(str(args.checkpoint))
    model.to(args.device)
    model.set_classes(labels, model.get_text_pe(labels))
    return model


def _export_scene(scene_name, model, args):
    scene_root = args.dataset_root / scene_name
    paths = _sample_color_paths(
        scene_root, args.frame_stride, args.source_frame_frequency, args.max_frames
    )
    if not paths:
        raise FileNotFoundError(f"未找到 RGB 帧：{scene_root / 'color'}")
    frames = []
    for frame_path in paths:
        result = model.predict(
            str(frame_path),
            conf=args.confidence,
            iou=args.nms_iou,
            max_det=args.max_detections,
            imgsz=args.image_size,
            verbose=False,
        )[0]
        frames.append(_result_record(frame_path, result))
    payload = {
        "scene_name": scene_name,
        "frame_count": len(frames),
        "detection_count": int(sum(len(frame["labels"]) for frame in frames)),
        "frames": frames,
    }
    target = args.output_root / scene_name
    target.mkdir(parents=True, exist_ok=False)
    (target / "yoloe_boxes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _load_existing_scene(output_root, scene_name):
    path = output_root / scene_name / "yoloe_boxes.json"
    if not path.is_file():
        raise FileNotFoundError(f"{scene_name} 的已有输出不完整：{path}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--yoloe_source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--source_frame_frequency", type=int)
    parser.add_argument("--confidence", type=float, default=0.08)
    parser.add_argument("--nms_iou", type=float, default=0.30)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "dataset_root", "config_path", "yoloe_source", "checkpoint", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.skip_existing:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    if not args.yoloe_source.is_dir() or not args.checkpoint.is_file():
        raise SystemExit("YOLOE 源码目录或 checkpoint 不存在。")
    with args.config_path.open() as handle:
        config = yaml.safe_load(handle)
    if args.source_frame_frequency is None:
        args.source_frame_frequency = int(config["openyolo3d"]["frequency"])
    labels = [str(item) for item in config["network2d"]["text_prompts"]]
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = _load_model(args, labels)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        if (args.output_root / scene_name).exists() and args.skip_existing:
            summary = _load_existing_scene(args.output_root, scene_name)
            state = "复用"
        else:
            summary = _export_scene(scene_name, model, args)
            state = "完成"
        summaries.append(summary)
        print(
            f"[场景{state}] {index}/{len(scenes)} {scene_name}: "
            f"{summary['frame_count']} 帧，{summary['detection_count']} 个二维框",
            flush=True,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload = {
        "gt_usage": "不读取 GT；不生成三维候选、不融合、不评分、不评测 AP。",
        "decision_state": "仅用于与冻结 YOLO-World 的独立二维覆盖对照。",
        "scene_count": len(summaries),
        "frame_count": int(sum(item["frame_count"] for item in summaries)),
        "detection_count": int(sum(item["detection_count"] for item in summaries)),
        "labels": labels,
        "params": {key: value for key, value in vars(args).items() if key != "output_root"},
    }
    (args.output_root / "yoloe_boxes_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
