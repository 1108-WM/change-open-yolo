#!/usr/bin/env python3
"""在固定 f30 RGB 帧导出 GroundingDINO 的二维框，不读取 GT。

这是 GD-SAM 对照的第一步：只建立 GroundingDINO 的独立二维观测账本。输出
不能直接进入三维候选、融合、评分或 AP 评测；是否继续把框送入 SAM，由随后
的 GT-only 覆盖诊断决定。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.ops import batched_nms, box_convert


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _frame_sort_key(path):
    try:
        return (0, int(path.stem), path.name)
    except ValueError:
        return (1, 0, path.name)


def _sample_color_paths(scene_root, frame_stride, source_frame_frequency, max_frames):
    paths = sorted((scene_root / "color").glob("*"), key=_frame_sort_key)
    paths = [path for path in paths if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    stride = max(1, int(frame_stride)) * max(1, int(source_frame_frequency))
    return paths[::stride][:int(max_frames)]


def _caption(labels):
    return ". ".join(labels).lower().strip() + "."


def _caption_token_count(tokenizer, labels):
    return len(tokenizer(_caption(labels), add_special_tokens=True, truncation=False)["input_ids"])


def _build_prompt_groups(labels, tokenizer, max_text_len):
    """只由固定类别文本构造不截断的提示词组，绝不根据 GT 或检测结果分组。"""
    groups = []
    current_ids = []
    for label_id, label in enumerate(labels):
        trial = current_ids + [label_id]
        trial_labels = [labels[index] for index in trial]
        if _caption_token_count(tokenizer, trial_labels) <= max_text_len:
            current_ids = trial
            continue
        if not current_ids:
            raise ValueError(f"单个类别提示超过文本上限 {max_text_len}: {label}")
        groups.append(current_ids)
        current_ids = [label_id]
        if _caption_token_count(tokenizer, [label]) > max_text_len:
            raise ValueError(f"单个类别提示超过文本上限 {max_text_len}: {label}")
    if current_ids:
        groups.append(current_ids)
    return groups


def _class_token_indices(tokenizer, labels, label_ids):
    """返回每个类别在本提示词中的 BERT token 位置，避免短语子串误匹配。"""
    group_labels = [labels[index] for index in label_ids]
    caption = _caption(group_labels)
    tokenized = tokenizer(
        caption,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = tokenized["offset_mapping"]
    spans = []
    start = 0
    for label in group_labels:
        end = start + len(label)
        indices = [
            index for index, (left, right) in enumerate(offsets)
            if left < end and right > start
        ]
        if not indices:
            raise ValueError(f"未找到类别 {label!r} 对应的 BERT token")
        spans.append(indices)
        start = end + 2  # 标签后的 ". "
    return caption, tokenized, spans


def _load_model(args):
    source = str(args.groundingdino_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("当前 CUDA 不可用；拒绝让 GroundingDINO 回退到 CPU。")
    config = SLConfig.fromfile(str(args.config_path))
    config.device = args.device
    config.text_encoder_type = str(args.bert_dir)
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval().to(args.device)
    return model, int(config.max_text_len)


def _predict_group(model, image, caption, token_spans, label_ids, box_threshold):
    """按类别 token 段取分数，而非按文本子串把 ``armchair`` 误归为 ``chair``。"""
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    class_scores = torch.stack([logits[:, span].max(dim=1).values for span in token_spans], dim=1)
    scores, local_class_ids = class_scores.max(dim=1)
    keep = scores >= box_threshold
    if not torch.any(keep):
        return (
            torch.empty((0, 4), dtype=torch.float32, device=image.device),
            torch.empty(0, dtype=torch.int64, device=image.device),
            torch.empty(0, dtype=torch.float32, device=image.device),
        )
    labels = torch.as_tensor(label_ids, dtype=torch.int64, device=image.device)[local_class_ids[keep]]
    return outputs["pred_boxes"][0][keep], labels, scores[keep]


def _predict_frame(model, image, image_shape, groups, labels, tokenizer, args):
    all_boxes, all_labels, all_scores = [], [], []
    for label_ids in groups:
        caption, _, spans = _class_token_indices(tokenizer, labels, label_ids)
        boxes, class_ids, scores = _predict_group(
            model, image, caption, spans, label_ids, args.box_threshold
        )
        if len(boxes):
            all_boxes.append(boxes)
            all_labels.append(class_ids)
            all_scores.append(scores)
    if not all_boxes:
        return np.empty((0, 4), np.float32), np.empty(0, np.int64), np.empty(0, np.float32)
    boxes = torch.cat(all_boxes)
    class_ids = torch.cat(all_labels)
    scores = torch.cat(all_scores)
    height, width = image_shape[:2]
    scale = boxes.new_tensor([width, height, width, height])
    xyxy = box_convert(boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")
    xyxy[:, 0::2].clamp_(0, width)
    xyxy[:, 1::2].clamp_(0, height)
    keep = batched_nms(xyxy, scores, class_ids, args.nms_iou)[:args.max_detections]
    return (
        xyxy[keep].detach().cpu().numpy().astype(np.float32, copy=False),
        class_ids[keep].detach().cpu().numpy().astype(np.int64, copy=False),
        scores[keep].detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _export_scene(scene_name, model, groups, labels, tokenizer, args):
    from groundingdino.util.inference import load_image

    scene_root = args.dataset_root / scene_name
    paths = _sample_color_paths(
        scene_root, args.frame_stride, args.source_frame_frequency, args.max_frames
    )
    if not paths:
        raise FileNotFoundError(f"未找到 RGB 帧：{scene_root / 'color'}")
    frames = []
    for frame_path in paths:
        image_source, image = load_image(str(frame_path))
        xyxy, class_ids, scores = _predict_frame(
            model, image.to(args.device), image_source.shape, groups, labels, tokenizer, args
        )
        frames.append({
            "frame_id": frame_path.stem,
            "boxes_xyxy": xyxy.tolist(),
            "labels": class_ids.tolist(),
            "scores": scores.tolist(),
        })
    payload = {
        "scene_name": scene_name,
        "frame_count": len(frames),
        "detection_count": int(sum(len(frame["labels"]) for frame in frames)),
        "frames": frames,
    }
    target = args.output_root / scene_name
    target.mkdir(parents=True, exist_ok=False)
    (target / "groundingdino_boxes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _load_existing_scene(output_root, scene_name):
    path = output_root / scene_name / "groundingdino_boxes.json"
    if not path.is_file():
        raise FileNotFoundError(f"{scene_name} 的已有输出不完整：{path}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--groundingdino_source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bert_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--scene_offset", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--source_frame_frequency", type=int)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--nms_iou", type=float, default=0.30)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list", "dataset_root", "config_path", "groundingdino_source", "checkpoint",
        "bert_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.skip_existing:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    if not args.groundingdino_source.is_dir() or not args.checkpoint.is_file() or not args.bert_dir.is_dir():
        raise SystemExit("GroundingDINO 源码、权重或本地 BERT 目录不存在。")
    with (PROJECT_ROOT / "pretrained/config_scannet200.yaml").open() as handle:
        baseline_config = yaml.safe_load(handle)
    if args.source_frame_frequency is None:
        args.source_frame_frequency = int(baseline_config["openyolo3d"]["frequency"])
    labels = [str(item) for item in baseline_config["network2d"]["text_prompts"]]
    model, max_text_len = _load_model(args)
    tokenizer = model.tokenizer
    groups = _build_prompt_groups(labels, tokenizer, max_text_len)
    group_metadata = [{
        "group_index": index,
        "label_ids": group,
        "labels": [labels[label_id] for label_id in group],
        "caption": _caption([labels[label_id] for label_id in group]),
        "token_count": _caption_token_count(tokenizer, [labels[label_id] for label_id in group]),
    } for index, group in enumerate(groups)]
    scenes = _read_scenes(args.scene_list)
    scenes = scenes[max(0, args.scene_offset):]
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        if (args.output_root / scene_name).exists() and args.skip_existing:
            summary, state = _load_existing_scene(args.output_root, scene_name), "复用"
        else:
            summary, state = _export_scene(scene_name, model, groups, labels, tokenizer, args), "完成"
        summaries.append(summary)
        print(f"[场景{state}] {index}/{len(scenes)} {scene_name}: {summary['frame_count']} 帧，{summary['detection_count']} 个二维框", flush=True)
    torch.cuda.empty_cache()
    payload = {
        "gt_usage": "不读取 GT；不生成三维候选、不融合、不评分、不评测 AP。",
        "decision_state": "仅用于与冻结 YOLO-World 的独立二维覆盖对照；通过后才考虑将框送入 SAM。",
        "scene_count": len(summaries),
        "frame_count": int(sum(item["frame_count"] for item in summaries)),
        "detection_count": int(sum(item["detection_count"] for item in summaries)),
        "labels": labels,
        "prompt_groups": group_metadata,
        "params": {key: value for key, value in vars(args).items() if key not in {"output_root"}},
    }
    (args.output_root / "groundingdino_boxes_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
