#!/usr/bin/env python3
"""为自动 SAM 的二维层级边界竞争关系记录冻结 DINOv2 外观一致性。"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "vit_small_patch14_dinov2"
NORMALIZE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


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


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def decode_binary_mask_rle(payload):
    height, width = (int(value) for value in payload["size"])
    counts = [int(value) for value in payload["counts"]]
    if sum(counts) != height * width:
        raise ValueError("RLE counts 与二维尺寸不一致")
    chunks, foreground = [], False
    for count in counts:
        chunks.append(np.full(count, foreground, dtype=bool))
        foreground = not foreground
    return np.concatenate(chunks).reshape((height, width), order="F")


def mask_crop_tensor(image, mask, bbox_xywh, output_size=224):
    """以 mask 约束的紧致 RGB 区域提取 DINO 输入，避免整帧背景主导特征。"""
    x, y, width, height = (int(value) for value in bbox_xywh)
    pad = max(2, int(round(max(width, height) * 0.05)))
    left, top = max(0, x - pad), max(0, y - pad)
    right, bottom = min(image.shape[1], x + width + pad), min(image.shape[0], y + height + pad)
    crop, crop_mask = image[top:bottom, left:right].copy(), mask[top:bottom, left:right]
    if not crop_mask.any():
        raise ValueError("mask bbox 中没有前景像素")
    crop[~crop_mask] = 127
    resized = np.asarray(Image.fromarray(crop).resize((output_size, output_size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(((resized - NORMALIZE_MEAN) / NORMALIZE_STD).transpose(2, 0, 1))


def cosine(left, right):
    return float(np.dot(left, right) / max(1e-12, np.linalg.norm(left) * np.linalg.norm(right)))


def summarize(values):
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {"count": len(values), "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def _track_pairs_by_observation(nodes):
    return {int(row["observation_id"]): {int(track) for track in row.get("existing_track_ids", [])} for row in nodes}


def _relation_key(row):
    return tuple(sorted((int(row["left_source_track_id"]), int(row["right_source_track_id"]))))


def matching_observation_pairs(plan_rows, nodes, same_frame_relations):
    """仅保留计划中的边界关系在真实同帧 mask 中实际出现的观测对。"""
    plans = {_relation_key(row): row for row in plan_rows if row["variant_plan_state"] == "boundary_competition_variant_hypothesis_keep_originals"}
    tracks = _track_pairs_by_observation(nodes)
    pairs = defaultdict(list)
    for relation in same_frame_relations:
        left_tracks = tracks.get(int(relation["left_observation_id"]), set())
        right_tracks = tracks.get(int(relation["right_observation_id"]), set())
        for left_track in left_tracks:
            for right_track in right_tracks:
                if left_track == right_track:
                    continue
                key = tuple(sorted((left_track, right_track)))
                if key in plans:
                    pairs[key].append(relation)
    return plans, pairs


def _load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or set(unexpected) != {"mask_token"}:
        raise ValueError(f"DINOv2 权重与 {MODEL_NAME} 不匹配：missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval()


def _features_for_scene(observations, required_ids, dataset_scene_root, model, device, batch_size):
    by_id = {int(row["observation_id"]): row for row in observations}
    tensors, keys = [], []
    input_size = int(model.patch_embed.img_size[0])
    for observation_id in sorted(required_ids):
        record = by_id[observation_id]
        image_path = dataset_scene_root / "color" / f"{record['frame_id']}.jpg"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        tensors.append(mask_crop_tensor(image, decode_binary_mask_rle(record["mask_rle"]), record["bbox_xywh"], output_size=input_size))
        keys.append(observation_id)
    features = {}
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(device)
            output = model(batch).detach().float().cpu().numpy()
            for observation_id, feature in zip(keys[start:start + batch_size], output):
                features[observation_id] = feature
    return features


def build_scene_ledger(plan_rows, nodes, observations, same_frame_relations, dataset_scene_root, model, device, batch_size):
    plans, pairs_by_track = matching_observation_pairs(plan_rows, nodes, same_frame_relations)
    required_ids = {int(relation[key]) for pairs in pairs_by_track.values() for relation in pairs for key in ("left_observation_id", "right_observation_id")}
    features = _features_for_scene(observations, required_ids, dataset_scene_root, model, device, batch_size) if required_ids else {}
    records = []
    for key, plan in sorted(plans.items()):
        frame_records = []
        for relation in pairs_by_track.get(key, []):
            left_id, right_id = int(relation["left_observation_id"]), int(relation["right_observation_id"])
            frame_records.append({
                "frame_index": int(relation["frame_index"]), "frame_id": str(relation["frame_id"]),
                "left_observation_id": left_id, "right_observation_id": right_id,
                "dino_vits14_masked_crop_cosine": cosine(features[left_id], features[right_id]),
                "exact_mask_iou": float(relation["iou"]),
                "left_coverage": float(relation["left_coverage"]), "right_coverage": float(relation["right_coverage"]),
            })
        scores = [row["dino_vits14_masked_crop_cosine"] for row in frame_records]
        records.append({
            "relation_kind": "automatic_sam_boundary_competition_dino_appearance_ledger",
            "left_candidate_id": int(plan["left_candidate_id"]), "right_candidate_id": int(plan["right_candidate_id"]),
            "left_source_track_id": key[0], "right_source_track_id": key[1],
            "class_id": int(plan["class_id"]), "semantic_js_divergence": float(plan["semantic_js_divergence"]),
            "exact_2d_evidence": plan["exact_2d_evidence"], "frame_records": frame_records,
            "dino_vits14_masked_crop_cosine": summarize(scores),
            "decision_state": "冻结外观连续证据；不设阈值、不选择同实例、不聚合、不删除、不改类别或分数，不输出预测或 AP。",
            "gt_usage": "none",
        })
    return records, len(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--variant-plan-root", required=True, type=Path)
    parser.add_argument("--automatic-root", required=True, type=Path)
    parser.add_argument("--evidence-graph-root", required=True, type=Path)
    parser.add_argument("--dataset-root", default=Path("data/scannet200"), type=Path)
    parser.add_argument("--checkpoint", default=Path("pretrained/checkpoints/dinov2_vits14_pretrain.pth"), type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    for name in ("scene_list", "variant_plan_root", "automatic_root", "evidence_graph_root", "dataset_root", "checkpoint", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("当前 CUDA 不可用；拒绝让全量 DINOv2 账本静默回退到 CPU。")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        raise SystemExit("DINOv2 权重不存在或为空。")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    model = _load_model(args.checkpoint, args.device)
    totals = Counter()
    for index, scene in enumerate(_scenes(args.scene_list), start=1):
        rows, feature_count = build_scene_ledger(
            _jsonl(args.variant_plan_root / scene / "automatic_sam_aggregation_boundary_variant_plan.jsonl"),
            _jsonl(args.evidence_graph_root / scene / "nodes.jsonl"),
            _jsonl(args.automatic_root / scene / "automatic_observations.jsonl"),
            _jsonl(args.automatic_root / scene / "same_frame_mask_relations.jsonl"),
            args.dataset_root / scene, model, args.device, args.batch_size,
        )
        target = args.output_root / scene
        target.mkdir()
        _write_jsonl(target / "automatic_sam_boundary_dino_appearance_ledger.jsonl", rows)
        totals["boundary_relation_count"] += len(rows)
        totals["encoded_observation_count"] += feature_count
        totals["frame_pair_count"] += sum(len(row["frame_records"]) for row in rows)
        print(f"[场景完成] {index}: {scene}，边界关系 {len(rows)}，编码观测 {feature_count}", flush=True)
    summary = {
        "purpose": "为真实二维层级边界竞争关系提供冻结 DINOv2 区域外观连续证据。",
        "model": MODEL_NAME, "checkpoint": str(args.checkpoint), "gt_usage": "none",
        "decision_state": "不设阈值、不选择同实例、不聚合、不删除、不改类别或分数，不输出预测或 AP。",
        "scene_count": len(_scenes(args.scene_list)), **dict(totals), "params": vars(args),
    }
    (args.output_root / "automatic_sam_boundary_dino_appearance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
