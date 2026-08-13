#!/usr/bin/env python3
"""Build a no-GT DINOv2 masked-crop appearance ledger for frozen N2 medoids."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "vit_small_patch14_dinov2"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
CONTRACT = (
    "No-GT N2 DINOv2 masked-crop appearance ledger only; frozen candidate geometry, "
    "components, scores, selection, materialization, and AP remain unchanged."
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def scenes(path):
    result = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or duplicated")
    return result


def decode_rle(payload):
    height, width = (int(value) for value in payload["size"])
    counts = np.asarray(payload["counts"], dtype=np.int64)
    if int(counts.sum()) != height * width:
        raise ValueError("RLE size mismatch")
    values = np.empty(height * width, dtype=bool)
    cursor, foreground = 0, False
    for count in counts:
        values[cursor:cursor + count] = foreground
        cursor += int(count)
        foreground = not foreground
    return values.reshape((height, width), order="F")


def crop_tensor(record, image, output_size):
    mask = decode_rle(record["source_mask_rle"])
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty source mask")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    pad = max(2, int(round(max(right - left, bottom - top) * .05)))
    left, right = max(0, left - pad), min(mask.shape[1], right + pad)
    top, bottom = max(0, top - pad), min(mask.shape[0], bottom + pad)
    crop, crop_mask = image[top:bottom, left:right].copy(), mask[top:bottom, left:right]
    crop[~crop_mask] = 127
    resized = np.asarray(Image.fromarray(crop).resize((output_size, output_size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(((resized - MEAN) / STD).transpose(2, 0, 1))


def cosine(left, right):
    return float(np.dot(left, right) / max(1e-12, np.linalg.norm(left) * np.linalg.norm(right)))


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or set(unexpected) != {"mask_token"}:
        raise ValueError(f"DINOv2 checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.cuda().eval()


def encode_records(records, dataset_scene_root, model, batch_size):
    pending, ids, features = [], [], {}
    input_size = int(model.patch_embed.img_size[0])

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            output = model(torch.stack(pending).cuda(non_blocking=True)).detach().float().cpu().numpy()
        for key, value in zip(ids, output):
            features[key] = value
        pending.clear()
        ids.clear()

    image_cache = {}
    for key, record in sorted(records.items()):
        frame_id = str(record["frame_id"])
        if frame_id not in image_cache:
            image_cache[frame_id] = np.asarray(Image.open(dataset_scene_root / "color" / f"{frame_id}.jpg").convert("RGB"))
        pending.append(crop_tensor(record, image_cache[frame_id], input_size))
        ids.append(key)
        if len(pending) >= batch_size:
            flush()
    flush()
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--n2-cache-root", type=Path, required=True)
    parser.add_argument("--variant-ledger-root", type=Path, required=True)
    parser.add_argument("--candidate-ledger-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/checkpoints/dinov2_vits14_pretrain.pth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "n2_cache_root", "variant_ledger_root", "candidate_ledger_root", "dataset_root", "checkpoint", "output_root"):
        setattr(args, name, resolve(getattr(args, name)))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing CPU fallback for full DINOv2 ledger")
    if not args.checkpoint.is_file() or not args.checkpoint.stat().st_size:
        raise SystemExit("DINOv2 checkpoint is missing")
    if args.scene_offset < 0:
        raise SystemExit("scene offset must be non-negative")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit("output root is non-empty; use --resume only for this exact frozen ledger")
    chosen_scenes = scenes(args.scene_list)[args.scene_offset:]
    if args.max_scenes is not None:
        chosen_scenes = chosen_scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint)
    summaries = []
    for ordinal, scene in enumerate(chosen_scenes, 1):
        existing = args.output_root / scene / "summary.json"
        if args.resume and existing.is_file():
            summaries.append(json.loads(existing.read_text()))
            print(f"[N2 DINOv2 外观账本] {ordinal}/{len(chosen_scenes)} {scene}: 已完成", flush=True)
            continue
        cache = json.loads((args.n2_cache_root / scene / "n2_medoid_candidates.json").read_text())["candidates"]
        variants = {row["candidate_family_key"]: row for row in (json.loads(line) for line in (args.variant_ledger_root / scene / "family_formation_variant_ledger.jsonl").read_text().splitlines() if line) if row["variant_name"] == "cross_view_consistency_medoid"}
        observations = {row["candidate_id"]: row for row in (json.loads(line) for line in (args.candidate_ledger_root / scene / "observation_candidate_ledger.jsonl").read_text().splitlines() if line)}
        needed = {}
        candidate_sources = {}
        for candidate_id, candidate in enumerate(cache):
            variant = variants[candidate["canonical_family_key"]]
            medoid = variant["source_candidate_ids"][0]
            selected = [str(value) for _, value in sorted(variant["selected_one_member_per_view"].items(), key=lambda item: int(item[0]))]
            source_ids = list(dict.fromkeys([medoid, *selected]))
            if any(source_id not in observations for source_id in source_ids):
                raise ValueError(f"{scene}: missing frozen source observation")
            candidate_sources[candidate_id] = (medoid, source_ids)
            needed.update({source_id: observations[source_id] for source_id in source_ids})
        features = encode_records(needed, args.dataset_root / scene, model, args.batch_size)
        rows = []
        for candidate_id, (medoid, source_ids) in sorted(candidate_sources.items()):
            similarities = [cosine(features[medoid], features[source_id]) for source_id in source_ids if source_id != medoid]
            rows.append({
                "scene_name": scene, "candidate_id": candidate_id,
                "canonical_family_key": cache[candidate_id]["canonical_family_key"],
                "medoid_source_candidate_id": medoid, "selected_view_source_candidate_ids": source_ids,
                "dino_selected_view_count": len(source_ids),
                "dino_medoid_to_view_cosine_mean": float(np.mean(similarities)) if similarities else 1.0,
                "dino_medoid_to_view_cosine_min": float(np.min(similarities)) if similarities else 1.0,
                "dino_medoid_to_view_cosine_std": float(np.std(similarities)) if similarities else 0.0,
                "ground_truth_usage": "none", "proposal_materialization_applied": False, "ap_computed": False,
                "decision_constraint": CONTRACT,
            })
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"
        stage.mkdir()
        with (stage / "n2_medoid_dinov2_appearance_ledger.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {"scene_name": scene, "candidate_count": len(rows), "encoded_observation_count": len(features), "ground_truth_usage": "none", "proposal_materialization_applied": False, "ap_computed": False}
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        target = args.output_root / scene
        if target.exists():
            if (target / "summary.json").is_file():
                shutil.rmtree(stage)
                summaries.append(json.loads((target / "summary.json").read_text()))
                print(f"[N2 DINOv2 外观账本] {ordinal}/{len(chosen_scenes)} {scene}: 并发发布已完成", flush=True)
                continue
            raise FileExistsError(f"{scene}: target exists without an atomic completion summary")
        os.replace(stage, target)
        summaries.append(summary)
        print(f"[N2 DINOv2 外观账本] {ordinal}/{len(chosen_scenes)} {scene}: {len(features)}", flush=True)
    complete_summaries = [json.loads(path.read_text()) for path in args.output_root.glob("scene*/summary.json")]
    root = {"diagnostic_type": "no-GT frozen N2 DINOv2 masked-crop appearance ledger", "decision_constraint": CONTRACT, "model": MODEL_NAME, "scene_count": len(complete_summaries), "candidate_count": sum(row["candidate_count"] for row in complete_summaries), "encoded_observation_count": sum(row["encoded_observation_count"] for row in complete_summaries), "proposal_materialization_applied": False, "ap_computed": False, "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
