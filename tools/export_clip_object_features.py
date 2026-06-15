import argparse
import json
import os
import os.path as osp
import sys

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizerFast

from utils.backprojection_fusion import load_backprojection_candidates


def _load_labels(dataset_name):
    with open(osp.join(REPO_ROOT, "pretrained", f"config_{dataset_name}.yaml")) as f:
        config = yaml.safe_load(f)
    return config["network2d"]["text_prompts"]


def _resolve_path(path, source_json=None):
    if path is None:
        return None
    if osp.exists(path):
        return path
    if source_json is not None:
        candidate = osp.join(osp.dirname(source_json), path)
        if osp.exists(candidate):
            return candidate
    return path


def _candidate_crop_path(candidate, image_field):
    evidence = candidate.get("evidence") or {}
    path = candidate.get(image_field) or evidence.get(image_field)
    return _resolve_path(path, candidate.get("_source_json"))


def _softmax(values):
    values = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.maximum(exp_values.sum(axis=-1, keepdims=True), 1e-12)


def _text_prompts(labels, template):
    return [template.format(label=label.replace("-", " ")) for label in labels]


def _load_clip_processor(clip_model_path):
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_path, local_files_only=True)
    image_processor = CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": 224},
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    return CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)


@torch.no_grad()
def export_clip_object_features(
    dataset_name,
    candidates_dir,
    output_dir,
    clip_model_path,
    image_field="crop_path",
    prompt_template="a photo of a {label}",
    batch_size=32,
    topk=5,
    device=None,
):
    labels = _load_labels(dataset_name)
    candidates_by_scene, summary = load_backprojection_candidates(candidates_dir)
    if summary["loaded"] == 0:
        raise RuntimeError(f"No backprojection candidates found under {candidates_dir}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = _load_clip_processor(clip_model_path)
    model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
    model.eval()

    text_inputs = processor(
        text=_text_prompts(labels, prompt_template),
        padding=True,
        return_tensors="pt",
    ).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp().detach().float()

    os.makedirs(output_dir, exist_ok=True)
    total_exported = 0
    per_scene_counts = {}
    for scene_name, candidates in tqdm(sorted(candidates_by_scene.items())):
        scene_dir = osp.join(output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        records = []

        batch_images = []
        batch_candidates = []
        for candidate in candidates:
            crop_path = _candidate_crop_path(candidate, image_field)
            if crop_path is None or not osp.exists(crop_path):
                continue
            try:
                image = Image.open(crop_path).convert("RGB")
            except Exception:
                continue
            batch_images.append(image)
            batch_candidates.append((candidate, crop_path))
            if len(batch_images) >= batch_size:
                records.extend(
                    _encode_batch(
                        model,
                        processor,
                        text_features,
                        logit_scale,
                        batch_images,
                        batch_candidates,
                        labels,
                        topk,
                        device,
                    )
                )
                batch_images = []
                batch_candidates = []

        if batch_images:
            records.extend(
                _encode_batch(
                    model,
                    processor,
                    text_features,
                    logit_scale,
                    batch_images,
                    batch_candidates,
                    labels,
                    topk,
                    device,
                )
            )

        output_json = osp.join(scene_dir, "clip_object_features.json")
        with open(output_json, "w") as f:
            json.dump(
                {
                    "scene_name": scene_name,
                    "labels": labels,
                    "clip_model_path": clip_model_path,
                    "image_field": image_field,
                    "prompt_template": prompt_template,
                    "features": records,
                },
                f,
                indent=2,
            )
        per_scene_counts[scene_name] = len(records)
        total_exported += len(records)

    summary_path = osp.join(output_dir, "clip_object_feature_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "candidates_dir": candidates_dir,
                "clip_model_path": clip_model_path,
                "image_field": image_field,
                "prompt_template": prompt_template,
                "candidate_summary": summary,
                "total_exported": total_exported,
                "per_scene_counts": per_scene_counts,
            },
            f,
            indent=2,
        )
    print(f"Saved {total_exported} CLIP object features to {output_dir}")
    print(f"Saved summary to {summary_path}")


def _encode_batch(
    model,
    processor,
    text_features,
    logit_scale,
    images,
    candidate_items,
    labels,
    topk,
    device,
):
    inputs = processor(images=images, return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = (logit_scale * image_features @ text_features.T).detach().cpu().numpy()
    probs = _softmax(logits).astype(np.float32)

    records = []
    for row, (candidate, crop_path) in zip(probs, candidate_items):
        top_ids = np.argsort(-row)[:topk]
        records.append(
            {
                "scene_name": candidate.get("scene_name"),
                "candidate_id": candidate.get("candidate_id"),
                "frame_id": candidate.get("frame_id"),
                "frame_index": candidate.get("frame_index"),
                "detection_id": candidate.get("detection_id"),
                "detector_class_id": candidate.get("class_id"),
                "detector_class_name": candidate.get("class_name"),
                "detector_score": candidate.get("score"),
                "fusion_score": candidate.get("fusion_score"),
                "support_view_count": candidate.get("support_view_count", 1),
                "num_seed_points": candidate.get("num_seed_points", 0),
                "seed_points_path": candidate.get("refined_seed_points_path") or candidate.get("seed_points_path"),
                "candidate_source_json": candidate.get("_source_json"),
                "crop_path": crop_path,
                "clip_probs": [float(v) for v in row.tolist()],
                "clip_topk": [
                    {
                        "class_id": int(class_id),
                        "class_name": labels[int(class_id)],
                        "prob": float(row[int(class_id)]),
                    }
                    for class_id in top_ids
                ],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--candidates_dir", default="./output/backprojection_candidates_replica_mv_m20")
    parser.add_argument("--output_dir", default="./output/clip_object_features_replica_mv_m20")
    parser.add_argument("--clip_model_path", default="./pretrained/clip-vit-base-patch32")
    parser.add_argument("--image_field", default="crop_path", choices=["crop_path", "overlay_path", "context_path"])
    parser.add_argument("--prompt_template", default="a photo of a {label}")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--topk", default=5, type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    export_clip_object_features(
        dataset_name=args.dataset_name,
        candidates_dir=args.candidates_dir,
        output_dir=args.output_dir,
        clip_model_path=args.clip_model_path,
        image_field=args.image_field,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
        topk=args.topk,
        device=args.device,
    )


if __name__ == "__main__":
    main()
