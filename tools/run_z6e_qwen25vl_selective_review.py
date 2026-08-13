#!/usr/bin/env python3
"""Run frozen Qwen2.5-VL multi-view reviews with strict abstain fallback."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"CURRENT", "PROPOSED", "ABSTAIN"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _crop(path: str, bbox: list[int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        left, top, right, bottom = bbox
        left = max(0, min(left, image.width - 1))
        top = max(0, min(top, image.height - 1))
        right = max(left + 1, min(right, image.width))
        bottom = max(top + 1, min(bottom, image.height))
        return image.crop((left, top, right, bottom))


def _prompt(row: dict) -> str:
    return (
        "The three images are fixed views of the same 3D object crop. "
        "Choose whether the object is better described by the CURRENT label or the PROPOSED label. "
        "Use only visible object appearance across all views. If the object is unclear, partly absent, "
        "or neither label is reliably supported, abstain.\n"
        f"CURRENT: {row['current_class_name']}\n"
        f"PROPOSED: {row['proposed_class_name']}\n"
        "Return exactly one word: CURRENT, PROPOSED, or ABSTAIN."
    )


def _parse(text: str) -> tuple[str, bool]:
    normalized = text.strip().upper().strip(". \n\t")
    if normalized in ALLOWED:
        return normalized, True
    return "ABSTAIN", False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root", type=Path,
        default=Path("docs/diagnostics/z6e_selective_vlm_review_manifest_official100_20260812"),
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("pretrained/checkpoints/Qwen2.5-VL-7B-Instruct"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--min-pixels", type=int, default=100352)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()
    for name in ("manifest_root", "model_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    manifest_path = args.manifest_root / "review_manifest.jsonl"
    manifest = _read_jsonl(manifest_path)
    stop = len(manifest) if args.count is None else min(len(manifest), args.start_index + args.count)
    selected = manifest[args.start_index:stop]
    if not selected:
        raise ValueError("empty requested review slice")

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Z6e Qwen review requires CUDA; CPU fallback is forbidden")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        local_files_only=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "review_outputs.jsonl"
    counts = Counter()
    with output_path.open("w") as handle:
        for offset, row in enumerate(selected, 1):
            images = [_crop(view["rgb_path"], view["bbox_xyxy"]) for view in row["views"]]
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": _prompt(row)})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            generated = generated[:, inputs.input_ids.shape[1]:]
            raw_output = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            decision, valid = _parse(raw_output)
            counts[decision] += 1
            counts["valid_output"] += int(valid)
            counts["invalid_fallback"] += int(not valid)
            handle.write(json.dumps({
                **row,
                "model_decision": decision,
                "raw_model_output": raw_output,
                "valid_strict_output": valid,
                "selected_class_index": (
                    int(row["proposed_class_index"])
                    if decision == "PROPOSED" else int(row["current_class_index"])
                ),
                "class_mutated": decision == "PROPOSED",
                "fallback_applied": decision != "PROPOSED",
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            }, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[Z6e Qwen review] {offset}/{len(selected)} index={row['review_index']} "
                f"decision={decision} valid={valid}", flush=True,
            )
            del inputs, generated, images, image_inputs, video_inputs
            gc.collect()
            torch.cuda.empty_cache()

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    payload = {
        "diagnostic_type": "official100 frozen selective Qwen2.5-VL multi-view review",
        "review_count": len(selected), "start_index": args.start_index,
        "decision_counts": dict(counts), "output_sha256": digest,
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "generation_contract": "deterministic greedy, max_new_tokens=8, exact CURRENT/PROPOSED/ABSTAIN parser",
        "fallback_contract": "ABSTAIN or invalid output keeps prediction-specific current class",
        "ground_truth_usage": "none", "candidate_mutation": False,
        "geometry_mutation": False, "score_mutation": False,
        "inference_plan_written": False, "safety60_read": False,
        "even48_read": False, "test60_read": False,
        "params": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
