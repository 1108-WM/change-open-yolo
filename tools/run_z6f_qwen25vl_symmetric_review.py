#!/usr/bin/env python3
"""Run two-order symmetric Qwen2.5-VL reviews with conservative agreement."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_z6e_qwen25vl_selective_review import _crop, _read_jsonl, _resolve  # noqa: E402


ALLOWED = {"A", "B", "ABSTAIN"}


def _prompt(label_a: str, label_b: str) -> str:
    return (
        "The three images are fixed views of the same 3D object crop. "
        "Choose whether the visible object is better described by Option A or Option B. "
        "If unclear, partly absent, or neither option is reliably supported, abstain.\n"
        f"Option A: {label_a}\nOption B: {label_b}\n"
        "Return exactly one token: A, B, or ABSTAIN."
    )


def _parse(text: str) -> tuple[str, bool]:
    normalized = text.strip().upper().strip(". \n\t")
    if normalized in ALLOWED:
        return normalized, True
    return "ABSTAIN", False


def _semantic_choice(output: str, proposed_is_a: bool) -> str:
    if output == "ABSTAIN":
        return "ABSTAIN"
    chose_a = output == "A"
    return "PROPOSED" if chose_a == proposed_is_a else "CURRENT"


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
    parser.add_argument("--safety60-transfer", action="store_true")
    args = parser.parse_args()
    for name in ("manifest_root", "model_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    manifest = _read_jsonl(args.manifest_root / "review_manifest.jsonl")
    stop = len(manifest) if args.count is None else min(len(manifest), args.start_index + args.count)
    selected = manifest[args.start_index:stop]
    if not selected:
        raise ValueError("empty requested review slice")

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Z6f symmetric Qwen review requires CUDA")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        local_files_only=True,
    )

    def infer(images: list[Image.Image], label_a: str, label_b: str) -> tuple[str, str, bool]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": _prompt(label_a, label_b)})
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
        raw = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed, valid = _parse(raw)
        del inputs, generated, image_inputs, video_inputs
        gc.collect(); torch.cuda.empty_cache()
        return parsed, raw, valid

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "review_outputs.jsonl"
    counts = Counter()
    with output_path.open("w") as handle:
        for offset, row in enumerate(selected, 1):
            images = [_crop(view["rgb_path"], view["bbox_xyxy"]) for view in row["views"]]
            first, raw_first, valid_first = infer(
                images, row["current_class_name"], row["proposed_class_name"]
            )
            second, raw_second, valid_second = infer(
                images, row["proposed_class_name"], row["current_class_name"]
            )
            semantic_first = _semantic_choice(first, proposed_is_a=False)
            semantic_second = _semantic_choice(second, proposed_is_a=True)
            if semantic_first == semantic_second and semantic_first in {"CURRENT", "PROPOSED"}:
                decision = semantic_first
                agreement_state = "symmetric_agreement"
            else:
                decision = "ABSTAIN"
                agreement_state = "order_disagreement_or_abstain"
            counts[decision] += 1
            counts[agreement_state] += 1
            counts["valid_both"] += int(valid_first and valid_second)
            handle.write(json.dumps({
                **row,
                "order_current_first": {
                    "raw_output": raw_first, "parsed_output": first,
                    "semantic_choice": semantic_first, "valid": valid_first,
                },
                "order_proposed_first": {
                    "raw_output": raw_second, "parsed_output": second,
                    "semantic_choice": semantic_second, "valid": valid_second,
                },
                "model_decision": decision,
                "agreement_state": agreement_state,
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
                f"[Z6f symmetric review] {offset}/{len(selected)} index={row['review_index']} "
                f"first={semantic_first} second={semantic_second} final={decision}", flush=True,
            )
            del images

    payload = {
        "diagnostic_type": "frozen symmetric-order selective Qwen2.5-VL review",
        "review_count": len(selected), "start_index": args.start_index,
        "decision_counts": dict(counts),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "decision_contract": "run current-first and proposed-first; mutate only if both semantic choices are PROPOSED; disagreement/abstain keeps current",
        "ground_truth_usage": "none", "candidate_mutation": False,
        "geometry_mutation": False, "score_mutation": False,
        "inference_plan_written": False, "safety60_read": bool(args.safety60_transfer),
        "even48_read": False, "test60_read": False,
        "params": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
