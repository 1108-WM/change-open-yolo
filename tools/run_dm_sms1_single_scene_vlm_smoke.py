#!/usr/bin/env python3
"""Run one real local Qwen2.5-VL no-GT attribute/arbitration smoke task."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError as error:
        raise ValueError("model output is not exactly one JSON object") from error
    raise ValueError("model output is not a JSON object")


def _attribute_prompt(row: dict) -> str:
    view_ranks = [int(view["view_rank"]) for view in row.get("view_inputs", [])]
    if not view_ranks or view_ranks != list(range(len(view_ranks))):
        raise ValueError("attribute input view ranks must be contiguous from zero")
    rank_text = "、".join(str(rank) for rank in view_ranks)
    return (
        f"你将看到 {len(view_ranks)} 张图片，每张图片左侧是完整场景红框，右侧是同一视角的局部裁剪；"
        f"图片的视角编号固定为 {rank_text}，不存在其他视角编号。"
        "只描述红框目标，不描述框外物体。禁止猜测类别，禁止使用家具、建筑构件或其他物体类别名词；"
        "只能写颜色、纹理、材料、几何形状、部件结构、可观察的使用方式和相对空间位置。"
        "所有 observation、counterevidence 和缺失证据文字必须使用简短英文，不得使用中文；"
        "不要把红框目标的组成部分命名为父类别；例如写 flat horizontal surface，不要写 table，"
        "写 vertical rectangular panel，不要写 door，写 supports sitting，不要写 chair。"
        "只有 spatial_context 可以提到周围物体，而且只能用于 on/near/next to 等位置关系；"
        "无法用纯属性表达时写 unknown。"
        "必须把所有视角的证据合并为一个顶层 JSON 对象；禁止输出数组，禁止为每张图片分别输出对象。\n"
        "\n只返回 JSON，不要使用 Markdown 代码块。字段必须严格为：" +
        json.dumps(row["response_schema"], ensure_ascii=False)
    )


def _candidate_prompt(prompt: str, attribute: dict, hypotheses: list[dict], order_names: list[str]) -> str:
    by_name = {item["class_name"]: item for item in hypotheses}
    template = {
        "candidate_results": [{
            "class_index": int(by_name[name]["class_index"]),
            "supported": False,
            "strong_counterevidence": False,
            "support_evidence": "",
            "counterevidence": "",
            "confidence": 0.0,
        } for name in order_names]
    }
    return (
        prompt + "\n候选名称与编号映射：" +
        json.dumps([{"class_index": item["class_index"], "class_name": item["class_name"]} for item in hypotheses], ensure_ascii=False) +
        "\n下面是前一步得到的无类别属性证据：\n" +
        json.dumps(attribute, ensure_ascii=False, sort_keys=True) +
        "\n顶层字段必须是 candidate_results，数组顺序和候选数量不得改变。严格复制下面模板后填写值：" +
        json.dumps(template, ensure_ascii=False) +
        "\n每项字段为 class_index、supported（布尔值）、"
        "strong_counterevidence（布尔值）、support_evidence、counterevidence、confidence（0到1）。"
        "只返回 JSON，不要使用 Markdown 代码块。"
    )


def _target_images(attribute_row: dict) -> list[Image.Image]:
    composites = []
    for view in attribute_row["view_inputs"]:
        with Image.open(view["rgb_path"]) as source:
            image = source.convert("RGB")
        x1, y1, x2, y2 = map(float, view["sam_box_prompt_xyxy"])
        width, height = image.size
        x1, x2 = max(0.0, x1), min(float(width - 1), x2)
        y1, y2 = max(0.0, y1), min(float(height - 1), y2)
        marked = image.copy()
        ImageDraw.Draw(marked).rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=6)
        pad_x = 0.2 * max(1.0, x2 - x1)
        pad_y = 0.2 * max(1.0, y2 - y1)
        crop_box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                    min(width, int(x2 + pad_x) + 1), min(height, int(y2 + pad_y) + 1))
        crop = image.crop(crop_box)
        # Give the full-scene panel and the local panel equal space.  Keeping
        # the crop at its native size made small targets almost invisible
        # after the multimodal processor resized the whole composite.
        resampling = getattr(Image, "Resampling", Image)
        crop = ImageOps.contain(crop, marked.size, method=resampling.LANCZOS)
        canvas = Image.new("RGB", (marked.width * 2, marked.height), (255, 255, 255))
        canvas.paste(marked, (0, 0))
        canvas.paste(crop, (
            marked.width + (marked.width - crop.width) // 2,
            (marked.height - crop.height) // 2,
        ))
        composites.append(canvas)
    return composites


def _validate_candidate_output(value: dict, expected_order: list[int]) -> None:
    items = value.get("candidate_results")
    if not isinstance(items, list) or len(items) != len(expected_order):
        raise ValueError("candidate output did not preserve the required ordered candidate array")
    observed_order = []
    for item in items:
        class_index = item.get("class_index") if isinstance(item, dict) else None
        if isinstance(class_index, bool) or not isinstance(class_index, int):
            raise ValueError("candidate output has a non-integer class_index")
        observed_order.append(class_index)
    if observed_order != expected_order:
        raise ValueError("candidate output did not preserve the required ordered candidate array")
    for item in items:
        if not isinstance(item.get("supported"), bool):
            raise ValueError("candidate output has an invalid supported flag")
        if not isinstance(item.get("strong_counterevidence"), bool):
            raise ValueError("candidate output has an invalid strong_counterevidence flag")
        if not isinstance(item.get("support_evidence"), str) or not isinstance(item.get("counterevidence"), str):
            raise ValueError("candidate output misses evidence text")
        if item["supported"] and not item["support_evidence"].strip():
            raise ValueError("candidate output marks support true without non-empty support evidence")
        if item["strong_counterevidence"] and not item["counterevidence"].strip():
            raise ValueError(
                "candidate output marks strong counterevidence true without non-empty counterevidence"
            )
        confidence = item.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0):
            raise ValueError("candidate output has invalid confidence")


def run(args: argparse.Namespace) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("single-scene VLM smoke requires CUDA; no CPU fallback is allowed")
    attributes = _read_rows(args.attribute_manifest)
    candidates = _read_rows(args.candidate_manifest)
    if not attributes or not candidates:
        raise ValueError("empty input manifest")
    attribute_row = attributes[args.index]
    candidate_row = next(row for row in candidates if row["task_id"] == attribute_row["task_id"])
    images = _target_images(attribute_row)
    args.output_root.mkdir(parents=True, exist_ok=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, min_pixels=100352, max_pixels=200704, local_files_only=True)

    def infer(prompt: str) -> tuple[str, dict]:
        messages = [{"role": "user", "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return raw, _json_object(raw)

    raw_attribute, attribute_output = infer(_attribute_prompt(attribute_row))
    (args.output_root / "attribute_raw.txt").write_text(raw_attribute)
    (args.output_root / "attribute_output.json").write_text(json.dumps(attribute_output, ensure_ascii=False, indent=2) + "\n")
    raw_ab, output_ab = infer(_candidate_prompt(candidate_row["evidence_prompt_ab"], attribute_output, candidate_row["candidate_hypotheses"], candidate_row["candidate_order_ab"]))
    (args.output_root / "order_ab_raw.txt").write_text(raw_ab)
    (args.output_root / "order_ab_output.json").write_text(json.dumps(output_ab, ensure_ascii=False, indent=2) + "\n")
    expected_ab = [next(item["class_index"] for item in candidate_row["candidate_hypotheses"] if item["class_name"] == name) for name in candidate_row["candidate_order_ab"]]
    _validate_candidate_output(output_ab, expected_ab)
    raw_ba, output_ba = infer(_candidate_prompt(candidate_row["evidence_prompt_ba"], attribute_output, candidate_row["candidate_hypotheses"], candidate_row["candidate_order_ba"]))
    (args.output_root / "order_ba_raw.txt").write_text(raw_ba)
    (args.output_root / "order_ba_output.json").write_text(json.dumps(output_ba, ensure_ascii=False, indent=2) + "\n")
    expected_ba = [next(item["class_index"] for item in candidate_row["candidate_hypotheses"] if item["class_name"] == name) for name in candidate_row["candidate_order_ba"]]
    _validate_candidate_output(output_ba, expected_ba)
    result = {
        "task_id": attribute_row["task_id"],
        "scene_name": attribute_row["scene_name"],
        "geometry_hash": attribute_row["geometry_hash"],
        "model_id": "Qwen2.5-VL-7B-Instruct",
        "model_path": str(args.model_dir),
        "attribute_raw_output": raw_attribute,
        "attribute_output": attribute_output,
        "order_ab_raw_output": raw_ab,
        "order_ab": output_ab,
        "order_ba_raw_output": raw_ba,
        "order_ba": output_ba,
        "candidate_labels_hidden": False,
        "class_decision_made": False,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    (args.output_root / "single_scene_vlm_output.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    summary = {
        "version": "dm_sms1_single_scene_vlm_smoke_v1",
        "task_id": result["task_id"],
        "scene_name": result["scene_name"],
        "geometry_hash": result["geometry_hash"],
        "model_id": result["model_id"],
        "attribute_output_present": True,
        "order_ab_output_present": True,
        "order_ba_output_present": True,
        "class_decision_made": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribute-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("pretrained/checkpoints/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
