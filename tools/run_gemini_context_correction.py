import argparse
import base64
import json
import mimetypes
import os
import os.path as osp
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import yaml

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_env_file(path):
    values = {}
    if path is None or not osp.exists(path):
        return values
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(path):
    if path is None:
        return None
    if osp.isabs(path):
        return path
    return osp.normpath(osp.join(REPO_ROOT, path))


def iter_candidate_jsons(path):
    if osp.isfile(path):
        yield path
        return

    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "context_candidates.json":
                yield osp.join(root, filename)


def load_candidates(path):
    with open(path) as f:
        payload = json.load(f)
    scene_name = payload.get("scene_name")
    for candidate in payload.get("candidates", []):
        item = dict(candidate)
        item.setdefault("scene_name", scene_name)
        item["_source_json"] = path
        yield item


def load_existing_keys(path):
    keys = set()
    if not osp.exists(path):
        return keys
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            keys.add((record.get("scene_name"), int(record.get("mask_id"))))
    return keys


def image_part(path):
    abs_path = resolve_path(path)
    if abs_path is None or not osp.exists(abs_path):
        return None
    mime_type, _ = mimetypes.guess_type(abs_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(abs_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return {"inline_data": {"mime_type": mime_type, "data": data}}


def build_prompt(candidate, labels):
    label_distribution = candidate.get("label_distribution", {})
    label_distribution_text = ", ".join(
        f"{name}: {count}" for name, count in sorted(label_distribution.items())
    )
    quality = candidate.get("quality", {})
    quality_text = ", ".join(f"{key}: {value}" for key, value in sorted(quality.items()))
    vocabulary = ", ".join(labels)
    return (
        "You are correcting open-vocabulary 3D instance segmentation labels.\n"
        "Use the full RGB context image, the red overlay image, and the crop. "
        "The red overlay marks the target 3D instance projection. "
        "Reason about surrounding scene context and the highlighted mask, not just one crop.\n\n"
        "Return JSON only with this schema:\n"
        "{"
        "\"corrected_class_name\": string, "
        "\"confidence\": number, "
        "\"decision\": \"change\" | \"keep\" | \"unknown\" | \"bad_mask\", "
        "\"reason\": string"
        "}\n\n"
        "Rules:\n"
        "- corrected_class_name must be exactly one item from the dataset vocabulary when decision is change or keep.\n"
        "- If the object is not a valid instance or is outside the vocabulary, set decision to unknown and corrected_class_name to the original predicted class.\n"
        "- If the red mask mainly covers a wall, ceiling, floor, large background surface, or multiple objects, set decision to bad_mask and corrected_class_name to the original predicted class.\n"
        "- Do not use change unless the target mask clearly isolates one object and the new class is more plausible than the original class.\n"
        "- Use keep when the original class is plausible but confidence should not be changed.\n"
        "- Use confidence from 0.0 to 1.0.\n"
        "- Prefer keeping the original class when evidence is weak.\n\n"
        f"Dataset vocabulary: {vocabulary}\n"
        f"Scene: {candidate.get('scene_name')}\n"
        f"Mask id: {candidate.get('mask_id')}\n"
        f"Original predicted class: {candidate.get('predicted_class_name')}\n"
        f"Original score: {candidate.get('predicted_score')}\n"
        f"Label margin: {candidate.get('label_margin')}\n"
        f"Selection reasons: {candidate.get('selection_reasons')}\n"
        f"Quality reasons: {candidate.get('quality_reasons')}\n"
        f"Quality metrics: {quality_text}\n"
        f"2D label distribution over visible points: {label_distribution_text}\n"
    )


def candidate_parts(candidate, labels):
    parts = [{"text": build_prompt(candidate, labels)}]
    views = candidate.get("views", [])
    if views:
        best_view = views[0]
        for key in ("color_path", "overlay_path", "crop_path", "mask_path"):
            part = image_part(best_view.get(key))
            if part is not None:
                parts.append(part)
    return parts


def _call_gemini_urllib(url, payload, api_key, timeout):
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_gemini_curl(url, payload, api_key, timeout):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as payload_file:
        json.dump(payload, payload_file)
        payload_path = payload_file.name
    with tempfile.NamedTemporaryFile("w", suffix=".curl", delete=False) as config_file:
        config_file.write(f'url = "{url}"\n')
        config_file.write('header = "Content-Type: application/json"\n')
        config_file.write(f'header = "x-goog-api-key: {api_key}"\n')
        config_path = config_file.name

    try:
        os.chmod(config_path, 0o600)
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "--fail-with-body",
                "--max-time",
                str(timeout),
                "-K",
                config_path,
                "--data-binary",
                f"@{payload_path}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        os.unlink(payload_path)
        os.unlink(config_path)

    if result.returncode != 0:
        raise RuntimeError((result.stderr + "\n" + result.stdout).strip())
    return json.loads(result.stdout)


def call_gemini(parts, api_key, model, temperature, timeout, retries, transport):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            if transport == "curl":
                return _call_gemini_curl(url, payload, api_key, timeout)
            return _call_gemini_urllib(url, payload, api_key, timeout)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {error_body}"
        except (urllib.error.URLError, RuntimeError) as exc:
            last_error = str(exc)

        if attempt < retries:
            time.sleep(2 ** attempt)

    raise RuntimeError(last_error)


def extract_response_text(response):
    candidates = response.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def parse_model_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def normalize_output(model_record, candidate, labels):
    label_set = set(labels)
    original = candidate.get("predicted_class_name")
    corrected = str(model_record.get("corrected_class_name", original)).strip()
    decision = str(model_record.get("decision", "keep")).strip().lower()
    if decision not in {"change", "keep", "unknown", "bad_mask"}:
        decision = "unknown"
    if corrected not in label_set:
        corrected = original
        if decision in {"change", "keep"}:
            decision = "unknown"
    if decision in {"unknown", "bad_mask"}:
        corrected = original

    try:
        confidence = float(model_record.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "scene_name": candidate.get("scene_name"),
        "mask_id": int(candidate.get("mask_id")),
        "predicted_class_name": original,
        "predicted_score": candidate.get("predicted_score"),
        "label_margin": candidate.get("label_margin"),
        "quality": candidate.get("quality", {}),
        "quality_reasons": candidate.get("quality_reasons", []),
        "corrected_class_name": corrected,
        "confidence": confidence,
        "decision": decision,
        "reason": str(model_record.get("reason", "")),
        "source_json": candidate.get("_source_json"),
        "model": model_record.get("_model"),
    }


def run(args):
    env_values = load_env_file(args.env_file)
    api_key = os.environ.get("GEMINI_API_KEY") or env_values.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. Export it or write it to --env_file."
        )

    config = load_yaml(osp.join(REPO_ROOT, f"pretrained/config_{args.dataset_name}.yaml"))
    labels = config["network2d"]["text_prompts"]
    candidate_jsons = list(iter_candidate_jsons(resolve_path(args.context_candidates)))
    if not candidate_jsons:
        raise FileNotFoundError(
            f"No context_candidates.json files found under {args.context_candidates}"
        )
    candidates = []
    for json_path in candidate_jsons:
        candidates.extend(load_candidates(json_path))

    candidates = sorted(
        candidates,
        key=lambda item: (
            item.get("scene_name") or "",
            int(item.get("mask_id", -1)),
        ),
    )
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    output_path = resolve_path(args.output_jsonl)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    existing = set() if args.force else load_existing_keys(output_path)

    written = 0
    with open(output_path, "a") as out:
        for index, candidate in enumerate(candidates, start=1):
            key = (candidate.get("scene_name"), int(candidate.get("mask_id")))
            if key in existing:
                continue

            print(
                f"[{index}/{len(candidates)}] {key[0]} mask {key[1]} "
                f"pred={candidate.get('predicted_class_name')}"
            )
            if args.dry_run:
                continue

            parts = candidate_parts(candidate, labels)
            response = call_gemini(
                parts,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
                transport=args.transport,
            )
            text = extract_response_text(response)
            model_record = parse_model_json(text)
            model_record["_model"] = args.model
            record = normalize_output(model_record, candidate, labels)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Saved {written} Gemini corrections to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--context_candidates", default="./output/context_candidates")
    parser.add_argument("--output_jsonl", default="./output/context_corrections_gemini.jsonl")
    parser.add_argument("--env_file", default="/tmp/openyolo3d_vlm.env")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max_candidates", default=None, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--timeout", default=90, type=int)
    parser.add_argument("--retries", default=2, type=int)
    parser.add_argument("--transport", default="curl", choices=["curl", "urllib"])
    parser.add_argument("--sleep", default=1.0, type=float)
    parser.add_argument("--force", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
