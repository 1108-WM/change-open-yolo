import argparse
import base64
import json
import mimetypes
import os
import os.path as osp
import sys
import urllib.error
import urllib.request


def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def image_to_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def post_json(url, api_key, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw": body[:1000]}
        return exc.code, parsed


def extract_responses_text(payload):
    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                parts.append(text)
    if parts:
        return "\n".join(parts)
    if payload.get("output_text"):
        return payload["output_text"]
    return ""


def extract_chat_text(payload):
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", default="/tmp/openyolo3d_vlm.env")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--endpoint", default="both", choices=["both", "responses", "chat"])
    parser.add_argument("--timeout", default=60, type=int)
    args = parser.parse_args()

    env = load_env(args.env_file)
    api_key = env.get("VLM_API_KEY")
    base_url = env.get("VLM_BASE_URL", "").rstrip("/")
    model = env.get("VLM_MODEL")
    if not api_key or not base_url or not model:
        raise SystemExit("Missing VLM_API_KEY, VLM_BASE_URL, or VLM_MODEL in env file.")
    if not osp.exists(args.image_path):
        raise SystemExit(f"Image not found: {args.image_path}")

    image_url = image_to_data_url(args.image_path)
    prompt = (
        "You are checking whether this API supports image input. "
        "Describe the main visible object in this image in one short sentence. "
        "Return JSON only: {\"supports_vision\": true, \"description\": \"...\"}"
    )

    attempts = []
    if args.endpoint in ("both", "responses"):
        attempts.append(
            (
                "responses",
                f"{base_url}/responses",
                {
                    "model": model,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": image_url},
                            ],
                        }
                    ],
                    "max_output_tokens": 120,
                },
                extract_responses_text,
            )
        )
    if args.endpoint in ("both", "chat"):
        attempts.append(
            (
                "chat.completions",
                f"{base_url}/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    "max_tokens": 120,
                },
                extract_chat_text,
            )
        )

    any_success = False
    for name, url, payload, extractor in attempts:
        print(f"[TEST] endpoint={name} model={model} url={url}")
        status, response = post_json(url, api_key, payload, args.timeout)
        text = extractor(response)
        if 200 <= status < 300 and text.strip():
            any_success = True
            print(f"[OK] HTTP {status}; image input appears supported.")
            print(text.strip()[:800])
        else:
            print(f"[FAIL] HTTP {status}; no usable vision response.")
            error = response.get("error", response)
            print(json.dumps(error, ensure_ascii=False)[:1000])

    if not any_success:
        sys.exit(2)


if __name__ == "__main__":
    main()
