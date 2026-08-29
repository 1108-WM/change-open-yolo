#!/usr/bin/env python3
"""Verify the pinned A100 runtime without running models or reading GT."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "numpy": "1.26.4",
    "PIL": "9.1.0",
    "transformers": "4.49.0",
    "accelerate": "1.4.0",
    "cv2": "4.8.1",
    "scipy": "1.15.3",
    "sklearn": "1.3.0",
    "yaml": "6.0.3",
}

EXPECTED_SCENE_LIST_SHA256 = (
    "d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083"
)
ALPHACLIP_COMMIT = "ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff"
SAM_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()

    errors: list[str] = []
    versions: dict[str, str] = {}
    for name, expected in EXPECTED.items():
        try:
            module = importlib.import_module(name)
            actual = str(getattr(module, "__version__", ""))
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")
            continue
        versions[name] = actual
        if actual != expected:
            errors.append(f"{name} version mismatch: expected {expected}, got {actual}")

    try:
        import qwen_vl_utils  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"cannot import qwen_vl_utils: {type(exc).__name__}: {exc}")

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        if args.require_cuda and not cuda_available:
            errors.append("CUDA is required but torch.cuda.is_available() is false")
    except Exception:
        cuda_available = False
        gpu_name = None

    scene_list = root / "configs/repro/scannetv2_val_312.txt"
    if not scene_list.is_file():
        errors.append(f"missing scene list: {scene_list}")
        scene_count = 0
        scene_sha = None
    else:
        scenes = [line.strip() for line in scene_list.read_text().splitlines() if line.strip()]
        scene_count = len(scenes)
        scene_sha = sha256(scene_list)
        if scene_count != 312:
            errors.append(f"scene count mismatch: expected 312, got {scene_count}")
        if scene_sha != EXPECTED_SCENE_LIST_SHA256:
            errors.append(f"scene-list SHA-256 mismatch: {scene_sha}")

    external = {
        "alpha_clip": root / "_external/AlphaCLIP/AlphaCLIP-main",
        "segment_anything": root / "_external/segment-anything/segment-anything-main",
    }
    source_heads = {name: git_head(path) for name, path in external.items()}
    if source_heads["alpha_clip"] not in (None, ALPHACLIP_COMMIT):
        errors.append(f"Alpha-CLIP commit mismatch: {source_heads['alpha_clip']}")
    if source_heads["segment_anything"] not in (None, SAM_COMMIT):
        errors.append(f"SAM commit mismatch: {source_heads['segment_anything']}")

    report = {
        "audit_valid": not errors,
        "errors": errors,
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": versions,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "scene_count": scene_count,
        "scene_list_sha256": scene_sha,
        "external_source_heads": source_heads,
        "ground_truth_read": False,
        "model_inference_run": False,
        "ap_computed": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
