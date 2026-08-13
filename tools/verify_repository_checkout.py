#!/usr/bin/env python3
"""Verify files that must survive a GitHub clone of this research fork."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "release_assets" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-submodules",
        action="store_true",
        help="also require the pinned MinkowskiEngine and ScanNet checkouts",
    )
    args = parser.parse_args()

    required_documents = [
        "README.md",
        "CURRENT_EXPERIMENT_STATUS.md",
        "新开对话阅读内容.md",
        "资料/当前基线修改方向.md",
        "资料/论文阅读记录.md",
        "docs/REMOTE_SERVER_SETUP.md",
        "docs/GITHUB_UPLOAD_SCOPE.md",
        ".gitmodules",
    ]
    errors: list[str] = []
    for relative in required_documents:
        if not (PROJECT_ROOT / relative).is_file():
            errors.append(f"missing required repository file: {relative}")

    if not MANIFEST_PATH.is_file():
        errors.append("missing release_assets/manifest.json")
        manifest = {"files": []}
    else:
        manifest = json.loads(MANIFEST_PATH.read_text())

    verified = 0
    for entry in manifest.get("files", []):
        relative = str(entry["path"])
        path = PROJECT_ROOT / relative
        if not path.is_file():
            errors.append(f"missing release asset: {relative}")
            continue
        actual_size = path.stat().st_size
        expected_size = int(entry["size_bytes"])
        if actual_size != expected_size:
            errors.append(
                f"size mismatch for {relative}: expected {expected_size}, got {actual_size}"
            )
            continue
        actual_hash = sha256(path)
        expected_hash = str(entry["sha256"])
        if actual_hash != expected_hash:
            errors.append(
                f"SHA-256 mismatch for {relative}: expected {expected_hash}, got {actual_hash}"
            )
            continue
        verified += 1

    if args.require_submodules:
        for relative in (
            "models/Mask3D/third_party/MinkowskiEngine/setup.py",
            "models/Mask3D/third_party/ScanNet/README.md",
        ):
            if not (PROJECT_ROOT / relative).is_file():
                errors.append(
                    f"submodule is not initialized: {relative}; run "
                    "git submodule update --init --recursive"
                )

    summary = {
        "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "verified_release_asset_count": verified,
        "require_submodules": args.require_submodules,
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
