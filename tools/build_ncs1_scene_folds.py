#!/usr/bin/env python3
"""Build the deterministic GT-free five-fold scene manifest for NCS-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_RE = re.compile(r"^(scene\d{4})_\d{2}$")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    for scene in scenes:
        if not SCENE_RE.fullmatch(scene):
            raise ValueError(f"invalid ScanNet scene name: {scene}")
    return scenes


def _main_scene(scene: str) -> str:
    match = SCENE_RE.fullmatch(scene)
    if match is None:
        raise ValueError(f"invalid ScanNet scene name: {scene}")
    return match.group(1)


def build_folds(scenes: list[str], fold_count: int, seed: str) -> list[list[str]]:
    """Assign main-scene groups deterministically while balancing scan counts."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    grouped: dict[str, list[str]] = defaultdict(list)
    for scene in scenes:
        grouped[_main_scene(scene)].append(scene)
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
            item[0],
        ),
    )
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for _, members in ordered_groups:
        destination = min(range(fold_count), key=lambda index: (len(folds[index]), index))
        folds[destination].extend(sorted(members))
    return [sorted(fold) for fold in folds]


def run(args: argparse.Namespace) -> dict:
    scene_list = _resolve(args.scene_list)
    output_path = _resolve(args.output_path)
    scenes = _read_scenes(scene_list)
    folds = build_folds(scenes, args.fold_count, args.seed)
    occurrences = Counter(scene for fold in folds for scene in fold)
    main_to_fold: dict[str, int] = {}
    for fold_index, fold in enumerate(folds):
        for scene in fold:
            main = _main_scene(scene)
            previous = main_to_fold.setdefault(main, fold_index)
            if previous != fold_index:
                raise ValueError(f"main scene leaked across folds: {main}")
    valid = (
        set(occurrences) == set(scenes)
        and all(count == 1 for count in occurrences.values())
        and all(fold for fold in folds)
    )
    manifest = {
        "manifest_type": "NCS-1 deterministic GT-free main-scene-isolated folds",
        "scene_list": str(scene_list),
        "scene_list_sha256": hashlib.sha256(scene_list.read_bytes()).hexdigest(),
        "seed": args.seed,
        "fold_count": args.fold_count,
        "scene_count": len(scenes),
        "main_scene_count": len(main_to_fold),
        "folds": [
            {
                "fold_index": index,
                "validation_scenes": fold,
                "training_scenes": sorted(set(scenes) - set(fold)),
            }
            for index, fold in enumerate(folds)
        ],
        "contracts": {
            "assignment_inputs": "scene names and fixed seed only",
            "main_scene_isolation": True,
            "ground_truth_usage": "none",
            "class_statistics_usage": "none",
        },
        "contract_valid": valid,
    }
    if not valid:
        raise ValueError("generated fold manifest is not an exact partition")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and json.loads(output_path.read_text()) != manifest:
        raise FileExistsError(f"refusing to overwrite a different manifest: {output_path}")
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"),
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=Path(
            "output/scannet200/scene_splits/ncs_independent_20260813/"
            "ncs_train100_folds_v1.json"
        ),
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", default="ncs1-main-scene-folds-v1")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "output_path": str(_resolve(args.output_path)),
        "scene_count": summary["scene_count"],
        "main_scene_count": summary["main_scene_count"],
        "fold_sizes": [len(fold["validation_scenes"]) for fold in summary["folds"]],
        "contract_valid": summary["contract_valid"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
