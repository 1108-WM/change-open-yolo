#!/usr/bin/env python3
"""Create a reproducible official-train 100-scene expansion and OOF manifest.

No data is downloaded or processed.  The existing official20 scenes are kept,
then deterministic extra official-train scenes are selected after excluding all
development/test scene lists.  The result contains a seeded 80/20 five-fold
scene manifest for future data collection and OOF.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_candidate_quality_head_oof import build_seeded_scene_folds


def read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError(f"scene list is empty or duplicate: {path}")
    return scenes


def build_manifest(official_train: list[str], current: list[str], excluded: set[str], target_count: int, selection_seed: int, fold_seed: int) -> tuple[list[str], dict]:
    if not set(current) <= set(official_train):
        raise ValueError("current scenes are not all official ScanNet200 train scenes")
    if set(current) & excluded:
        raise ValueError("current scenes overlap an evaluation split")
    if target_count < len(current) or target_count % 5:
        raise ValueError("target count must retain current scenes and be divisible by five")
    pool = sorted(set(official_train) - set(current) - excluded)
    required = target_count - len(current)
    if len(pool) < required:
        raise ValueError("not enough non-evaluation official train scenes for requested expansion")
    additions = sorted(random.Random(selection_seed).sample(pool, required))
    scenes = current + additions
    folds = build_seeded_scene_folds(scenes, fold_count=5, seed=fold_seed)
    return scenes, {
        "version": "official_train100_manifest_v1", "scene_count": len(scenes),
        "existing_scene_count": len(current), "new_scene_count": required,
        "selection_strategy": "keep_existing_then_seeded_sample_from_official_train_excluding_evaluation",
        "selection_seed": selection_seed, "evaluation_overlap_count": 0,
        "split_strategy": "seeded_explicit_scene_permutation_equal_chunks", "split_seed": fold_seed,
        "split_seed_applied": True, "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-train-list", type=Path, required=True)
    parser.add_argument("--current-scene-list", type=Path, required=True)
    parser.add_argument("--evaluation-scene-list", type=Path, action="append", required=True)
    parser.add_argument("--target-scene-count", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=20260808)
    parser.add_argument("--fold-seed", type=int, default=20260808)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    official_train, current = read_scenes(args.official_train_list), read_scenes(args.current_scene_list)
    excluded = set().union(*(set(read_scenes(path)) for path in args.evaluation_scene_list))
    scenes, manifest = build_manifest(official_train, current, excluded, args.target_scene_count, args.selection_seed, args.fold_seed)
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "official_train100.txt").write_text("\n".join(scenes) + "\n")
        (staging / "oof_5fold_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
