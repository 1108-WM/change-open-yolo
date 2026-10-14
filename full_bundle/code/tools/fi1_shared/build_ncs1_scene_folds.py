"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import hashlib

import re

from collections import defaultdict

SCENE_RE = re.compile(r"^(scene\d{4})_\d{2}$")

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
