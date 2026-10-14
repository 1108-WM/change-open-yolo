#!/usr/bin/env python3
"""Prepare the frozen independent NCS ScanNet scenes.

The train split is allowed to read official labels and writes a GT file for
development/training.  The validation split deliberately takes the geometry-
only path and never opens ``*_labels.ply`` or ``*.aggregation.json``.  Each
scene is committed atomically, so an interrupted run can be resumed safely.
No model inference, candidate generation or AP evaluation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_scannet200_train_scene import (
    DEFAULT_LABEL_MAP,
    DEFAULT_TRAIN_LIST,
    build_geometry_only_scene,
    build_processed_scene,
    decode_sens,
    validate_split,
)


DEFAULT_RAW_ROOT = Path("/media/jia/软件1/scannet_train_stream/scans")
DEFAULT_OUTPUT_ROOT = Path("/media/jia/软件1/scannet_train_stream/prepared_ncs")
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "output/scannet200/scene_splits/ncs_independent_20260813"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError(f"scene list must be non-empty and duplicate-free: {path}")
    if any(not scene.startswith("scene") for scene in scenes):
        raise ValueError(f"invalid scene name in {path}")
    return scenes


def _required_files(raw_scene: Path, *, with_gt: bool) -> list[Path]:
    scene = raw_scene.name
    required = [
        raw_scene / f"{scene}.sens",
        raw_scene / f"{scene}_vh_clean_2.0.010000.segs.json",
        raw_scene / f"{scene}_vh_clean_2.ply",
    ]
    if with_gt:
        required += [
            raw_scene / f"{scene}.aggregation.json",
            raw_scene / f"{scene}_vh_clean_2.labels.ply",
        ]
    return required


def _existing_is_compatible(scene_root: Path, scene: str, split: str, with_gt: bool) -> bool:
    manifest_path = scene_root / "stream_prepare_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("scene_name") == scene
        and manifest.get("ncs_split") == split
        and manifest.get("gt_policy") == ("train_allowed" if with_gt else "validation_forbidden")
        and manifest.get("operations", {}).get("candidate_generation") is False
    )


def prepare_scene(
    raw_scene: Path,
    output_root: Path,
    *,
    split: str,
    with_gt: bool,
    label_map: Path,
    resume: bool,
) -> dict:
    scene = raw_scene.name
    missing = [str(path) for path in _required_files(raw_scene, with_gt=with_gt) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{scene}: missing required files: {missing}")
    final_root = output_root / scene
    if final_root.exists():
        if resume and _existing_is_compatible(final_root, scene, split, with_gt):
            return {"scene_name": scene, "state": "already_prepared", "split": split}
        raise FileExistsError(f"output scene already exists or has incompatible manifest: {final_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{scene}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    gt_stage = None
    try:
        gt_final = output_root / "ground_truth" / f"{scene}.txt"
        if with_gt and gt_final.exists():
            raise FileExistsError(f"ground truth already exists: {gt_final}")
        sens_summary = decode_sens(raw_scene / f"{scene}.sens", staging)
        if with_gt:
            processed_summary = build_processed_scene(raw_scene, staging, label_map)
            gt_root = output_root / "ground_truth"
            gt_root.mkdir(exist_ok=True)
            gt_stage = gt_root / f".{scene}.tmp.{os.getpid()}.txt"
            import numpy as np

            np.savetxt(gt_stage, processed_summary.pop("gt"), fmt="%d")
        else:
            processed_summary = build_geometry_only_scene(raw_scene, staging)

        shutil.copy2(raw_scene / f"{scene}_vh_clean_2.ply", staging / f"{scene}_vh_clean_2.ply")
        scene_txt = raw_scene / f"{scene}.txt"
        if scene_txt.is_file():
            shutil.copy2(scene_txt, staging / scene_txt.name)
        manifest = {
            "scene_name": scene,
            "ncs_split": split,
            "source_raw_scene": str(raw_scene.resolve()),
            "source_scene_sha256": _sha256(raw_scene / f"{scene}.sens"),
            "sens": sens_summary,
            "processed": processed_summary,
            "operations": {
                "mask3d_inference": False,
                "yoloworld_inference": False,
                "sam_inference": False,
                "candidate_generation": False,
                "training_label_generation": bool(with_gt),
                "gt_access": "train_allowed" if with_gt else "validation_forbidden",
            },
            "gt_policy": "train_allowed" if with_gt else "validation_forbidden",
        }
        (staging / "stream_prepare_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(final_root)
        if gt_stage is not None:
            gt_stage.rename(gt_final)
        return {"scene_name": scene, "state": "prepared", "split": split, "manifest": manifest}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if gt_stage is not None:
            gt_stage.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ncs-train", action="store_true")
    group.add_argument("--ncs-validation", action="store_true")
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-list", type=Path, default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    split = "NCS-train" if args.ncs_train else "NCS-validation"
    default_list = DEFAULT_SPLIT_DIR / (
        "ncs_train100.txt" if args.ncs_train else "ncs_validation60.txt"
    )
    scene_list = (args.scene_list or default_list).resolve()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    train_list = args.train_list.resolve()
    scenes = _read_scenes(scene_list)
    for scene in scenes:
        validate_split(scene, train_list, [])

    with_gt = args.ncs_train
    results = []
    for index, scene in enumerate(scenes, 1):
        print(f"[{index}/{len(scenes)}] {split} {scene}", flush=True)
        results.append(
            prepare_scene(
                raw_root / scene,
                output_root,
                split=split,
                with_gt=with_gt,
                label_map=args.label_map.resolve(),
                resume=args.resume,
            )
        )
    summary = {
        "dataset": "ScanNet v2 independent NCS",
        "split": split,
        "scene_list": str(scene_list),
        "scene_list_sha256": _sha256(scene_list),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "gt_policy": "train_only" if with_gt else "forbidden",
        "scene_count": len(scenes),
        "prepared_count": sum(item["state"] == "prepared" for item in results),
        "already_prepared_count": sum(item["state"] == "already_prepared" for item in results),
        "scenes": results,
    }
    summary_path = (args.summary or output_root / f"{split.lower().replace('-', '_')}_prepare_summary.json").resolve()
    summary["summary_path"] = str(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("split", "scene_count", "prepared_count", "already_prepared_count", "summary_path")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
