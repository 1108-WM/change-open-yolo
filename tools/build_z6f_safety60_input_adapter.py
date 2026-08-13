#!/usr/bin/env python3
"""Build a no-GT filesystem adapter for frozen Z6f safety60 transfer.

The adapter does not recompute or mutate candidates.  It exposes the frozen
600-candidate Mask3D+YOLO native cache, the track subset retained by the
champion score plan, the legacy YOLO-World frame cache in the signed Z1
container, and the per-scene pair-union rows through the official100 stream
interface expected by the Z1/Z2/Z6b tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_symlink(source: Path, target: Path) -> None:
    target.symlink_to(os.path.relpath(source, target.parent))


def _signed_cache(scene: str, source: Path, prepared_root: Path, prompts: list[str]) -> dict:
    payload = torch.load(source, map_location="cpu")
    if isinstance(payload, dict) and set(payload) >= {"metadata", "predictions"}:
        predictions = payload["predictions"]
        source_schema = "signed"
    elif isinstance(payload, dict):
        predictions = payload
        source_schema = "legacy_predictions_mapping"
    else:
        raise ValueError(f"{scene}: unsupported YOLO cache payload")
    return {
        "metadata": {
            "schema_version": 2,
            "scene_name": scene,
            "scene_path": str(prepared_root / scene),
            "datatype": "mesh",
            "text_prompts": prompts,
            "adapter_provenance": str(source),
            "adapter_source_schema": source_schema,
            "ground_truth_usage": "none",
        },
        "predictions": predictions,
    }


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    prompts = [
        str(value)
        for value in yaml.safe_load(args.config_path.read_text())["network2d"]["text_prompts"]
    ]
    if len(prompts) != 198:
        raise ValueError(f"expected 198 prompts, got {len(prompts)}")
    if args.output_root.exists() or args.combined_plan_root.exists():
        raise FileExistsError("adapter output already exists")
    stream_stage = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    plan_stage = args.combined_plan_root.parent / f".{args.combined_plan_root.name}.tmp.{os.getpid()}"
    stream_stage.mkdir(parents=True)
    plan_stage.mkdir(parents=True)
    scene_summaries = []
    all_unions = []
    try:
        for ordinal, scene in enumerate(scenes, 1):
            scene_root = stream_stage / scene
            native_root = scene_root / "native_cache"
            track_root = scene_root / "d2b_tracks_filtered" / scene
            yolo_root = scene_root / "yoloworld_bboxes_2d"
            native_root.mkdir(parents=True)
            track_root.mkdir(parents=True)
            yolo_root.mkdir(parents=True)

            native_shapes = {}
            for suffix in ("masks", "classes", "scores"):
                source = args.native_cache_root / f"{scene}_pred_{suffix}.npy"
                if not source.is_file():
                    raise FileNotFoundError(source)
                native_shapes[suffix] = list(np.load(source, mmap_mode="r").shape)
                _relative_symlink(source, native_root / f"{scene}_pred_{suffix}.npy")
            native_count = int(native_shapes["classes"][0])
            if native_shapes["masks"][1] != native_count or native_shapes["scores"] != [native_count]:
                raise ValueError(f"{scene}: native cache dimensions disagree")

            plan_dir = args.frozen_plan_root / scene
            score_rows = _read_jsonl(plan_dir / "frozen_score_plan.jsonl")
            selected_track_ids = {
                int(row["candidate_id"])
                for row in score_rows if str(row["candidate_source"]) == "d2b_track"
            }
            source_tracks = json.loads(
                (args.track_root / scene / "automatic_tracks.json").read_text()
            )
            tracks = [
                row for row in source_tracks.get("tracks", [])
                if int(row["track_id"]) in selected_track_ids
            ]
            if {int(row["track_id"]) for row in tracks} != selected_track_ids:
                raise ValueError(f"{scene}: frozen track subset is incomplete")
            (track_root / "automatic_tracks.json").write_text(
                json.dumps({**source_tracks, "tracks": tracks}, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )

            legacy_yolo = args.yoloworld_cache_root / f"{scene}.pt"
            if not legacy_yolo.is_file():
                raise FileNotFoundError(legacy_yolo)
            torch.save(
                _signed_cache(scene, legacy_yolo, args.prepared_dataset_root, prompts),
                yolo_root / f"{scene}.pt",
            )

            scene_unions = []
            for row in _read_jsonl(plan_dir / "pair_union_append_candidates.jsonl"):
                adapted = dict(row)
                adapted["selected_track_id"] = int(row.get("selected_track_id", row["track_id"]))
                if adapted["selected_track_id"] not in selected_track_ids:
                    raise ValueError(f"{scene}: union references a non-retained track")
                scene_unions.append(adapted)
                all_unions.append(adapted)
            scene_summaries.append({
                "scene_name": scene,
                "native_count": native_count,
                "retained_track_count": len(tracks),
                "pair_union_count": len(scene_unions),
                "signed_yoloworld_frame_count": len(torch.load(
                    yolo_root / f"{scene}.pt", map_location="cpu"
                )["predictions"]),
            })
            print(
                f"[Z6f safety adapter] {ordinal}/{len(scenes)} {scene}: "
                f"native={native_count} track={len(tracks)} union={len(scene_unions)}",
                flush=True,
            )

        union_path = plan_stage / "pair_union_append_candidates.jsonl"
        with union_path.open("w") as handle:
            for row in all_unions:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "adapter_type": "frozen Z6f safety60 stream/combined-plan compatibility adapter",
            "scene_count": len(scenes),
            "native_count": sum(row["native_count"] for row in scene_summaries),
            "retained_track_count": sum(row["retained_track_count"] for row in scene_summaries),
            "pair_union_count": len(all_unions),
            "scene_summaries": scene_summaries,
            "pair_union_jsonl_sha256": _sha256(union_path),
            "contracts": {
                "native": "exact frozen Mask3D+YOLO 600-candidate cache; relative symlinks only",
                "track": "exact subset present in frozen champion score plan; IDs and point paths unchanged",
                "pair_union": "exact per-scene frozen v2 prior-corrected rows; selected_track_id aliases frozen track_id",
                "yoloworld": "legacy prediction mapping wrapped with signed metadata; tensors unchanged",
            },
            "ground_truth_usage": "none",
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
            "params": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        }
        (stream_stage / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (plan_stage / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stream_stage, args.output_root)
        os.replace(plan_stage, args.combined_plan_root)
        return summary
    except Exception:
        shutil.rmtree(stream_stage, ignore_errors=True)
        shutil.rmtree(plan_stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native-cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--yoloworld-cache-root", type=Path, required=True)
    parser.add_argument("--prepared-dataset-root", type=Path, required=True)
    parser.add_argument("--frozen-plan-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, _resolve(value))
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
