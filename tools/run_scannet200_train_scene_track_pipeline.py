#!/usr/bin/env python3
"""Run the frozen D1/Details/D2b track pipeline for one train scene."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _read_scenes(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _run(command: list[str]) -> None:
    print(f"[run] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _validate_inputs(scene: str, prepared_root: Path, scene_root: Path) -> tuple[int, int]:
    prepared = _read_json(prepared_root / scene / "stream_prepare_manifest.json")
    if prepared.get("scene_name") != scene or prepared.get("split") != "official_scannet200_train":
        raise ValueError("prepared manifest is not an official train scene")

    native = _read_json(scene_root / "native_export_manifest.json")
    if native.get("scene_name") != scene:
        raise ValueError("native manifest scene differs")
    if native.get("cache_contract") != "Mask3D + YOLO-World only":
        raise ValueError("native cache is not the frozen Mask3D + YOLO-World baseline")
    if native.get("ground_truth_usage") != "none":
        raise ValueError("native cache unexpectedly used ground truth")

    sam_root = scene_root / "sam_automatic_uniform30"
    sam = _read_json(sam_root / scene / "summary.json")
    if int(sam.get("frame_count", 0)) != 30:
        raise ValueError("automatic SAM input is not uniform30")
    if not sam.get("mask_rle_saved") or not sam.get("exact_same_frame_relations_saved"):
        raise ValueError("automatic SAM input lacks exact RLE or same-frame relations")
    observation_count = int(sam.get("observation_count", 0))
    if observation_count <= 0:
        raise ValueError("automatic SAM input contains no observations")
    observations = sam_root / scene / "automatic_observations.jsonl"
    points = list((sam_root / scene / "points").glob("*.npz"))
    line_count = sum(bool(line.strip()) for line in observations.read_text().splitlines())
    if line_count != observation_count or len(points) != observation_count:
        raise ValueError("automatic SAM observation files are incomplete")
    return int(native["point_count"]), observation_count


def _validate_outputs(scene: str, scene_root: Path, observation_count: int) -> dict:
    d1 = _read_json(scene_root / "d1_hierarchy_safe" / scene / "summary.json")
    if d1.get("policy") != "hierarchy_safe":
        raise ValueError("D1 policy differs from the frozen hierarchy_safe contract")
    if int(d1.get("source_observation_count", -1)) != observation_count:
        raise ValueError("D1 did not conserve the source observation ledger")
    if int(d1.get("changed_mask_count", -1)) != 0 or int(
        d1.get("changed_point_indices_count", -1)
    ) != 0:
        raise ValueError("hierarchy_safe D1 changed kept observation geometry")

    tracks = _read_json(scene_root / "details_siou_tracks" / scene / "automatic_tracks.json")
    consensus = _read_json(
        scene_root / "details_consensus_tracks" / scene / "automatic_tracks.json"
    )
    d2b = _read_json(scene_root / "d2b_tracks" / scene / "summary.json")
    track_count = int(tracks.get("track_count", 0))
    consensus_count = int(consensus.get("track_count", 0))
    d2b_count = int(d2b.get("final_proposal_count", 0))
    if track_count <= 0 or consensus_count <= 0 or d2b_count <= 0:
        raise ValueError("one of the frozen track stages produced no candidates")
    if int(consensus.get("source_track_count", -1)) != track_count:
        raise ValueError("consensus input count differs from Details tracks")
    if int(d2b.get("source_proposal_count", -1)) != consensus_count:
        raise ValueError("D2b input count differs from consensus tracks")
    if d2b.get("ground_truth_usage") != "none":
        raise ValueError("D2b unexpectedly used ground truth")
    return {
        "d1_output_observation_count": int(d1["output_observation_count"]),
        "d1_suppressed_observation_count": int(d1["suppressed_observation_count"]),
        "details_track_count": track_count,
        "consensus_track_count": consensus_count,
        "d2b_track_count": d2b_count,
        "d2b_merge_action_count": int(d2b["merge_action_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument(
        "--train-list",
        type=Path,
        default=PROJECT_ROOT
        / "_external/ESAM/ESAM-main/data/scannet200/meta_data/scannetv2_train.txt",
    )
    parser.add_argument(
        "--config-path", type=Path, default=PROJECT_ROOT / "pretrained/config_scannet200.yaml"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    scene = args.scene
    prepared_root = args.prepared_root.resolve()
    records_root = args.records_root.resolve()
    train_list = args.train_list.resolve()
    config_path = args.config_path.resolve()
    if scene not in _read_scenes(train_list):
        raise SystemExit(f"{scene} is not an official ScanNet200 train scene")
    scene_root = records_root / scene
    point_count, observation_count = _validate_inputs(scene, prepared_root, scene_root)

    list_root = records_root / ".pipeline_scene_lists"
    list_root.mkdir(parents=True, exist_ok=True)
    scene_list = list_root / f"{scene}.txt"
    scene_list.write_text(f"{scene}\n")

    python = sys.executable
    sam_root = scene_root / "sam_automatic_uniform30"
    d1_root = scene_root / "d1_hierarchy_safe"
    track_root = scene_root / "details_siou_tracks"
    consensus_root = scene_root / "details_consensus_tracks"
    d2b_root = scene_root / "d2b_tracks"
    resume = ["--resume"] if args.resume else []

    _run(
        [
            python,
            "tools/build_details_same_frame_hierarchy_preprocessor.py",
            "--scene-list",
            str(scene_list),
            "--automatic-root",
            str(sam_root),
            "--output-root",
            str(d1_root),
            "--policy",
            "hierarchy_safe",
            "--duplicate-min-coverage",
            "0.95",
            *resume,
        ]
    )
    _run(
        [
            python,
            "tools/build_details_frame_siou_tracks.py",
            "--scene_list",
            str(scene_list),
            "--automatic_root",
            str(d1_root),
            "--processed_scene_root",
            str(prepared_root),
            "--dataset_root",
            str(prepared_root),
            "--config_path",
            str(config_path),
            "--output_root",
            str(track_root),
            *resume,
        ]
    )
    _run(
        [
            python,
            "tools/refine_details_automatic_tracks_consensus.py",
            "--scene_list",
            str(scene_list),
            "--track_root",
            str(track_root),
            "--automatic_root",
            str(d1_root),
            "--processed_scene_root",
            str(prepared_root),
            "--dataset_root",
            str(prepared_root),
            "--config_path",
            str(config_path),
            "--output_root",
            str(consensus_root),
            *resume,
        ]
    )
    _run(
        [
            python,
            "tools/build_details_iterative_proposal_merges.py",
            "--scene-list",
            str(scene_list),
            "--track-root",
            str(consensus_root),
            "--automatic-root",
            str(d1_root),
            "--processed-scene-root",
            str(prepared_root),
            "--dataset-root",
            str(prepared_root),
            "--config-path",
            str(config_path),
            "--output-root",
            str(d2b_root),
            *resume,
        ]
    )

    stage_counts = _validate_outputs(scene, scene_root, observation_count)
    manifest = {
        "scene_name": scene,
        "split": "official_scannet200_train",
        "point_count": point_count,
        "automatic_sam_frame_count": 30,
        "automatic_sam_observation_count": observation_count,
        **stage_counts,
        "ground_truth_usage": "none",
        "native_candidate_mutation": False,
        "training_label_generation": False,
        "pipeline_contract": "hierarchy_safe D1 -> Details sIoU -> frozen consensus -> D2b",
    }
    path = scene_root / "track_pipeline_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
