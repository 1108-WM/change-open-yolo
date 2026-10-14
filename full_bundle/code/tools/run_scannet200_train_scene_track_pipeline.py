#!/usr/bin/env python3
"""Run the frozen D1/Details/D2b track pipeline for one train scene."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _read_scenes(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _run(command: list[str]) -> None:
    print(f"[run] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _configured_view_count(
    prepared: dict, prepared_scene_root: Path, config_path: Path | None
) -> int:
    """Return the view count exposed by WORLD_2_CAM before max-30 sampling."""
    if config_path is None:
        color_root = prepared_scene_root / "color"
        return (
            sum(path.is_file() for path in color_root.iterdir())
            if color_root.is_dir()
            else 0
        )

    config = yaml.safe_load(config_path.read_text())
    frequency = int(config.get("openyolo3d", {}).get("frequency", 1))
    if frequency <= 0:
        raise ValueError("OpenYOLO3D frame frequency must be positive")
    raw_frame_count = int(prepared.get("sens", {}).get("frame_count", 0))
    if raw_frame_count <= 0:
        poses_root = prepared_scene_root / "poses"
        raw_frame_count = (
            sum(path.is_file() for path in poses_root.iterdir())
            if poses_root.is_dir()
            else 0
        )
    return len(range(0, raw_frame_count, frequency))


def _validate_inputs(
    scene: str,
    prepared_root: Path,
    scene_root: Path,
    dataset_contract: str = "official_scannet200_train",
    automatic_root: Path | None = None,
    config_path: Path | None = None,
) -> tuple[int, int]:
    prepared = _read_json(prepared_root / scene / "stream_prepare_manifest.json")
    if dataset_contract == "official_scannet200_train":
        prepared_valid = (
            prepared.get("scene_name") == scene
            and prepared.get("split") == "official_scannet200_train"
        )
        expected_cache_contract = "Mask3D + YOLO-World only"
    elif dataset_contract in {"NCS-train", "NCS-validation"}:
        expected_split = dataset_contract
        expected_policy = "train_allowed" if dataset_contract == "NCS-train" else "validation_forbidden"
        prepared_valid = (
            prepared.get("scene_name") == scene
            and prepared.get("ncs_split") == expected_split
            and prepared.get("gt_policy") == expected_policy
            and prepared.get("operations", {}).get("candidate_generation") is False
        )
        expected_cache_contract = "frozen Mask3D + YOLO-World only"
    else:
        raise ValueError(f"unsupported dataset contract: {dataset_contract}")
    if not prepared_valid:
        raise ValueError(f"prepared manifest does not satisfy {dataset_contract}")

    native = _read_json(scene_root / "native_export_manifest.json")
    if native.get("scene_name") != scene or (
        dataset_contract in {"NCS-train", "NCS-validation"}
        and native.get("ncs_split") != dataset_contract
    ):
        raise ValueError("native manifest scene differs")
    if native.get("cache_contract") != expected_cache_contract:
        raise ValueError("native cache is not the frozen Mask3D + YOLO-World baseline")
    if native.get("ground_truth_usage") != "none":
        raise ValueError("native cache unexpectedly used ground truth")

    sam_root = automatic_root or scene_root / "sam_automatic_uniform30"
    sam = _read_json(sam_root / scene / "summary.json")
    available_frame_count = _configured_view_count(
        prepared, prepared_root / scene, config_path
    )
    expected_frame_count = min(30, available_frame_count) if available_frame_count else 30
    if int(sam.get("frame_count", 0)) != expected_frame_count:
        raise ValueError("automatic SAM input is not the uniform30-or-all-available contract")
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


def _validate_outputs(
    scene: str,
    observation_count: int,
    d1_root: Path,
    track_root: Path,
    consensus_root: Path,
    d2b_root: Path,
) -> dict:
    d1 = _read_json(d1_root / scene / "summary.json")
    if d1.get("policy") != "hierarchy_safe":
        raise ValueError("D1 policy differs from the frozen hierarchy_safe contract")
    if int(d1.get("source_observation_count", -1)) != observation_count:
        raise ValueError("D1 did not conserve the source observation ledger")
    if int(d1.get("changed_mask_count", -1)) != 0 or int(
        d1.get("changed_point_indices_count", -1)
    ) != 0:
        raise ValueError("hierarchy_safe D1 changed kept observation geometry")

    tracks = _read_json(track_root / scene / "automatic_tracks.json")
    consensus = _read_json(consensus_root / scene / "automatic_tracks.json")
    d2b = _read_json(d2b_root / scene / "summary.json")
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
    parser.add_argument(
        "--ncs-train", action="store_true",
        help="Validate NCS-train manifests and membership instead of the legacy official100 contract.",
    )
    parser.add_argument(
        "--ncs-validation", action="store_true",
        help="Validate NCS-validation manifests and membership without reading validation GT.",
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument(
        "--pipeline-root", type=Path,
        help="Optional shared stage root; omitted preserves the historical per-scene records layout.",
    )
    parser.add_argument(
        "--automatic-root", type=Path,
        help="Optional shared uniform30 automatic-SAM root containing <scene>/summary.json.",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=PROJECT_ROOT
        / "full_bundle/metadata/scannet/scannetv2_train.txt",
    )
    parser.add_argument(
        "--config-path", type=Path, default=PROJECT_ROOT / "pretrained/config_scannet200.yaml"
    )
    parser.add_argument(
        "--ncs-scene-list", type=Path,
        default=PROJECT_ROOT
        / "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.ncs_train and args.ncs_validation:
        raise SystemExit("--ncs-train and --ncs-validation are mutually exclusive")

    scene = args.scene
    prepared_root = args.prepared_root.resolve()
    records_root = args.records_root.resolve()
    train_list = args.train_list.resolve()
    config_path = args.config_path.resolve()
    membership_list = args.ncs_scene_list.resolve() if args.ncs_train else train_list
    if scene not in _read_scenes(membership_list):
        contract_name = "NCS-train100" if args.ncs_train else ("NCS-validation60" if args.ncs_validation else "official ScanNet200 train")
        raise SystemExit(f"{scene} is not a {contract_name} scene")
    scene_root = records_root / scene
    dataset_contract = "NCS-train" if args.ncs_train else ("NCS-validation" if args.ncs_validation else "official_scannet200_train")
    pipeline_root = args.pipeline_root.resolve() if args.pipeline_root else None
    automatic_root = (
        args.automatic_root.resolve()
        if args.automatic_root else (
            pipeline_root / "sam_automatic_uniform30"
            if pipeline_root else scene_root / "sam_automatic_uniform30"
        )
    )
    point_count, observation_count = _validate_inputs(
        scene, prepared_root, scene_root, dataset_contract, automatic_root, config_path
    )

    list_root = (pipeline_root or records_root) / ".pipeline_scene_lists"
    list_root.mkdir(parents=True, exist_ok=True)
    scene_list = list_root / f"{scene}.txt"
    scene_list.write_text(f"{scene}\n")

    python = sys.executable
    sam_root = automatic_root
    stage_parent = pipeline_root or scene_root
    d1_root = stage_parent / "d1_hierarchy_safe"
    track_root = stage_parent / "details_siou_tracks"
    consensus_root = stage_parent / "details_consensus_tracks"
    d2b_root = stage_parent / "d2b_tracks"
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

    stage_counts = _validate_outputs(
        scene, observation_count, d1_root, track_root, consensus_root, d2b_root
    )
    manifest = {
        "scene_name": scene,
        "split": dataset_contract,
        "point_count": point_count,
        "automatic_sam_frame_count": int(
            _read_json(sam_root / scene / "summary.json")["frame_count"]
        ),
        "automatic_sam_frame_contract": "uniform max-30; all available when fewer than 30",
        "automatic_sam_observation_count": observation_count,
        **stage_counts,
        "ground_truth_usage": "none",
        "native_candidate_mutation": False,
        "training_label_generation": False,
        "pipeline_contract": "hierarchy_safe D1 -> Details sIoU -> frozen consensus -> D2b",
    }
    if pipeline_root:
        manifest_root = pipeline_root / "track_pipeline_manifests"
        manifest_root.mkdir(parents=True, exist_ok=True)
        path = manifest_root / f"{scene}.json"
    else:
        path = scene_root / "track_pipeline_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
