#!/usr/bin/env python3
"""Build a val48 Grounded-SAM fixed-stride frontend from frozen stride-5 masks.

This adapter never instantiates SAM or GroundingDINO.  It derives a deterministic
``frame_id % stride == 0`` subset of arm D, reprojects retained mask RLEs onto
the frozen mesh, and reruns the four CPU tracking stages in an isolated output
directory.  The compatible arm name remains ``D`` solely so the frozen full
backend can consume the result. One file supports G10, G15, and G20.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.fi1_frontend_protocol import digest, freeze_json, sha256_file, write_json
from tools.run_fi1_frontend_comparison import (
    assemble_arm,
    check_point_order,
    fixed_selection,
    track_arm,
)

COMPATIBLE_ARM = "D"
FRONTEND = "grounded_sam"


def read(path: Path):
    return json.loads(Path(path).read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_file(path: Path, expected: str) -> None:
    require(path.is_file(), "missing bound file: " + str(path))
    require(sha256_file(path) == expected, "changed bound file: " + str(path))


def stride_frames(source_d_frames: list[str], stride: int) -> list[str]:
    """Return a canonical fixed-stride subset of frozen stride-5 frame IDs."""
    require(stride in (10, 15, 20), "supported stride must be 10, 15, or 20")
    values = list(map(str, source_d_frames))
    require(values and len(values) == len(set(values)), "invalid source D frame list")
    require(all(value.isdigit() for value in values), "nonnumeric source D frame ID")
    require([int(value) for value in values] == sorted(map(int, values)),
            "source D frame IDs are not sorted")
    selected = [value for value in values if int(value) % stride == 0]
    require(selected, f"stride-{stride} subset is empty")
    require(set(selected) < set(values), f"stride-{stride} must be a strict subset of stride-5")
    return selected


def stride10_frames(source_d_frames: list[str]) -> list[str]:
    """Compatibility wrapper retained for prior tests and G10 callers."""
    return stride_frames(source_d_frames, 10)


def source_observation_inputs(source_frontend: Path, manifest: dict) -> dict[str, str]:
    """Bind the retained D aggregate and every raw input used for reprojection."""
    scene = manifest["scene_name"]
    root = source_frontend / "arms" / COMPATIBLE_ARM / "observations" / scene
    inputs = {str(path): sha256_file(path) for path in (
        root / "summary.json",
        root / "automatic_observations.jsonl",
        root / "same_frame_mask_relations.jsonl",
    )}
    for item in manifest["files"].values():
        verify_file(Path(item["path"]), item["sha256"])
        inputs[item["path"]] = item["sha256"]
    for frame in manifest["frames"]:
        for name in ("color", "depth", "pose"):
            item = frame[name]
            verify_file(Path(item["path"]), item["sha256"])
            inputs[item["path"]] = item["sha256"]
    return inputs


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def project_frame_cpu(manifest: dict, frame: dict, point_count: int):
    """Replay the frozen projection kernel on CPU; no vision-model inference."""
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from utils import WORLD_2_CAM

    if not frame.get("pose_valid", True):
        return (np.zeros((point_count, 2), dtype=np.int16),
                np.zeros(point_count, dtype=bool), (1.0, 1.0))
    world = WORLD_2_CAM.__new__(WORLD_2_CAM)
    world.poses = [frame["pose"]["path"]]
    world.intrinsics = [manifest["files"]["intrinsics"]["path"]]
    world.mesh = manifest["files"]["mesh"]["path"]
    world.depth_maps_paths = [frame["depth"]["path"]]
    world.color_paths = [frame["color"]["path"]]
    world.image_resolution = imageio.imread(world.color_paths[0]).shape[:2]
    world.depth_resolution = imageio.imread(world.depth_maps_paths[0]).shape
    world.height, world.width = world.depth_resolution
    world.depth_scale, world.vis_depth_threshold = 1000.0, 0.05
    world.device = torch.device("cpu")
    projections, visibility = world.get_mesh_projections()
    scaling = tuple(d / c for d, c in zip(world.depth_resolution, world.image_resolution))
    return projections.numpy()[0], visibility.numpy()[0], scaling


def materialize_reprojected_cache(output: Path, source_frontend: Path, manifest: dict,
                                  selected: list[str], identity: str, point_count: int) -> Path:
    """Recover selected pointsets from retained mask RLEs after frame-cache cleanup."""
    import numpy as np
    from tools.fi1_mask_projection import decode_binary_mask_rle, mask_to_visible_points

    scene = manifest["scene_name"]
    target_root = output / "reprojected_frame_cache" / FRONTEND / scene
    source_root = source_frontend / "arms" / COMPATIBLE_ARM / "observations" / scene
    observations = jsonl(source_root / "automatic_observations.jsonl")
    relations = jsonl(source_root / "same_frame_mask_relations.jsonl")
    by_frame: dict[str, list[dict]] = {}
    for row in observations:
        by_frame.setdefault(str(row["frame_id"]), []).append(row)
    relations_by_frame: dict[str, list[dict]] = {}
    for row in relations:
        relations_by_frame.setdefault(str(row["frame_id"]), []).append(row)
    source_summary = read(source_root / "summary.json")
    frame_summaries = {str(row["frame_id"]): row for row in source_summary["frames"]}
    frames = {str(frame["frame_id"]): frame for frame in manifest["frames"]}
    require(set(by_frame) <= set(frames) and set(selected) <= set(frame_summaries),
            "retained D observation frame mismatch")

    for frame_id in selected:
        target = target_root / f"frame{int(frame_id):06d}"
        if (target / "receipt.json").exists():
            receipt = read(target / "receipt.json")
            require(receipt["identity"] == identity, "reprojected frame identity differs")
            for name, expected in receipt["files"].items():
                verify_file(target / name, expected)
            continue
        require(not target.exists(), "incomplete reprojected frame retained: " + str(target))
        staging = target.with_name(target.name + f".writing.{os.getpid()}")
        staging.mkdir(parents=True, exist_ok=False)
        frame = frames[frame_id]
        projection, visibility, scaling = project_frame_cpu(manifest, frame, point_count)
        rows, old_to_new = [], {}
        for local_id, old in enumerate(by_frame.get(frame_id, [])):
            mask = decode_binary_mask_rle(old["mask_rle"])
            points = mask_to_visible_points(mask, 0, projection[None], visibility[None], scaling)
            require(len(points) >= 20, "retained eligible mask reprojects below 20 points")
            filename = f"obs{local_id:03d}_points.npz"
            np.savez_compressed(staging / filename, point_indices=points)
            old_to_new[int(old["observation_id"])] = local_id
            rows.append({**old, "observation_id": local_id,
                         "point_indices_path": str(target / filename)})
        local_relations = []
        for old in relations_by_frame.get(frame_id, []):
            left, right = int(old["left_observation_id"]), int(old["right_observation_id"])
            require(left in old_to_new and right in old_to_new, "same-frame relation parent absent")
            local_relations.append({key: value for key, value in old.items()
                                    if key not in ("scene_name", "frame_id", "frame_index",
                                                   "left_observation_id", "right_observation_id")} |
                                   {"left_observation_id": old_to_new[left],
                                    "right_observation_id": old_to_new[right]})
        np.save(staging / "visibility.npy", visibility, allow_pickle=False)
        write_json(staging / "observations.json", rows)
        write_json(staging / "relations.json", local_relations)
        old_summary = frame_summaries[frame_id]
        receipt = {
            "identity": identity,
            "generated_mask_count": old_summary["generated_mask_count"],
            "eligible_mask_count": old_summary["eligible_mask_count"],
            "kept_observation_count": old_summary["kept_observation_count"],
            "inference_seconds": old_summary["inference_seconds"],
            "inference_skipped_invalid_pose": old_summary["inference_skipped_invalid_pose"],
            "projection_replayed_on_cpu": True,
            "visual_inference_replayed": False,
            "source_mask_rle_count": len(rows),
            "files": {path.name: sha256_file(path) for path in staging.iterdir() if path.is_file()},
        }
        require(len(rows) == receipt["kept_observation_count"], "retained frame count differs")
        write_json(staging / "receipt.json", receipt)
        staging.rename(target)
    return target_root


def build_frontend(args) -> None:
    source = args.source_frontend.resolve()
    output = args.output.resolve()
    selection_root = args.selection_root.resolve()
    selection = fixed_selection(selection_root)
    stride = int(args.stride)
    variant = f"grounded_stride{stride}"
    requested = selection["scenes"] if args.all_scenes else args.scenes
    require(requested and len(requested) == len(set(requested)),
            "provide a unique nonempty scene subset")
    require(set(requested) <= set(selection["scenes"]), "scene outside frozen val48")
    scenes = [scene for scene in selection["scenes"] if scene in requested]
    require(output != source and not output.is_relative_to(source) and not source.is_relative_to(output),
            "derived frontend must be isolated from source frontend")

    source_files = {
        str(Path(__file__).resolve()): sha256_file(__file__),
        str(ROOT / "tools/run_fi1_frontend_comparison.py"):
            sha256_file(ROOT / "tools/run_fi1_frontend_comparison.py"),
        str(ROOT / "tools/fi1_frontend_protocol.py"):
            sha256_file(ROOT / "tools/fi1_frontend_protocol.py"),
        str(source / "contract.json"): sha256_file(source / "contract.json"),
        str(source / "selection.json"): sha256_file(source / "selection.json"),
        str(selection_root / "selection_receipt.json"):
            sha256_file(selection_root / "selection_receipt.json"),
    }
    derived_selection = {
        **selection,
        "variant": variant,
        "stride": stride,
        "compatible_arm": COMPATIBLE_ARM,
        "frontend": FRONTEND,
        "sampling": f"retained D mask RLE frames satisfying int(frame_id) % {stride} == 0",
        "source_frontend": str(source),
        "requested_scenes": scenes,
        "new_visual_inference": False,
    }
    contract = {
        "version": 1,
        "variant": variant,
        "stride": stride,
        "compatible_arm": COMPATIBLE_ARM,
        "selection_sha256": digest(derived_selection),
        "source_files": source_files,
        "source_frontend": str(source),
        "cache_policy": "verified retained mask RLE reuse plus frozen raw-input reprojection; no Frontends construction or generate call",
        "new_visual_inference": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    freeze_json(output / "selection.json", derived_selection)
    freeze_json(output / "contract.json", contract)
    (output / "scenes.txt").write_text("\n".join(scenes) + "\n")

    records = []
    for scene in scenes:
        source_manifest_path = source / "manifests" / f"{scene}.json"
        source_manifest = read(source_manifest_path)
        require(source_manifest["scene_name"] == scene, "source manifest scene mismatch")
        selected = stride_frames(source_manifest["arms"][COMPATIBLE_ARM], stride)
        source_inputs = source_observation_inputs(source, source_manifest)
        manifest = copy.deepcopy(source_manifest)
        manifest["arms"][COMPATIBLE_ARM] = selected
        manifest["derived_variant"] = {
            "name": variant,
            "stride": stride,
            "compatible_arm": COMPATIBLE_ARM,
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_D_frame_count": len(source_manifest["arms"][COMPATIBLE_ARM]),
            "selected_frame_count": len(selected),
            "selection_rule": f"int(frame_id) % {stride} == 0",
            "new_visual_inference": False,
        }
        manifest_path = output / "manifests" / f"{scene}.json"
        freeze_json(manifest_path, manifest)
        point_count = check_point_order(manifest)
        identity = digest({
            "contract": digest(contract),
            "manifest": digest(manifest),
            "frontend": FRONTEND,
            "arm": COMPATIBLE_ARM,
        })
        frame_root = materialize_reprojected_cache(output, source, manifest, selected,
                                                   identity, point_count)
        assemble_arm(output, COMPATIBLE_ARM, manifest, frame_root, identity)
        track_arm(output, COMPATIBLE_ARM, manifest, frame_root, point_count, identity)
        summary = read(output / "arms" / COMPATIBLE_ARM / "observations" / scene / "summary.json")
        require([row["frame_id"] for row in summary["frames"]] == selected,
            f"assembled frames differ from deterministic stride-{stride} subset")
        records.append({
            "scene": scene,
            "variant": variant,
            "stride": stride,
            "compatible_arm": COMPATIBLE_ARM,
            "source_D_frame_count": len(source_manifest["arms"][COMPATIBLE_ARM]),
            "selected_frame_count": len(selected),
            "selected_frames": selected,
            "observation_count": summary["observation_count"],
            "point_count": point_count,
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_input_count": len(source_inputs),
            "source_inputs_digest": digest(source_inputs),
            "projection_replayed_on_cpu": True,
            "new_visual_inference": False,
        })
        print(f"G{stride} frontend complete", scene, len(selected), summary["observation_count"], flush=True)
    write_json(output / "summary.json", {
        "complete": True,
        "variant": variant,
        "stride": stride,
        "compatible_arm": COMPATIBLE_ARM,
        "scene_count": len(records),
        "records": records,
        "new_visual_inference": False,
        "ground_truth_read": False,
        "ap_computed": False,
    })


def audit_frontend(args) -> None:
    source = args.source_frontend.resolve()
    output = args.output.resolve()
    contract = read(output / "contract.json")
    selection = read(output / "selection.json")
    summary = read(output / "summary.json")
    stride = int(contract["stride"])
    variant = f"grounded_stride{stride}"
    require(int(args.stride) == stride, "requested audit stride differs from frozen output")
    require(contract["variant"] == selection["variant"] == summary["variant"] == variant,
            "variant identity mismatch")
    require(contract["selection_sha256"] == digest(selection), "selection identity differs")
    require(not contract["new_visual_inference"] and not summary["new_visual_inference"],
            "visual inference incorrectly declared")
    for path, expected in contract["source_files"].items():
        verify_file(Path(path), expected)
    audited = []
    for record in summary["records"]:
        scene = record["scene"]
        source_manifest = read(source / "manifests" / f"{scene}.json")
        manifest = read(output / "manifests" / f"{scene}.json")
        selected = stride_frames(source_manifest["arms"][COMPATIBLE_ARM], stride)
        require(manifest["arms"][COMPATIBLE_ARM] == selected == record["selected_frames"],
                f"stride-{stride} frame derivation differs")
        source_inputs = source_observation_inputs(source, source_manifest)
        require(record["source_input_count"] == len(source_inputs)
                and record["source_inputs_digest"] == digest(source_inputs),
                "source aggregate/raw-input binding differs")
        identity = digest({"contract": digest(contract), "manifest": digest(manifest),
                           "frontend": FRONTEND, "arm": COMPATIBLE_ARM})
        obs = read(output / "arms" / COMPATIBLE_ARM / "observations" / scene / "summary.json")
        require(obs["identity"] == identity, "observation identity differs")
        require([row["frame_id"] for row in obs["frames"]] == selected,
                "observation frame order differs")
        for frame_id in selected:
            cache = output / "reprojected_frame_cache" / FRONTEND / scene / f"frame{int(frame_id):06d}"
            receipt = read(cache / "receipt.json")
            require(receipt["identity"] == identity and receipt["projection_replayed_on_cpu"]
                    and not receipt["visual_inference_replayed"], "reprojection receipt differs")
            for name, expected in receipt["files"].items():
                verify_file(cache / name, expected)
        for name, expected in obs["output_sha256"].items():
            verify_file(output / "arms" / COMPATIBLE_ARM / "observations" / scene / name, expected)
        for stage in ("hierarchy", "tracks", "consensus", "merged"):
            receipt_path = output / "arms" / COMPATIBLE_ARM / "stage_receipts" / f"{scene}_{stage}.json"
            receipt = read(receipt_path)
            require(receipt["identity"] == identity, "tracking stage identity differs")
            for name, expected in receipt["files"].items():
                verify_file(output / "arms" / COMPATIBLE_ARM / stage / scene / name, expected)
        audited.append({"scene": scene, "frames": len(selected),
                        "observations": obs["observation_count"]})
    report = {"complete": True, "variant": variant, "stride": stride,
              "scene_count": len(audited), "records": audited,
              "strict_fixed_stride_subset": True,
              "retained_mask_rles_and_raw_projection_inputs_verified": True,
              "projection_replayed_on_cpu": True, "new_visual_inference": False}
    write_json(output / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("frontend", "audit-frontend"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-frontend", type=Path, default=
                        ROOT / ".local/fi1_val48_frontend_full_pipeline_20260914/frontend")
    parser.add_argument("--selection-root", type=Path, default=
                        ROOT / ".local/fi1_val48_frontend_full_pipeline_20260914")
    parser.add_argument("--scenes", nargs="+", default=["scene0153_00"])
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--stride", type=int, choices=(10, 15, 20), default=20,
                        help="Fixed raw-frame stride; supports G10, G15, and G20")
    args = parser.parse_args()
    {"frontend": build_frontend, "audit-frontend": audit_frontend}[args.command](args)


if __name__ == "__main__":
    main()
