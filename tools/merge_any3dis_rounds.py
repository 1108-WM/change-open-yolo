#!/usr/bin/env python3
"""Combine Any3DIS SAM2 tracking rounds before cross-round 3D cleanup.

Each tracking round starts local track ids at zero.  This utility assigns
scene-local global ids while preserving the original mask and lifted-point
artifacts, so the existing Details-Matter-style postprocessor can compare
candidates produced in different rounds without re-running SAM2.
"""

import argparse
import json
import shutil
from pathlib import Path


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _parse_roots(value, flag):
    roots = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not roots:
        raise ValueError(f"{flag} cannot be empty.")
    return roots


def _scene_names(args, track_roots):
    if args.scene_names:
        return [item.strip() for item in args.scene_names.split(",") if item.strip()]
    if args.scene_split:
        return [line.strip() for line in Path(args.scene_split).read_text().splitlines() if line.strip()]
    return sorted(path.name for path in track_roots[0].glob("scene*") if path.is_dir())


def _merge_scene(scene_name, track_roots, lift_roots, output_track_root, output_lift_root):
    merged_tracks = []
    merged_lifted = []
    merged_instances = []
    source_rounds = []
    frame_ids = None
    image_shape = None
    reliable_ids = set()
    track_id_map = {}

    for round_index, (track_root, lift_root) in enumerate(zip(track_roots, lift_roots)):
        track_scene = track_root / scene_name
        lift_scene = lift_root / scene_name
        track_summary_path = track_scene / "summary.json"
        lift_summary_path = lift_scene / "summary.json"
        if not track_summary_path.is_file() or not lift_summary_path.is_file():
            raise FileNotFoundError(f"Missing round {round_index} artifacts for {scene_name}")
        track_summary = json.loads(track_summary_path.read_text())
        lift_summary = json.loads(lift_summary_path.read_text())
        current_frames = track_summary["frame_ids"]
        if frame_ids is None:
            frame_ids = current_frames
            image_shape = track_summary.get("image_shape")
        elif current_frames != frame_ids:
            raise ValueError(f"Frame sampling differs across rounds for {scene_name}")
        reliable_ids.update(int(item) for item in track_summary.get("reliable_superpoint_ids", []))

        records = _read_jsonl(track_scene / "tracks.jsonl")
        local_map = {}
        for record in records:
            local_id = int(record["track_id"])
            global_id = len(merged_tracks)
            local_map[local_id] = global_id
            updated = dict(record)
            updated["track_id"] = global_id
            updated["source_round"] = round_index
            updated["source_track_id"] = local_id
            merged_tracks.append(updated)
        track_id_map[round_index] = local_map

        for record in _read_jsonl(lift_scene / "lifted_tracks.jsonl"):
            local_id = int(record["track_id"])
            if local_id not in local_map:
                continue
            updated = dict(record)
            updated["track_id"] = local_map[local_id]
            updated["source_round"] = round_index
            updated["source_track_id"] = local_id
            merged_lifted.append(updated)
        for record in _read_jsonl(lift_scene / "instances.jsonl"):
            local_id = int(record["track_id"])
            if local_id not in local_map:
                continue
            updated = dict(record)
            updated["track_id"] = local_map[local_id]
            updated["source_round"] = round_index
            updated["source_track_id"] = local_id
            merged_instances.append(updated)
        source_rounds.append(
            {
                "round_index": round_index,
                "track_root": str(track_root),
                "lift_root": str(lift_root),
                "saved_track_count": len(records),
                "lifted_instance_count": int(lift_summary.get("kept_instance_count", 0)),
            }
        )

    output_track_scene = output_track_root / scene_name
    output_lift_scene = output_lift_root / scene_name
    output_track_scene.mkdir(parents=True, exist_ok=True)
    output_lift_scene.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_track_scene / "tracks.jsonl", merged_tracks)
    _write_jsonl(output_lift_scene / "lifted_tracks.jsonl", merged_lifted)
    _write_jsonl(output_lift_scene / "instances.jsonl", merged_instances)
    track_summary = {
        "scene_name": scene_name,
        "frame_ids": frame_ids,
        "image_shape": image_shape,
        "reliable_superpoint_ids": sorted(reliable_ids),
        "reliable_superpoint_count": len(reliable_ids),
        "saved_track_count": len(merged_tracks),
        "source_rounds": source_rounds,
    }
    lift_summary = {
        "scene_name": scene_name,
        "raw_track_count": len(merged_lifted),
        "kept_instance_count": len(merged_instances),
        "source_rounds": source_rounds,
    }
    (output_track_scene / "summary.json").write_text(json.dumps(track_summary, indent=2, sort_keys=True) + "\n")
    (output_lift_scene / "summary.json").write_text(json.dumps(lift_summary, indent=2, sort_keys=True) + "\n")
    return {
        "scene_name": scene_name,
        "track_count": len(merged_tracks),
        "lifted_instance_count": len(merged_instances),
        "round_count": len(source_rounds),
    }


def main():
    parser = argparse.ArgumentParser(description="Merge multiple Any3DIS SAM2 tracking/lifting rounds.")
    parser.add_argument("--track_roots", required=True, help="Comma-separated tracking output roots in round order.")
    parser.add_argument("--lift_roots", required=True, help="Comma-separated lift output roots in matching round order.")
    parser.add_argument("--output_track_root", required=True)
    parser.add_argument("--output_lift_root", required=True)
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    track_roots = _parse_roots(args.track_roots, "--track_roots")
    lift_roots = _parse_roots(args.lift_roots, "--lift_roots")
    if len(track_roots) != len(lift_roots):
        raise ValueError("--track_roots and --lift_roots must have the same number of rounds.")
    for root in [*track_roots, *lift_roots]:
        if not root.is_dir():
            raise FileNotFoundError(root)
    output_track_root = Path(args.output_track_root)
    output_lift_root = Path(args.output_lift_root)
    for root in (output_track_root, output_lift_root):
        if root.exists() and args.overwrite:
            shutil.rmtree(root)
        if root.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {root}; use --overwrite")
        root.mkdir(parents=True)
    scenes = _scene_names(args, track_roots)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    summaries = [
        _merge_scene(scene_name, track_roots, lift_roots, output_track_root, output_lift_root)
        for scene_name in scenes
    ]
    payload = {"scenes": summaries, "params": vars(args)}
    (output_track_root / "merge_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_lift_root / "merge_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
