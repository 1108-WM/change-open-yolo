#!/usr/bin/env python3
"""Compare two fixed track geometry sets without GT, semantics, or AP."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def geometry_signature(track):
    values = [int(value) for value in track.get("superpoint_ids", [])]
    if not values:
        raise ValueError(f"track {track.get('track_id')} has no superpoint_ids")
    if values != sorted(set(values)) or values[0] < 0:
        raise ValueError(f"track {track.get('track_id')} has invalid superpoint_ids")
    return tuple(values)


def compare_scene_tracks(scene_name, source_tracks, candidate_tracks):
    source_by_geometry = defaultdict(list)
    candidate_by_geometry = defaultdict(list)
    for track in source_tracks:
        source_by_geometry[geometry_signature(track)].append(int(track["track_id"]))
    for track in candidate_tracks:
        candidate_by_geometry[geometry_signature(track)].append(int(track["track_id"]))

    source_counts = Counter(
        {signature: len(track_ids) for signature, track_ids in source_by_geometry.items()}
    )
    candidate_counts = Counter(
        {signature: len(track_ids) for signature, track_ids in candidate_by_geometry.items()}
    )
    shared = source_counts & candidate_counts
    source_only = source_counts - candidate_counts
    candidate_only = candidate_counts - source_counts
    changed = []
    for signature in sorted(set(source_only) | set(candidate_only)):
        changed.append(
            {
                "scene_name": scene_name,
                "superpoint_ids": list(signature),
                "superpoint_count": len(signature),
                "source_count": int(source_counts[signature]),
                "candidate_count": int(candidate_counts[signature]),
                "source_only_count": int(source_only[signature]),
                "candidate_only_count": int(candidate_only[signature]),
                "source_track_ids": sorted(source_by_geometry.get(signature, [])),
                "candidate_track_ids": sorted(candidate_by_geometry.get(signature, [])),
                "gt_usage": "none",
            }
        )
    summary = {
        "scene_name": scene_name,
        "source_track_count": len(source_tracks),
        "candidate_track_count": len(candidate_tracks),
        "exact_shared_track_count": int(sum(shared.values())),
        "source_only_track_count": int(sum(source_only.values())),
        "candidate_only_track_count": int(sum(candidate_only.values())),
        "source_unique_geometry_count": len(source_counts),
        "candidate_unique_geometry_count": len(candidate_counts),
        "changed_geometry_count": len(changed),
        "gt_usage": "none",
    }
    if summary["exact_shared_track_count"] + summary["source_only_track_count"] != len(
        source_tracks
    ):
        raise ValueError("source geometry multiset does not conserve track count")
    if summary["exact_shared_track_count"] + summary[
        "candidate_only_track_count"
    ] != len(candidate_tracks):
        raise ValueError("candidate geometry multiset does not conserve track count")
    return summary, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--source-track-root", type=Path, required=True)
    parser.add_argument("--candidate-track-root", type=Path, required=True)
    parser.add_argument("--source-name", default="source")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "source_track_root", "candidate_track_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    scenes = _read_scenes(args.scene_list)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries, changed_rows = [], []
    for scene_name in scenes:
        source = json.loads(
            (args.source_track_root / scene_name / "automatic_tracks.json").read_text()
        )
        candidate = json.loads(
            (args.candidate_track_root / scene_name / "automatic_tracks.json").read_text()
        )
        summary, changed = compare_scene_tracks(
            scene_name, source.get("tracks", []), candidate.get("tracks", [])
        )
        summaries.append(summary)
        changed_rows.extend(changed)

    with (args.output_root / "changed_geometries.jsonl").open("w") as handle:
        for row in changed_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    payload = {
        "scene_count": len(scenes),
        "source_name": args.source_name,
        "candidate_name": args.candidate_name,
        "source_track_count": sum(row["source_track_count"] for row in summaries),
        "candidate_track_count": sum(row["candidate_track_count"] for row in summaries),
        "exact_shared_track_count": sum(
            row["exact_shared_track_count"] for row in summaries
        ),
        "source_only_track_count": sum(
            row["source_only_track_count"] for row in summaries
        ),
        "candidate_only_track_count": sum(
            row["candidate_only_track_count"] for row in summaries
        ),
        "changed_geometry_count": sum(row["changed_geometry_count"] for row in summaries),
        "unchanged_scene_count": sum(
            row["changed_geometry_count"] == 0 for row in summaries
        ),
        "scene_summaries": summaries,
        "gt_usage": "none",
        "decision_state": "Exact superpoint geometry multiset comparison only; no selection or scoring.",
        "params": vars(args),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "scene_count",
                    "source_track_count",
                    "candidate_track_count",
                    "exact_shared_track_count",
                    "source_only_track_count",
                    "candidate_only_track_count",
                    "changed_geometry_count",
                    "unchanged_scene_count",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
