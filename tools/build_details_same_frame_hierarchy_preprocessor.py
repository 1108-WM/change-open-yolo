#!/usr/bin/env python3
"""Build an auditable same-frame hierarchy preprocessing branch for SAM masks.

The default ``hierarchy_safe`` policy suppresses only near-duplicate masks.
Containment and partial-overlap relations are recorded but never subtracted.
The ``details_exact`` policy is a paper-faithful ablation: masks are processed
from small to large and pixels already claimed by smaller masks are removed.

This tool does not read GT, classes, semantics, Mask3D candidates, or AP.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def decode_binary_mask_rle(payload):
    size = payload.get("size", [])
    counts = [int(value) for value in payload.get("counts", [])]
    if len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError("invalid RLE size")
    if any(value < 0 for value in counts):
        raise ValueError("negative RLE count")
    pixel_count = int(size[0]) * int(size[1])
    if sum(counts) != pixel_count:
        raise ValueError("RLE counts do not match its size")
    values = np.empty(pixel_count, dtype=bool)
    offset = 0
    foreground = False
    for count in counts:
        values[offset : offset + count] = foreground
        offset += count
        foreground = not foreground
    return values.reshape((int(size[0]), int(size[1])), order="F")


def encode_binary_mask_rle(mask):
    mask = np.asarray(mask, dtype=bool)
    pixels = np.asfortranarray(mask).reshape(-1, order="F").astype(np.uint8)
    padded = np.concatenate(
        (np.asarray([0], dtype=np.uint8), pixels, np.asarray([0], dtype=np.uint8))
    )
    changes = np.flatnonzero(np.diff(padded))
    counts = np.diff(
        np.concatenate((np.asarray([0], dtype=np.int64), changes, np.asarray([len(pixels)], dtype=np.int64)))
    )
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": [int(value) for value in counts],
    }


def mask_bbox_xywh(mask):
    ys, xs = np.nonzero(mask) if np.any(mask) else ([], [])
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _validate_observations(scene_name, observations):
    by_id = {}
    for row in observations:
        observation_id = int(row["observation_id"])
        if observation_id in by_id:
            raise ValueError(f"duplicate observation id: {observation_id}")
        if str(row["scene_name"]) != scene_name:
            raise ValueError(f"observation {observation_id} has a mismatched scene")
        mask = decode_binary_mask_rle(row["mask_rle"])
        if int(mask.sum()) != int(row["area"]):
            raise ValueError(f"observation {observation_id} has inconsistent RLE area")
        point_path = Path(row["point_indices_path"])
        if not point_path.is_file():
            raise FileNotFoundError(f"missing point indices: {point_path}")
        by_id[observation_id] = row
    return by_id


def classify_relation(iou, left_coverage, right_coverage, duplicate_min_coverage):
    if iou <= 0.0:
        return "disjoint"
    left_duplicate = left_coverage >= duplicate_min_coverage
    right_duplicate = right_coverage >= duplicate_min_coverage
    if left_duplicate and right_duplicate:
        return "duplicate"
    if left_duplicate or right_duplicate:
        return "containment"
    return "partial"


def build_complete_relation_graph(observations, sparse_relations, duplicate_min_coverage):
    by_id = {int(row["observation_id"]): row for row in observations}
    by_frame = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame_index"])].append(int(row["observation_id"]))

    sparse = {}
    for raw in sparse_relations:
        left_id, right_id = sorted(
            (int(raw["left_observation_id"]), int(raw["right_observation_id"]))
        )
        if left_id not in by_id or right_id not in by_id:
            raise ValueError("same-frame relation references an unknown observation")
        left, right = by_id[left_id], by_id[right_id]
        if int(left["frame_index"]) != int(right["frame_index"]):
            raise ValueError("same-frame relation crosses frames")
        if (left_id, right_id) in sparse:
            raise ValueError("duplicate same-frame relation")

        intersection = int(raw["intersection_pixel_count"])
        left_area, right_area = int(left["area"]), int(right["area"])
        union = left_area + right_area - intersection
        expected = {
            "iou": intersection / max(1, union),
            "left_coverage": intersection / max(1, left_area),
            "right_coverage": intersection / max(1, right_area),
        }
        raw_left_id = int(raw["left_observation_id"])
        if raw_left_id != left_id:
            expected["left_coverage"], expected["right_coverage"] = (
                expected["right_coverage"], expected["left_coverage"]
            )
        for key, value in expected.items():
            if not np.isclose(float(raw[key]), value, atol=1e-6):
                raise ValueError(f"relation {(left_id, right_id)} has inconsistent {key}")

        if raw_left_id == left_id:
            left_coverage = float(raw["left_coverage"])
            right_coverage = float(raw["right_coverage"])
        else:
            left_coverage = float(raw["right_coverage"])
            right_coverage = float(raw["left_coverage"])
        sparse[(left_id, right_id)] = {
            "intersection_pixel_count": intersection,
            "iou": float(raw["iou"]),
            "left_coverage": left_coverage,
            "right_coverage": right_coverage,
        }

    records = []
    for frame_index in sorted(by_frame):
        ids = sorted(by_frame[frame_index])
        for left_offset, left_id in enumerate(ids):
            for right_id in ids[left_offset + 1 :]:
                values = sparse.get(
                    (left_id, right_id),
                    {
                        "intersection_pixel_count": 0,
                        "iou": 0.0,
                        "left_coverage": 0.0,
                        "right_coverage": 0.0,
                    },
                )
                kind = classify_relation(
                    values["iou"],
                    values["left_coverage"],
                    values["right_coverage"],
                    duplicate_min_coverage,
                )
                records.append(
                    {
                        "scene_name": str(by_id[left_id]["scene_name"]),
                        "frame_id": str(by_id[left_id]["frame_id"]),
                        "frame_index": frame_index,
                        "left_observation_id": left_id,
                        "right_observation_id": right_id,
                        "left_area": int(by_id[left_id]["area"]),
                        "right_area": int(by_id[right_id]["area"]),
                        **values,
                        "relation_kind": kind,
                        "gt_usage": "none",
                    }
                )
    if set(sparse) - {
        (row["left_observation_id"], row["right_observation_id"]) for row in records
    }:
        raise ValueError("relation graph lost a sparse overlap relation")
    return records


def observation_quality(row):
    return float(row["predicted_iou"]) * float(row["stability_score"])


def select_duplicate_representatives(observations, relations):
    parent = {int(row["observation_id"]): int(row["observation_id"]) for row in observations}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for relation in relations:
        if relation["relation_kind"] == "duplicate":
            union(int(relation["left_observation_id"]), int(relation["right_observation_id"]))

    components = defaultdict(list)
    by_id = {int(row["observation_id"]): row for row in observations}
    for observation_id in sorted(by_id):
        components[find(observation_id)].append(observation_id)

    representatives = {}
    component_ids = {}
    for component_id, members in enumerate(sorted(components.values(), key=lambda item: item[0])):
        representative = min(
            members,
            key=lambda item: (-observation_quality(by_id[item]), item),
        )
        for observation_id in members:
            representatives[observation_id] = representative
            component_ids[observation_id] = component_id
    return representatives, component_ids


def preprocess_hierarchy_safe(observations, relations):
    representatives, component_ids = select_duplicate_representatives(observations, relations)
    kept, actions = [], []
    for row in sorted(observations, key=lambda item: int(item["observation_id"])):
        observation_id = int(row["observation_id"])
        representative = int(representatives[observation_id])
        is_kept = observation_id == representative
        if is_kept:
            kept.append(dict(row))
        actions.append(
            {
                "scene_name": str(row["scene_name"]),
                "frame_id": str(row["frame_id"]),
                "frame_index": int(row["frame_index"]),
                "observation_id": observation_id,
                "policy": "hierarchy_safe",
                "action": "kept" if is_kept else "suppressed_near_duplicate",
                "representative_observation_id": representative,
                "duplicate_component_id": int(component_ids[observation_id]),
                "sam_quality": observation_quality(row),
                "mask_changed": False,
                "point_indices_changed": False,
                "gt_usage": "none",
            }
        )
    return kept, actions


def preprocess_details_exact(observations, point_output_root, published_point_root):
    by_frame = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame_index"])].append(row)

    kept, actions = [], []
    point_output_root.mkdir()
    for frame_index in sorted(by_frame):
        claimed_mask = None
        claimed_points = set()
        ordered = sorted(
            by_frame[frame_index],
            key=lambda row: (int(row["area"]), int(row["observation_id"])),
        )
        for row in ordered:
            observation_id = int(row["observation_id"])
            original_mask = decode_binary_mask_rle(row["mask_rle"])
            if claimed_mask is None:
                claimed_mask = np.zeros_like(original_mask, dtype=bool)
            if original_mask.shape != claimed_mask.shape:
                raise ValueError("same-frame observations have different RLE sizes")
            processed_mask = np.logical_and(original_mask, np.logical_not(claimed_mask))
            claimed_mask |= original_mask

            with np.load(row["point_indices_path"]) as payload:
                original_points = np.unique(
                    np.asarray(payload["point_indices"], dtype=np.int64)
                )
            processed_points = np.asarray(
                [point for point in original_points if int(point) not in claimed_points],
                dtype=np.int64,
            )
            claimed_points.update(int(point) for point in original_points)
            processed_area = int(processed_mask.sum())
            is_kept = processed_area > 0 and len(processed_points) > 0
            action = "kept_unchanged"
            if not is_kept:
                action = "suppressed_empty_after_overlap_removal"
            elif processed_area != int(row["area"]) or len(processed_points) != len(original_points):
                action = "kept_after_larger_mask_overlap_removal"

            output_path = None
            if is_kept:
                filename = f"obs{observation_id:06d}_points.npz"
                np.savez_compressed(
                    point_output_root / filename,
                    point_indices=processed_points,
                )
                output_path = published_point_root / filename
                transformed = dict(row)
                transformed.update(
                    {
                        "area": processed_area,
                        "bbox_xywh": mask_bbox_xywh(processed_mask),
                        "mask_rle": encode_binary_mask_rle(processed_mask),
                        "point_indices_path": str(output_path),
                    }
                )
                kept.append(transformed)
            actions.append(
                {
                    "scene_name": str(row["scene_name"]),
                    "frame_id": str(row["frame_id"]),
                    "frame_index": frame_index,
                    "observation_id": observation_id,
                    "policy": "details_exact",
                    "action": action,
                    "representative_observation_id": observation_id if is_kept else None,
                    "sam_quality": observation_quality(row),
                    "original_area": int(row["area"]),
                    "output_area": processed_area,
                    "removed_pixel_count": int(row["area"]) - processed_area,
                    "original_point_count": int(len(original_points)),
                    "output_point_count": int(len(processed_points)),
                    "removed_point_count": int(len(original_points) - len(processed_points)),
                    "mask_changed": processed_area != int(row["area"]),
                    "point_indices_changed": len(processed_points) != len(original_points),
                    "output_point_indices_path": str(output_path) if output_path else None,
                    "gt_usage": "none",
                }
            )
    return sorted(kept, key=lambda row: int(row["observation_id"])), sorted(
        actions, key=lambda row: int(row["observation_id"])
    )


def validate_preprocessed_output(policy, source, output, actions, staging_point_root=None):
    source_by_id = {int(row["observation_id"]): row for row in source}
    output_by_id = {int(row["observation_id"]): row for row in output}
    action_by_id = {int(row["observation_id"]): row for row in actions}
    if len(source_by_id) != len(source) or len(output_by_id) != len(output):
        raise ValueError("preprocessor output contains duplicate observation ids")
    if set(action_by_id) != set(source_by_id) or len(action_by_id) != len(actions):
        raise ValueError("action ledger does not conserve source observation ids")
    if not set(output_by_id).issubset(source_by_id):
        raise ValueError("preprocessor created an unknown observation id")

    invariant_keys = ("observation_id", "scene_name", "frame_id", "frame_index")
    for observation_id, output_row in output_by_id.items():
        source_row = source_by_id[observation_id]
        if any(output_row[key] != source_row[key] for key in invariant_keys):
            raise ValueError(f"observation {observation_id} changed an invariant field")
        if int(decode_binary_mask_rle(output_row["mask_rle"]).sum()) != int(
            output_row["area"]
        ):
            raise ValueError(f"observation {observation_id} has an invalid output RLE")

    if policy == "hierarchy_safe":
        if any(output_by_id[item] != source_by_id[item] for item in output_by_id):
            raise ValueError("hierarchy_safe changed a kept observation")
        for observation_id, action in action_by_id.items():
            representative = int(action["representative_observation_id"])
            if representative not in output_by_id:
                raise ValueError(
                    f"observation {observation_id} points to a suppressed representative"
                )
            expected_kept = action["action"] == "kept"
            if expected_kept != (observation_id in output_by_id):
                raise ValueError(f"observation {observation_id} action/output mismatch")
    else:
        if staging_point_root is None:
            raise ValueError("details_exact validation requires a staging point root")
        for observation_id, action in action_by_id.items():
            is_kept = observation_id in output_by_id
            if is_kept != (action["representative_observation_id"] == observation_id):
                raise ValueError(f"observation {observation_id} action/output mismatch")
            if is_kept:
                point_path = staging_point_root / Path(
                    output_by_id[observation_id]["point_indices_path"]
                ).name
                if not point_path.is_file():
                    raise FileNotFoundError(f"missing transformed points: {point_path}")


def _build_scene(scene_name, args):
    source_root = args.automatic_root / scene_name
    observations = _read_jsonl(source_root / "automatic_observations.jsonl")
    _validate_observations(scene_name, observations)
    sparse_relations = _read_jsonl(source_root / "same_frame_mask_relations.jsonl")
    relations = build_complete_relation_graph(
        observations, sparse_relations, args.duplicate_min_coverage
    )

    published_root = args.output_root / scene_name
    staging_root = args.output_root / f".{scene_name}.writing"
    if published_root.exists() or staging_root.exists():
        raise FileExistsError(f"output scene already exists: {published_root}")
    staging_root.mkdir(parents=True)
    _write_jsonl(staging_root / "original_automatic_observations.jsonl", observations)
    _write_jsonl(staging_root / "same_frame_hierarchy_relations.jsonl", relations)

    if args.policy == "hierarchy_safe":
        kept, actions = preprocess_hierarchy_safe(observations, relations)
    else:
        kept, actions = preprocess_details_exact(
            observations,
            staging_root / "points",
            args.output_root / scene_name / "points",
        )
    validate_preprocessed_output(
        args.policy,
        observations,
        kept,
        actions,
        staging_root / "points" if args.policy == "details_exact" else None,
    )
    _write_jsonl(staging_root / "automatic_observations.jsonl", kept)
    _write_jsonl(staging_root / "observation_action_ledger.jsonl", actions)

    relation_counts = {
        kind: sum(row["relation_kind"] == kind for row in relations)
        for kind in ("duplicate", "containment", "partial", "disjoint")
    }
    summary = {
        "scene_name": scene_name,
        "policy": args.policy,
        "source_observation_count": len(observations),
        "output_observation_count": len(kept),
        "suppressed_observation_count": len(observations) - len(kept),
        "changed_mask_count": sum(bool(row["mask_changed"]) for row in actions),
        "changed_point_indices_count": sum(
            bool(row["point_indices_changed"]) for row in actions
        ),
        "relation_count": len(relations),
        "relation_counts": relation_counts,
        "duplicate_min_coverage": args.duplicate_min_coverage,
        "gt_usage": "none",
        "decision_state": (
            "Near-duplicate suppression only; containment and partial overlap are ledger-only."
            if args.policy == "hierarchy_safe"
            else "Paper-faithful small-to-large 2D overlap removal ablation."
        ),
    }
    (staging_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging_root, published_root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("hierarchy_safe", "details_exact"),
        default="hierarchy_safe",
    )
    parser.add_argument("--duplicate-min-coverage", type=float, default=0.95)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "automatic_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if not 0.0 < args.duplicate_min_coverage <= 1.0:
        raise SystemExit("--duplicate-min-coverage must be in (0, 1]")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file():
            if not args.resume:
                raise SystemExit(f"output scene already exists: {scene_name}")
            summary = json.loads(existing.read_text())
            if summary.get("policy") != args.policy:
                raise SystemExit(f"resume policy mismatch for {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: "
                f"{summary['source_observation_count']} -> {summary['output_observation_count']}",
                flush=True,
            )
        summaries.append(summary)

    payload = {
        "scene_count": len(summaries),
        "policy": args.policy,
        "source_observation_count": sum(
            row["source_observation_count"] for row in summaries
        ),
        "output_observation_count": sum(
            row["output_observation_count"] for row in summaries
        ),
        "suppressed_observation_count": sum(
            row["suppressed_observation_count"] for row in summaries
        ),
        "changed_mask_count": sum(row["changed_mask_count"] for row in summaries),
        "changed_point_indices_count": sum(
            row["changed_point_indices_count"] for row in summaries
        ),
        "relation_counts": {
            kind: sum(row["relation_counts"][kind] for row in summaries)
            for kind in ("duplicate", "containment", "partial", "disjoint")
        },
        "params": {key: value for key, value in vars(args).items()},
        "gt_usage": "none",
    }
    (args.output_root / "same_frame_hierarchy_preprocessor_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
