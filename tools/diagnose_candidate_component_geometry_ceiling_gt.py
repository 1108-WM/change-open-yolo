#!/usr/bin/env python3
"""Diagnose the candidate-level high-IoU geometry ceiling on official100.

This GT-only offline diagnostic never computes AP.  For every valid GT
instance it records the best IoU already reachable by frozen native and track
candidates, then the best IoU after append-only geometry families derived from
native--track relation components.  Candidate scores, classes, and files are
never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    candidate_ledger_path,
    read_jsonl,
    read_scene_list,
)
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    EXPECTED_SCENE_LIST_SHA256,
    _canonical_native_groups,
)
from tools.build_train_candidate_component_union_feature_ledger import (  # noqa: E402
    _track_points,
)
from tools.diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids  # noqa: E402
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    MIN_REGION_SIZE,
)


VERSION = "official100_component_geometry_ceiling_gt_v1"
THRESHOLDS = (0.25, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
FAMILIES = (
    "pair_union",
    "pair_intersection",
    "pair_superpoint_vote",
    "pair_superpoint_agreement",
    "track_component_native_union",
    "track_component_native_intersection",
    "component_all_union",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_digest(points: np.ndarray) -> str:
    points = np.asarray(points, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(len(points).to_bytes(8, "little"))
    digest.update(np.ascontiguousarray(points).tobytes())
    return digest.hexdigest()


def iou_by_gt(
    points: np.ndarray, gt_ids: np.ndarray, gt_sizes: dict[int, int],
) -> dict[int, float]:
    points = np.asarray(points, dtype=np.int64)
    if not len(points):
        return {}
    ids, intersections = np.unique(gt_ids[points], return_counts=True)
    result = {}
    for gt_id, intersection in zip(ids, intersections):
        gt_id = int(gt_id)
        if gt_id <= 0 or gt_id not in gt_sizes:
            continue
        union = len(points) + gt_sizes[gt_id] - int(intersection)
        result[gt_id] = float(intersection / max(1, union))
    return result


def _superpoint_index(superpoints: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    result = {
        int(superpoint_id): np.flatnonzero(superpoints == superpoint_id).astype(np.int64)
        for superpoint_id in np.unique(superpoints)
    }
    return result, {superpoint_id: len(points) for superpoint_id, points in result.items()}


def _occupancy(points: np.ndarray, superpoints: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(superpoints[points], return_counts=True)
    return {int(superpoint_id): int(count) for superpoint_id, count in zip(ids, counts)}


def superpoint_vote_fusion(
    left: np.ndarray,
    right: np.ndarray,
    superpoints: np.ndarray,
    points_by_superpoint: dict[int, np.ndarray],
    superpoint_sizes: dict[int, int],
) -> np.ndarray:
    """Full superpoints receiving at least one normalized vote across two sources.

    Each source contributes its occupancy fraction in the superpoint.  The
    fixed two-source consensus rule keeps a superpoint when the two fractions
    sum to at least one.  There is no learned or scanned threshold.
    """
    left_counts = _occupancy(left, superpoints)
    right_counts = _occupancy(right, superpoints)
    selected = []
    for superpoint_id in sorted(set(left_counts) | set(right_counts)):
        size = superpoint_sizes[superpoint_id]
        support = (
            left_counts.get(superpoint_id, 0) / size
            + right_counts.get(superpoint_id, 0) / size
        )
        if support >= 1.0 - 1e-12:
            selected.append(points_by_superpoint[superpoint_id])
    return (
        np.concatenate(selected).astype(np.int64)
        if selected else np.empty(0, dtype=np.int64)
    )


def superpoint_agreement_fusion(
    left: np.ndarray,
    right: np.ndarray,
    superpoints: np.ndarray,
    points_by_superpoint: dict[int, np.ndarray],
) -> np.ndarray:
    """Expand only raw superpoints touched by both frozen source masks."""
    left_ids = set(map(int, np.unique(superpoints[left])))
    right_ids = set(map(int, np.unique(superpoints[right])))
    selected = [points_by_superpoint[item] for item in sorted(left_ids & right_ids)]
    return (
        np.concatenate(selected).astype(np.int64)
        if selected else np.empty(0, dtype=np.int64)
    )


def _coverage(best: dict[int, float]) -> dict[str, dict]:
    count = len(best)
    result = {}
    for threshold in THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        covered = sum(value >= threshold for value in best.values())
        result[tag] = {
            "iou_threshold": threshold,
            "covered_gt_count": covered,
            "covered_gt_fraction": float(covered / max(1, count)),
        }
    return result


def _quality_band(value: float) -> str:
    if value >= 0.90:
        return "iou90_plus"
    if value >= 0.75:
        return "iou75_90"
    if value >= 0.50:
        return "iou50_75"
    if value >= 0.25:
        return "iou25_50"
    return "below_iou25"


def _scene(scene: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    ledger_rows = read_jsonl(candidate_ledger_path(args.records_root, scene))
    native_rows = sorted(
        (row for row in ledger_rows if row["candidate_source"] == NATIVE_SOURCE),
        key=lambda row: int(row["candidate_id"]),
    )
    track_rows = sorted(
        (row for row in ledger_rows if row["candidate_source"] == TRACK_SOURCE),
        key=lambda row: int(row["candidate_id"]),
    )
    cache_root = args.records_root / scene / "native_cache"
    masks_path = cache_root / f"{scene}_pred_masks.npy"
    scores_path = cache_root / f"{scene}_pred_scores.npy"
    masks = np.load(masks_path, mmap_mode="r")
    scores = np.asarray(np.load(scores_path, mmap_mode="r"), dtype=np.float64)
    groups, _ = _canonical_native_groups(scene, masks, native_rows, scores)
    native_points = {
        group_id: np.flatnonzero(np.asarray(
            masks[:, group["representative_candidate_id"]], dtype=bool
        )).astype(np.int64)
        for group_id, group in groups.items()
    }

    track_path = (
        args.records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
    )
    tracks = json.loads(track_path.read_text()).get("tracks", [])
    track_by_id = {int(row["track_id"]): row for row in tracks}
    if set(track_by_id) != {int(row["candidate_id"]) for row in track_rows}:
        raise ValueError(f"{scene}: filtered tracks differ from candidate ledger")
    all_track_points = {
        track_id: _track_points(track, masks.shape[0])
        for track_id, track in track_by_id.items()
    }

    scene_stem = scene[len("scene"):] if scene.startswith("scene") else scene
    processed_path = args.prepared_root / scene / f"{scene_stem}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    if len(processed) != masks.shape[0] or processed.shape[1] < 10:
        raise ValueError(f"{scene}: prepared points differ from native masks")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    points_by_superpoint, superpoint_sizes = _superpoint_index(superpoints)

    gt_path = args.gt_dir / f"{scene}.txt"
    raw_gt_ids = np.loadtxt(gt_path, dtype=np.int64)
    gt_ids = _class_agnostic_gt_ids(raw_gt_ids)
    if len(gt_ids) != masks.shape[0]:
        raise ValueError(f"{scene}: GT point count differs from candidates")
    valid_gt_ids, valid_gt_sizes = np.unique(gt_ids[gt_ids > 0], return_counts=True)
    gt_sizes = {
        int(gt_id): int(size) for gt_id, size in zip(valid_gt_ids, valid_gt_sizes)
        if int(size) >= MIN_REGION_SIZE
    }
    if not gt_sizes:
        raise ValueError(f"{scene}: no valid class-agnostic GT instances")

    existing_best = {gt_id: 0.0 for gt_id in gt_sizes}
    existing_best_source = {gt_id: None for gt_id in gt_sizes}
    existing_digests = set()

    def update_existing(points: np.ndarray, source: str) -> None:
        if len(points) < MIN_REGION_SIZE:
            return
        existing_digests.add(_geometry_digest(points))
        for gt_id, iou in iou_by_gt(points, gt_ids, gt_sizes).items():
            if iou > existing_best[gt_id]:
                existing_best[gt_id] = iou
                existing_best_source[gt_id] = source

    for points in native_points.values():
        update_existing(points, NATIVE_SOURCE)
    for points in all_track_points.values():
        update_existing(points, TRACK_SOURCE)

    family_best = {
        family: dict(existing_best) for family in FAMILIES
    }
    combined_best = dict(existing_best)
    family_stats = {
        family: Counter({
            "raw_proposal_count": 0,
            "accepted_min_region_count": 0,
            "novel_geometry_count": 0,
            "proposal_improving_any_gt_count": 0,
        })
        for family in FAMILIES
    }

    def add_proposal(family: str, points: np.ndarray) -> None:
        points = np.unique(np.asarray(points, dtype=np.int64))
        stats = family_stats[family]
        stats["raw_proposal_count"] += 1
        if len(points) < MIN_REGION_SIZE:
            return
        stats["accepted_min_region_count"] += 1
        digest = _geometry_digest(points)
        if digest not in existing_digests:
            stats["novel_geometry_count"] += 1
        improves = False
        for gt_id, iou in iou_by_gt(points, gt_ids, gt_sizes).items():
            if iou > existing_best[gt_id] + 1e-12:
                improves = True
            if iou > family_best[family][gt_id]:
                family_best[family][gt_id] = iou
            if iou > combined_best[gt_id]:
                combined_best[gt_id] = iou
        stats["proposal_improving_any_gt_count"] += int(improves)

    relations = read_jsonl(
        args.relation_feature_ledger_root / scene / "relation_features.jsonl"
    )
    components = read_jsonl(
        args.relation_feature_ledger_root / scene / "relation_components.jsonl"
    )
    relations_by_component = defaultdict(list)
    for row in relations:
        relations_by_component[int(row["relation_component_id"])].append(row)

    for row in relations:
        track = all_track_points[int(row["track_id"])]
        native = native_points[str(row["native_exact_geometry_group_id"])]
        add_proposal("pair_union", np.union1d(track, native))
        add_proposal("pair_intersection", np.intersect1d(track, native, assume_unique=True))
        add_proposal(
            "pair_superpoint_vote",
            superpoint_vote_fusion(
                track, native, superpoints, points_by_superpoint, superpoint_sizes
            ),
        )
        add_proposal(
            "pair_superpoint_agreement",
            superpoint_agreement_fusion(
                track, native, superpoints, points_by_superpoint
            ),
        )

    for component in components:
        component_id = int(component["relation_component_id"])
        component_native = np.unique(np.concatenate([
            native_points[str(group_id)]
            for group_id in component["native_exact_geometry_group_ids"]
        ]))
        component_tracks = [
            all_track_points[int(track_id)] for track_id in component["track_ids"]
        ]
        for track in component_tracks:
            add_proposal(
                "track_component_native_union", np.union1d(track, component_native)
            )
            add_proposal(
                "track_component_native_intersection",
                np.intersect1d(track, component_native, assume_unique=True),
            )
        add_proposal(
            "component_all_union",
            np.unique(np.concatenate([component_native, *component_tracks])),
        )
        if len(relations_by_component[component_id]) != int(component["relation_count"]):
            raise ValueError(f"{scene}/{component_id}: component relation count mismatch")

    gt_rows = []
    for gt_id in sorted(gt_sizes):
        family_values = {
            family: float(family_best[family][gt_id]) for family in FAMILIES
        }
        best_family = max(FAMILIES, key=lambda family: family_values[family])
        gt_rows.append({
            "scene_name": scene,
            "class_agnostic_gt_instance_id": gt_id,
            "gt_point_count": gt_sizes[gt_id],
            "existing_best_iou": float(existing_best[gt_id]),
            "existing_best_source": existing_best_source[gt_id],
            "family_best_iou_after_append": family_values,
            "combined_geometry_best_iou_after_append": float(combined_best[gt_id]),
            "combined_iou_gain": float(combined_best[gt_id] - existing_best[gt_id]),
            "best_geometry_family": best_family,
            "existing_quality_band": _quality_band(existing_best[gt_id]),
            "combined_quality_band": _quality_band(combined_best[gt_id]),
        })

    summary = {
        "scene_name": scene,
        "gt_instance_count": len(gt_sizes),
        "native_representative_count": len(native_points),
        "track_count": len(all_track_points),
        "relation_count": len(relations),
        "relation_component_count": len(components),
        "existing_coverage": _coverage(existing_best),
        "family_coverage_after_append": {
            family: _coverage(family_best[family]) for family in FAMILIES
        },
        "combined_geometry_coverage_after_append": _coverage(combined_best),
        "family_proposal_stats": {
            family: dict(stats) for family, stats in family_stats.items()
        },
        "input_provenance": {
            "candidate_ledger_sha256": _sha256(candidate_ledger_path(args.records_root, scene)),
            "native_masks_sha256": _sha256(masks_path),
            "native_scores_sha256": _sha256(scores_path),
            "filtered_tracks_sha256": _sha256(track_path),
            "prepared_points_sha256": _sha256(processed_path),
            "ground_truth_sha256": _sha256(gt_path),
            "relation_features_sha256": _sha256(
                args.relation_feature_ledger_root / scene / "relation_features.jsonl"
            ),
            "relation_components_sha256": _sha256(
                args.relation_feature_ledger_root / scene / "relation_components.jsonl"
            ),
        },
    }
    return gt_rows, summary


def run(args: argparse.Namespace) -> dict:
    all_scenes = read_scene_list(args.scene_list)
    if len(all_scenes) != args.expected_scene_count:
        raise ValueError(
            f"expected {args.expected_scene_count} scenes, got {len(all_scenes)}"
        )
    scene_list_sha = _sha256(args.scene_list)
    if args.expected_scene_count == 100 and scene_list_sha != EXPECTED_SCENE_LIST_SHA256:
        raise ValueError("official100 scene-list SHA-256 mismatch")
    scenes = all_scenes if args.max_scenes is None else all_scenes[:args.max_scenes]
    if not scenes:
        raise ValueError("smoke scene count must be positive")
    relation_summary = json.loads(
        (args.relation_feature_ledger_root / "summary.json").read_text()
    )
    if relation_summary.get("feature_ground_truth_usage") != "none":
        raise ValueError("relation ledger violates no-GT feature contract")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    all_gt_rows = []
    scene_summaries = []
    try:
        for index, scene in enumerate(scenes, start=1):
            rows, summary = _scene(scene, args)
            scene_root = staging / scene
            scene_root.mkdir()
            (scene_root / "gt_geometry_ceiling.jsonl").write_text("".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ))
            (scene_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            all_gt_rows.extend(rows)
            scene_summaries.append(summary)
            print(f"[geometry ceiling] {index}/{len(scenes)} {scene}", flush=True)

        gt_count = len(all_gt_rows)
        existing_best = {
            index: float(row["existing_best_iou"]) for index, row in enumerate(all_gt_rows)
        }
        combined_best = {
            index: float(row["combined_geometry_best_iou_after_append"])
            for index, row in enumerate(all_gt_rows)
        }
        family_best = {
            family: {
                index: float(row["family_best_iou_after_append"][family])
                for index, row in enumerate(all_gt_rows)
            }
            for family in FAMILIES
        }
        existing_coverage = _coverage(existing_best)
        combined_coverage = _coverage(combined_best)
        family_coverage = {
            family: _coverage(values) for family, values in family_best.items()
        }
        coverage_delta = {
            tag: {
                "covered_gt_count": (
                    combined_coverage[tag]["covered_gt_count"]
                    - existing_coverage[tag]["covered_gt_count"]
                ),
                "covered_gt_fraction": (
                    combined_coverage[tag]["covered_gt_fraction"]
                    - existing_coverage[tag]["covered_gt_fraction"]
                ),
            }
            for tag in existing_coverage
        }
        family_delta = {
            family: {
                tag: {
                    "covered_gt_count": (
                        coverage[tag]["covered_gt_count"]
                        - existing_coverage[tag]["covered_gt_count"]
                    ),
                    "covered_gt_fraction": (
                        coverage[tag]["covered_gt_fraction"]
                        - existing_coverage[tag]["covered_gt_fraction"]
                    ),
                }
                for tag in existing_coverage
            }
            for family, coverage in family_coverage.items()
        }
        gains = np.asarray([
            row["combined_iou_gain"] for row in all_gt_rows
        ], dtype=np.float64)
        payload = {
            "version": VERSION,
            "diagnostic_type": "official train GT-only append-only candidate geometry IoU ceiling; no AP",
            "scene_count": len(scenes),
            "expected_full_scene_count": args.expected_scene_count,
            "is_smoke_subset": len(scenes) != args.expected_scene_count,
            "gt_instance_count": gt_count,
            "geometry_families": list(FAMILIES),
            "existing_coverage": existing_coverage,
            "family_coverage_after_append": family_coverage,
            "family_coverage_delta_vs_existing": family_delta,
            "combined_geometry_coverage_after_append": combined_coverage,
            "combined_coverage_delta_vs_existing": coverage_delta,
            "combined_iou_gain_summary": {
                "improved_gt_count": int(np.count_nonzero(gains > 1e-12)),
                "mean": float(gains.mean()),
                "median": float(np.median(gains)),
                "max": float(gains.max(initial=0.0)),
            },
            "existing_quality_band_counts": dict(sorted(Counter(
                row["existing_quality_band"] for row in all_gt_rows
            ).items())),
            "combined_quality_band_counts": dict(sorted(Counter(
                row["combined_quality_band"] for row in all_gt_rows
            ).items())),
            "best_geometry_family_counts": dict(sorted(Counter(
                row["best_geometry_family"] for row in all_gt_rows
                if row["combined_iou_gain"] > 1e-12
            ).items())),
            "proposal_stats": {
                family: dict(sum((
                    Counter(summary["family_proposal_stats"][family])
                    for summary in scene_summaries
                ), Counter()))
                for family in FAMILIES
            },
            "contracts": {
                "ap_computed": False,
                "candidate_scores_read_for_geometry_choice": False,
                "candidate_files_modified": False,
                "candidate_geometry_materialized": False,
                "threshold_scanning": False,
                "ground_truth_usage": "offline candidate-IoU ceiling labels only",
                "inference_geometry_inputs_use_ground_truth": False,
            },
            "safety60_read": False,
            "even48_read": False,
            "test60_read": False,
            "input_provenance": {
                "scene_list_sha256": scene_list_sha,
                "relation_summary_sha256": _sha256(
                    args.relation_feature_ledger_root / "summary.json"
                ),
            },
            "scene_summaries": scene_summaries,
        }
        (staging / "gt_geometry_ceiling.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in all_gt_rows
        ))
        (staging / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "records_root", "prepared_root", "relation_feature_ledger_root",
        "gt_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "scene_count": summary["scene_count"],
        "gt_instance_count": summary["gt_instance_count"],
        "combined_coverage_delta_vs_existing": summary[
            "combined_coverage_delta_vs_existing"
        ],
        "combined_iou_gain_summary": summary["combined_iou_gain_summary"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
