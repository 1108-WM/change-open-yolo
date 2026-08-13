#!/usr/bin/env python3
"""Build offline utility labels and no-GT features for pair proposals."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
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
    _sha256,
    _write_jsonl,
)
from tools.build_train_candidate_component_union_feature_ledger import _track_points  # noqa: E402
from tools.diagnose_candidate_component_geometry_ceiling_gt import (  # noqa: E402
    _geometry_digest,
    iou_by_gt,
)
from tools.diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids  # noqa: E402
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    MIN_REGION_SIZE,
)


VERSIONS = {
    "pair_union": "official100_pair_union_utility_ledger_v1",
    "pair_intersection": "official100_pair_intersection_utility_ledger_v1",
}
VERSION = VERSIONS["pair_union"]
OFFICIAL_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
LEARNED_SCORE_TOKENS = (
    "C_plus_geometry_track_structure",
    "D_plus_gvc",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_proposal_target(
    proposal_iou: dict[int, float], existing_best: dict[int, float],
) -> dict:
    candidates = []
    for gt_id, iou in proposal_iou.items():
        base = float(existing_best[gt_id])
        crossings = sum(
            iou >= threshold and base < threshold
            for threshold in OFFICIAL_THRESHOLDS
        )
        candidates.append((crossings, iou - base, iou, -int(gt_id), int(gt_id), base))
    if not candidates:
        return {
            "target_gt_instance_id": None,
            "proposal_iou": 0.0,
            "existing_best_iou": 0.0,
            "iou_gain": 0.0,
            "official_threshold_cross_count": 0,
            "official_threshold_cross_fraction": 0.0,
        }
    crossings, gain, iou, _, gt_id, base = max(candidates)
    return {
        "target_gt_instance_id": gt_id,
        "proposal_iou": float(iou),
        "existing_best_iou": float(base),
        "iou_gain": float(gain),
        "official_threshold_cross_count": int(crossings),
        "official_threshold_cross_fraction": float(
            crossings / len(OFFICIAL_THRESHOLDS)
        ),
    }


def select_union_target(
    proposal_iou: dict[int, float], existing_best: dict[int, float],
) -> dict:
    """Backward-compatible name retained for the frozen pair-union tests."""
    return select_proposal_target(proposal_iou, existing_best)


def proposal_points(kind: str, track: np.ndarray, native: np.ndarray) -> np.ndarray:
    if kind == "pair_union":
        return np.union1d(track, native).astype(np.int64)
    if kind == "pair_intersection":
        return np.intersect1d(track, native, assume_unique=True).astype(np.int64)
    raise ValueError(f"unsupported pair proposal kind: {kind}")


def _pure_inference_relation_features(features: dict) -> dict[str, float]:
    output = {}
    for name, value in features.items():
        if any(token in name for token in LEARNED_SCORE_TOKENS):
            continue
        if isinstance(value, bool):
            output[f"relation__{name}"] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            output[f"relation__{name}"] = float(value)
    if not output:
        raise ValueError("relation row has no pure inference features")
    return output


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
        raise ValueError(f"{scene}: track file differs from candidate ledger")
    track_points = {
        track_id: _track_points(track, masks.shape[0])
        for track_id, track in track_by_id.items()
    }

    gt_path = args.gt_dir / f"{scene}.txt"
    gt_ids = _class_agnostic_gt_ids(np.loadtxt(gt_path, dtype=np.int64))
    valid_ids, valid_sizes = np.unique(gt_ids[gt_ids > 0], return_counts=True)
    gt_sizes = {
        int(gt_id): int(size) for gt_id, size in zip(valid_ids, valid_sizes)
        if int(size) >= MIN_REGION_SIZE
    }
    existing_best = {gt_id: 0.0 for gt_id in gt_sizes}
    existing_digests = set()

    def add_existing(points: np.ndarray) -> None:
        if len(points) < MIN_REGION_SIZE:
            return
        existing_digests.add(_geometry_digest(points))
        for gt_id, iou in iou_by_gt(points, gt_ids, gt_sizes).items():
            existing_best[gt_id] = max(existing_best[gt_id], iou)

    for points in native_points.values():
        add_existing(points)
    for points in track_points.values():
        add_existing(points)

    relations_path = args.relation_feature_ledger_root / scene / "relation_features.jsonl"
    relations = read_jsonl(relations_path)
    output = []
    for relation in relations:
        track_id = int(relation["track_id"])
        group_id = str(relation["native_exact_geometry_group_id"])
        track = track_points[track_id]
        native = native_points[group_id]
        intersection_count = int(len(np.intersect1d(track, native, assume_unique=True)))
        union = np.union1d(track, native).astype(np.int64)
        proposal = proposal_points(args.proposal_kind, track, native)
        proposal_iou = iou_by_gt(proposal, gt_ids, gt_sizes)
        label = select_proposal_target(proposal_iou, existing_best)
        target_id = label["target_gt_instance_id"]
        track_target_iou = (
            iou_by_gt(track, gt_ids, gt_sizes).get(target_id, 0.0)
            if target_id is not None else 0.0
        )
        native_target_iou = (
            iou_by_gt(native, gt_ids, gt_sizes).get(target_id, 0.0)
            if target_id is not None else 0.0
        )
        features = _pure_inference_relation_features(relation["features"])
        features.update({
            "proposal__union_point_count": float(len(union)),
            "proposal__intersection_point_count": float(intersection_count),
            "proposal__track_exclusive_point_count": float(len(track) - intersection_count),
            "proposal__native_exclusive_point_count": float(len(native) - intersection_count),
            "proposal__union_growth_over_track_fraction": float(
                (len(union) - len(track)) / max(1, len(track))
            ),
            "proposal__union_growth_over_native_fraction": float(
                (len(union) - len(native)) / max(1, len(native))
            ),
            "proposal__geometry_novel_vs_existing": float(
                _geometry_digest(proposal) not in existing_digests
            ),
            "proposal__accepted_min_region": float(len(proposal) >= MIN_REGION_SIZE),
        })
        if args.proposal_kind == "pair_intersection":
            features.update({
                "proposal__intersection_retention_of_track_fraction": float(
                    len(proposal) / max(1, len(track))
                ),
                "proposal__intersection_retention_of_native_fraction": float(
                    len(proposal) / max(1, len(native))
                ),
                "proposal__intersection_fraction_of_union": float(
                    len(proposal) / max(1, len(union))
                ),
            })
        output.append({
            "scene_name": scene,
            "relation_component_id": int(relation["relation_component_id"]),
            "track_id": track_id,
            "native_exact_geometry_group_id": group_id,
            "native_member_candidate_ids": [
                int(value) for value in relation["native_member_candidate_ids"]
            ],
            "proposal_kind": args.proposal_kind,
            "proposal_point_count": len(proposal),
            "proposal_geometry_sha256": _geometry_digest(proposal),
            "model_features": features,
            "labels": {
                "ground_truth_usage": (
                    f"official_train_offline_{args.proposal_kind}_utility_only"
                ),
                **label,
                "track_iou_to_selected_target": float(track_target_iou),
                "native_iou_to_selected_target": float(native_target_iou),
                "improves_existing_best_iou": bool(label["iou_gain"] > 1e-12),
                "crosses_any_official_threshold": bool(
                    label["official_threshold_cross_count"] > 0
                ),
            },
            "contracts": {
                "feature_ground_truth_usage": "none",
                "proposal_materialized": False,
                "candidate_files_modified": False,
                "ap_computed": False,
            },
        })
    if len(output) != len(relations):
        raise AssertionError(f"{scene}: {args.proposal_kind} proposal conservation failed")
    summary = {
        "scene_name": scene,
        "proposal_count": len(output),
        "accepted_min_region_count": sum(
            row["model_features"]["proposal__accepted_min_region"] for row in output
        ),
        "novel_geometry_count": sum(
            row["model_features"]["proposal__geometry_novel_vs_existing"] for row in output
        ),
        "improves_existing_best_iou_count": sum(
            row["labels"]["improves_existing_best_iou"] for row in output
        ),
        "crosses_any_official_threshold_count": sum(
            row["labels"]["crosses_any_official_threshold"] for row in output
        ),
        "official_threshold_cross_count_distribution": dict(sorted(Counter(
            row["labels"]["official_threshold_cross_count"] for row in output
        ).items())),
        "input_provenance": {
            "candidate_ledger_sha256": _sha256(candidate_ledger_path(args.records_root, scene)),
            "native_masks_sha256": _sha256(masks_path),
            "native_scores_sha256": _sha256(scores_path),
            "filtered_tracks_sha256": _sha256(track_path),
            "ground_truth_sha256": _sha256(gt_path),
            "relation_features_sha256": _sha256(relations_path),
        },
    }
    return output, summary


def run(args: argparse.Namespace) -> dict:
    all_scenes = read_scene_list(args.scene_list)
    if len(all_scenes) != args.expected_scene_count:
        raise ValueError("scene count differs from fixed protocol")
    scene_list_sha = _sha256(args.scene_list)
    if args.expected_scene_count == 100 and scene_list_sha != EXPECTED_SCENE_LIST_SHA256:
        raise ValueError("official100 scene-list SHA-256 mismatch")
    scenes = all_scenes if args.max_scenes is None else all_scenes[:args.max_scenes]
    if not scenes:
        raise ValueError("smoke scene count must be positive")
    relation_summary_path = args.relation_feature_ledger_root / "summary.json"
    relation_summary = json.loads(relation_summary_path.read_text())
    if relation_summary.get("feature_ground_truth_usage") != "none":
        raise ValueError("relation ledger violates no-GT feature contract")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    all_rows, summaries = [], []
    try:
        for index, scene in enumerate(scenes, start=1):
            rows, summary = _scene(scene, args)
            scene_root = staging / scene
            scene_root.mkdir()
            _write_jsonl(scene_root / f"{args.proposal_kind}_utilities.jsonl", rows)
            (scene_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            all_rows.extend(rows)
            summaries.append(summary)
            print(
                f"[{args.proposal_kind} utility] {index}/{len(scenes)} {scene}",
                flush=True,
            )
        feature_names = sorted(all_rows[0]["model_features"]) if all_rows else []
        if any(sorted(row["model_features"]) != feature_names for row in all_rows):
            raise ValueError("pair-union feature schema differs across proposals")
        _write_jsonl(staging / f"{args.proposal_kind}_utilities.jsonl", all_rows)
        payload = {
            "version": VERSIONS[args.proposal_kind],
            "proposal_kind": args.proposal_kind,
            "scene_count": len(scenes),
            "expected_full_scene_count": args.expected_scene_count,
            "is_smoke_subset": len(scenes) != args.expected_scene_count,
            "proposal_count": len(all_rows),
            "accepted_min_region_count": int(sum(
                row["accepted_min_region_count"] for row in summaries
            )),
            "novel_geometry_count": int(sum(
                row["novel_geometry_count"] for row in summaries
            )),
            "improves_existing_best_iou_count": int(sum(
                row["improves_existing_best_iou_count"] for row in summaries
            )),
            "crosses_any_official_threshold_count": int(sum(
                row["crosses_any_official_threshold_count"] for row in summaries
            )),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "learned_candidate_quality_scores_excluded_from_features": True,
            "feature_ground_truth_usage": "none",
            "label_ground_truth_usage": (
                f"official_train_offline_{args.proposal_kind}_utility_only"
            ),
            "proposal_materialized": False,
            "ap_computed": False,
            "threshold_scanning": False,
            "safety60_read": False,
            "even48_read": False,
            "test60_read": False,
            "input_provenance": {
                "scene_list_sha256": scene_list_sha,
                "relation_summary_sha256": _sha256(relation_summary_path),
            },
            "scene_summaries": summaries,
        }
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
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument(
        "--proposal-kind", choices=tuple(VERSIONS), default="pair_union"
    )
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "records_root", "relation_feature_ledger_root", "gt_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        key: summary[key] for key in (
            "scene_count", "proposal_count", "novel_geometry_count",
            "improves_existing_best_iou_count", "crosses_any_official_threshold_count",
        )
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
