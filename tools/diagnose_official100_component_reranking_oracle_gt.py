#!/usr/bin/env python3
"""GT-only reranking oracle audit for the current official100 coexist candidates.

The candidate set is exactly the native exact-geometry representatives plus
the filtered D2b tracks used by the structured action head.  This diagnostic
never adds, removes, or changes a mask/class.  It reports three distinct
questions which must not be conflated:

* fixed-mask threshold-specific maximum-matching coverage ceilings;
* realizable shared-score permutations that preserve every component's score
  multiset while assigning larger scores to larger best-GT-IoU candidates;
* optimistic threshold-specific component score controls that put a maximum
  one-to-one matched subset before the frozen candidates and the remaining
  controlled candidates after them.

The last family uses a different GT-only score control at every IoU threshold,
so its averaged AP is an oracle diagnostic, not one deployable shared score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
    _scene_inputs,
    _scene_records,
    _sha256,
    configure_track_score_context,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    MIN_REGION_SIZE,
    _set_match_scores,
)
from tools.train_candidate_component_action_head_oof import _metrics_from_records  # noqa: E402


VERSION = "official100_component_reranking_oracle_gt_v2"
THRESHOLDS = (0.25, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
OFFICIAL_THRESHOLDS = tuple(value for value in THRESHOLDS if 0.50 <= value <= 0.90)
PERMUTATION_VARIANTS = (
    "component_native_only",
    "component_track_only",
    "component_source_separate",
    "component_joint",
)
THRESHOLD_ORACLE_VARIANTS = (
    "component_native_only",
    "component_track_only",
    "component_source_separate",
    "component_joint",
)
SHARED_ORACLE_VARIANTS = THRESHOLD_ORACLE_VARIANTS


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def maximum_matching_selection(
    candidate_edges: dict[tuple[str, int], list[tuple[int, float]]],
) -> dict[tuple[str, int], tuple[int, float]]:
    """Return a deterministic maximum candidate--GT matching.

    Candidate adjacency is visited from larger IoU to smaller IoU.  The
    cardinality is exact; IoU is only a deterministic preference among equal
    cardinality matchings.
    """
    candidates = sorted(candidate_edges, key=lambda key: (key[0], key[1]))
    gt_owner: dict[int, tuple[str, int]] = {}
    gt_edge_iou: dict[int, float] = {}

    def visit(candidate: tuple[str, int], seen: set[int]) -> bool:
        edges = sorted(candidate_edges[candidate], key=lambda item: (-item[1], item[0]))
        for gt_index, iou in edges:
            if gt_index in seen:
                continue
            seen.add(gt_index)
            previous = gt_owner.get(gt_index)
            if previous is None or visit(previous, seen):
                gt_owner[gt_index] = candidate
                gt_edge_iou[gt_index] = float(iou)
                return True
        return False

    for candidate in candidates:
        visit(candidate, set())
    return {owner: (gt_index, gt_edge_iou[gt_index]) for gt_index, owner in gt_owner.items()}


def component_permutation_scores(
    base_scores: dict[tuple[str, int], float],
    qualities: dict[tuple[str, int], float],
    components: list[dict[str, list[tuple[str, int]]]],
    variant: str,
) -> dict[tuple[str, int], float]:
    """Permute existing score levels only within disjoint relation components."""
    if variant not in PERMUTATION_VARIANTS:
        raise ValueError(f"unknown permutation variant: {variant}")
    result = dict(base_scores)
    for component in components:
        native = [key for key in component["native"] if key in base_scores]
        track = [key for key in component["track"] if key in base_scores]
        if variant == "component_native_only":
            groups = [native]
        elif variant == "component_track_only":
            groups = [track]
        elif variant == "component_source_separate":
            groups = [native, track]
        else:
            groups = [native + track]
        for keys in groups:
            if len(keys) < 2:
                continue
            levels = sorted(float(base_scores[key]) for key in keys)
            ordered = sorted(
                keys,
                key=lambda key: (
                    float(qualities[key]),
                    float(base_scores[key]),
                    key[0],
                    key[1],
                ),
            )
            for key, score in zip(ordered, levels):
                result[key] = score
    return result


def threshold_oracle_scores(
    base_scores: dict[tuple[str, int], float],
    qualities: dict[tuple[str, int], float],
    edges: dict[tuple[str, int], list[tuple[int, float]]],
    native_controlled: set[tuple[str, int]],
    track_controlled: set[tuple[str, int]],
    variant: str,
) -> tuple[dict[tuple[str, int], float], set[tuple[str, int]]]:
    """Build one optimistic threshold-specific score control."""
    if variant not in THRESHOLD_ORACLE_VARIANTS:
        raise ValueError(f"unknown threshold oracle variant: {variant}")
    native = native_controlled & set(base_scores)
    track = track_controlled & set(base_scores)
    if variant == "component_native_only":
        controlled = native
        selections = [maximum_matching_selection({key: edges.get(key, []) for key in native})]
    elif variant == "component_track_only":
        controlled = track
        selections = [maximum_matching_selection({key: edges.get(key, []) for key in track})]
    elif variant == "component_source_separate":
        controlled = native | track
        selections = [
            maximum_matching_selection({key: edges.get(key, []) for key in native}),
            maximum_matching_selection({key: edges.get(key, []) for key in track}),
        ]
    else:
        controlled = native | track
        selections = [
            maximum_matching_selection({key: edges.get(key, []) for key in controlled})
        ]
    selected = set().union(*(set(selection) for selection in selections))
    result = dict(base_scores)
    for key in controlled:
        quality = float(qualities[key])
        if key in selected:
            # All selected candidates precede frozen [0,1] scores.
            result[key] = 2.0 + quality
        else:
            # All remaining controlled candidates stay present at the tail.
            result[key] = -2.0 + quality
    return result, selected


def shared_gt_quality_scores(
    base_scores: dict[tuple[str, int], float],
    qualities: dict[tuple[str, int], float],
    target_ids: dict[tuple[str, int], int],
    native_controlled: set[tuple[str, int]],
    track_controlled: set[tuple[str, int]],
    variant: str,
) -> tuple[dict[tuple[str, int], float], set[tuple[str, int]]]:
    """Build one shared GT-only score order for all evaluation thresholds.

    Within each controlled pool, at most one candidate per best-GT identity is
    promoted.  Promotion requires best IoU > 0.25; every other controlled
    candidate remains present at the score tail.  ``source_separate`` chooses
    one representative independently per source, while ``joint`` resolves the
    native/track competition jointly.
    """
    if variant not in SHARED_ORACLE_VARIANTS:
        raise ValueError(f"unknown shared oracle variant: {variant}")
    native = native_controlled & set(base_scores)
    track = track_controlled & set(base_scores)
    if variant == "component_native_only":
        pools = [native]
        controlled = native
    elif variant == "component_track_only":
        pools = [track]
        controlled = track
    elif variant == "component_source_separate":
        pools = [native, track]
        controlled = native | track
    else:
        pools = [native | track]
        controlled = native | track
    selected = set()
    for pool in pools:
        by_target: dict[int, list[tuple[str, int]]] = {}
        for key in pool:
            target_id = int(target_ids[key])
            if target_id < 0 or float(qualities[key]) <= 0.25:
                continue
            by_target.setdefault(target_id, []).append(key)
        for keys in by_target.values():
            selected.add(max(
                keys,
                key=lambda key: (
                    float(qualities[key]),
                    float(base_scores[key]),
                    -key[1],
                    key[0],
                ),
            ))
    result = dict(base_scores)
    for key in controlled:
        quality = float(qualities[key])
        result[key] = (2.0 if key in selected else -2.0) + quality
    return result, selected


def _candidate_components(cache: dict) -> list[dict[str, list[tuple[str, int]]]]:
    result = []
    seen = set()
    for component in cache["components"]:
        native = [
            ("native", int(cache["groups"][str(group_id)]["representative_candidate_id"]))
            for group_id in component["native_exact_geometry_group_ids"]
        ]
        track = [("track", int(track_id)) for track_id in component["track_ids"]]
        keys = native + track
        if seen & set(keys):
            raise ValueError("relation components are not candidate-disjoint")
        seen.update(keys)
        result.append({"native": native, "track": track})
    return result


def _label_maps(args, scene: str, cache: dict) -> tuple[dict, dict]:
    rows = read_jsonl(candidate_ledger_path(args.records_root, scene))
    by_key = {
        (
            "native" if row["candidate_source"] == NATIVE_SOURCE else "track",
            int(row["candidate_id"]),
        ): (
            float(row["label_best_gt_iou"]),
            (
                -1
                if row["label_best_gt_instance_id"] is None
                else int(row["label_best_gt_instance_id"])
            ),
        )
        for row in rows
    }
    qualities = {}
    target_ids = {}
    for key in cache["uuid_by_candidate"]:
        if key not in by_key:
            raise ValueError(f"{scene}: missing GT quality for {key}")
        value, target_id = by_key[key]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{scene}: invalid GT quality for {key}: {value}")
        qualities[key] = value
        target_ids[key] = target_id
    return qualities, target_ids


def _base_score_map(cache: dict) -> dict[tuple[str, int], float]:
    uuid_to_key = {uuid: key for key, uuid in cache["uuid_by_candidate"].items()}
    result = {}
    for row in cache["pred"]["chair"]:
        key = uuid_to_key[row["uuid"]]
        result[key] = float(row["confidence"])
    if set(result) != set(cache["uuid_by_candidate"]):
        raise ValueError("candidate UUID score map is incomplete")
    return result


def _valid_gt_and_edges(cache: dict, threshold: float) -> tuple[int, dict]:
    valid_gt = []
    for gt in cache["gt"]["chair"]:
        if (
            int(gt["instance_id"]) >= 1000
            and int(gt["vert_count"]) >= MIN_REGION_SIZE
            and float(gt["med_dist"]) <= float(instance_eval.opt["distance_threshes"][0])
            and float(gt["dist_conf"]) >= float(instance_eval.opt["distance_confs"][0])
        ):
            valid_gt.append(gt)
    uuid_to_key = {uuid: key for key, uuid in cache["uuid_by_candidate"].items()}
    edges = {key: [] for key in cache["uuid_by_candidate"]}
    for gt_index, gt in enumerate(valid_gt):
        for pred in gt["matched_pred"]:
            key = uuid_to_key.get(pred["uuid"])
            if key is None:
                continue
            iou = float(pred["intersection"]) / (
                int(gt["vert_count"]) + int(pred["vert_count"]) - int(pred["intersection"])
            )
            if iou > threshold:
                edges[key].append((gt_index, iou))
    return len(valid_gt), edges


def _set_cache_scores(cache: dict, score_map: dict[tuple[str, int], float]) -> None:
    uuid_scores = {
        cache["uuid_by_candidate"][key]: float(score)
        for key, score in score_map.items()
    }
    _set_match_scores({"scene": {"gt": cache["gt"], "pred": cache["pred"]}}, uuid_scores)


def _matching_count(edges: dict, keys: set[tuple[str, int]]) -> int:
    return len(maximum_matching_selection({key: edges.get(key, []) for key in keys}))


def _metric_triplet(metrics: dict) -> dict[str, float]:
    return {
        "ap": float(metrics["official_ap"]),
        "ap50": float(metrics["threshold_metrics"]["50"]["ap"]),
        "ap25": float(metrics["threshold_metrics"]["25"]["ap"]),
    }


def _delta(actual: dict, baseline: dict) -> dict[str, float]:
    return {key: float(actual[key] - baseline[key]) for key in baseline}


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("this diagnostic requires official100")
    configure_track_score_context(args, scenes)
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original_load_ids(path))

    baseline_records = {}
    permutation_records = {name: {} for name in PERMUTATION_VARIANTS}
    shared_oracle_records = {name: {} for name in SHARED_ORACLE_VARIANTS}
    threshold_records = {name: {} for name in THRESHOLD_ORACLE_VARIANTS}
    coverage_totals = {
        name: {str(int(round(t * 100))): 0 for t in THRESHOLDS}
        for name in (
            "all_native", "all_track", "all_combined",
            "component_native", "component_track", "component_combined",
        )
    }
    gt_totals = {str(int(round(t * 100))): 0 for t in THRESHOLDS}
    selected_totals = {
        name: Counter() for name in THRESHOLD_ORACLE_VARIANTS
    }
    shared_selected_totals = Counter()
    candidate_totals = Counter()
    component_count = 0
    try:
        for scene_index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            base = _base_score_map(cache)
            quality, target_ids = _label_maps(args, scene, cache)
            components = _candidate_components(cache)
            component_count += len(components)
            native_all = {key for key in base if key[0] == "native"}
            track_all = {key for key in base if key[0] == "track"}
            native_controlled = {
                key for component in components for key in component["native"] if key in base
            }
            track_controlled = {
                key for component in components for key in component["track"] if key in base
            }
            candidate_totals.update({
                "native": len(native_all),
                "track": len(track_all),
                "component_native": len(native_controlled),
                "component_track": len(track_controlled),
            })

            _set_cache_scores(cache, base)
            baseline_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            for variant in PERMUTATION_VARIANTS:
                scores = component_permutation_scores(base, quality, components, variant)
                _set_cache_scores(cache, scores)
                permutation_records[variant][scene] = _scene_records(
                    cache, cache["all_native_representatives"], cache["all_track_ids"]
                )
            for variant in SHARED_ORACLE_VARIANTS:
                scores, selected = shared_gt_quality_scores(
                    base, quality, target_ids, native_controlled, track_controlled, variant
                )
                shared_selected_totals[variant] += len(selected)
                _set_cache_scores(cache, scores)
                shared_oracle_records[variant][scene] = _scene_records(
                    cache, cache["all_native_representatives"], cache["all_track_ids"]
                )

            threshold_records_scene = {name: {} for name in THRESHOLD_ORACLE_VARIANTS}
            for threshold in THRESHOLDS:
                tag = str(int(round(threshold * 100)))
                gt_count, edges = _valid_gt_and_edges(cache, threshold)
                gt_totals[tag] += gt_count
                subsets = {
                    "all_native": native_all,
                    "all_track": track_all,
                    "all_combined": native_all | track_all,
                    "component_native": native_controlled,
                    "component_track": track_controlled,
                    "component_combined": native_controlled | track_controlled,
                }
                for name, keys in subsets.items():
                    coverage_totals[name][tag] += _matching_count(edges, keys)
                for variant in THRESHOLD_ORACLE_VARIANTS:
                    scores, selected = threshold_oracle_scores(
                        base, quality, edges, native_controlled, track_controlled, variant
                    )
                    selected_totals[variant][tag] += len(selected)
                    _set_cache_scores(cache, scores)
                    records = _scene_records(
                        cache, cache["all_native_representatives"], cache["all_track_ids"]
                    )
                    threshold_records_scene[variant][tag] = records[tag]
            for variant in THRESHOLD_ORACLE_VARIANTS:
                threshold_records[variant][scene] = threshold_records_scene[variant]
            print(f"[official100 reranking oracle] {scene_index}/100 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    baseline_metrics = _metrics_from_records(baseline_records)
    baseline_triplet = _metric_triplet(baseline_metrics)
    permutation_metrics = {
        name: _metrics_from_records(records) for name, records in permutation_records.items()
    }
    shared_oracle_metrics = {
        name: _metrics_from_records(records) for name, records in shared_oracle_records.items()
    }
    threshold_metrics = {
        name: _metrics_from_records(records) for name, records in threshold_records.items()
    }
    coverage = {}
    for name, counts in coverage_totals.items():
        values = {tag: counts[tag] / max(1, gt_totals[tag]) for tag in counts}
        coverage[name] = {
            "ap": float(np.mean([values[str(int(t * 100))] for t in OFFICIAL_THRESHOLDS])),
            "ap50": float(values["50"]),
            "ap25": float(values["25"]),
            "threshold_recall_ceiling": values,
            "match_counts": counts,
        }

    permutation_triplets = {
        name: _metric_triplet(metrics) for name, metrics in permutation_metrics.items()
    }
    threshold_triplets = {
        name: _metric_triplet(metrics) for name, metrics in threshold_metrics.items()
    }
    shared_oracle_triplets = {
        name: _metric_triplet(metrics) for name, metrics in shared_oracle_metrics.items()
    }
    result = {
        "version": VERSION,
        "diagnostic_type": "official100 current-candidate GT-only component reranking oracle audit",
        "scene_count": len(scenes),
        "relation_component_count": component_count,
        "candidate_counts": dict(sorted(candidate_totals.items())),
        "ground_truth_counts": gt_totals,
        "candidate_set_contract": "native exact-geometry representatives plus filtered D2b tracks; track score context is frozen OOF quality",
        "candidate_count_modified": False,
        "candidate_geometry_modified": False,
        "candidate_class_modified": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "baseline": baseline_triplet,
        "fixed_mask_threshold_specific_coverage_ceilings": coverage,
        "component_score_multiset_permutation": {
            "definition": "shared-score realizable control; preserve each component/source score multiset and assign larger existing levels to larger best-GT-IoU candidates",
            "metrics": permutation_triplets,
            "delta_vs_baseline": {
                name: _delta(values, baseline_triplet)
                for name, values in permutation_triplets.items()
            },
            "cross_source_joint_increment_vs_source_separate": _delta(
                permutation_triplets["component_joint"],
                permutation_triplets["component_source_separate"],
            ),
        },
        "component_shared_gt_quality_score_oracle": {
            "definition": "one shared score for AP/AP50/AP25; per controlled pool promote only the highest-best-IoU candidate for each best-GT identity when best IoU > 0.25, while all other controlled candidates remain at the tail",
            "deployable": False,
            "shared_score_across_thresholds": True,
            "metrics": shared_oracle_triplets,
            "delta_vs_baseline": {
                name: _delta(values, baseline_triplet)
                for name, values in shared_oracle_triplets.items()
            },
            "cross_source_joint_increment_vs_source_separate": _delta(
                shared_oracle_triplets["component_joint"],
                shared_oracle_triplets["component_source_separate"],
            ),
            "selected_candidate_counts": dict(sorted(shared_selected_totals.items())),
        },
        "component_threshold_specific_score_oracle": {
            "definition": "different GT-only score control per IoU threshold; maximum-matched controlled candidates score above frozen candidates and all remaining controlled candidates stay present at the tail",
            "deployable_shared_score": False,
            "metrics": threshold_triplets,
            "delta_vs_baseline": {
                name: _delta(values, baseline_triplet)
                for name, values in threshold_triplets.items()
            },
            "cross_source_joint_increment_vs_source_separate": _delta(
                threshold_triplets["component_joint"],
                threshold_triplets["component_source_separate"],
            ),
            "selected_candidate_counts_by_threshold": {
                name: dict(sorted(counts.items())) for name, counts in selected_totals.items()
            },
        },
        "input_provenance": {
            "scene_list": str(args.scene_list),
            "scene_list_sha256": _sha256(args.scene_list),
            "relation_feature_ledger_root": str(args.relation_feature_ledger_root),
            "relation_feature_ledger_summary_sha256": _sha256(
                args.relation_feature_ledger_root / "summary.json"
            ),
            "oof_predictions": str(args.oof_predictions),
            "oof_predictions_sha256": _sha256(args.oof_predictions),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "records_root", "relation_feature_ledger_root",
        "gt_dir", "oof_predictions", "output",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    args.track_score_mode = "oof_quality"
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
