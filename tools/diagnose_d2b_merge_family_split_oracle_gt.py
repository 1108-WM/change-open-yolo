#!/usr/bin/env python3
"""GT-only feasible-oracle diagnostics for frozen D2b merge-family splits.

The D2b history and its pre-merge tracks are immutable.  This program first
replays every frozen merge with the original consensus function and verifies
that every final ``keep`` geometry equals the published D2b geometry.  It then
measures the pre-registered keep/undo-last/cut-edge/restore-all partitions
against GT.  GT remains offline-only: no proposal, score, class, selector, or
inference decision is written.

The final system result is deliberately called a *GT-greedy feasible geometry
ceiling*.  One action is chosen for each merge family, all selected partitions
are geometrically real, and the reported AP-like value is the usual
threshold-specific ideal-ranking ceiling (maximum one-to-one matching).  It is
not an exact combinatorial AP maximization and cannot be deployed as a rule.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_details_iterative_proposal_merges import (  # noqa: E402
    CONSENSUS_PARAMS,
    _merge_proposals,
    initialize_proposals,
)
from tools.refine_details_automatic_tracks_consensus import _load_observations  # noqa: E402
from tools.diagnose_n1_sampro3d_candidate_space_oracle_gt import (  # noqa: E402
    _geometry_for_superpoints,
    _ious,
    _load_gt,
    _scenes,
    _sp_gt_counts,
)
from tools.diagnose_n2_medoid_candidate_oracle_gt import (  # noqa: E402
    OFFICIAL_THRESHOLDS,
    maximum_matching,
)


CONTRACT = (
    "GT-only D2b merge-family split oracle. GT selects only offline diagnostic "
    "actions and ideal matching; it must not become an inference action, score, "
    "threshold, selector, proposal, or class."
)
STRICT_MUTUAL_COVERAGE = 0.99
EPS = 1e-12
CACHE_MODES = ("mask3d_yoloworld_only", "strong_native")


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _lineage_key(values) -> tuple[int, ...]:
    return tuple(sorted(map(int, values)))


def _validate_expected_cache_mode(cache_root: Path, expected_mode: str) -> dict:
    """Reject cache/source combinations that would invalidate a baseline audit."""
    manifest_path = cache_root / "native_cache_no_gt_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"native cache manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    inputs = manifest.get("candidate_inputs", {})
    explicit_mode = inputs.get("mode")
    mask3d_only = bool(manifest.get("params", {}).get("mask3d_yoloworld_only"))
    if expected_mode == "mask3d_yoloworld_only":
        if explicit_mode != expected_mode or not mask3d_only:
            raise ValueError(
                "native cache mode mismatch: expected manifest candidate_inputs.mode="
                "mask3d_yoloworld_only and params.mask3d_yoloworld_only=true"
            )
    elif expected_mode == "strong_native":
        # The historical strong cache predates an explicit mode field.  Its
        # manifest instead records loaded SAM-fused/BPR candidate inputs.
        if explicit_mode is not None or mask3d_only or int(inputs.get("loaded", 0)) <= 0:
            raise ValueError(
                "native cache mode mismatch: expected historical strong_native "
                "manifest with non-empty fused/BPR candidate inputs"
            )
    else:
        raise ValueError(f"unknown expected cache mode: {expected_mode}")
    return {
        "expected_cache_mode": expected_mode,
        "manifest_path": str(manifest_path),
        "manifest_candidate_inputs": inputs,
    }


def _load_scene_inputs(scene: str, args):
    """Load raw inputs and replay the frozen merge sequence exactly."""
    from utils import WORLD_2_CAM

    processed = np.load(
        args.processed_scene_root / scene / f"{scene.replace('scene', '')}.npy",
        mmap_mode="r",
    )
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene}: missing raw superpoint column")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {int(item): int(count) for item, count in zip(ids, counts)}
    source_tracks = json.loads(
        (args.premerge_track_root / scene / "automatic_tracks.json").read_text()
    )["tracks"]
    frozen_actions = _read_jsonl(args.d2b_merge_root / scene / "merge_actions.jsonl")
    world = WORLD_2_CAM(str(args.dataset_root / scene), args.depth_scale, args.config)
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    observations = _load_observations(args.d1_observation_root / scene, superpoints, visibility)

    active = {int(row["proposal_id"]): row for row in initialize_proposals(source_tracks, superpoint_sizes)}
    nodes = {f"leaf_{proposal_id}": dict(row) for proposal_id, row in active.items()}
    for frozen in sorted(frozen_actions, key=lambda row: int(row["action_index"])):
        anchor_id, absorbed_id = int(frozen["anchor_proposal_id"]), int(frozen["absorbed_proposal_id"])
        if anchor_id not in active or absorbed_id not in active:
            raise ValueError(f"{scene}: replay cannot resolve D2b action {frozen['action_index']}")
        anchor, absorbed = active[anchor_id], active[absorbed_id]
        if _lineage_key(anchor["lineage_proposal_ids"]) != _lineage_key(frozen["anchor_lineage_before"]):
            raise ValueError(f"{scene}: frozen anchor lineage mismatch at action {frozen['action_index']}")
        if _lineage_key(absorbed["lineage_proposal_ids"]) != _lineage_key(frozen["absorbed_lineage"]):
            raise ValueError(f"{scene}: frozen absorbed lineage mismatch at action {frozen['action_index']}")
        refined, audit = _merge_proposals(
            anchor, absorbed, observations, superpoint_sizes, int(frozen["round_index"])
        )
        expected = {
            "anchor_lineage_after": list(refined["lineage_proposal_ids"]),
            "refined_superpoint_count": len(refined["superpoint_ids"]),
            "refined_point_count": int(refined["point_count"]),
            "union_superpoint_count": audit["union_superpoint_count"],
            "removed_by_refinement_count": audit["removed_by_refinement_count"],
            "merged_observation_count": audit["merged_observation_count"],
            "merged_frame_count": audit["merged_frame_count"],
        }
        for key, value in expected.items():
            if frozen.get(key) != value:
                raise ValueError(f"{scene}: frozen replay differs at action {frozen['action_index']} field {key}")
        active[anchor_id] = refined
        del active[absorbed_id]
        nodes[f"merge_{int(frozen['action_index'])}"] = dict(refined)

    published = json.loads((args.d2b_merge_root / scene / "automatic_tracks.json").read_text())["tracks"]
    replayed_by_lineage = {_lineage_key(row["lineage_proposal_ids"]): row for row in active.values()}
    published_by_lineage = {_lineage_key(row["lineage_proposal_ids"]): row for row in published}
    if set(replayed_by_lineage) != set(published_by_lineage):
        raise ValueError(f"{scene}: final D2b lineage differs after exact replay")
    for lineage, replayed in replayed_by_lineage.items():
        target = published_by_lineage[lineage]
        if sorted(map(int, replayed["superpoint_ids"])) != sorted(map(int, target["superpoint_ids"])):
            raise ValueError(f"{scene}: keep geometry differs for lineage {lineage}")
        if int(replayed["point_count"]) != int(target["point_count"]):
            raise ValueError(f"{scene}: keep point count differs for lineage {lineage}")
    return processed, superpoints, superpoint_sizes, nodes, published_by_lineage


def _native_masks(scene: str, args, point_count: int):
    masks = np.load(args.native_prediction_cache / f"{scene}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene}: native masks must be 2-D")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene}: native point dimension differs")
    return masks


def _strict_native_duplicate(superpoint_ids, superpoints, native_masks):
    """Apply the frozen strict track--native mutual-duplicate contract anew."""
    points = np.flatnonzero(np.isin(superpoints, np.asarray(sorted(superpoint_ids), dtype=np.int64)))
    if not len(points):
        return None
    point_count = len(points)
    native_sizes = np.asarray(native_masks.sum(axis=0, dtype=np.int64)).reshape(-1)
    # Bidirectional coverage >0.99 implies almost identical cardinality.  This
    # necessary no-GT prefilter avoids intersecting every node with every
    # native mask, while preserving the exact strict comparison below.
    candidate_ids = np.flatnonzero(
        (native_sizes > STRICT_MUTUAL_COVERAGE * point_count)
        & (native_sizes < point_count / STRICT_MUTUAL_COVERAGE)
    )
    if not len(candidate_ids):
        return None
    intersections = np.asarray(
        native_masks[np.ix_(points, candidate_ids)].sum(axis=0, dtype=np.int64)
    ).reshape(-1)
    track_inside = intersections / max(1, point_count)
    native_inside = intersections / np.maximum(1, native_sizes[candidate_ids])
    eligible = np.flatnonzero(
        (track_inside > STRICT_MUTUAL_COVERAGE)
        & (native_inside > STRICT_MUTUAL_COVERAGE)
    )
    if not len(eligible):
        return None
    scores = intersections[eligible] / np.maximum(
        1, point_count + native_sizes[candidate_ids[eligible]] - intersections[eligible]
    )
    selected = int(eligible[np.argmax(scores)])
    candidate = int(candidate_ids[selected])
    return {
        "native_candidate_id": candidate,
        "point_iou": float(intersections[selected] / max(1, point_count + native_sizes[candidate] - intersections[selected])),
        "track_inside_native_ratio": float(track_inside[selected]),
        "native_inside_track_ratio": float(native_inside[selected]),
    }


def _node_geometry(
    node_id, nodes, sp_sizes, sp_gt, gt_sizes, superpoints, native_masks,
    evaluate_native_duplicate=False,
):
    node = nodes[node_id]
    ids, size, intersections = _geometry_for_superpoints(node["superpoint_ids"], sp_sizes, sp_gt)
    iou_by_gt = _ious(size, intersections, gt_sizes)
    best_gt = min(iou_by_gt, key=lambda gt: (-iou_by_gt[gt], gt)) if iou_by_gt else -1
    # Native duplicate filtering is expensive but only affects the one
    # globally selected, mutually exclusive partition per family.  Deferring
    # it for unselected counterfactual actions does not alter their frozen
    # geometry or GT labels; the final feasible system below evaluates it
    # exactly before any matching ceiling is reported.
    duplicate = (
        _strict_native_duplicate(ids, superpoints, native_masks)
        if evaluate_native_duplicate else None
    )
    return {
        "node_id": node_id,
        "lineage_proposal_ids": list(map(int, node["lineage_proposal_ids"])),
        "superpoint_ids": ids,
        "point_count": int(size),
        "best_gt_instance_id": int(best_gt),
        "best_gt_iou": float(iou_by_gt.get(best_gt, 0.0)),
        "iou_by_gt": {int(key): float(value) for key, value in iou_by_gt.items()},
        "strict_native_duplicate": duplicate is not None,
        "native_duplicate_witness": duplicate,
        "native_duplicate_evaluation": (
            "evaluated_exact_strict_099" if evaluate_native_duplicate
            else "deferred_until_globally_selected_feasible_partition"
        ),
    }


def _best_by_gt(rows, gt_sizes):
    result = {int(gt): 0.0 for gt in gt_sizes}
    for row in rows:
        for gt, iou in row["iou_by_gt"].items():
            result[int(gt)] = max(result[int(gt)], float(iou))
    return result


def _action_metrics(partition, gt_sizes):
    retained = [row for row in partition if not row["strict_native_duplicate"]]
    best = _best_by_gt(retained, gt_sizes)
    result = {
        "partition_node_count": len(partition),
        "partition_candidate_count_after_native_duplicate_filter": len(retained),
        "filtered_strict_native_duplicate_count": len(partition) - len(retained),
        "best_iou_by_gt": best,
    }
    for threshold in (0.25, 0.50):
        tag = str(int(threshold * 100))
        covered = [gt for gt, iou in best.items() if iou >= threshold]
        by_candidate = defaultdict(int)
        for gt in covered:
            matching = [row for row in retained if row["iou_by_gt"].get(gt, 0.0) >= threshold]
            if matching:
                winner = min(matching, key=lambda row: (-row["iou_by_gt"][gt], row["node_id"]))
                by_candidate[winner["node_id"]] += 1
        result[f"covered_gt_count_iou{tag}"] = len(covered)
        result[f"fragment_multi_gt_candidate_count_iou{tag}"] = sum(value > 1 for value in by_candidate.values())
        result[f"duplicate_same_gt_candidate_excess_iou{tag}"] = sum(
            max(0, sum(row["iou_by_gt"].get(gt, 0.0) >= threshold for row in retained) - 1)
            for gt in covered
        )
    return result


def _edges(prefix, rows):
    return {f"{prefix}{row['node_id']}": row["iou_by_gt"] for row in rows}


def _choose_feasible_actions(families, static_rows, gt_sizes):
    """GT-greedy, but one already-real partition per tree: always feasible."""
    current = _best_by_gt(static_rows, gt_sizes)
    selected = {}
    for family_id in sorted(families):
        choices = families[family_id]
        best_key, best_rows, best_utility = None, None, None
        for action_key, rows in choices.items():
            candidate = dict(current)
            for gt, value in _best_by_gt(rows, gt_sizes).items():
                candidate[gt] = max(candidate[gt], value)
            utility = sum(candidate.values())
            ranking = (utility, -len(rows), action_key)
            if best_utility is None or ranking > best_utility:
                best_key, best_rows, best_utility = action_key, rows, ranking
        selected[family_id] = (best_key, best_rows)
        for gt, value in _best_by_gt(best_rows, gt_sizes).items():
            current[gt] = max(current[gt], value)
    return selected


def _cross_tree_conflicts(selected):
    rows = [(family, row) for family, (_, partition) in selected.items() for row in partition]
    nonzero = strict = 0
    for left_index, (left_family, left) in enumerate(rows):
        left_ids = set(left["superpoint_ids"])
        for right_family, right in rows[left_index + 1:]:
            if left_family == right_family:
                continue
            shared = len(left_ids & set(right["superpoint_ids"]))
            if not shared:
                continue
            nonzero += 1
            left_cover = shared / max(1, len(left_ids))
            right_cover = shared / max(1, len(right["superpoint_ids"]))
            strict += int(left_cover > STRICT_MUTUAL_COVERAGE and right_cover > STRICT_MUTUAL_COVERAGE)
    return {"cross_tree_nonzero_superpoint_overlap_pair_count": nonzero, "cross_tree_strict_mutual_duplicate_pair_count": strict}


def _scene_oracle(scene: str, args):
    if args.verbose_phase:
        print(f"[phase] {scene} replay", flush=True)
    processed, superpoints, sp_sizes, nodes, replayed_final = _load_scene_inputs(scene, args)
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene}.txt", args.min_region_size)
    if len(gt_ids) != len(superpoints):
        raise ValueError(f"{scene}: GT and processed points differ")
    sp_gt = _sp_gt_counts(superpoints, gt_ids, gt_sizes)
    native_masks = _native_masks(scene, args, len(superpoints))
    trees = _read_jsonl(args.action_ledger_root / scene / "merge_family_tree_ledger.jsonl")
    actions = _read_jsonl(args.action_ledger_root / scene / "split_action_ledger.jsonl")
    tree_ids = {row["merge_family_id"] for row in trees}
    if any(row["merge_family_id"] not in tree_ids for row in actions):
        raise ValueError(f"{scene}: split action references unknown tree")

    required_nodes = {node for action in actions for node in action["partition_node_ids"]}
    if args.verbose_phase:
        print(f"[phase] {scene} node_geometry={len(required_nodes)}", flush=True)
    cache = {
        node: _node_geometry(node, nodes, sp_sizes, sp_gt, gt_sizes, superpoints, native_masks)
        for node in sorted(required_nodes)
    }
    keep_checks = []
    for tree in trees:
        root_id = tree["final_d2b_node"]["node_id"]
        root = nodes[root_id]
        target = replayed_final[_lineage_key(tree["final_lineage_proposal_ids"])]
        same = sorted(root["superpoint_ids"]) == sorted(target["superpoint_ids"])
        keep_checks.append(bool(same))
        if not same:
            raise ValueError(f"{scene}: tree keep geometry does not equal frozen D2b")

    action_rows, family_choices = [], defaultdict(dict)
    if args.verbose_phase:
        print(f"[phase] {scene} local_actions={len(actions)}", flush=True)
    for action in actions:
        partition = [cache[node] for node in action["partition_node_ids"]]
        metrics = _action_metrics(partition, gt_sizes)
        family_id = action["merge_family_id"]
        action_key = f"{action['split_action_kind']}:{action['cut_merge_action_index']}"
        family_choices[family_id][action_key] = [row for row in partition if not row["strict_native_duplicate"]]
        keep_action = next(item for item in actions if item["merge_family_id"] == family_id and item["split_action_kind"] == "keep")
        keep_best = _action_metrics([cache[node] for node in keep_action["partition_node_ids"]], gt_sizes)["best_iou_by_gt"]
        crossing = {}
        for threshold in (0.25, 0.50):
            tag = str(int(threshold * 100))
            after = metrics["best_iou_by_gt"]
            crossing[f"up_crossing_count_iou{tag}"] = sum(after[gt] >= threshold > keep_best[gt] for gt in gt_sizes)
            crossing[f"down_crossing_count_iou{tag}"] = sum(keep_best[gt] >= threshold > after[gt] for gt in gt_sizes)
        action_rows.append({
            "scene_name": scene, "merge_family_id": family_id,
            "split_action_kind": action["split_action_kind"], "cut_merge_action_index": action["cut_merge_action_index"],
            "partition_node_ids": action["partition_node_ids"],
            "partition_lineages": action["partition_lineages"],
            "partition_nodes": partition, **metrics, **crossing,
            "recovered_overmerge_evidence": bool(crossing["up_crossing_count_iou25"] or crossing["up_crossing_count_iou50"]),
            "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False, "ap_computed": False,
        })

    family_roots = {tuple(tree["final_lineage_proposal_ids"]) for tree in trees}
    filtered_tracks = json.loads((args.filtered_d2b_track_root / scene / "automatic_tracks.json").read_text())["tracks"]
    static_rows = []
    for track in filtered_tracks:
        lineage = tuple(sorted(map(int, track["lineage_proposal_ids"])))
        if lineage in family_roots:
            continue
        node = {"lineage_proposal_ids": lineage, "superpoint_ids": track["superpoint_ids"]}
        temporary_id = "static_" + "_".join(map(str, lineage))
        nodes[temporary_id] = node
        static_rows.append(_node_geometry(temporary_id, nodes, sp_sizes, sp_gt, gt_sizes, superpoints, native_masks))
    native_edges = {}
    native_sizes = np.asarray(native_masks.sum(axis=0, dtype=np.int64)).reshape(-1)
    for gt, gt_size in gt_sizes.items():
        intersections = np.asarray(native_masks[gt_ids == gt].sum(axis=0, dtype=np.int64)).reshape(-1)
        ious = intersections / np.maximum(1, native_sizes + gt_size - intersections)
        for index in np.flatnonzero(ious > 0):
            native_edges.setdefault(f"native_{int(index)}", {})[int(gt)] = float(ious[index])
    static_edges = {**native_edges, **_edges("track_", static_rows)}
    selected = _choose_feasible_actions(family_choices, static_rows, gt_sizes)
    if args.verbose_phase:
        print(f"[phase] {scene} selected_native_duplicate={sum(len(rows) for _, rows in selected.values())}", flush=True)
    selected_rows = [dict(row) for _, partition in selected.values() for row in partition]
    for row in selected_rows:
        duplicate = _strict_native_duplicate(
            row["superpoint_ids"], superpoints, native_masks
        )
        row["strict_native_duplicate"] = duplicate is not None
        row["native_duplicate_witness"] = duplicate
        row["native_duplicate_evaluation"] = "evaluated_exact_strict_099"
    selected_rows = [row for row in selected_rows if not row["strict_native_duplicate"]]
    all_edges = {**static_edges, **_edges("split_", selected_rows)}
    if args.verbose_phase:
        print(f"[phase] {scene} baseline_edges", flush=True)
    baseline_edges = {**native_edges, **_edges("track_", [
        _node_geometry("baseline_" + "_".join(map(str, row["lineage_proposal_ids"])),
                       {**nodes, "baseline_" + "_".join(map(str, row["lineage_proposal_ids"])): {"lineage_proposal_ids": row["lineage_proposal_ids"], "superpoint_ids": row["superpoint_ids"]}},
                       sp_sizes, sp_gt, gt_sizes, superpoints, native_masks)
        for row in filtered_tracks
    ])}
    threshold_metrics = {}
    if args.verbose_phase:
        print(f"[phase] {scene} matching", flush=True)
    for threshold in OFFICIAL_THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        base_count, _ = maximum_matching(baseline_edges, threshold)
        selected_count, _ = maximum_matching(all_edges, threshold)
        threshold_metrics[tag] = {
            "threshold": threshold, "valid_gt_instance_count": len(gt_sizes),
            "frozen_base_d2b_maximum_matching": base_count,
            "feasible_split_system_maximum_matching": selected_count,
            "feasible_increment_vs_frozen_base_d2b": selected_count - base_count,
            "threshold_specific_ideal_ranking_ceiling": selected_count / max(1, len(gt_sizes)),
        }
    selected_by_family = {family: key for family, (key, _) in selected.items()}
    for row in action_rows:
        key = f"{row['split_action_kind']}:{row['cut_merge_action_index']}"
        row["selected_by_gt_greedy_feasible_combination"] = selected_by_family[row["merge_family_id"]] == key
    gt_rows = []
    baseline_best = _best_by_gt([row for row in static_rows] + [
        {"iou_by_gt": edges} for edges in native_edges.values()
    ], gt_sizes)
    selected_best = _best_by_gt(selected_rows + static_rows + [{"iou_by_gt": edges} for edges in native_edges.values()], gt_sizes)
    for gt in sorted(gt_sizes):
        gt_rows.append({
            "scene_name": scene, "gt_instance_id": gt, "gt_point_count": gt_sizes[gt],
            "best_frozen_base_d2b_iou": baseline_best[gt], "best_feasible_split_system_iou": selected_best[gt],
            "new_iou25": selected_best[gt] >= .25 > baseline_best[gt],
            "new_iou50": selected_best[gt] >= .50 > baseline_best[gt],
            "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False,
        })
    summary = {
        "scene_name": scene, "valid_gt_instance_count": len(gt_sizes), "merge_family_count": len(trees),
        "split_action_count": len(actions), "keep_geometry_verified_tree_count": sum(keep_checks),
        "keep_geometry_verified": all(keep_checks), "selected_action_counts": dict(Counter(selected_by_family.values())),
        "selected_split_candidate_count": len(selected_rows), "threshold_metrics": threshold_metrics,
        **_cross_tree_conflicts(selected), "ground_truth_usage": "offline_diagnostic_only",
        "proposal_materialization_applied": False, "ap_computed": False,
    }
    return action_rows, gt_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--d2b-merge-root", type=Path, required=True)
    parser.add_argument("--premerge-track-root", type=Path, required=True)
    parser.add_argument("--d1-observation-root", type=Path, required=True)
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-mode", required=True, choices=CACHE_MODES)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true", help="reuse already atomically published scene diagnostics")
    parser.add_argument("--verbose-phase", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required; GT is offline diagnosis only")
    for name in ("scene_list", "action_ledger_root", "d2b_merge_root", "premerge_track_root", "d1_observation_root", "filtered_d2b_track_root", "native_prediction_cache", "processed_scene_root", "dataset_root", "config_path", "gt_instance_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    args.cache_mode_audit = _validate_expected_cache_mode(
        args.native_prediction_cache, args.expected_cache_mode
    )
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ordinal, scene in enumerate(scenes, 1):
        published = args.output_root / scene / "summary.json"
        if published.is_file() and args.resume:
            summary = json.loads(published.read_text())
            if not summary.get("keep_geometry_verified"):
                raise ValueError(f"{scene}: existing resume result lacks keep verification")
            summaries.append(summary)
            print(f"[resume D2b split GT oracle] {ordinal}/{len(scenes)} {scene}", flush=True)
            continue
        if (args.output_root / scene).exists():
            raise FileExistsError(f"{scene}: incomplete or unexpected existing output")
        actions, gt_rows, summary = _scene_oracle(scene, args)
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        stage.mkdir()
        _write_jsonl(stage / "split_action_oracle_gt.jsonl", actions)
        _write_jsonl(stage / "gt_instance_oracle_gt.jsonl", gt_rows)
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        destination = args.output_root / scene
        try:
            os.replace(stage, destination)
        except OSError as error:
            # A previously interrupted launcher can finish the same atomic
            # scene while this diagnostic is running.  Never overwrite it:
            # accept only a fully published, independently verified result.
            published_summary = destination / "summary.json"
            if not published_summary.is_file():
                raise error
            published = json.loads(published_summary.read_text())
            if not published.get("keep_geometry_verified"):
                raise ValueError(f"{scene}: concurrent publication lacks keep verification") from error
            shutil.rmtree(stage)
            summary = published
        summaries.append(summary)
        print(f"[D2b split GT oracle] {ordinal}/{len(scenes)} {scene}", flush=True)
    totals = {}
    for threshold in OFFICIAL_THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        keys = ("valid_gt_instance_count", "frozen_base_d2b_maximum_matching", "feasible_split_system_maximum_matching", "feasible_increment_vs_frozen_base_d2b")
        totals[tag] = {key: sum(row["threshold_metrics"][tag][key] for row in summaries) for key in keys}
        totals[tag]["threshold_specific_ideal_ranking_ceiling"] = totals[tag]["feasible_split_system_maximum_matching"] / max(1, totals[tag]["valid_gt_instance_count"])
    official = [str(int(round(t * 100))) for t in OFFICIAL_THRESHOLDS if t >= .50]
    root = {
        "diagnostic_type": "GT-only D2b merge-family split feasible geometry oracle",
        "decision_constraint": CONTRACT,
        "oracle_scope": "one real partition per tree selected by GT-greedy utility; threshold-specific maximum-matching ideal-ranking ceiling, not exact global AP maximization",
        "native_cache_mode_audit": args.cache_mode_audit,
        "scene_count": len(summaries), "threshold_totals": totals,
        "aggregate_feasible_ideal_ranking_ceiling": {
            "ap": float(np.mean([totals[tag]["threshold_specific_ideal_ranking_ceiling"] for tag in official])),
            "ap50": totals["50"]["threshold_specific_ideal_ranking_ceiling"],
            "ap25": totals["25"]["threshold_specific_ideal_ranking_ceiling"],
            "increment_ap": float(np.mean([totals[tag]["feasible_increment_vs_frozen_base_d2b"] / max(1, totals[tag]["valid_gt_instance_count"]) for tag in official])),
        },
        "keep_geometry_verified_tree_count": sum(row["keep_geometry_verified_tree_count"] for row in summaries),
        "merge_family_count": sum(row["merge_family_count"] for row in summaries),
        "split_action_count": sum(row["split_action_count"] for row in summaries),
        "proposal_materialization_applied": False, "ap_computed": False,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
