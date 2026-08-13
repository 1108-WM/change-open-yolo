#!/usr/bin/env python3
"""GT-only upper-bound diagnostics for the frozen N2 medoid candidate cache.

The input masks, their no-GT competition components, and all baseline masks
are immutable.  GT is used only to measure fixed geometry and ideal matching;
none of the selected rows is written back to inference.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from diagnose_n1_sampro3d_candidate_space_oracle_gt import (
    _geometry_for_superpoints, _ious, _load_gt, _resolve, _scenes,
    _sp_gt_counts,
)

CONTRACT = (
    "GT-only frozen-N2 oracle: GT labels fixed masks and computes ideal matching "
    "only; it must not alter candidate geometry, components, scores, or inference."
)
OFFICIAL_THRESHOLDS = (0.25, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


class Dinic:
    def __init__(self, node_count):
        self.graph = [[] for _ in range(node_count)]

    def add(self, left, right, capacity):
        forward = [right, len(self.graph[right]), int(capacity)]
        backward = [left, len(self.graph[left]), 0]
        self.graph[left].append(forward)
        self.graph[right].append(backward)
        return forward

    def flow(self, source, sink):
        total = 0
        while True:
            level = [-1] * len(self.graph)
            queue = [source]
            level[source] = 0
            for node in queue:
                for target, _, capacity in self.graph[node]:
                    if capacity and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return total
            work = [0] * len(self.graph)

            def push(node, amount):
                if node == sink:
                    return amount
                while work[node] < len(self.graph[node]):
                    edge = self.graph[node][work[node]]
                    target, reverse, capacity = edge
                    if capacity and level[target] == level[node] + 1:
                        result = push(target, min(amount, capacity))
                        if result:
                            edge[2] -= result
                            self.graph[target][reverse][2] += result
                            return result
                    work[node] += 1
                return 0

            while True:
                pushed = push(source, 10 ** 9)
                if not pushed:
                    break
                total += pushed


def track_edges(track_root, scene, gt_ids, gt_sizes):
    rows = json.loads((track_root / scene / "automatic_tracks.json").read_text())["tracks"]
    result = {}
    for row in rows:
        points = np.unique(np.asarray(np.load(row["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < len(gt_ids))]
        if not len(points):
            continue
        values, counts = np.unique(gt_ids[points], return_counts=True)
        iou = {
            int(gt): float(inter / max(1, len(points) + gt_sizes[int(gt)] - inter))
            for gt, inter in zip(values, counts) if int(gt) in gt_sizes
        }
        result[f"d2b_{int(row['track_id'])}"] = iou
    return result


def native_edges(native_root, scene, gt_ids, gt_sizes):
    masks = np.load(native_root / f"{scene}_pred_masks.npy", mmap_mode="r")
    if masks.shape[0] != len(gt_ids):
        masks = masks.T
    if masks.shape[0] != len(gt_ids):
        raise ValueError(f"{scene}: native point dimension mismatches GT")
    sizes = masks.sum(axis=0, dtype=np.int64)
    result = {f"native_{index}": {} for index in range(masks.shape[1])}
    for gt, gt_size in gt_sizes.items():
        inter = np.asarray(masks[gt_ids == gt].sum(axis=0, dtype=np.int64)).reshape(-1)
        iou = inter / np.maximum(1, sizes + gt_size - inter)
        for index in np.flatnonzero(iou > 0):
            result[f"native_{int(index)}"][int(gt)] = float(iou[index])
    return result


def maximum_matching(prediction_edges, threshold, groups=None):
    """Maximum cardinality GT matching, optionally with one prediction/group."""
    prediction_edges = {
        key: {gt: value for gt, value in edges.items() if value >= threshold}
        for key, edges in prediction_edges.items()
    }
    prediction_edges = {key: edges for key, edges in prediction_edges.items() if edges}
    gt_ids = sorted({gt for edges in prediction_edges.values() for gt in edges})
    keys = sorted(prediction_edges)
    group_keys = sorted(set(groups[key] for key in keys)) if groups else []
    source, sink = 0, 1
    offset = 2
    group_nodes = {key: offset + index for index, key in enumerate(group_keys)}
    offset += len(group_nodes)
    prediction_nodes = {key: offset + index for index, key in enumerate(keys)}
    offset += len(prediction_nodes)
    gt_nodes = {gt: offset + index for index, gt in enumerate(gt_ids)}
    graph = Dinic(offset + len(gt_nodes))
    if groups:
        for group, node in group_nodes.items():
            graph.add(source, node, 1)
        for key, node in prediction_nodes.items():
            graph.add(group_nodes[groups[key]], node, 1)
    else:
        for node in prediction_nodes.values():
            graph.add(source, node, 1)
    for node in gt_nodes.values():
        graph.add(node, sink, 1)
    chosen_edges = {}
    for key, edges in prediction_edges.items():
        for gt in sorted(edges):
            chosen_edges[(key, gt)] = graph.add(prediction_nodes[key], gt_nodes[gt], 1)
    result = graph.flow(source, sink)
    selected = {
        key: gt for (key, gt), edge in chosen_edges.items()
        if edge[2] == 0
    }
    return result, selected


def best_by_gt(prediction_edges, gt_sizes):
    output = {gt: 0.0 for gt in gt_sizes}
    for edges in prediction_edges.values():
        for gt, value in edges.items():
            if gt in output:
                output[gt] = max(output[gt], float(value))
    return output


def scene_oracle(scene, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene}.txt", args.min_region_size)
    processed = np.load(args.processed_scene_root / scene / f"{scene.replace('scene', '')}.npy", mmap_mode="r")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    sp_sizes = {int(sp): int(count) for sp, count in zip(ids, counts)}
    sp_gt = _sp_gt_counts(superpoints, gt_ids, gt_sizes)
    quality = {
        int(row["candidate_id"]): row
        for row in (
            json.loads(line) for line in
            (args.quality_ledger_root / scene / "candidate_quality_competition_ledger.jsonl").read_text().splitlines() if line
        )
    }
    candidates = json.loads((args.n2_cache_root / scene / "n2_medoid_candidates.json").read_text())["candidates"]
    if set(quality) != set(range(len(candidates))):
        raise ValueError(f"{scene}: quality ledger IDs do not equal frozen cache IDs")
    n2_edges, candidate_rows = {}, []
    groups = {}
    for candidate_id, candidate in enumerate(candidates):
        _, size, intersections = _geometry_for_superpoints(candidate["superpoint_ids"], sp_sizes, sp_gt)
        edges = _ious(size, intersections, gt_sizes)
        n2_edges[f"n2_{candidate_id}"] = edges
        groups[f"n2_{candidate_id}"] = f"component_{quality[candidate_id]['near_duplicate_component_id']}"
        best_gt = min(edges, key=lambda gt: (-edges[gt], gt)) if edges else -1
        best_iou = float(edges.get(best_gt, 0.0))
        candidate_rows.append({
            "scene_name": scene,
            "candidate_id": candidate_id,
            "near_duplicate_component_id": quality[candidate_id]["near_duplicate_component_id"],
            "near_duplicate_component_size": quality[candidate_id]["near_duplicate_component_size"],
            "candidate_point_count": size,
            "best_gt_instance_id": best_gt,
            "best_gt_iou": best_iou,
            "iou_by_gt": {str(gt): float(value) for gt, value in sorted(edges.items())},
            "ground_truth_usage": "offline_diagnostic_only",
            "proposal_materialization_applied": False,
            "ap_computed": False,
        })
    baseline_edges = {**native_edges(args.native_prediction_cache, scene, gt_ids, gt_sizes), **track_edges(args.d2b_track_root, scene, gt_ids, gt_sizes)}
    baseline_best = best_by_gt(baseline_edges, gt_sizes)
    n2_best = best_by_gt(n2_edges, gt_sizes)
    all_edges = {**baseline_edges, **n2_edges}
    metrics = {}
    selected_by_threshold = {}
    for threshold in OFFICIAL_THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        base_count, base_selected = maximum_matching(baseline_edges, threshold)
        n2_raw_count, n2_raw_selected = maximum_matching(n2_edges, threshold)
        n2_component_count, n2_component_selected = maximum_matching(n2_edges, threshold, groups)
        all_raw_count, all_raw_selected = maximum_matching(all_edges, threshold)
        all_groups = {**{key: f"baseline_{key}" for key in baseline_edges}, **groups}
        all_component_count, all_component_selected = maximum_matching(all_edges, threshold, all_groups)
        metrics[tag] = {
            "threshold": threshold,
            "valid_gt_instance_count": len(gt_sizes),
            "baseline_maximum_matching": base_count,
            "n2_raw_maximum_matching": n2_raw_count,
            "n2_one_per_near_duplicate_component_maximum_matching": n2_component_count,
            "fixed_mask_ideal_ranking_ceiling": all_raw_count / max(1, len(gt_sizes)),
            "baseline_ideal_ranking_ceiling": base_count / max(1, len(gt_sizes)),
            "fixed_mask_ideal_ranking_ceiling_increment_vs_baseline": (all_raw_count - base_count) / max(1, len(gt_sizes)),
            "system_raw_maximum_increment_vs_baseline": all_raw_count - base_count,
            "system_one_per_component_maximum_increment_vs_baseline": all_component_count - base_count,
            "system_one_per_component_maximum_matching": all_component_count,
        }
        selected_by_threshold[tag] = {
            "n2_raw": n2_raw_selected,
            "n2_component": n2_component_selected,
            "system_raw": all_raw_selected,
            "system_component": all_component_selected,
            "baseline": base_selected,
        }
    for row in candidate_rows:
        for threshold in (0.25, 0.50):
            tag = str(int(threshold * 100))
            positive = row["best_gt_iou"] >= threshold
            baseline_covers = bool(row["best_gt_instance_id"] > 0 and baseline_best[row["best_gt_instance_id"]] >= threshold)
            row[f"iou{tag}"] = positive
            row[f"target_already_covered_by_base_d2b_iou{tag}"] = baseline_covers
            row[f"classification_iou{tag}"] = (
                "pure_fp" if not positive else ("baseline_duplicate" if baseline_covers else "new_geometry_eligible")
            )
            row[f"selected_by_raw_n2_oracle_iou{tag}"] = f"n2_{row['candidate_id']}" in selected_by_threshold[tag]["n2_raw"]
            row[f"selected_by_component_n2_oracle_iou{tag}"] = f"n2_{row['candidate_id']}" in selected_by_threshold[tag]["n2_component"]
            row[f"selected_by_component_system_oracle_iou{tag}"] = f"n2_{row['candidate_id']}" in selected_by_threshold[tag]["system_component"]
    component_rows = []
    grouped = defaultdict(list)
    for row in candidate_rows:
        grouped[row["near_duplicate_component_id"]].append(row)
    for component_id, rows in sorted(grouped.items()):
        component_rows.append({
            "scene_name": scene,
            "near_duplicate_component_id": component_id,
            "candidate_count": len(rows),
            "candidate_ids": [row["candidate_id"] for row in rows],
            "best_member_iou": max(row["best_gt_iou"] for row in rows),
            "raw_oracle_selected_candidate_iou25": next((row["candidate_id"] for row in rows if row["selected_by_raw_n2_oracle_iou25"]), None),
            "one_per_component_selected_candidate_iou25": next((row["candidate_id"] for row in rows if row["selected_by_component_n2_oracle_iou25"]), None),
            "raw_oracle_selected_candidate_iou50": next((row["candidate_id"] for row in rows if row["selected_by_raw_n2_oracle_iou50"]), None),
            "one_per_component_selected_candidate_iou50": next((row["candidate_id"] for row in rows if row["selected_by_component_n2_oracle_iou50"]), None),
            "ground_truth_usage": "offline_diagnostic_only",
            "proposal_materialization_applied": False,
        })
    gt_rows = []
    for gt in sorted(gt_sizes):
        row = {
            "scene_name": scene, "gt_instance_id": gt, "gt_point_count": gt_sizes[gt],
            "best_base_d2b_iou": baseline_best[gt], "best_n2_iou": n2_best[gt],
            "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False,
        }
        for threshold in (0.25, 0.50):
            tag = str(int(threshold * 100))
            row[f"base_d2b_covered_iou{tag}"] = baseline_best[gt] >= threshold
            row[f"n2_covered_iou{tag}"] = n2_best[gt] >= threshold
            row[f"n2_new_vs_base_d2b_iou{tag}"] = n2_best[gt] >= threshold and baseline_best[gt] < threshold
        gt_rows.append(row)
    summary = {
        "scene_name": scene,
        "valid_gt_instance_count": len(gt_sizes),
        "candidate_count": len(candidate_rows),
        "near_duplicate_component_count": len(component_rows),
        "threshold_metrics": metrics,
        "ground_truth_usage": "offline_diagnostic_only",
        "proposal_materialization_applied": False,
        "ap_computed": False,
    }
    return candidate_rows, component_rows, gt_rows, summary


def write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--n2-cache-root", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required; GT is offline diagnosis only")
    for name in ("scene_list", "n2_cache_root", "quality_ledger_root", "d2b_track_root", "native_prediction_cache", "processed_scene_root", "gt_instance_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("output root is non-empty")
    selected_scenes = _scenes(args.scene_list)[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ordinal, scene in enumerate(selected_scenes, 1):
        candidate_rows, component_rows, gt_rows, summary = scene_oracle(scene, args)
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"
        stage.mkdir()
        write_jsonl(stage / "candidate_oracle_gt.jsonl", candidate_rows)
        write_jsonl(stage / "component_oracle_gt.jsonl", component_rows)
        write_jsonl(stage / "gt_instance_oracle_gt.jsonl", gt_rows)
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output_root / scene)
        summaries.append(summary)
        print(f"[N2 medoid GT oracle] {ordinal}/{len(selected_scenes)} {scene}", flush=True)
    totals = {}
    for threshold in OFFICIAL_THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        keys = ("valid_gt_instance_count", "baseline_maximum_matching", "n2_raw_maximum_matching", "n2_one_per_near_duplicate_component_maximum_matching", "system_raw_maximum_increment_vs_baseline", "system_one_per_component_maximum_increment_vs_baseline", "system_one_per_component_maximum_matching")
        totals[tag] = {key: sum(item["threshold_metrics"][tag][key] for item in summaries) for key in keys}
        total_gt = totals[tag]["valid_gt_instance_count"]
        totals[tag]["fixed_mask_ideal_ranking_ceiling"] = sum(item["threshold_metrics"][tag]["fixed_mask_ideal_ranking_ceiling"] * item["threshold_metrics"][tag]["valid_gt_instance_count"] for item in summaries) / max(1, total_gt)
    official_tags = [str(int(round(threshold * 100))) for threshold in OFFICIAL_THRESHOLDS if threshold >= .50]
    aggregate = {
        "fixed_mask_ideal_ranking_ceiling_ap": float(np.mean([totals[tag]["fixed_mask_ideal_ranking_ceiling"] for tag in official_tags])),
        "fixed_mask_ideal_ranking_ceiling_ap50": totals["50"]["fixed_mask_ideal_ranking_ceiling"],
        "fixed_mask_ideal_ranking_ceiling_ap25": totals["25"]["fixed_mask_ideal_ranking_ceiling"],
        "raw_system_increment_ceiling_ap": float(np.mean([totals[tag]["system_raw_maximum_increment_vs_baseline"] / max(1, totals[tag]["valid_gt_instance_count"]) for tag in official_tags])),
        "one_per_component_system_increment_ceiling_ap": float(np.mean([totals[tag]["system_one_per_component_maximum_increment_vs_baseline"] / max(1, totals[tag]["valid_gt_instance_count"]) for tag in official_tags])),
    }
    root = {
        "diagnostic_type": "GT-only frozen N2 medoid candidate flood/competition oracle",
        "decision_constraint": CONTRACT,
        "scene_count": len(summaries),
        "threshold_totals": totals,
        "aggregate_ideal_ranking_ceiling": aggregate,
        "proposal_materialization_applied": False,
        "ap_computed": False,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
