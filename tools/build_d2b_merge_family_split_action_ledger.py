#!/usr/bin/env python3
"""Build a no-GT split-action ledger from the frozen D2b merge history."""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    "No-GT D2b merge-family split-action ledger only; D1, consensus, D2b, scores, "
    "native proposals, and AP remain frozen and no split proposal is materialized."
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def scenes(path):
    rows = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or duplicated")
    return rows


def lineage_key(values):
    return tuple(sorted(map(int, values)))


def leaf_summary(track, observations):
    observation_ids = sorted(map(int, track["observation_ids"]))
    frames = sorted({int(observations[item]["frame_index"]) for item in observation_ids})
    return {
        "node_id": f"leaf_{int(track['track_id'])}",
        "node_kind": "frozen_premerge_leaf",
        "lineage_proposal_ids": [int(track["track_id"])],
        "observation_ids": observation_ids,
        "frame_indices": frames,
        "superpoint_ids": sorted(map(int, track["superpoint_ids"])),
        "superpoint_count": int(track["superpoint_count"]),
        "point_count": int(track["point_count"]),
        "source_points_path": track["points_path"],
    }


def leaves(node):
    if node["node_kind"] == "frozen_premerge_leaf":
        return [node]
    return leaves(node["left_child"]) + leaves(node["right_child"])


def cut_partition(node, cut_node_id):
    if node["node_id"] == cut_node_id:
        return [node["left_child"], node["right_child"]]
    if node["node_kind"] == "frozen_premerge_leaf":
        return [node]
    return cut_partition(node["left_child"], cut_node_id) + cut_partition(node["right_child"], cut_node_id)


def compact_node(node):
    return {
        "node_id": node["node_id"], "node_kind": node["node_kind"],
        "lineage_proposal_ids": node["lineage_proposal_ids"],
        "observation_ids": node["observation_ids"], "frame_indices": node["frame_indices"],
        "superpoint_ids": node.get("superpoint_ids"), "superpoint_count": node.get("superpoint_count"),
        "point_count": node.get("point_count"), "merge_action_index": node.get("merge_action_index"),
        "merge_round_index": node.get("merge_round_index"),
    }


def pair_evidence(left, right, relation, same_frame):
    left_observations = {item for leaf in leaves(left) for item in leaf["observation_ids"]}
    right_observations = {item for leaf in leaves(right) for item in leaf["observation_ids"]}
    same = []
    for left_id in left_observations:
        for right_id in right_observations:
            item = same_frame.get(tuple(sorted((left_id, right_id))))
            if item is not None:
                same.append(item)
    kinds = Counter(item["relation_kind"] for item in same)
    return {
        "merge_point_iou": float(relation["point_iou"]),
        "merge_left_point_coverage": float(relation["left_point_coverage"]),
        "merge_right_point_coverage": float(relation["right_point_coverage"]),
        "merge_point_intersection_count": int(relation["point_intersection_count"]),
        "spatial_contact_proxy": "shared_raw_superpoint" if int(relation["point_intersection_count"]) else "no_shared_raw_superpoint",
        "same_frame_relation_count": len(same),
        "same_frame_separation_counterevidence_count": int(kinds.get("disjoint", 0)),
        "same_frame_containment_count": int(kinds.get("containment", 0)),
        "same_frame_partial_overlap_count": int(kinds.get("partial_overlap", 0)),
        "left_view_count": len({frame for leaf in leaves(left) for frame in leaf["frame_indices"]}),
        "right_view_count": len({frame for leaf in leaves(right) for frame in leaf["frame_indices"]}),
        "shared_view_count": len({frame for leaf in leaves(left) for frame in leaf["frame_indices"]} & {frame for leaf in leaves(right) for frame in leaf["frame_indices"]}),
        "depth_consistency_state": "not_available_in_frozen_D1_merge_sources",
        "depth_consistency_value": None,
    }


def scene_ledger(scene, args):
    source_tracks = json.loads((args.premerge_track_root / scene / "automatic_tracks.json").read_text())["tracks"]
    observations = {int(row["observation_id"]): row for row in (json.loads(line) for line in (args.d1_observation_root / scene / "automatic_observations.jsonl").read_text().splitlines() if line)}
    actions = [json.loads(line) for line in (args.d2b_merge_root / scene / "merge_actions.jsonl").read_text().splitlines() if line]
    relation_rows = [json.loads(line) for line in (args.d2b_merge_root / scene / "merge_round_relations.jsonl").read_text().splitlines() if line]
    same_frame_rows = [json.loads(line) for line in (args.d1_observation_root / scene / "same_frame_hierarchy_relations.jsonl").read_text().splitlines() if line]
    relations = {(int(row["round_index"]), min(int(row["left_proposal_id"]), int(row["right_proposal_id"])), max(int(row["left_proposal_id"]), int(row["right_proposal_id"]))): row for row in relation_rows}
    same_frame = {tuple(sorted((int(row["left_observation_id"]), int(row["right_observation_id"])))): row for row in same_frame_rows}
    active = {}
    for track in source_tracks:
        leaf = leaf_summary(track, observations)
        active[lineage_key(leaf["lineage_proposal_ids"])] = leaf
    all_internal = []
    for action in sorted(actions, key=lambda row: int(row["action_index"])):
        left_key, right_key = lineage_key(action["anchor_lineage_before"]), lineage_key(action["absorbed_lineage"])
        if left_key not in active or right_key not in active:
            raise ValueError(f"{scene}: merge action {action['action_index']} cannot resolve active children")
        left, right = active.pop(left_key), active.pop(right_key)
        parent_key = lineage_key(action["anchor_lineage_after"])
        if parent_key != lineage_key(list(left_key) + list(right_key)):
            raise ValueError(f"{scene}: merge lineage is not conserved")
        relation_key = (int(action["round_index"]), min(int(action["anchor_proposal_id"]), int(action["absorbed_proposal_id"])), max(int(action["anchor_proposal_id"]), int(action["absorbed_proposal_id"])))
        if relation_key not in relations:
            raise ValueError(f"{scene}: missing frozen relation for merge action {action['action_index']}")
        node = {
            "node_id": f"merge_{int(action['action_index'])}", "node_kind": "frozen_D2b_merge",
            "lineage_proposal_ids": list(parent_key),
            "observation_ids": sorted(set(left["observation_ids"]) | set(right["observation_ids"])),
            "frame_indices": sorted(set(left["frame_indices"]) | set(right["frame_indices"])),
            "superpoint_ids": None, "superpoint_count": int(action["refined_superpoint_count"]),
            "point_count": int(action["refined_point_count"]), "merge_action_index": int(action["action_index"]),
            "merge_round_index": int(action["round_index"]), "left_child": left, "right_child": right,
            "action_record": action, "merge_evidence": pair_evidence(left, right, relations[relation_key], same_frame),
        }
        active[parent_key] = node
        all_internal.append(node)
    trees, action_rows = [], []
    for final_key, root in sorted(active.items()):
        if root["node_kind"] != "frozen_D2b_merge":
            continue
        tree_id = f"{scene}_root_{'_'.join(map(str, final_key))}"
        leaf_nodes = leaves(root)
        internal_nodes = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node["node_kind"] == "frozen_D2b_merge":
                internal_nodes.append(node)
                stack.extend((node["left_child"], node["right_child"]))
        trees.append({
            "scene_name": scene, "merge_family_id": tree_id, "final_lineage_proposal_ids": list(final_key),
            "final_d2b_node": compact_node(root), "leaf_subcandidates": [compact_node(node) for node in leaf_nodes],
            "merge_nodes": [{**compact_node(node), "left_child_node_id": node["left_child"]["node_id"], "right_child_node_id": node["right_child"]["node_id"], "merge_evidence": node["merge_evidence"]} for node in sorted(internal_nodes, key=lambda item: item["merge_action_index"])],
            "ground_truth_usage": "none", "proposal_materialization_applied": False, "ap_computed": False,
            "decision_constraint": CONTRACT,
        })
        def add_action(kind, partition, cut_node=None):
            action_rows.append({
                "scene_name": scene, "merge_family_id": tree_id, "split_action_kind": kind,
                "cut_merge_action_index": None if cut_node is None else cut_node["merge_action_index"],
                "partition_node_ids": [node["node_id"] for node in partition],
                "partition_lineages": [node["lineage_proposal_ids"] for node in partition],
                "partition_count": len(partition), "ground_truth_usage": "none",
                "proposal_materialization_applied": False, "ap_computed": False, "decision_constraint": CONTRACT,
            })
        add_action("keep", [root])
        add_action("undo_last", [root["left_child"], root["right_child"]])
        add_action("restore_all", leaf_nodes)
        for node in internal_nodes:
            add_action("cut_edge", cut_partition(root, node["node_id"]), node)
    summary = {"scene_name": scene, "source_premerge_track_count": len(source_tracks), "d2b_merge_action_count": len(actions), "merge_family_count": len(trees), "split_action_count": len(action_rows), "ground_truth_usage": "none", "proposal_materialization_applied": False, "ap_computed": False}
    return trees, action_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--d2b-merge-root", type=Path, required=True)
    parser.add_argument("--premerge-track-root", type=Path, required=True)
    parser.add_argument("--d1-observation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "d2b_merge_root", "premerge_track_root", "d1_observation_root", "output_root"):
        setattr(args, name, resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("output root is non-empty")
    chosen_scenes = scenes(args.scene_list)[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ordinal, scene in enumerate(chosen_scenes, 1):
        trees, actions, summary = scene_ledger(scene, args)
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"
        stage.mkdir()
        for name, rows in (("merge_family_tree_ledger.jsonl", trees), ("split_action_ledger.jsonl", actions)):
            with (stage / name).open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output_root / scene)
        summaries.append(summary)
        print(f"[D2b 拆分账本] {ordinal}/{len(chosen_scenes)} {scene}: {summary['d2b_merge_action_count']} 次合并", flush=True)
    root = {"diagnostic_type": "no-GT D2b merge-family split-action ledger", "decision_constraint": CONTRACT, "scene_count": len(summaries), "d2b_merge_action_count": sum(row["d2b_merge_action_count"] for row in summaries), "merge_family_count": sum(row["merge_family_count"] for row in summaries), "split_action_count": sum(row["split_action_count"] for row in summaries), "proposal_materialization_applied": False, "ap_computed": False, "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
