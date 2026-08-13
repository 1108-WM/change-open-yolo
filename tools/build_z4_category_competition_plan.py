#!/usr/bin/env python3
"""Build a GT-free class-conditional local competition plan.

The plan is diagnostic only: it never changes candidate membership, masks,
classes, or scores.  Candidates compete only when they have the same emitted
class and their geometry IoU is at least the frozen threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402
from tools.evaluate_z3_yoloworld_control_group_gt import (  # noqa: E402
    _load_native,
    _load_tracks,
    _load_union_rows,
    _points,
    _read_scenes,
    _resolve,
)


SOURCE_NAMES = ("native", "track", "pair_union")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _valid_class(value: int) -> bool:
    return int(value) in instance_eval.PRED_ID_TO_ID and int(instance_eval.PRED_ID_TO_ID[int(value)]) >= 0


def _components(indices: list[int], intersections: np.ndarray, sizes: np.ndarray, threshold: float) -> list[list[int]]:
    adjacency: dict[int, list[int]] = {index: [] for index in indices}
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1:]:
            union = float(sizes[left] + sizes[right] - intersections[left, right])
            iou = float(intersections[left, right] / union) if union > 0 else 0.0
            if iou + 1e-12 >= threshold:
                adjacency[left].append(right)
                adjacency[right].append(left)
    result, seen = [], set()
    for root in indices:
        if root in seen:
            continue
        queue, seen_component = deque([root]), set()
        while queue:
            current = queue.popleft()
            if current in seen_component:
                continue
            seen_component.add(current); seen.add(current)
            queue.extend(adjacency[current])
        result.append(sorted(seen_component))
    return result


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    rows_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if str(row["scene_name"]) in scenes:
            rows_by_scene[str(row["scene_name"])].append(row)
    if set(rows_by_scene) != set(scenes):
        raise ValueError("OOF rows do not exactly cover scene list")

    all_component_rows, all_candidate_rows, scene_summaries = [], [], []
    total_candidate_count = 0
    for scene_index, scene in enumerate(scenes, 1):
        native = _load_native(args.stream_records_root, scene)
        point_count = native["pred_masks"].shape[0]
        tracks = _load_tracks(args.stream_records_root, scene)
        unions = _load_union_rows(args.combined_plan_root, scene)
        scene_rows = sorted(rows_by_scene[scene], key=lambda row: (SOURCE_NAMES.index(str(row["candidate_source"])), int(row["candidate_id"])))
        masks, sizes, candidate_records = [], [], []
        for row in scene_rows:
            source = str(row["candidate_source"])
            candidate_id = int(row["candidate_id"])
            class_index = int(row["class_index"])
            if not _valid_class(class_index):
                raise ValueError(f"{scene}:{source}:{candidate_id}: invalid emitted class")
            if source == "native":
                if candidate_id >= native["pred_masks"].shape[1]:
                    raise ValueError(f"{scene}: native candidate outside cache: {candidate_id}")
                points = np.flatnonzero(native["pred_masks"][:, candidate_id])
                score = float(row["oof_predictions"]["C_joint_yolo_alpha"])
            elif source == "track":
                points = _points(Path(tracks[candidate_id]["points_path"]), point_count)
                score = float(row["oof_predictions"]["C_joint_yolo_alpha"])
            elif source == "pair_union":
                points = _points(Path(unions[candidate_id]["points_path"]), point_count)
                score = float(row["original_score"])
            else:
                raise ValueError(f"unknown candidate source: {source}")
            mask_index = len(masks)
            masks.append(points)
            sizes.append(len(points))
            candidate_records.append({
                "scene_name": scene, "local_candidate_index": mask_index,
                "candidate_source": source, "candidate_id": candidate_id,
                "class_index": class_index, "semantic_class_id": int(instance_eval.PRED_ID_TO_ID[class_index]),
                "semantic_evidence_node_key": str(row["semantic_evidence_node_key"]),
                "original_score": float(row["original_score"]), "competition_score": score,
                "geometry_point_count": len(points),
            })
        if not masks:
            raise ValueError(f"{scene}: no valid candidates")
        matrix = sparse.csr_matrix(
            (np.ones(sum(sizes), dtype=np.uint8),
             (np.concatenate([np.full(len(points), index, dtype=np.int64) for index, points in enumerate(masks)]),
              np.concatenate(masks))),
            shape=(len(masks), point_count), dtype=np.uint8,
        )
        intersections = (matrix @ matrix.T).toarray().astype(np.int64)
        sizes_array = np.asarray(sizes, dtype=np.int64)
        by_class: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(candidate_records):
            by_class[int(record["class_index"])].append(index)
        scene_component_count = 0
        for class_index, class_indices in sorted(by_class.items()):
            for component_local_index, component in enumerate(
                _components(class_indices, intersections, sizes_array, args.iou_threshold)
            ):
                component_id = f"{scene}:class{class_index}:component{scene_component_count:05d}"
                ordered = sorted(component, key=lambda index: (-candidate_records[index]["competition_score"], SOURCE_NAMES.index(candidate_records[index]["candidate_source"]), candidate_records[index]["candidate_id"]))
                winner = ordered[0]
                winner_score = candidate_records[winner]["competition_score"]
                second_score = candidate_records[ordered[1]]["competition_score"] if len(ordered) > 1 else 0.0
                component_sources = Counter(candidate_records[index]["candidate_source"] for index in component)
                component_summary = {
                    "scene_name": scene, "component_id": component_id,
                    "class_index": class_index, "semantic_class_id": int(instance_eval.PRED_ID_TO_ID[class_index]),
                    "component_candidate_count": len(component),
                    "component_native_count": component_sources["native"],
                    "component_track_count": component_sources["track"],
                    "component_pair_union_count": component_sources["pair_union"],
                    "winner_local_candidate_index": winner,
                    "winner_candidate_source": candidate_records[winner]["candidate_source"],
                    "winner_candidate_id": candidate_records[winner]["candidate_id"],
                    "winner_competition_score": winner_score,
                    "winner_second_margin": winner_score - second_score,
                    "winner_is_unique": len(component) == 1,
                    "competition_iou_threshold": args.iou_threshold,
                    "decision_state": "plan_only; no candidate selected, removed, or modified",
                }
                all_component_rows.append(component_summary)
                for rank, index in enumerate(ordered, 1):
                    record = candidate_records[index]
                    all_candidate_rows.append({
                        **record, "component_id": component_id,
                        "component_candidate_count": len(component),
                        "component_rank": rank, "component_winner": index == winner,
                        "component_score_margin_vs_winner": winner_score - record["competition_score"],
                        "decision_state": "plan_only; no candidate selected, removed, or modified",
                    })
                scene_component_count += 1
        scene_summaries.append({
            "scene_name": scene, "candidate_count": len(candidate_records),
            "class_count": len(by_class), "component_count": scene_component_count,
            "multi_candidate_component_count": sum(row["component_candidate_count"] > 1 for row in all_component_rows if row["scene_name"] == scene),
        })
        total_candidate_count += len(candidate_records)
        print(f"[Z4 plan] {scene_index}/{len(scenes)} {scene}: candidates={len(candidate_records)} components={scene_component_count}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "components.jsonl").open("w") as handle:
        for row in all_component_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "candidates.jsonl").open("w") as handle:
        for row in all_candidate_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "scene_summaries.jsonl").open("w") as handle:
        for row in scene_summaries:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "diagnostic_type": "Z4 GT-free class-conditional local competition plan",
        "ground_truth_usage": "none", "candidate_membership_modified": False,
        "candidate_scores_modified": False, "candidate_classes_modified": False,
        "scene_count": len(scenes), "candidate_count": total_candidate_count,
        "component_count": len(all_component_rows),
        "multi_candidate_component_count": sum(row["component_candidate_count"] > 1 for row in all_component_rows),
        "multi_candidate_candidate_count": sum(row["component_candidate_count"] for row in all_component_rows if row["component_candidate_count"] > 1),
        "source_candidate_counts": dict(Counter(row["candidate_source"] for row in all_candidate_rows)),
        "component_source_counts": dict(Counter(
            source for row in all_component_rows for source, count in (
                ("native", row["component_native_count"]), ("track", row["component_track_count"]),
                ("pair_union", row["component_pair_union_count"]),
            ) for _ in range(count)
        )),
        "competition_score_contract": "C_joint_yolo_alpha OOF for native/track; frozen pair-union original score for union",
        "competition_contract": {
            "same_class_only": True, "local_geometry_iou_threshold": args.iou_threshold,
            "exact_geometry_identity": "semantic_evidence_node_key retained on every candidate",
            "cross_class_nms": False, "action_application": False,
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    args = parser.parse_args()
    if not 0.0 < args.iou_threshold <= 1.0:
        raise SystemExit("--iou-threshold must be in (0,1]")
    for name in ("scene_list", "oof_root", "stream_records_root", "combined_plan_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
