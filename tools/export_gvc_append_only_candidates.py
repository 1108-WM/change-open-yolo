#!/usr/bin/env python3
"""将自动 SAM 轨迹导出为 GVC append-only 候选。

本工具只读取已冻结的自动轨迹、YOLO-World 投票与 GVC 账本。每条候选保留轨迹
原始点并集，不重建、裁剪、填充 superpoint，也不读取 GT。与 native 的关系只参与
连续分数；候选不会因与 native 重叠而被删除。去重只发生在本次导出的同类 GVC 候选间。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_KIND = "gvc_append_only"
SCORE_WEIGHTS = {
    "gvc_score": 0.55,
    "semantic_reliability": 0.20,
    "support_view_rank": 0.15,
    "native_novelty": 0.10,
}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError("场景列表含重复场景")
    return scenes


def _bounded(value):
    return float(min(1.0, max(0.0, float(value or 0.0))))


def _point_iou(left, right):
    shared = np.intersect1d(left, right, assume_unique=True).size
    return float(shared / max(1, len(left) + len(right) - shared))


def _scene_percentile(records, field):
    """仅用当前场景候选池的推理期值，输出稳定的中位秩百分位。"""
    if not records:
        return {}
    ordered = sorted(records, key=lambda item: (float(item[field]), int(item["track_id"])))
    if len(ordered) == 1:
        return {int(ordered[0]["track_id"]): 1.0}
    output = {}
    start = 0
    while start < len(ordered):
        value = float(ordered[start][field])
        end = start + 1
        while end < len(ordered) and float(ordered[end][field]) == value:
            end += 1
        percentile = float(((start + end - 1) * 0.5) / (len(ordered) - 1))
        for record in ordered[start:end]:
            output[int(record["track_id"])] = percentile
        start = end
    return output


def _semantic_reliability(semantic):
    total = max(0.0, float(semantic.get("vote_total", 0.0) or 0.0))
    top = max(0.0, float(semantic.get("top_vote", 0.0) or 0.0))
    vote_share = _bounded(top / total) if total > 0.0 else 0.0
    margin = _bounded(semantic.get("vote_margin", 0.0))
    return float(0.5 * vote_share + 0.5 * margin), vote_share


def _native_novelty(gvc):
    """native 重叠只降分，绝不成为候选过滤条件。"""
    return float((1.0 - _bounded(gvc.get("native_top_iou", 0.0))) * (1.0 - _bounded(gvc.get("track_inside_top_native_ratio", 0.0))))


def _load_scene_records(scene_name, args):
    track_path = args.track_root / scene_name / "automatic_tracks.json"
    semantic_path = args.semantic_root / scene_name / "automatic_track_yoloworld_semantics.json"
    gvc_path = args.gvc_root / scene_name / "track_gvc_feature_ledger.json"
    tracks = {int(item["track_id"]): item for item in json.loads(track_path.read_text())["tracks"]}
    semantics = {int(item["track_id"]): item for item in json.loads(semantic_path.read_text())}
    gvcs = {int(item["track_id"]): item for item in json.loads(gvc_path.read_text())}
    records = []
    skipped = Counter()
    for track_id, track in sorted(tracks.items()):
        semantic = semantics.get(track_id)
        gvc = gvcs.get(track_id)
        if semantic is None or gvc is None:
            skipped["missing_semantic_or_gvc"] += 1
            continue
        class_id = int(semantic.get("voted_class_index", -1))
        if not 0 <= class_id < len(args.labels):
            skipped["missing_valid_yoloworld_class"] += 1
            continue
        points_path = Path(track["points_path"])
        if not points_path.is_absolute():
            points_path = (args.track_root / scene_name / points_path).resolve()
        if not points_path.is_file():
            skipped["missing_track_points"] += 1
            continue
        points = np.unique(np.asarray(np.load(points_path)["point_indices"], dtype=np.int64))
        if len(points) == 0:
            skipped["empty_track_points"] += 1
            continue
        semantic_reliability, vote_share = _semantic_reliability(semantic)
        records.append({
            "track_id": track_id,
            "class_id": class_id,
            "class_name": str(args.labels[class_id]),
            "points": points,
            "points_path": points_path,
            "gvc_score": _bounded(gvc.get("gvc_score", 0.0)),
            "semantic_reliability": semantic_reliability,
            "vote_share": vote_share,
            "vote_margin": _bounded(semantic.get("vote_margin", 0.0)),
            "native_novelty": _native_novelty(gvc),
            "support_view_count": int(track.get("support_view_count", 0)),
            "semantic_frame_count": int(semantic.get("semantic_frame_count", 0)),
            "voted_class_support_views": int(semantic.get("voted_class_support_views", 0)),
            "native_top_candidate_id": int(gvc.get("native_top_candidate_id", -1)),
            "native_top_iou": _bounded(gvc.get("native_top_iou", 0.0)),
            "track_inside_top_native_ratio": _bounded(gvc.get("track_inside_top_native_ratio", 0.0)),
            "native_overlap_count": int(gvc.get("native_overlap_count", 0)),
            "gvc_selected_view_count": int(gvc.get("gvc_selected_view_count", 0)),
            "gvc_matched_view_count": int(gvc.get("gvc_matched_view_count", 0)),
            "gvc_selected_match_ratio": _bounded(gvc.get("gvc_selected_match_ratio", 0.0)),
            "gvc_selected_frames": list(gvc.get("gvc_selected_frames", [])),
        })
    return records, skipped


def _score_records(records):
    support_ranks = _scene_percentile(records, "support_view_count")
    for record in records:
        record["support_view_rank"] = float(support_ranks[int(record["track_id"])])
        record["score"] = float(sum(SCORE_WEIGHTS[name] * record[name] for name in SCORE_WEIGHTS))


def _deduplicate_same_class(records, threshold):
    kept = []
    by_class = defaultdict(list)
    skipped = []
    for record in sorted(records, key=lambda item: (-item["score"], int(item["track_id"]))):
        duplicate = next(
            (
                previous for previous in by_class[record["class_id"]]
                if _point_iou(record["points"], previous["points"]) >= threshold
            ),
            None,
        )
        if duplicate is not None:
            skipped.append({
                "track_id": int(record["track_id"]),
                "reason": "same_class_gvc_duplicate",
                "kept_track_id": int(duplicate["track_id"]),
                "iou": _point_iou(record["points"], duplicate["points"]),
            })
            continue
        kept.append(record)
        by_class[record["class_id"]].append(record)
    return kept, skipped


def _numeric_summary(candidates, field):
    values = np.asarray([float(item.get(field, 0.0) or 0.0) for item in candidates], dtype=np.float64)
    if len(values) == 0:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def _candidate(record, scene_name, seed_path):
    return {
        "scene_name": scene_name,
        "candidate_id": int(record["track_id"]),
        "source_kind": SOURCE_KIND,
        "candidate_source": "automatic_sam_track",
        "class_id": int(record["class_id"]),
        "class_name": record["class_name"],
        "score": float(record["score"]),
        "fusion_score": float(record["score"]),
        "proposal_priority": float(record["score"]),
        "seed_points_path": str(seed_path),
        "num_seed_points": int(len(record["points"])),
        "support_view_count": int(record["support_view_count"]),
        "support_views": record["gvc_selected_frames"],
        "gvc_score": float(record["gvc_score"]),
        "semantic_reliability": float(record["semantic_reliability"]),
        "vote_share": float(record["vote_share"]),
        "yoloworld_vote_margin": float(record["vote_margin"]),
        "support_view_rank": float(record["support_view_rank"]),
        "native_novelty": float(record["native_novelty"]),
        "native_top_candidate_id": int(record["native_top_candidate_id"]),
        "native_top_iou": float(record["native_top_iou"]),
        "track_inside_top_native_ratio": float(record["track_inside_top_native_ratio"]),
        "native_overlap_count": int(record["native_overlap_count"]),
        "semantic_frame_count": int(record["semantic_frame_count"]),
        "voted_class_support_views": int(record["voted_class_support_views"]),
        "gvc_selected_view_count": int(record["gvc_selected_view_count"]),
        "gvc_matched_view_count": int(record["gvc_matched_view_count"]),
        "gvc_selected_match_ratio": float(record["gvc_selected_match_ratio"]),
        "score_components": {name: float(record[name]) for name in SCORE_WEIGHTS},
    }


def _export_scene(scene_name, args):
    records, structural_skips = _load_scene_records(scene_name, args)
    _score_records(records)
    kept, duplicate_skips = _deduplicate_same_class(records, args.same_class_dedup_iou)
    scene_root = args.output_root / scene_name
    seed_root = scene_root / "seed_points"
    seed_root.mkdir(parents=True, exist_ok=False)
    candidates = []
    for record in kept:
        relative_seed_path = Path("seed_points") / f"gvc_track{int(record['track_id']):05d}.npz"
        np.savez_compressed(scene_root / relative_seed_path, point_indices=record["points"].astype(np.int32))
        candidates.append(_candidate(record, scene_name, relative_seed_path))
    payload = {
        "scene_name": scene_name,
        "source_kind": SOURCE_KIND,
        "gt_usage": "none",
        "append_only_contract": {
            "native_candidates_mutated": False,
            "native_overlap_filtering": False,
            "same_class_gvc_dedup_iou": float(args.same_class_dedup_iou),
        },
        "score_policy": {"weights": SCORE_WEIGHTS, "normalization": "support_view_rank 仅使用当前场景候选池"},
        "candidates": candidates,
    }
    (scene_root / "backprojection_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    audit = {
        "scene_name": scene_name,
        "input_track_count": int(len(records) + sum(structural_skips.values())),
        "scored_candidate_count": int(len(records)),
        "exported_candidate_count": int(len(candidates)),
        "structural_skip_counts": dict(sorted(structural_skips.items())),
        "same_class_duplicate_skips": duplicate_skips,
        "exported_distributions": {
            "class_counts": dict(sorted(Counter(item["class_name"] for item in candidates).items())),
            "score": _numeric_summary(candidates, "score"),
            "gvc_score": _numeric_summary(candidates, "gvc_score"),
            "num_seed_points": _numeric_summary(candidates, "num_seed_points"),
            "support_view_count": _numeric_summary(candidates, "support_view_count"),
            "native_top_iou": _numeric_summary(candidates, "native_top_iou"),
            "track_inside_top_native_ratio": _numeric_summary(candidates, "track_inside_top_native_ratio"),
            "native_overlap_count": _numeric_summary(candidates, "native_overlap_count"),
        },
    }
    (scene_root / "gvc_export_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--gvc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--same-class-dedup-iou", type=float, default=0.50)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "semantic_root", "gvc_root", "output_root", "config_path"):
        setattr(args, name, _resolve(getattr(args, name)))
    if not 0.0 < args.same_class_dedup_iou <= 1.0:
        raise SystemExit("--same-class-dedup-iou 必须位于 (0, 1]。")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    config = yaml.safe_load(args.config_path.read_text())
    args.labels = list(config["network2d"]["text_prompts"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    audits = []
    for index, scene_name in enumerate(scenes, start=1):
        audit = _export_scene(scene_name, args)
        audits.append(audit)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {audit['exported_candidate_count']} 条 GVC 候选", flush=True)
    summary = {
        "gt_usage": "none",
        "decision_state": "仅导出 GVC append-only 候选；不读取 GT，不修改 native 候选，不运行 AP。",
        "source_kind": SOURCE_KIND,
        "scene_count": len(audits),
        "exported_candidate_count": sum(item["exported_candidate_count"] for item in audits),
        "same_class_dedup_iou": float(args.same_class_dedup_iou),
        "score_policy": {"weights": SCORE_WEIGHTS, "normalization": "support_view_rank 仅使用当前场景候选池"},
        "scenes": audits,
    }
    (args.output_root / "gvc_append_only_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
