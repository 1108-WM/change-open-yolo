import argparse
import gc
import json
import os
import os.path as osp
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200
from evaluate.replica.eval_semantic_instance import CLASS_LABELS, ID_TO_LABEL, PRED_ID_TO_ID, VALID_CLASS_IDS
from utils import OpenYolo3D
from utils.backprojection_fusion import (
    _candidate_source_kind,
    _candidate_source_name,
    _load_seed_indices,
    append_backprojection_proposals,
    load_backprojection_candidates,
)


def _load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_gt_instances(gt_path, min_region_size=100):
    gt_ids = np.loadtxt(gt_path, dtype=np.int64)
    valid_ids = set(int(item) for item in VALID_CLASS_IDS)
    instances = []
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        label_id = int(instance_id // 1000)
        if label_id not in valid_ids:
            continue
        count = int((gt_ids == instance_id).sum())
        if count < min_region_size:
            continue
        instances.append(
            {
                "instance_id": instance_id,
                "label_id": label_id,
                "label_name": ID_TO_LABEL[label_id],
                "vert_count": count,
            }
        )
    counts = {item["instance_id"]: item["vert_count"] for item in instances}
    meta = {item["instance_id"]: item for item in instances}
    return gt_ids, instances, counts, meta


def _best_gt_for_mask(mask, gt_ids, gt_counts, gt_meta):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return {
            "best_iou": 0.0,
            "best_intersection": 0,
            "best_gt": None,
        }

    hit_ids, hit_counts = np.unique(gt_ids[indices], return_counts=True)
    best_iou = 0.0
    best_intersection = 0
    best_gt = None
    candidate_size = int(len(indices))
    for instance_id, intersection in zip(hit_ids, hit_counts):
        instance_id = int(instance_id)
        if instance_id not in gt_counts:
            continue
        intersection = int(intersection)
        union = candidate_size + int(gt_counts[instance_id]) - intersection
        iou = float(intersection / max(1, union))
        if iou > best_iou:
            best_iou = iou
            best_intersection = intersection
            best_gt = gt_meta[instance_id]
    return {
        "best_iou": best_iou,
        "best_intersection": best_intersection,
        "best_gt": best_gt,
    }


def _pred_class_name(class_id, labels):
    class_id = int(class_id)
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return "unknown"


def _pred_to_eval_label_id(class_id):
    try:
        return int(PRED_ID_TO_ID[int(class_id)])
    except Exception:
        return -1


def _eval_to_pred_id(label_id):
    matches = np.where(VALID_CLASS_IDS == int(label_id))[0]
    if len(matches) == 0:
        return -1
    return int(matches[0])


def _mask_geometry_features(mask, points_xyz):
    indices = np.flatnonzero(mask)
    if points_xyz is None or len(indices) == 0:
        return {
            "bbox_dx": 0.0,
            "bbox_dy": 0.0,
            "bbox_dz": 0.0,
            "bbox_volume": 0.0,
        }
    pts = points_xyz[indices]
    extent = pts.max(axis=0) - pts.min(axis=0)
    return {
        "bbox_dx": float(extent[0]),
        "bbox_dy": float(extent[1]),
        "bbox_dz": float(extent[2]),
        "bbox_volume": float(max(extent[0], 0.0) * max(extent[1], 0.0) * max(extent[2], 0.0)),
    }


def _noisy_or_update(old_value, contribution):
    old_value = float(np.clip(old_value, 0.0, 1.0))
    contribution = float(np.clip(contribution, 0.0, 1.0))
    return 1.0 - (1.0 - old_value) * (1.0 - contribution)


def _candidate_score(candidate, use_fusion_score):
    if use_fusion_score and candidate.get("fusion_score") is not None:
        return _safe_float(candidate.get("fusion_score"))
    return _safe_float(candidate.get("score"))


def _support_weight(candidate, support_scale):
    if support_scale <= 0:
        return 1.0
    support = max(1.0, _safe_float(candidate.get("support_view_count"), 1.0))
    return min(1.0, support / float(support_scale))


def _candidate_evidence_features(
    mask,
    scene_candidates,
    num_points,
    current_class_id,
    min_candidate_score=0.30,
    min_seed_points=80,
    min_seed_overlap=0.05,
    min_support_views=1,
    support_scale=8.0,
    overlap_power=1.0,
    use_fusion_score=True,
):
    evidence = defaultdict(float)
    source_evidence = defaultdict(lambda: defaultdict(float))
    matched = 0
    used = 0
    max_seed_overlap = 0.0
    max_candidate_score = 0.0
    max_support = 0

    for candidate in scene_candidates:
        class_id = _safe_int(candidate.get("class_id"), -1)
        if class_id < 0:
            continue
        score = _candidate_score(candidate, use_fusion_score)
        if score < min_candidate_score:
            continue
        if _safe_int(candidate.get("num_seed_points")) < min_seed_points:
            continue
        if _safe_int(candidate.get("support_view_count"), 1) < min_support_views:
            continue

        seed_indices = _load_seed_indices(candidate, num_points)
        if seed_indices is None or len(seed_indices) < min_seed_points:
            continue
        used += 1

        seed_overlap = float(mask[seed_indices].sum() / max(1, len(seed_indices)))
        if seed_overlap < min_seed_overlap:
            continue

        matched += 1
        max_seed_overlap = max(max_seed_overlap, seed_overlap)
        max_candidate_score = max(max_candidate_score, score)
        max_support = max(max_support, _safe_int(candidate.get("support_view_count"), 1))
        support = _support_weight(candidate, support_scale)
        contribution = score * support * (seed_overlap ** float(overlap_power))
        evidence[class_id] = _noisy_or_update(evidence[class_id], contribution)
        source_kind = _candidate_source_kind(candidate)
        source_evidence[source_kind][class_id] = _noisy_or_update(source_evidence[source_kind][class_id], contribution)

    ranked = sorted(evidence.items(), key=lambda item: item[1], reverse=True)
    top1_class, top1_score = (int(ranked[0][0]), float(ranked[0][1])) if ranked else (-1, 0.0)
    top2_class, top2_score = (int(ranked[1][0]), float(ranked[1][1])) if len(ranked) > 1 else (-1, 0.0)

    features = {
        "object_evidence_used_candidates": int(used),
        "object_evidence_matched_candidates": int(matched),
        "object_evidence_top1_class_id": int(top1_class),
        "object_evidence_top1_score": float(top1_score),
        "object_evidence_top2_class_id": int(top2_class),
        "object_evidence_top2_score": float(top2_score),
        "object_evidence_margin": float(top1_score - top2_score),
        "object_evidence_current_class_score": float(evidence.get(int(current_class_id), 0.0)),
        "object_evidence_max_seed_overlap": float(max_seed_overlap),
        "object_evidence_max_candidate_score": float(max_candidate_score),
        "object_evidence_max_support_views": int(max_support),
    }
    for source_kind in ("sam_fused", "bpr"):
        source_ranked = sorted(source_evidence[source_kind].items(), key=lambda item: item[1], reverse=True)
        features[f"{source_kind}_evidence_top1_class_id"] = int(source_ranked[0][0]) if source_ranked else -1
        features[f"{source_kind}_evidence_top1_score"] = float(source_ranked[0][1]) if source_ranked else 0.0
        features[f"{source_kind}_evidence_current_class_score"] = float(
            source_evidence[source_kind].get(int(current_class_id), 0.0)
        )
    return features


def _proposal_feature_defaults():
    return {
        "candidate_id": None,
        "component_id": None,
        "candidate_score": 0.0,
        "candidate_fusion_score": 0.0,
        "candidate_quality_score": 0.0,
        "candidate_support_view_count": 0,
        "candidate_support_mean_iou": 0.0,
        "candidate_support_best_iou": 0.0,
        "candidate_seed_in_existing_mask_ratio": 0.0,
        "candidate_best_existing_iou": 0.0,
        "candidate_box_area_ratio": 0.0,
        "candidate_source_score_scale": 1.0,
        "candidate_num_seed_points": 0,
        "candidate_growth_ratio": 1.0,
    }


def _proposal_features(applied_record):
    features = _proposal_feature_defaults()
    if applied_record is None:
        return features
    features.update(
        {
            "candidate_id": applied_record.get("candidate_id"),
            "component_id": applied_record.get("component_id"),
            "candidate_score": _safe_float(applied_record.get("score")),
            "candidate_fusion_score": _safe_float(applied_record.get("fusion_score")),
            "candidate_quality_score": _safe_float(applied_record.get("quality_score")),
            "candidate_support_view_count": _safe_int(applied_record.get("support_view_count")),
            "candidate_support_mean_iou": _safe_float(applied_record.get("support_mean_iou")),
            "candidate_support_best_iou": _safe_float(applied_record.get("support_best_iou")),
            "candidate_seed_in_existing_mask_ratio": _safe_float(applied_record.get("seed_in_existing_mask_ratio")),
            "candidate_best_existing_iou": _safe_float(applied_record.get("best_existing_iou")),
            "candidate_box_area_ratio": _safe_float(applied_record.get("box_area_ratio")),
            "candidate_source_score_scale": _safe_float(applied_record.get("source_score_scale"), 1.0),
            "candidate_num_seed_points": _safe_int(applied_record.get("num_seed_points")),
            "candidate_growth_ratio": _safe_float(applied_record.get("growth_ratio"), 1.0),
        }
    )
    return features


def _clear_openyolo_state(openyolo3d, unload_3d_network=False):
    for attr in (
        "world2cam",
        "mesh_projections",
        "preds_3d",
        "preds_2d",
        "predicted_masks",
        "predicated_scores",
        "predicated_classes",
    ):
        if hasattr(openyolo3d, attr):
            setattr(openyolo3d, attr, None)
    if unload_3d_network and hasattr(openyolo3d, "network_3d"):
        openyolo3d.network_3d = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def export_dataset(args):
    if args.dataset_name != "replica":
        raise NotImplementedError("Semantic fusion dataset export currently supports Replica first.")

    config = _load_yaml(osp.join("./pretrained", f"config_{args.dataset_name}.yaml"))
    labels = config["network2d"]["text_prompts"]
    depth_scale = config["openyolo3d"]["depth_scale"]
    path_2_dataset = osp.join("./data", args.dataset_name)
    gt_dir = osp.join("./data", args.dataset_name, "ground_truth")
    datatype = "point cloud"

    if args.scene_names:
        scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    elif args.dataset_name == "replica":
        scene_names = SCENE_NAMES_REPLICA
    else:
        scene_names = SCENE_NAMES_SCANNET200

    candidates_by_scene, candidate_summary = load_backprojection_candidates(args.backprojection_candidates)
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = osp.join(args.output_dir, args.output_jsonl)
    summary_path = osp.join(args.output_dir, args.summary_json)

    openyolo3d = OpenYolo3D(f"./pretrained/config_{args.dataset_name}.yaml")
    summary = {
        "dataset_name": args.dataset_name,
        "scene_names": scene_names,
        "output_jsonl": jsonl_path,
        "candidate_summary": candidate_summary,
        "params": vars(args),
        "scenes": {},
        "overall": {},
    }
    overall_source_counts = Counter()
    overall_gt_counts = Counter()
    overall_pred_counts = Counter()
    num_records = 0
    num_positive_25 = 0
    num_positive_50 = 0
    num_class_correct_25 = 0
    num_class_correct_50 = 0

    with open(jsonl_path, "w") as writer:
        for scene_name in tqdm(scene_names):
            gt_path = osp.join(gt_dir, f"{scene_name}.txt")
            gt_ids, gt_instances, gt_counts, gt_meta = _load_gt_instances(gt_path, args.min_region_size)

            prediction = openyolo3d.predict(
                path_2_scene_data=osp.join(path_2_dataset, scene_name),
                depth_scale=depth_scale,
                datatype=datatype,
                processed_scene=None,
                path_to_3d_masks=args.path_to_3d_masks,
                is_gt=args.is_gt,
                path_to_2d_preds=args.path_to_2d_preds,
                save_2d_preds=args.save_2d_preds,
                reuse_2d_preds=args.reuse_2d_preds,
            )
            scene_prediction = prediction[scene_name]
            original_count = int(scene_prediction[0].shape[1])
            points_xyz, _ = openyolo3d.world2cam.load_ply(openyolo3d.world2cam.mesh)
            points_xyz = _to_numpy(points_xyz)[:, :3]

            fusion_report = {"loaded": 0, "applied": [], "skipped": []}
            if args.backprojection_candidates is not None:
                fused = append_backprojection_proposals(
                    scene_name,
                    scene_prediction[0],
                    scene_prediction[1],
                    scene_prediction[2],
                    candidates_by_scene,
                    points_xyz=points_xyz,
                    min_score=args.backprojection_min_score,
                    min_seed_points=args.backprojection_min_seed_points,
                    max_existing_iou=args.backprojection_max_existing_iou,
                    max_seed_in_existing_mask_ratio=args.backprojection_max_seed_in_existing_mask_ratio,
                    max_proposal_iou=args.backprojection_max_proposal_iou,
                    max_candidates=args.backprojection_max_candidates_per_scene,
                    score_scale=args.backprojection_score_scale,
                    use_candidate_fusion_score=args.backprojection_use_candidate_fusion_score,
                    allowed_classes=args.backprojection_allowed_classes,
                    blocked_classes=args.backprojection_blocked_classes,
                    min_support_views=args.backprojection_min_support_views,
                    min_support_mean_iou=args.backprojection_min_support_mean_iou,
                    min_support_best_iou=args.backprojection_min_support_best_iou,
                    min_fusion_score=args.backprojection_min_fusion_score,
                    max_box_area_ratio=args.backprojection_max_box_area_ratio,
                    min_quality_score=args.backprojection_min_quality_score,
                    quality_sort=args.backprojection_quality_sort,
                    grow_radius=args.backprojection_grow_radius,
                    max_growth_ratio=args.backprojection_max_growth_ratio,
                    cc_cleanup=args.backprojection_cc_cleanup,
                    cc_radius=args.backprojection_cc_radius,
                    cc_min_component_points=args.backprojection_cc_min_component_points,
                    cc_keep_topk=args.backprojection_cc_keep_topk,
                    cc_max_points=args.backprojection_cc_max_points,
                    cc_split_components=args.backprojection_cc_split_components,
                    source_priorities=args.backprojection_source_priorities,
                    source_max_candidates=args.backprojection_source_max_candidates,
                    source_score_scales=args.backprojection_source_score_scales,
                )
                scene_prediction = fused[:3]
                fusion_report = fused[3]

            pred_masks = _to_numpy(scene_prediction[0]).astype(bool)
            pred_classes = _to_numpy(scene_prediction[1]).astype(np.int64)
            pred_scores = _to_numpy(scene_prediction[2]).astype(np.float32)
            applied_records = fusion_report.get("applied", [])
            scene_source_counts = Counter()
            scene_gt_counts = Counter(item["label_name"] for item in gt_instances)
            scene_positive_25 = 0
            scene_positive_50 = 0
            scene_class_correct_25 = 0
            scene_class_correct_50 = 0
            scene_candidates = candidates_by_scene.get(scene_name, [])

            for pred_id in range(pred_masks.shape[1]):
                score = float(pred_scores[pred_id])
                if args.drop_below_score_threshold and score < args.score_threshold:
                    continue

                mask = pred_masks[:, pred_id]
                source_kind = "mask3d"
                source_name = "mask3d"
                applied = None
                if pred_id >= original_count:
                    applied_idx = pred_id - original_count
                    if 0 <= applied_idx < len(applied_records):
                        applied = applied_records[applied_idx]
                        source_kind = applied.get("source_kind") or "proposal"
                        source_name = applied.get("source_name") or source_kind
                    else:
                        source_kind = "proposal"
                        source_name = "proposal"

                pred_class_id = int(pred_classes[pred_id])
                pred_eval_label_id = _pred_to_eval_label_id(pred_class_id)
                match = _best_gt_for_mask(mask, gt_ids, gt_counts, gt_meta)
                best_gt = match["best_gt"]
                gt_pred_class_id = _eval_to_pred_id(best_gt["label_id"]) if best_gt is not None else -1
                best_iou = float(match["best_iou"])
                is_positive_25 = best_iou >= 0.25
                is_positive_50 = best_iou >= 0.50
                class_correct = best_gt is not None and pred_eval_label_id == int(best_gt["label_id"])
                class_correct_25 = bool(is_positive_25 and class_correct)
                class_correct_50 = bool(is_positive_50 and class_correct)

                features = {
                    "base_score": score,
                    "keep_after_score_threshold": bool(score >= args.score_threshold),
                    "mask_point_count": int(mask.sum()),
                    "mask_point_ratio": float(mask.sum() / max(1, pred_masks.shape[0])),
                    "source_is_mask3d": 1.0 if source_kind == "mask3d" else 0.0,
                    "source_is_sam_fused": 1.0 if source_kind == "sam_fused" else 0.0,
                    "source_is_bpr": 1.0 if source_kind == "bpr" else 0.0,
                }
                features.update(_mask_geometry_features(mask, points_xyz))
                features.update(_proposal_features(applied))
                features.update(
                    _candidate_evidence_features(
                        mask,
                        scene_candidates,
                        pred_masks.shape[0],
                        pred_class_id,
                        min_candidate_score=args.object_evidence_min_candidate_score,
                        min_seed_points=args.object_evidence_min_seed_points,
                        min_seed_overlap=args.object_evidence_min_seed_overlap,
                        min_support_views=args.object_evidence_min_support_views,
                        support_scale=args.object_evidence_support_scale,
                        overlap_power=args.object_evidence_overlap_power,
                        use_fusion_score=args.object_evidence_use_candidate_fusion_score,
                    )
                )

                record = {
                    "scene_name": scene_name,
                    "prediction_id": int(pred_id),
                    "source_kind": source_kind,
                    "source_name": source_name,
                    "pred_class_id": pred_class_id,
                    "pred_class_name": _pred_class_name(pred_class_id, labels),
                    "pred_eval_label_id": pred_eval_label_id,
                    "gt_pred_class_id": gt_pred_class_id,
                    "gt_label_id": int(best_gt["label_id"]) if best_gt is not None else -1,
                    "gt_label_name": best_gt["label_name"] if best_gt is not None else "background",
                    "gt_instance_id": int(best_gt["instance_id"]) if best_gt is not None else -1,
                    "gt_instance_points": int(best_gt["vert_count"]) if best_gt is not None else 0,
                    "best_iou": best_iou,
                    "best_intersection": int(match["best_intersection"]),
                    "is_positive_25": bool(is_positive_25),
                    "is_positive_50": bool(is_positive_50),
                    "class_correct": bool(class_correct),
                    "class_correct_25": class_correct_25,
                    "class_correct_50": class_correct_50,
                    "features": features,
                }
                writer.write(json.dumps(record) + "\n")

                num_records += 1
                scene_positive_25 += int(is_positive_25)
                scene_positive_50 += int(is_positive_50)
                scene_class_correct_25 += int(class_correct_25)
                scene_class_correct_50 += int(class_correct_50)
                scene_source_counts[source_kind] += 1
                overall_source_counts[source_kind] += 1
                overall_pred_counts[record["pred_class_name"]] += 1
                if best_gt is not None:
                    overall_gt_counts[best_gt["label_name"]] += 1

            summary["scenes"][scene_name] = {
                "num_gt_instances": len(gt_instances),
                "gt_class_counts": dict(scene_gt_counts),
                "num_original_predictions": original_count,
                "num_fused_predictions": int(pred_masks.shape[1]),
                "num_exported_records": int(sum(scene_source_counts.values())),
                "source_counts": dict(scene_source_counts),
                "fusion_loaded": int(fusion_report.get("loaded", 0)),
                "fusion_applied": int(len(fusion_report.get("applied", []))),
                "fusion_skipped": int(len(fusion_report.get("skipped", []))),
                "positive_at_25": int(scene_positive_25),
                "positive_at_50": int(scene_positive_50),
                "class_correct_at_25": int(scene_class_correct_25),
                "class_correct_at_50": int(scene_class_correct_50),
            }

            _clear_openyolo_state(openyolo3d, unload_3d_network=args.path_to_3d_masks is None)

    summary["overall"] = {
        "num_records": int(num_records),
        "source_counts": dict(overall_source_counts),
        "matched_gt_class_counts": dict(overall_gt_counts),
        "pred_class_counts": dict(overall_pred_counts),
        "positive_at_25": int(num_positive_25),
        "positive_at_50": int(num_positive_50),
        "class_correct_at_25": int(num_class_correct_25),
        "class_correct_at_50": int(num_class_correct_50),
    }

    # Recompute global counters from per-scene summaries to avoid accidental drift
    # when rows are filtered by score threshold.
    summary["overall"]["positive_at_25"] = int(sum(item["positive_at_25"] for item in summary["scenes"].values()))
    summary["overall"]["positive_at_50"] = int(sum(item["positive_at_50"] for item in summary["scenes"].values()))
    summary["overall"]["class_correct_at_25"] = int(
        sum(item["class_correct_at_25"] for item in summary["scenes"].values())
    )
    summary["overall"]["class_correct_at_50"] = int(
        sum(item["class_correct_at_50"] for item in summary["scenes"].values())
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved semantic fusion dataset to {jsonl_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "Records: "
        f"{summary['overall']['num_records']} | "
        f"sources={summary['overall']['source_counts']} | "
        f"pos@25={summary['overall']['positive_at_25']} | "
        f"class@25={summary['overall']['class_correct_at_25']} | "
        f"class@50={summary['overall']['class_correct_at_50']}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica"])
    parser.add_argument("--scene_names", default=None, help="Comma-separated scene names; default: all Replica scenes")
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--is_gt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--save_2d_preds", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", default="./output/semantic_fusion_dataset_replica")
    parser.add_argument("--output_jsonl", default="features.jsonl")
    parser.add_argument("--summary_json", default="summary.json")
    parser.add_argument("--min_region_size", default=100, type=int)
    parser.add_argument("--score_threshold", default=0.20, type=float)
    parser.add_argument("--drop_below_score_threshold", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--backprojection_candidates", default=None)
    parser.add_argument("--backprojection_min_score", default=0.40, type=float)
    parser.add_argument("--backprojection_min_seed_points", default=80, type=int)
    parser.add_argument("--backprojection_max_existing_iou", default=0.30, type=float)
    parser.add_argument("--backprojection_max_seed_in_existing_mask_ratio", default=0.70, type=float)
    parser.add_argument("--backprojection_max_proposal_iou", default=0.50, type=float)
    parser.add_argument("--backprojection_max_candidates_per_scene", default=30, type=int)
    parser.add_argument("--backprojection_score_scale", default=0.50, type=float)
    parser.add_argument("--backprojection_use_candidate_fusion_score", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_allowed_classes", default=None)
    parser.add_argument("--backprojection_blocked_classes", default=None)
    parser.add_argument("--backprojection_min_support_views", default=0, type=int)
    parser.add_argument("--backprojection_min_support_mean_iou", default=0.0, type=float)
    parser.add_argument("--backprojection_min_support_best_iou", default=0.0, type=float)
    parser.add_argument("--backprojection_min_fusion_score", default=0.0, type=float)
    parser.add_argument("--backprojection_max_box_area_ratio", default=None, type=float)
    parser.add_argument("--backprojection_min_quality_score", default=0.0, type=float)
    parser.add_argument("--backprojection_quality_sort", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_grow_radius", default=0.0, type=float)
    parser.add_argument("--backprojection_max_growth_ratio", default=4.0, type=float)
    parser.add_argument("--backprojection_cc_cleanup", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_cc_radius", default=0.03, type=float)
    parser.add_argument("--backprojection_cc_min_component_points", default=50, type=int)
    parser.add_argument("--backprojection_cc_keep_topk", default=1, type=int)
    parser.add_argument("--backprojection_cc_max_points", default=30000, type=int)
    parser.add_argument("--backprojection_cc_split_components", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_source_priorities", default=None)
    parser.add_argument("--backprojection_source_max_candidates", default=None)
    parser.add_argument("--backprojection_source_score_scales", default=None)

    parser.add_argument("--object_evidence_min_candidate_score", default=0.30, type=float)
    parser.add_argument("--object_evidence_min_seed_points", default=80, type=int)
    parser.add_argument("--object_evidence_min_seed_overlap", default=0.05, type=float)
    parser.add_argument("--object_evidence_min_support_views", default=1, type=int)
    parser.add_argument("--object_evidence_support_scale", default=8.0, type=float)
    parser.add_argument("--object_evidence_overlap_power", default=1.0, type=float)
    parser.add_argument("--object_evidence_use_candidate_fusion_score", default=True, action=argparse.BooleanOptionalAction)
    return parser


if __name__ == "__main__":
    export_dataset(build_parser().parse_args())
