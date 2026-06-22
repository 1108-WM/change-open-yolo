import argparse
import gc
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200, evaluate_replica, evaluate_scannet200
from utils import OpenYolo3D
from utils.backprojection_fusion import append_backprojection_proposals, load_backprojection_candidates


def _load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _iter_feature_paths(path):
    if path is None:
        return
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "multiview_object_clip_features.json":
                yield osp.join(root, filename)


def load_multiview_clip_features(path):
    grouped = {}
    summary = {"files": [], "loaded": 0}
    if path is None:
        return {}, summary
    if not osp.exists(path):
        raise FileNotFoundError(f"Multiview CLIP feature path does not exist: {path}")
    for json_path in _iter_feature_paths(path):
        summary["files"].append(json_path)
        with open(json_path) as f:
            payload = json.load(f)
        scene_name = payload.get("scene_name")
        scene_records = grouped.setdefault(str(scene_name), {})
        for record in payload.get("features", []):
            scene_records[int(record["prediction_id"])] = record
            summary["loaded"] += 1
    return grouped, summary


def _parse_class_filter(value):
    if value is None:
        return None
    parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def _parse_pair_filter(value):
    if value is None:
        return None
    pairs = set()
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "->" in item:
            left, right = item.split("->", 1)
        elif ":" in item:
            left, right = item.split(":", 1)
        else:
            continue
        left = left.strip()
        right = right.strip()
        if left and right:
            pairs.add((left, right))
    return pairs or None


def _parse_pair_rules(value):
    if value is None:
        return None
    rules = {}
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 5:
            continue
        pair, confidence, margin, gain, max_base_score = parts
        if "->" not in pair:
            continue
        old_name, new_name = [part.strip() for part in pair.split("->", 1)]
        if not old_name or not new_name:
            continue
        try:
            rules[(old_name, new_name)] = {
                "confidence": float(confidence),
                "margin": float(margin),
                "gain": float(gain),
                "max_base_score": float(max_base_score),
            }
        except ValueError:
            continue
    return rules or None


def _parse_confusion_groups(value):
    if value is None:
        return None
    groups = []
    for item in str(value).split(";"):
        names = {part.strip() for part in item.split(",") if part.strip()}
        if len(names) >= 2:
            groups.append(names)
    return groups or None


def _same_confusion_group(groups, left, right):
    if groups is None:
        return True
    return any(left in group and right in group for group in groups)


def _record_source_allowed(record, value):
    allowed = _parse_class_filter(value)
    if allowed is None:
        return True
    source_kind = str(record.get("source_kind") or "")
    source_name = str(record.get("source_name") or "")
    return source_kind in allowed or source_name in allowed


def _score_field_for_mode(mode):
    if mode == "logits":
        return "clip_logits"
    if mode == "similarities":
        return "clip_similarities"
    return "clip_probs"


def _get_clip_scores(record, mode):
    field = _score_field_for_mode(mode)
    values = record.get(field)
    if values is None and mode != "probs":
        values = record.get("clip_probs")
    if values is None:
        return None
    scores = np.asarray(values, dtype=np.float32)
    if scores.ndim != 1 or len(scores) == 0:
        return None
    return scores


def _standardize(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def _build_sms_scores(feature_records, args):
    if args.clip_filter_policy != "sms":
        return {}
    raw_items = []
    for pred_id, record in feature_records.items():
        if not _record_source_allowed(record, args.clip_filter_source_kinds):
            continue
        scores = _get_clip_scores(record, args.clip_decision_score_mode)
        if scores is None:
            continue
        top_class = int(np.argmax(scores))
        raw_items.append((int(pred_id), top_class, float(scores[top_class])))
    if not raw_items:
        return {}

    by_pred = {}
    if args.clip_sms_scope == "scene":
        z_values = _standardize([item[2] for item in raw_items])
        for (pred_id, _, _), z_value in zip(raw_items, z_values):
            by_pred[pred_id] = float(z_value)
        return by_pred

    by_class = {}
    for pred_id, top_class, value in raw_items:
        by_class.setdefault(top_class, []).append((pred_id, value))
    fallback = _standardize([item[2] for item in raw_items])
    fallback_by_pred = {pred_id: float(z_value) for (pred_id, _, _), z_value in zip(raw_items, fallback)}
    for class_items in by_class.values():
        pred_ids = [item[0] for item in class_items]
        values = [item[1] for item in class_items]
        if len(values) < int(args.clip_sms_min_class_count):
            for pred_id in pred_ids:
                by_pred[pred_id] = fallback_by_pred[pred_id]
            continue
        for pred_id, z_value in zip(pred_ids, _standardize(values)):
            by_pred[pred_id] = float(z_value)
    return by_pred


def _apply_final_classifier(scene_name, pred_classes, pred_scores, feature_records, labels, args):
    classes = pred_classes.copy()
    scores = pred_scores.copy()
    blocked = _parse_class_filter(args.clip_blocked_classes)
    allowed = _parse_class_filter(args.clip_allowed_classes)
    sms_scores = _build_sms_scores(feature_records, args)
    report = {
        "loaded": len(feature_records),
        "applied": [],
        "skipped": [],
        "filtered": [],
    }
    keep_overrides = {}
    for pred_id in range(len(classes)):
        record = feature_records.get(int(pred_id))
        if not record:
            if args.clip_filter_missing_features:
                keep_overrides[pred_id] = False
                report["filtered"].append({"prediction_id": int(pred_id), "reason": "missing_feature"})
            continue
        if not _record_source_allowed(record, args.clip_apply_source_kinds):
            continue
        clip_scores = _get_clip_scores(record, args.clip_decision_score_mode)
        if clip_scores is None:
            if args.clip_filter_missing_features:
                keep_overrides[pred_id] = False
                report["filtered"].append({"prediction_id": int(pred_id), "reason": "empty_feature"})
            continue
        decision_scores = clip_scores.copy()
        current_class = int(classes[pred_id])
        if args.clip_yolo_prior_weight != 0.0 and 0 <= current_class < len(decision_scores):
            decision_scores[current_class] += float(args.clip_yolo_prior_weight)
        order = np.argsort(-decision_scores)
        top_class = int(order[0])
        second_score = float(decision_scores[order[1]]) if len(order) > 1 else 0.0
        top_score = float(decision_scores[top_class])
        margin = top_score - second_score
        current_name = labels[current_class] if 0 <= current_class < len(labels) else str(current_class)
        top_name = labels[top_class] if 0 <= top_class < len(labels) else str(top_class)
        sms_score = sms_scores.get(int(pred_id))

        if args.clip_filter_policy == "sms":
            if sms_score is None or sms_score < float(args.clip_sms_threshold):
                keep_overrides[pred_id] = False
                report["filtered"].append(
                    {
                        "prediction_id": int(pred_id),
                        "reason": "sms_below_threshold",
                        "sms_score": None if sms_score is None else float(sms_score),
                        "top_class": top_name,
                        "top_score": top_score,
                    }
                )
                continue
        if blocked is not None and top_name in blocked:
            report["skipped"].append({"prediction_id": int(pred_id), "reason": "class_blocked", "top_class": top_name})
            continue
        if allowed is not None and top_name not in allowed:
            report["skipped"].append({"prediction_id": int(pred_id), "reason": "class_not_allowed", "top_class": top_name})
            continue
        if args.clip_final_min_score is not None and top_score < float(args.clip_final_min_score):
            report["skipped"].append(
                {"prediction_id": int(pred_id), "reason": "below_final_min_score", "top_score": top_score}
            )
            continue
        if args.clip_final_min_margin is not None and margin < float(args.clip_final_min_margin):
            report["skipped"].append(
                {"prediction_id": int(pred_id), "reason": "below_final_min_margin", "margin": margin}
            )
            continue

        if top_class != current_class:
            classes[pred_id] = top_class
            report["applied"].append(
                {
                    "prediction_id": int(pred_id),
                    "old_class_id": current_class,
                    "old_class_name": current_name,
                    "new_class_id": top_class,
                    "new_class_name": top_name,
                    "base_score": float(pred_scores[pred_id]),
                    "top_clip_score": top_score,
                    "margin": margin,
                    "sms_score": None if sms_score is None else float(sms_score),
                }
            )
        if args.clip_score_policy == "replace":
            scores[pred_id] = top_score
        elif args.clip_score_policy == "boost":
            scores[pred_id] = max(float(scores[pred_id]), top_score)
        elif args.clip_score_policy == "blend":
            scores[pred_id] = float(args.clip_score_alpha) * float(scores[pred_id]) + (1.0 - float(args.clip_score_alpha)) * top_score
    return classes, scores, report, keep_overrides


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


def _apply_clip_corrections(scene_name, pred_classes, pred_scores, feature_records, labels, args):
    classes = pred_classes.copy()
    scores = pred_scores.copy()
    blocked = _parse_class_filter(args.clip_blocked_classes)
    allowed = _parse_class_filter(args.clip_allowed_classes)
    allowed_pairs = _parse_pair_filter(args.clip_allowed_pairs)
    pair_rules = _parse_pair_rules(args.clip_pair_rules)
    confusion_groups = _parse_confusion_groups(args.clip_confusion_groups)
    report = {"loaded": len(feature_records), "applied": [], "skipped": []}
    for pred_id in range(len(classes)):
        record = feature_records.get(int(pred_id))
        if not record or not record.get("clip_probs"):
            continue
        if args.clip_feature_max_base_score is not None and float(record.get("pred_score", pred_scores[pred_id])) > float(
            args.clip_feature_max_base_score
        ):
            report["skipped"].append(
                {
                    "prediction_id": pred_id,
                    "reason": "feature_high_base_score",
                    "base_score": float(record.get("pred_score", pred_scores[pred_id])),
                }
            )
            continue
        probs = np.asarray(record["clip_probs"], dtype=np.float32)
        if len(probs) == 0:
            continue
        order = np.argsort(-probs)
        top_class = int(order[0])
        second_score = float(probs[order[1]]) if len(order) > 1 else 0.0
        top_score = float(probs[top_class])
        margin = top_score - second_score
        current_class = int(classes[pred_id])
        current_name = labels[current_class] if 0 <= current_class < len(labels) else str(current_class)
        top_name = labels[top_class] if 0 <= top_class < len(labels) else str(top_class)
        current_clip_score = float(probs[current_class]) if 0 <= current_class < len(probs) else 0.0

        if top_class == current_class:
            continue
        if allowed is not None and top_name not in allowed:
            report["skipped"].append({"prediction_id": pred_id, "reason": "class_not_allowed", "top_class": top_name})
            continue
        if allowed_pairs is not None and (current_name, top_name) not in allowed_pairs:
            report["skipped"].append(
                {
                    "prediction_id": pred_id,
                    "reason": "pair_not_allowed",
                    "old_class": current_name,
                    "top_class": top_name,
                }
            )
            continue
        if confusion_groups is not None and not _same_confusion_group(confusion_groups, current_name, top_name):
            report["skipped"].append(
                {
                    "prediction_id": pred_id,
                    "reason": "outside_confusion_group",
                    "old_class": current_name,
                    "top_class": top_name,
                }
            )
            continue
        pair_rule = pair_rules.get((current_name, top_name)) if pair_rules is not None else None
        if pair_rules is not None and pair_rule is None:
            report["skipped"].append(
                {
                    "prediction_id": pred_id,
                    "reason": "pair_rule_missing",
                    "old_class": current_name,
                    "top_class": top_name,
                }
            )
            continue
        min_confidence = args.clip_min_confidence if pair_rule is None else pair_rule["confidence"]
        min_margin = args.clip_min_margin if pair_rule is None else pair_rule["margin"]
        min_gain = args.clip_min_gain_over_current if pair_rule is None else pair_rule["gain"]
        max_base_score = args.clip_max_base_score if pair_rule is None else pair_rule["max_base_score"]
        if blocked is not None and top_name in blocked:
            report["skipped"].append({"prediction_id": pred_id, "reason": "class_blocked", "top_class": top_name})
            continue
        if float(pred_scores[pred_id]) > max_base_score:
            report["skipped"].append({"prediction_id": pred_id, "reason": "high_base_score", "base_score": float(pred_scores[pred_id])})
            continue
        if top_score < min_confidence:
            continue
        if margin < min_margin:
            continue
        if min_gain is not None and (top_score - current_clip_score) < min_gain:
            continue

        classes[pred_id] = top_class
        if args.clip_score_policy == "boost":
            scores[pred_id] = max(float(scores[pred_id]), top_score)
        elif args.clip_score_policy == "blend":
            scores[pred_id] = float(args.clip_score_alpha) * float(scores[pred_id]) + (1.0 - float(args.clip_score_alpha)) * top_score
        report["applied"].append(
            {
                "prediction_id": int(pred_id),
                "old_class_id": current_class,
                "old_class_name": current_name,
                "new_class_id": top_class,
                "new_class_name": top_name,
                "base_score": float(pred_scores[pred_id]),
                "top_clip_score": top_score,
                "current_clip_score": current_clip_score,
                "margin": margin,
            }
        )
    return classes, scores, report, {}


def evaluate_with_clip_correction(args):
    config = _load_yaml(osp.join("./pretrained", f"config_{args.dataset_name}.yaml"))
    labels = config["network2d"]["text_prompts"]
    depth_scale = config["openyolo3d"]["depth_scale"]
    path_2_dataset = osp.join("./data", args.dataset_name)
    gt_dir = osp.join("./data", args.dataset_name, "ground_truth")
    datatype = "point cloud" if args.dataset_name == "replica" else "mesh"
    if args.scene_names:
        scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    elif args.dataset_name == "replica":
        scene_names = SCENE_NAMES_REPLICA
    else:
        scene_names = SCENE_NAMES_SCANNET200

    candidates_by_scene, candidate_summary = load_backprojection_candidates(args.backprojection_candidates)
    clip_features, clip_summary = load_multiview_clip_features(args.multiview_clip_features)
    print(f"[INFO] Loaded multiview object CLIP features: {clip_summary['loaded']} records from {len(clip_summary['files'])} files.")

    openyolo3d = OpenYolo3D(f"./pretrained/config_{args.dataset_name}.yaml")
    preds = {}
    reports = {}
    for scene_name in tqdm(scene_names):
        scene_id = scene_name.replace("scene", "")
        processed_file = (
            osp.join(path_2_dataset, scene_name, f"{scene_id}.npy")
            if args.dataset_name == "scannet200"
            else None
        )
        prediction = openyolo3d.predict(
            path_2_scene_data=osp.join(path_2_dataset, scene_name),
            depth_scale=depth_scale,
            datatype=datatype,
            processed_scene=processed_file,
            path_to_3d_masks=args.path_to_3d_masks,
            is_gt=args.is_gt,
            path_to_2d_preds=args.path_to_2d_preds,
            save_2d_preds=args.save_2d_preds,
            reuse_2d_preds=args.reuse_2d_preds,
        )
        scene_prediction = prediction[scene_name]
        if args.backprojection_candidates is not None:
            points_xyz, _ = openyolo3d.world2cam.load_ply(openyolo3d.world2cam.mesh)
            point_segments = None
            point_visibility = None
            superpoint_box_context = None
            if args.backprojection_superpoint_refine:
                if processed_file is None:
                    raise ValueError("--backprojection_superpoint_refine requires a ScanNet200 processed scene file.")
                point_segments = np.load(processed_file, mmap_mode="r")[:, 9].astype(np.int64)
                projections_for_sp, point_visibility = openyolo3d.mesh_projections
                if float(args.backprojection_superpoint_min_box_positive_ratio or 0.0) > 0.0:
                    superpoint_box_context = {
                        "projections": projections_for_sp,
                        "scaling_params": openyolo3d.scaling_params,
                    }
            fused = append_backprojection_proposals(
                scene_name,
                scene_prediction[0],
                scene_prediction[1],
                scene_prediction[2],
                candidates_by_scene,
                points_xyz=points_xyz[:, :3],
                point_segments=point_segments,
                point_visibility=point_visibility,
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
                cc_cleanup=args.backprojection_cc_cleanup,
                cc_radius=args.backprojection_cc_radius,
                cc_min_component_points=args.backprojection_cc_min_component_points,
                cc_keep_topk=args.backprojection_cc_keep_topk,
                source_priorities=args.backprojection_source_priorities,
                source_max_candidates=args.backprojection_source_max_candidates,
                source_score_scales=args.backprojection_source_score_scales,
                superpoint_refine=args.backprojection_superpoint_refine,
                superpoint_min_coverage=args.backprojection_superpoint_min_coverage,
                superpoint_max_expansion_ratio=args.backprojection_superpoint_max_expansion_ratio,
                superpoint_min_view_siou=args.backprojection_superpoint_min_view_siou,
                superpoint_box_context=superpoint_box_context,
                superpoint_min_box_positive_ratio=args.backprojection_superpoint_min_box_positive_ratio,
                superpoint_max_box_negative_ratio=args.backprojection_superpoint_max_box_negative_ratio,
                superpoint_box_min_visible_points=args.backprojection_superpoint_box_min_visible_points,
                superpoint_box_min_views=args.backprojection_superpoint_box_min_views,
            )
            scene_prediction = fused[:3]
        pred_masks = _to_numpy(scene_prediction[0]).astype(bool)
        pred_classes = _to_numpy(scene_prediction[1]).astype(np.int64)
        pred_scores = _to_numpy(scene_prediction[2]).astype(np.float32)

        if args.clip_application_mode == "final_classifier":
            pred_classes, corrected_scores, report, keep_overrides = _apply_final_classifier(
                scene_name,
                pred_classes,
                pred_scores,
                clip_features.get(scene_name, {}),
                labels,
                args,
            )
        else:
            pred_classes, corrected_scores, report, keep_overrides = _apply_clip_corrections(
                scene_name,
                pred_classes,
                pred_scores,
                clip_features.get(scene_name, {}),
                labels,
                args,
            )
        reports[scene_name] = report

        keep = pred_scores >= args.score_threshold
        for pred_id, keep_value in keep_overrides.items():
            if 0 <= int(pred_id) < len(keep):
                keep[int(pred_id)] = bool(keep_value)
        if keep.sum() == 0 and args.keep_one_if_empty and len(pred_scores) > 0:
            keep[int(pred_scores.argmax())] = True
        if args.base_eval_score_mode == "baseline":
            eval_scores = np.ones_like(pred_scores, dtype=np.float32)
            num_added = len(pred_scores) - int(_to_numpy(prediction[scene_name][0]).shape[1])
            if num_added > 0:
                eval_scores[-num_added:] = corrected_scores[-num_added:]
        else:
            eval_scores = corrected_scores
        preds[scene_name] = {
            "pred_masks": pred_masks[:, keep],
            "pred_scores": eval_scores[keep],
            "pred_classes": pred_classes[keep],
        }
        _clear_openyolo_state(openyolo3d, unload_3d_network=args.path_to_3d_masks is None)

    output_dir = osp.dirname(args.eval_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.dataset_name == "replica":
        inst_ap = evaluate_replica(preds, gt_dir, output_file=args.eval_output_file, dataset=args.dataset_name)
    else:
        inst_ap = evaluate_scannet200(preds, gt_dir, output_file=args.eval_output_file, dataset=args.dataset_name)
    if args.report_path:
        report_dir = osp.dirname(args.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_path, "w") as f:
            json.dump(
                {
                    "candidate_summary": candidate_summary,
                    "clip_summary": clip_summary,
                    "params": vars(args),
                    "scene_reports": reports,
                    "inst_ap": inst_ap,
                },
                f,
                indent=2,
            )
        print(f"Saved correction report to {args.report_path}")
    return inst_ap


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--is_gt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--save_2d_preds", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--score_threshold", default=0.20, type=float)
    parser.add_argument("--keep_one_if_empty", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--base_eval_score_mode", default="baseline", choices=["baseline", "openyolo"])
    parser.add_argument("--multiview_clip_features", required=True)
    parser.add_argument("--eval_output_file", default="./output/multiview_clip_correction_eval/eval.csv")
    parser.add_argument("--report_path", default="./output/multiview_clip_correction_eval/report.json")

    parser.add_argument("--clip_application_mode", default="correction", choices=["correction", "final_classifier"])
    parser.add_argument("--clip_decision_score_mode", default="probs", choices=["probs", "logits", "similarities"])
    parser.add_argument("--clip_apply_source_kinds", default=None)
    parser.add_argument("--clip_filter_policy", default="none", choices=["none", "sms"])
    parser.add_argument("--clip_filter_source_kinds", default=None)
    parser.add_argument("--clip_filter_missing_features", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--clip_sms_scope", default="class", choices=["scene", "class"])
    parser.add_argument("--clip_sms_threshold", default=-1.0, type=float)
    parser.add_argument("--clip_sms_min_class_count", default=5, type=int)
    parser.add_argument("--clip_yolo_prior_weight", default=0.0, type=float)
    parser.add_argument("--clip_final_min_score", default=None, type=float)
    parser.add_argument("--clip_final_min_margin", default=None, type=float)
    parser.add_argument("--clip_min_confidence", default=0.60, type=float)
    parser.add_argument("--clip_min_margin", default=0.10, type=float)
    parser.add_argument("--clip_min_gain_over_current", default=0.10, type=float)
    parser.add_argument("--clip_max_base_score", default=1.10, type=float)
    parser.add_argument(
        "--clip_feature_max_base_score",
        default=None,
        type=float,
        help="Ignore cached CLIP features whose original prediction score is above this value. Used to simulate selective export.",
    )
    parser.add_argument("--clip_allowed_classes", default=None)
    parser.add_argument("--clip_allowed_pairs", default=None, help="Comma-separated old->new class pairs, e.g. desk-organizer->shelf")
    parser.add_argument(
        "--clip_confusion_groups",
        default=None,
        help="Semicolon-separated class groups. Corrections are allowed only within a group, e.g. pillow,cushion,blanket;shelf,cabinet",
    )
    parser.add_argument(
        "--clip_pair_rules",
        default=None,
        help="Semicolon-separated old->new:confidence:margin:gain:max_base_score rules",
    )
    parser.add_argument("--clip_blocked_classes", default="rug")
    parser.add_argument("--clip_score_policy", default="keep", choices=["keep", "replace", "boost", "blend"])
    parser.add_argument("--clip_score_alpha", default=0.50, type=float)

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
    parser.add_argument("--backprojection_cc_cleanup", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_cc_radius", default=0.03, type=float)
    parser.add_argument("--backprojection_cc_min_component_points", default=50, type=int)
    parser.add_argument("--backprojection_cc_keep_topk", default=1, type=int)
    parser.add_argument("--backprojection_source_priorities", default=None)
    parser.add_argument("--backprojection_source_max_candidates", default=None)
    parser.add_argument("--backprojection_source_score_scales", default=None)
    parser.add_argument("--backprojection_superpoint_refine", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_superpoint_min_coverage", default=0.30, type=float)
    parser.add_argument("--backprojection_superpoint_max_expansion_ratio", default=2.0, type=float)
    parser.add_argument("--backprojection_superpoint_min_view_siou", default=0.0, type=float)
    parser.add_argument("--backprojection_superpoint_min_box_positive_ratio", default=0.0, type=float)
    parser.add_argument("--backprojection_superpoint_max_box_negative_ratio", default=1.0, type=float)
    parser.add_argument("--backprojection_superpoint_box_min_visible_points", default=5, type=int)
    parser.add_argument("--backprojection_superpoint_box_min_views", default=1, type=int)
    return parser


if __name__ == "__main__":
    evaluate_with_clip_correction(build_parser().parse_args())
