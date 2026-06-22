import argparse
import contextlib
import gc
import io
import json
import os
import os.path as osp
import sys
from itertools import product

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
from evaluate_multiview_object_clip_correction import _apply_clip_corrections, load_multiview_clip_features


def _load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _parse_list(value, cast=str):
    if value is None:
        return []
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


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


def _build_base_predictions(args, labels, scene_names, candidates_by_scene):
    config = _load_yaml(osp.join("./pretrained", f"config_{args.dataset_name}.yaml"))
    depth_scale = config["openyolo3d"]["depth_scale"]
    path_2_dataset = osp.join("./data", args.dataset_name)
    datatype = "point cloud" if args.dataset_name == "replica" else "mesh"
    openyolo3d = OpenYolo3D(f"./pretrained/config_{args.dataset_name}.yaml")
    base = {}
    for scene_name in tqdm(scene_names, desc="build_predictions"):
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
        original_count = int(_to_numpy(scene_prediction[0]).shape[1])
        if args.backprojection_candidates is not None:
            points_xyz, _ = openyolo3d.world2cam.load_ply(openyolo3d.world2cam.mesh)
            fused = append_backprojection_proposals(
                scene_name,
                scene_prediction[0],
                scene_prediction[1],
                scene_prediction[2],
                candidates_by_scene,
                points_xyz=points_xyz[:, :3],
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
                source_score_scales=args.backprojection_source_score_scales,
            )
            scene_prediction = fused[:3]
        base[scene_name] = {
            "pred_masks": _to_numpy(scene_prediction[0]).astype(bool),
            "pred_classes": _to_numpy(scene_prediction[1]).astype(np.int64),
            "pred_scores": _to_numpy(scene_prediction[2]).astype(np.float32),
            "original_count": original_count,
        }
        _clear_openyolo_state(openyolo3d, unload_3d_network=args.path_to_3d_masks is None)
    return base


def _make_eval_args(args, confidence, margin, gain):
    return argparse.Namespace(
        clip_blocked_classes=args.clip_blocked_classes,
        clip_allowed_classes=args.clip_allowed_classes,
        clip_allowed_pairs=args.clip_allowed_pairs,
        clip_confusion_groups=args.clip_confusion_groups,
        clip_pair_rules=None,
        clip_min_confidence=float(confidence),
        clip_min_margin=float(margin),
        clip_min_gain_over_current=float(gain),
        clip_max_base_score=float(args.clip_max_base_score),
        clip_feature_max_base_score=args.clip_feature_max_base_score,
        clip_score_policy=args.clip_score_policy,
        clip_score_alpha=float(args.clip_score_alpha),
    )


def _build_corrected_preds(
    base_predictions,
    clip_features,
    labels,
    eval_args,
    score_threshold,
    base_eval_score_mode,
    keep_one_if_empty=False,
):
    preds = {}
    reports = {}
    for scene_name, base in base_predictions.items():
        pred_masks = base["pred_masks"]
        pred_classes = base["pred_classes"].copy()
        pred_scores = base["pred_scores"].copy()
        corrected_classes, corrected_scores, report = _apply_clip_corrections(
            scene_name,
            pred_classes,
            pred_scores,
            clip_features.get(scene_name, {}),
            labels,
            eval_args,
        )
        reports[scene_name] = report
        keep = pred_scores >= score_threshold
        if keep.sum() == 0 and keep_one_if_empty and len(pred_scores) > 0:
            keep[int(pred_scores.argmax())] = True
        if base_eval_score_mode == "baseline":
            eval_scores = np.ones_like(pred_scores, dtype=np.float32)
            num_added = len(pred_scores) - int(base["original_count"])
            if num_added > 0:
                eval_scores[-num_added:] = corrected_scores[-num_added:]
        else:
            eval_scores = corrected_scores
        preds[scene_name] = {
            "pred_masks": pred_masks[:, keep],
            "pred_scores": eval_scores[keep],
            "pred_classes": corrected_classes[keep],
        }
    return preds, reports


def _evaluate_subset(preds, scene_names, gt_dir, dataset_name):
    subset = {scene: preds[scene] for scene in scene_names if scene in preds}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        if dataset_name == "replica":
            return evaluate_replica(subset, gt_dir, output_file=None, dataset=dataset_name)
        return evaluate_scannet200(subset, gt_dir, output_file=None, dataset=dataset_name)


def search_thresholds(args):
    config = _load_yaml(osp.join("./pretrained", f"config_{args.dataset_name}.yaml"))
    labels = config["network2d"]["text_prompts"]
    gt_dir = osp.join("./data", args.dataset_name, "ground_truth")
    if args.scene_names:
        scene_names = _parse_list(args.scene_names)
    elif args.dataset_name == "replica":
        scene_names = SCENE_NAMES_REPLICA
    else:
        scene_names = SCENE_NAMES_SCANNET200
    train_scenes = _parse_list(args.train_scenes)
    val_scenes = _parse_list(args.val_scenes)
    if not train_scenes or not val_scenes:
        midpoint = max(1, len(scene_names) // 2)
        train_scenes = scene_names[:midpoint]
        val_scenes = scene_names[midpoint:]

    candidates_by_scene, candidate_summary = load_backprojection_candidates(args.backprojection_candidates)
    clip_features, clip_summary = load_multiview_clip_features(args.multiview_clip_features)
    print(f"[INFO] Loaded CLIP features: {clip_summary['loaded']} records from {len(clip_summary['files'])} files.")
    base_predictions = _build_base_predictions(args, labels, scene_names, candidates_by_scene)

    confidences = _parse_list(args.confidences, float)
    margins = _parse_list(args.margins, float)
    gains = _parse_list(args.gains, float)
    rows = []
    for confidence, margin, gain in tqdm(list(product(confidences, margins, gains)), desc="threshold_grid"):
        eval_args = _make_eval_args(args, confidence, margin, gain)
        preds, reports = _build_corrected_preds(
            base_predictions,
            clip_features,
            labels,
            eval_args,
            args.score_threshold,
            args.base_eval_score_mode,
            args.keep_one_if_empty,
        )
        train_ap = _evaluate_subset(preds, train_scenes, gt_dir, args.dataset_name)
        val_ap = _evaluate_subset(preds, val_scenes, gt_dir, args.dataset_name)
        all_ap = _evaluate_subset(preds, scene_names, gt_dir, args.dataset_name)
        applied = sum(len(report.get("applied", [])) for report in reports.values())
        rows.append(
            {
                "confidence": confidence,
                "margin": margin,
                "gain": gain,
                "applied": applied,
                "train": {
                    "ap": train_ap["all_ap"],
                    "ap50": train_ap["all_ap_50%"],
                    "ap25": train_ap["all_ap_25%"],
                },
                "val": {
                    "ap": val_ap["all_ap"],
                    "ap50": val_ap["all_ap_50%"],
                    "ap25": val_ap["all_ap_25%"],
                },
                "all": {
                    "ap": all_ap["all_ap"],
                    "ap50": all_ap["all_ap_50%"],
                    "ap25": all_ap["all_ap_25%"],
                },
            }
        )
    rows = sorted(rows, key=lambda row: (-row["train"][args.selection_metric], -row["val"][args.selection_metric]))
    os.makedirs(osp.dirname(args.report_path), exist_ok=True)
    payload = {
        "dataset_name": args.dataset_name,
        "scene_names": scene_names,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "candidate_summary": candidate_summary,
        "clip_summary": clip_summary,
        "params": vars(args),
        "best_by_train": rows[0] if rows else None,
        "rows": rows,
    }
    with open(args.report_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Best by train {args.selection_metric}: {payload['best_by_train']}")
    print(f"Saved threshold search report to {args.report_path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--train_scenes", default="office0,office1,office2,room0")
    parser.add_argument("--val_scenes", default="office3,office4,room1,room2")
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--is_gt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--save_2d_preds", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--score_threshold", default=0.20, type=float)
    parser.add_argument("--keep_one_if_empty", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--base_eval_score_mode", default="baseline", choices=["baseline", "openyolo"])
    parser.add_argument("--multiview_clip_features", required=True)
    parser.add_argument("--report_path", default="./output/multiview_clip_correction_eval/search_alphaclip_thresholds.json")
    parser.add_argument("--selection_metric", default="ap", choices=["ap", "ap50", "ap25"])

    parser.add_argument("--confidences", default="0.60,0.70,0.75,0.80")
    parser.add_argument("--margins", default="0.10,0.20,0.30")
    parser.add_argument("--gains", default="0.10,0.20,0.30")
    parser.add_argument("--clip_max_base_score", default=1.10, type=float)
    parser.add_argument("--clip_feature_max_base_score", default=None, type=float)
    parser.add_argument("--clip_allowed_classes", default=None)
    parser.add_argument("--clip_allowed_pairs", default=None)
    parser.add_argument("--clip_confusion_groups", default=None)
    parser.add_argument("--clip_blocked_classes", default="rug")
    parser.add_argument("--clip_score_policy", default="keep", choices=["keep", "boost", "blend"])
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
    parser.add_argument("--backprojection_source_score_scales", default=None)
    return parser


if __name__ == "__main__":
    search_thresholds(build_parser().parse_args())
