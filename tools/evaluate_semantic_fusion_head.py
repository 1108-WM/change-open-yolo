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
TOOLS_ROOT = osp.dirname(osp.abspath(__file__))
if TOOLS_ROOT not in sys.path:
    sys.path.insert(0, TOOLS_ROOT)

from evaluate import SCENE_NAMES_REPLICA, evaluate_replica
from export_semantic_fusion_dataset import (
    _candidate_evidence_features,
    _mask_geometry_features,
    _pred_class_name,
    _proposal_features,
    _to_numpy,
)
from train_semantic_fusion_head import FusionHead, _build_feature_matrix
from utils import OpenYolo3D
from utils.backprojection_fusion import append_backprojection_proposals, load_backprojection_candidates


def _load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


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


def _load_head(path, device):
    payload = torch.load(path, map_location="cpu")
    model = FusionHead(
        len(payload["feature_names"]),
        hidden_dim=int(payload.get("hidden_dim", 0)),
        dropout=float(payload.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def _score_records(records, model, payload, device):
    x = _build_feature_matrix(
        records,
        payload["numeric_feature_names"],
        payload["source_values"],
        int(payload.get("num_classes", 49)),
    )
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x).to(device))).detach().cpu().numpy()
    return scores.astype(np.float32)


def _build_inference_records(
    scene_name,
    pred_masks,
    pred_classes,
    pred_scores,
    labels,
    original_count,
    applied_records,
    points_xyz,
    scene_candidates,
    args,
):
    records = []
    for pred_id in range(pred_masks.shape[1]):
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

        mask = pred_masks[:, pred_id]
        pred_class_id = int(pred_classes[pred_id])
        features = {
            "base_score": float(pred_scores[pred_id]),
            "keep_after_score_threshold": bool(float(pred_scores[pred_id]) >= args.keep_score_threshold),
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
        records.append(
            {
                "scene_name": scene_name,
                "prediction_id": int(pred_id),
                "source_kind": source_kind,
                "source_name": source_name,
                "pred_class_id": pred_class_id,
                "pred_class_name": _pred_class_name(pred_class_id, labels),
                "features": features,
            }
        )
    return records


def _parse_sources(value):
    if value is None:
        return None
    parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def _apply_score_policy(base_scores, head_scores, source_kinds, policy, blend_alpha, rescore_sources=None):
    base_scores = np.asarray(base_scores, dtype=np.float32)
    head_scores = np.asarray(head_scores, dtype=np.float32)
    output = base_scores.copy()
    rescore_sources = _parse_sources(rescore_sources)
    if rescore_sources is None:
        rescore_mask = np.ones_like(base_scores, dtype=bool)
    else:
        rescore_mask = np.asarray([source in rescore_sources for source in source_kinds], dtype=bool)

    if policy == "replace":
        rescored = head_scores
    elif policy == "blend":
        rescored = np.clip(float(blend_alpha) * base_scores + (1.0 - float(blend_alpha)) * head_scores, 0.0, 1.0)
    elif policy == "multiply":
        rescored = np.clip(base_scores * head_scores, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported score policy: {policy}")

    output[rescore_mask] = rescored[rescore_mask]
    return output


def evaluate_with_head(args):
    if args.dataset_name != "replica":
        raise NotImplementedError("Only Replica is supported for this semantic fusion head evaluator.")

    config = _load_yaml(osp.join("./pretrained", f"config_{args.dataset_name}.yaml"))
    labels = config["network2d"]["text_prompts"]
    depth_scale = config["openyolo3d"]["depth_scale"]
    path_2_dataset = osp.join("./data", args.dataset_name)
    gt_dir = osp.join("./data", args.dataset_name, "ground_truth")
    datatype = "point cloud"

    scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()] if args.scene_names else SCENE_NAMES_REPLICA
    candidates_by_scene, candidate_summary = load_backprojection_candidates(args.backprojection_candidates)
    device = torch.device(args.device)
    model, payload = _load_head(args.semantic_fusion_head, device)

    openyolo3d = OpenYolo3D(f"./pretrained/config_{args.dataset_name}.yaml")
    preds = {}
    report = {
        "semantic_fusion_head": args.semantic_fusion_head,
        "candidate_summary": candidate_summary,
        "params": vars(args),
        "scenes": {},
    }

    for scene_name in tqdm(scene_names):
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
        records = _build_inference_records(
            scene_name,
            pred_masks,
            pred_classes,
            pred_scores,
            labels,
            original_count,
            fusion_report.get("applied", []),
            points_xyz,
            candidates_by_scene.get(scene_name, []),
            args,
        )
        head_scores = _score_records(records, model, payload, device)
        source_kinds = [record["source_kind"] for record in records]
        if args.base_eval_score_mode == "baseline":
            base_eval_scores = pred_scores.copy()
            base_eval_scores[np.asarray([source == "mask3d" for source in source_kinds], dtype=bool)] = 1.0
        else:
            base_eval_scores = pred_scores
        eval_scores = _apply_score_policy(
            base_eval_scores,
            head_scores,
            source_kinds,
            args.score_policy,
            args.blend_alpha,
            rescore_sources=args.rescore_sources,
        )

        keep = pred_scores >= args.keep_score_threshold
        if args.keep_with_head_score:
            keep = eval_scores >= args.keep_score_threshold
        if keep.sum() == 0 and args.keep_one_if_empty and len(eval_scores) > 0:
            keep[int(eval_scores.argmax())] = True

        preds[scene_name] = {
            "pred_masks": pred_masks[:, keep],
            "pred_scores": eval_scores[keep],
            "pred_classes": pred_classes[keep],
        }
        report["scenes"][scene_name] = {
            "num_original_predictions": original_count,
            "num_fused_predictions": int(pred_masks.shape[1]),
            "num_kept": int(keep.sum()),
            "fusion_loaded": int(fusion_report.get("loaded", 0)),
            "fusion_applied": int(len(fusion_report.get("applied", []))),
            "head_score_mean": float(head_scores.mean()) if len(head_scores) else 0.0,
            "head_score_max": float(head_scores.max()) if len(head_scores) else 0.0,
            "eval_score_mean": float(eval_scores.mean()) if len(eval_scores) else 0.0,
            "eval_score_max": float(eval_scores.max()) if len(eval_scores) else 0.0,
        }
        _clear_openyolo_state(openyolo3d, unload_3d_network=args.path_to_3d_masks is None)

    output_dir = osp.dirname(args.eval_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    inst_ap = evaluate_replica(preds, gt_dir, output_file=args.eval_output_file, dataset=args.dataset_name)
    report["inst_ap"] = inst_ap
    if args.report_path:
        report_dir = osp.dirname(args.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved semantic fusion eval report to {args.report_path}")
    return inst_ap


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica"])
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--is_gt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--save_2d_preds", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--semantic_fusion_head", required=True)
    parser.add_argument("--score_policy", default="replace", choices=["replace", "blend", "multiply"])
    parser.add_argument("--blend_alpha", default=0.35, type=float)
    parser.add_argument("--base_eval_score_mode", default="baseline", choices=["baseline", "openyolo"])
    parser.add_argument("--rescore_sources", default=None, help="Comma-separated source kinds to rescore; default: all")
    parser.add_argument("--keep_score_threshold", default=0.20, type=float)
    parser.add_argument("--keep_one_if_empty", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--keep_with_head_score", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_output_file", default="./output/semantic_fusion_head_eval/replica_eval.csv")
    parser.add_argument("--report_path", default="./output/semantic_fusion_head_eval/report.json")
    parser.add_argument("--device", default="cpu")

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
    evaluate_with_head(build_parser().parse_args())
