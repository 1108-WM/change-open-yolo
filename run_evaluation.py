import torch
from tqdm import tqdm
import argparse
from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200, evaluate_scannet200, evaluate_replica
from utils import OpenYolo3D
from utils.backprojection_fusion import (
    append_backprojection_proposals,
    load_backprojection_candidates,
    load_backprojection_verifications,
)
from utils.context_fusion import apply_context_corrections, load_context_corrections
from utils.object_query_rescore import rescore_with_object_queries, save_object_query_report
from utils.clip_object_rescore import (
    load_clip_object_features,
    rescore_with_clip_object_features,
    save_clip_object_report,
)
import yaml
import os
import os.path as osp
import json
import gc
import shutil
import numpy as np

class InstSegEvaluator():
    def __init__(self, dataset_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type

    def evaluate_full(self, preds, scene_gt_dir, dataset, output_file='temp_output.txt', pretrained_on_scannet200=True):
        if dataset == "replica":
            inst_AP = evaluate_replica(preds, scene_gt_dir, output_file=output_file, dataset=dataset)
        elif dataset == "scannet200":
            inst_AP = evaluate_scannet200(preds, scene_gt_dir, output_file=output_file, dataset=dataset, pretrained_on_scannet200 = pretrained_on_scannet200)
        else:
            print("DATASET NOT SUPPORTED!")
            exit()
        return inst_AP

def test_pipeline_full(
    dataset_type,
    path_to_3d_masks,
    is_gt,
    use_pred_scores=False,
    score_threshold=0.0,
    path_to_2d_preds=None,
    save_2d_preds=False,
    reuse_2d_preds=True,
    backprojection_candidates=None,
    backprojection_min_score=0.35,
    backprojection_min_seed_points=80,
    backprojection_max_existing_iou=0.30,
    backprojection_max_seed_in_existing_mask_ratio=0.70,
    backprojection_max_proposal_iou=0.50,
    backprojection_max_candidates_per_scene=None,
    backprojection_score_scale=0.50,
    backprojection_use_candidate_fusion_score=True,
    backprojection_allowed_classes=None,
    backprojection_blocked_classes=None,
    backprojection_min_support_views=0,
    backprojection_min_support_mean_iou=0.0,
    backprojection_min_support_best_iou=0.0,
    backprojection_min_fusion_score=0.0,
    backprojection_max_box_area_ratio=None,
    backprojection_min_quality_score=0.0,
    backprojection_min_scene_source_quality_z=None,
    backprojection_quality_sort=False,
    backprojection_verifications=None,
    backprojection_verifier_min_confidence=0.0,
    backprojection_verifier_suppress_decisions="suppress,bad_mask,invalid,reject",
    backprojection_verifier_strict=False,
    backprojection_grow_radius=0.0,
    backprojection_max_growth_ratio=4.0,
    backprojection_seed_cc_cleanup=False,
    backprojection_seed_cc_radius=0.03,
    backprojection_seed_cc_min_component_points=30,
    backprojection_seed_cc_keep_topk=1,
    backprojection_seed_cc_max_points=30000,
    backprojection_seed_cc_min_keep_ratio=0.0,
    backprojection_cc_cleanup=False,
    backprojection_cc_radius=0.03,
    backprojection_cc_min_component_points=50,
    backprojection_cc_keep_topk=1,
    backprojection_cc_max_points=30000,
    backprojection_cc_split_components=False,
    backprojection_cc_source_kinds=None,
    backprojection_cc_min_keep_ratio=0.0,
    backprojection_cc_keep_ratio_score_weight=0.0,
    backprojection_source_priorities=None,
    backprojection_source_max_candidates=None,
    backprojection_source_score_scales=None,
    backprojection_source_min_scores=None,
    backprojection_max_candidates_per_class=None,
    backprojection_class_max_candidates=None,
    backprojection_quality_calibration_weight=0.0,
    backprojection_novelty_calibration_weight=0.0,
    backprojection_label_consensus_calibration_weight=0.0,
    backprojection_score_calibration_min=0.2,
    backprojection_score_calibration_max=1.2,
    backprojection_max_proposal_score=None,
    backprojection_min_label_consensus_score=0.0,
    backprojection_max_label_conflict_score=1.0,
    backprojection_label_consensus_iou_threshold=0.25,
    backprojection_label_consensus_min_visible_points=30,
    backprojection_label_consensus_frame_mode="support",
    backprojection_projection_consistency_min_box_iou=0.0,
    backprojection_projection_consistency_min_point_ratio=0.0,
    backprojection_projection_consistency_min_views=1,
    backprojection_projection_consistency_min_visible_points=30,
    backprojection_projection_consistency_frame_mode="support",
    backprojection_projection_consistency_box_padding_ratio=0.05,
    backprojection_projection_consistency_score_weight=0.0,
    backprojection_superpoint_refine=False,
    backprojection_superpoint_min_coverage=0.30,
    backprojection_superpoint_max_expansion_ratio=2.0,
    backprojection_superpoint_max_segment_ratio=None,
    backprojection_superpoint_large_segment_min_coverage=None,
    backprojection_superpoint_min_seed_retention=0.0,
    backprojection_superpoint_min_support_views=0,
    backprojection_superpoint_min_support_ratio=0.0,
    backprojection_superpoint_min_view_siou=0.0,
    backprojection_superpoint_view_siou_min_views=2,
    backprojection_superpoint_view_siou_min_visible_points=1,
    backprojection_superpoint_min_box_positive_ratio=0.0,
    backprojection_superpoint_max_box_negative_ratio=1.0,
    backprojection_superpoint_box_min_visible_points=5,
    backprojection_superpoint_box_min_views=1,
    backprojection_superpoint_box_padding_ratio=0.05,
    backprojection_local_superpoint_refine=False,
    backprojection_local_superpoint_knn=10,
    backprojection_local_superpoint_merge_k=0.25,
    backprojection_local_superpoint_min_size=10,
    backprojection_local_superpoint_min_coverage=0.25,
    backprojection_local_superpoint_max_expansion_ratio=1.0,
    backprojection_local_superpoint_min_seed_retention=0.80,
    backprojection_local_superpoint_max_points=30000,
    backprojection_merge_iou=0.0,
    backprojection_inclusion_threshold=0.0,
    backprojection_postprocess_same_class_only=True,
    backprojection_containment_action="none",
    backprojection_containment_threshold=0.85,
    backprojection_containment_min_area_ratio=1.5,
    backprojection_containment_score_ratio=0.75,
    backprojection_containment_quality_margin=0.0,
    backprojection_containment_score_factor=0.5,
    backprojection_containment_min_points=50,
    backprojection_report_path=None,
    object_query_rescore=False,
    object_query_candidates=None,
    object_query_min_candidate_score=0.30,
    object_query_min_seed_points=80,
    object_query_min_seed_overlap=0.35,
    object_query_min_support_views=2,
    object_query_support_scale=8.0,
    object_query_overlap_power=1.0,
    object_query_min_evidence=0.45,
    object_query_min_margin=0.10,
    object_query_max_base_score=1.01,
    object_query_score_alpha=0.50,
    object_query_use_candidate_fusion_score=True,
    object_query_allowed_classes=None,
    object_query_blocked_classes=None,
    object_query_report_path=None,
    clip_object_rescore=False,
    clip_object_features=None,
    clip_object_min_seed_points=80,
    clip_object_min_seed_overlap=0.45,
    clip_object_min_support_views=2,
    clip_object_support_scale=8.0,
    clip_object_topk_classes=5,
    clip_object_min_clip_prob=0.10,
    clip_object_min_evidence=0.35,
    clip_object_min_margin=0.12,
    clip_object_max_base_score=1.01,
    clip_object_score_alpha=0.50,
    clip_object_allowed_classes=None,
    clip_object_blocked_classes=None,
    clip_object_report_path=None,
    context_corrections=None,
    correction_min_confidence=0.0,
    correction_score_policy="keep",
    correction_score_blend_alpha=0.5,
    correction_apply_decisions="change,keep",
    correction_apply_min_confidence=None,
    correction_apply_min_score=None,
    correction_apply_max_score=None,
    correction_bad_mask_policy="skip",
    correction_bad_mask_score=0.0,
    correction_score_boost=0.0,
    correction_allowed_classes=None,
    correction_blocked_classes=None,
    correction_strict=False,
    correction_report_path=None,
    eval_output_file='temp_output.txt',
    scene_list=None,
    max_scenes=None,
    eval_prediction_cache_dir=None,
    eval_cleanup_prediction_cache=False,
    processed_scene_root=None,
):
    config = load_yaml(osp.join(f'./pretrained/config_{dataset_type}.yaml'))
    path_2_dataset = osp.join('./data', dataset_type)
    gt_dir = osp.join('./data', dataset_type, 'ground_truth')
    depth_scale = config["openyolo3d"]["depth_scale"]
    labels = config["network2d"]["text_prompts"]
    corrections, correction_summary = load_context_corrections(
        context_corrections,
        labels,
        min_confidence=correction_min_confidence,
        strict=correction_strict,
    )
    if context_corrections is not None:
        print(
            "[INFO] Loaded context corrections: "
            f"{correction_summary['used']} used / "
            f"{correction_summary['loaded']} records from "
            f"{len(correction_summary['files'])} file(s)."
    )
    bpr_candidates, bpr_summary = load_backprojection_candidates(backprojection_candidates)
    bpr_reports = {}
    bpr_added_counts = {}
    if backprojection_candidates is not None:
        print(
            "[INFO] Loaded back-projection candidates: "
            f"{bpr_summary['loaded']} candidates from "
            f"{len(bpr_summary['files'])} file(s)."
        )
    bpr_verifications, bpr_verification_summary = load_backprojection_verifications(
        backprojection_verifications,
        min_confidence=backprojection_verifier_min_confidence,
        strict=backprojection_verifier_strict,
    )
    if backprojection_verifications is not None:
        print(
            "[INFO] Loaded back-projection verifier decisions: "
            f"{bpr_verification_summary['used']} used / "
            f"{bpr_verification_summary['loaded']} records from "
            f"{len(bpr_verification_summary['files'])} file(s)."
        )
    object_candidates = {}
    object_summary = {"files": [], "loaded": 0}
    if object_query_rescore:
        if object_query_candidates is None and backprojection_candidates is not None:
            object_candidates = bpr_candidates
            object_summary = bpr_summary
        else:
            object_candidates, object_summary = load_backprojection_candidates(object_query_candidates)
        print(
            "[INFO] Loaded object-query candidates: "
            f"{object_summary['loaded']} candidates from "
            f"{len(object_summary['files'])} file(s)."
        )
    clip_features = {}
    clip_summary = {"files": [], "loaded": 0}
    if clip_object_rescore:
        clip_features, clip_summary = load_clip_object_features(clip_object_features)
        print(
            "[INFO] Loaded CLIP object features: "
            f"{clip_summary['loaded']} records from "
            f"{len(clip_summary['files'])} file(s)."
        )
    
    if dataset_type == "replica":
        scene_names = SCENE_NAMES_REPLICA
        datatype="point cloud"
    elif dataset_type == "scannet200":
        scene_names = SCENE_NAMES_SCANNET200
        datatype="mesh"

    if scene_list is not None:
        if osp.isfile(scene_list):
            with open(scene_list) as f:
                requested_scenes = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        else:
            requested_scenes = [scene.strip() for scene in scene_list.split(",") if scene.strip()]
        requested_set = set(requested_scenes)
        scene_names = [scene for scene in scene_names if scene in requested_set]
        missing_scenes = sorted(requested_set.difference(scene_names))
        if missing_scenes:
            print(f"[WARN] Ignoring {len(missing_scenes)} unknown scene(s): {missing_scenes[:5]}")
    if max_scenes is not None:
        scene_names = scene_names[:max_scenes]
    if len(scene_names) == 0:
        raise ValueError("No scenes selected for evaluation.")
    print(f"[INFO] Evaluating {len(scene_names)} scene(s).")
        
    evaluator = InstSegEvaluator(dataset_type)
    openyolo3d = OpenYolo3D(f"./pretrained/config_{dataset_type}.yaml")
    predictions = {}
    preds = {}
    correction_reports = {}
    object_query_reports = {}
    clip_object_reports = {}
    stream_eval_predictions = (
        not object_query_rescore
        and not clip_object_rescore
        and context_corrections is None
    )
    cached_eval_prediction_paths = {}
    if eval_prediction_cache_dir is not None:
        if not stream_eval_predictions:
            raise ValueError("--eval_prediction_cache_dir is only supported without object/CLIP/context rescoring.")
        os.makedirs(eval_prediction_cache_dir, exist_ok=True)

    def save_eval_prediction(scene_name, eval_prediction):
        scene_prefix = osp.join(eval_prediction_cache_dir, scene_name)
        paths = {
            "pred_masks": f"{scene_prefix}_pred_masks.npy",
            "pred_scores": f"{scene_prefix}_pred_scores.npy",
            "pred_classes": f"{scene_prefix}_pred_classes.npy",
        }
        for key, path in paths.items():
            np.save(path, eval_prediction[key])
        return paths

    def load_eval_prediction(paths):
        return {
            "pred_masks": np.load(paths["pred_masks"], mmap_mode="r"),
            "pred_scores": np.load(paths["pred_scores"], mmap_mode="r"),
            "pred_classes": np.load(paths["pred_classes"], mmap_mode="r"),
        }

    def build_eval_prediction(scene_name, scene_prediction, pred_classes=None, pred_scores=None):
        pred_masks = scene_prediction[0]
        if pred_classes is None:
            pred_classes = scene_prediction[1]
        if pred_scores is None:
            pred_scores = scene_prediction[2]

        keep = pred_scores >= score_threshold
        if keep.sum() == 0:
            keep[pred_scores.argmax()] = True
        if use_pred_scores:
            eval_scores = pred_scores
        else:
            eval_scores = torch.ones_like(torch.from_numpy(pred_scores)).numpy()
            if backprojection_candidates is not None:
                num_added = bpr_added_counts.get(
                    scene_name,
                    len(bpr_reports.get(scene_name, {}).get("applied", [])),
                )
                if num_added > 0:
                    eval_scores[-num_added:] = pred_scores[-num_added:]
        return {
            'pred_masks': pred_masks[:, keep].astype(bool, copy=False),
            'pred_scores': eval_scores[keep],
            'pred_classes': pred_classes[keep],
        }

    for scene_name in tqdm(scene_names):
        scene_id = scene_name.replace("scene", "")
        processed_scene_base = processed_scene_root or path_2_dataset
        processed_file = osp.join(processed_scene_base, scene_name, f"{scene_id}.npy") if dataset_type == "scannet200" else None
        prediction = openyolo3d.predict(path_2_scene_data = osp.join(path_2_dataset, scene_name), 
                                        depth_scale = depth_scale,
                                        datatype = datatype, 
                                        processed_scene = processed_file,
                                        path_to_3d_masks = path_to_3d_masks,
                                        is_gt = is_gt,
                                        path_to_2d_preds = path_to_2d_preds,
                                        save_2d_preds = save_2d_preds,
                                        reuse_2d_preds = reuse_2d_preds)
        scene_prediction = prediction[scene_name]
        points_xyz = None
        if backprojection_candidates is not None:
            points_xyz, _ = openyolo3d.world2cam.load_ply(openyolo3d.world2cam.mesh)
            point_segments = None
            point_visibility = None
            label_consensus_context = None
            projection_consistency_context = None
            superpoint_box_context = None
            if backprojection_superpoint_refine:
                if processed_file is None:
                    raise ValueError("--backprojection_superpoint_refine requires a ScanNet200 processed scene file.")
                point_segments = np.load(processed_file, mmap_mode="r")[:, 9].astype(np.int64)
                projections, point_visibility = openyolo3d.mesh_projections
                if float(backprojection_superpoint_min_box_positive_ratio or 0.0) > 0.0:
                    superpoint_box_context = {
                        "projections": projections,
                        "scaling_params": openyolo3d.scaling_params,
                    }
            if (
                backprojection_label_consensus_calibration_weight > 0.0
                or backprojection_min_label_consensus_score > 0.0
                or backprojection_max_label_conflict_score < 1.0
            ):
                projections, visibility = openyolo3d.mesh_projections
                point_visibility = visibility if point_visibility is None else point_visibility
                label_consensus_context = {
                    "projections": projections,
                    "point_visibility": visibility,
                    "preds_2d": openyolo3d.preds_2d,
                    "color_paths": openyolo3d.world2cam.color_paths,
                    "scaling_params": openyolo3d.scaling_params,
                }
            if (
                backprojection_projection_consistency_min_box_iou > 0.0
                or backprojection_projection_consistency_min_point_ratio > 0.0
                or backprojection_projection_consistency_score_weight > 0.0
            ):
                projections, visibility = openyolo3d.mesh_projections
                point_visibility = visibility if point_visibility is None else point_visibility
                projection_consistency_context = {
                    "projections": projections,
                    "point_visibility": visibility,
                    "preds_2d": openyolo3d.preds_2d,
                    "color_paths": openyolo3d.world2cam.color_paths,
                    "scaling_params": openyolo3d.scaling_params,
                }
            original_num_scene_predictions = int(scene_prediction[0].shape[1])
            scene_prediction = append_backprojection_proposals(
                scene_name,
                scene_prediction[0],
                scene_prediction[1],
                scene_prediction[2],
                bpr_candidates,
                points_xyz=points_xyz[:, :3],
                point_segments=point_segments,
                point_visibility=point_visibility,
                min_score=backprojection_min_score,
                min_seed_points=backprojection_min_seed_points,
                max_existing_iou=backprojection_max_existing_iou,
                max_seed_in_existing_mask_ratio=backprojection_max_seed_in_existing_mask_ratio,
                max_proposal_iou=backprojection_max_proposal_iou,
                max_candidates=backprojection_max_candidates_per_scene,
                score_scale=backprojection_score_scale,
                use_candidate_fusion_score=backprojection_use_candidate_fusion_score,
                allowed_classes=backprojection_allowed_classes,
                blocked_classes=backprojection_blocked_classes,
                min_support_views=backprojection_min_support_views,
                min_support_mean_iou=backprojection_min_support_mean_iou,
                min_support_best_iou=backprojection_min_support_best_iou,
                min_fusion_score=backprojection_min_fusion_score,
                max_box_area_ratio=backprojection_max_box_area_ratio,
                min_quality_score=backprojection_min_quality_score,
                min_scene_source_quality_z=backprojection_min_scene_source_quality_z,
                quality_sort=backprojection_quality_sort,
                verifications_by_scene=bpr_verifications,
                verifier_suppress_decisions=backprojection_verifier_suppress_decisions,
                grow_radius=backprojection_grow_radius,
                max_growth_ratio=backprojection_max_growth_ratio,
                seed_cc_cleanup=backprojection_seed_cc_cleanup,
                seed_cc_radius=backprojection_seed_cc_radius,
                seed_cc_min_component_points=backprojection_seed_cc_min_component_points,
                seed_cc_keep_topk=backprojection_seed_cc_keep_topk,
                seed_cc_max_points=backprojection_seed_cc_max_points,
                seed_cc_min_keep_ratio=backprojection_seed_cc_min_keep_ratio,
                cc_cleanup=backprojection_cc_cleanup,
                cc_radius=backprojection_cc_radius,
                cc_min_component_points=backprojection_cc_min_component_points,
                cc_keep_topk=backprojection_cc_keep_topk,
                cc_max_points=backprojection_cc_max_points,
                cc_split_components=backprojection_cc_split_components,
                cc_source_kinds=backprojection_cc_source_kinds,
                cc_min_keep_ratio=backprojection_cc_min_keep_ratio,
                cc_keep_ratio_score_weight=backprojection_cc_keep_ratio_score_weight,
                source_priorities=backprojection_source_priorities,
                source_max_candidates=backprojection_source_max_candidates,
                source_score_scales=backprojection_source_score_scales,
                source_min_scores=backprojection_source_min_scores,
                max_candidates_per_class=backprojection_max_candidates_per_class,
                class_max_candidates=backprojection_class_max_candidates,
                quality_calibration_weight=backprojection_quality_calibration_weight,
                novelty_calibration_weight=backprojection_novelty_calibration_weight,
                label_consensus_calibration_weight=backprojection_label_consensus_calibration_weight,
                score_calibration_min=backprojection_score_calibration_min,
                score_calibration_max=backprojection_score_calibration_max,
                max_proposal_score=backprojection_max_proposal_score,
                min_label_consensus_score=backprojection_min_label_consensus_score,
                max_label_conflict_score=backprojection_max_label_conflict_score,
                label_consensus_context=label_consensus_context,
                label_consensus_iou_threshold=backprojection_label_consensus_iou_threshold,
                label_consensus_min_visible_points=backprojection_label_consensus_min_visible_points,
                label_consensus_frame_mode=backprojection_label_consensus_frame_mode,
                projection_consistency_context=projection_consistency_context,
                projection_consistency_min_box_iou=backprojection_projection_consistency_min_box_iou,
                projection_consistency_min_point_ratio=backprojection_projection_consistency_min_point_ratio,
                projection_consistency_min_views=backprojection_projection_consistency_min_views,
                projection_consistency_min_visible_points=backprojection_projection_consistency_min_visible_points,
                projection_consistency_frame_mode=backprojection_projection_consistency_frame_mode,
                projection_consistency_box_padding_ratio=backprojection_projection_consistency_box_padding_ratio,
                projection_consistency_score_weight=backprojection_projection_consistency_score_weight,
                superpoint_refine=backprojection_superpoint_refine,
                superpoint_min_coverage=backprojection_superpoint_min_coverage,
                superpoint_max_expansion_ratio=backprojection_superpoint_max_expansion_ratio,
                superpoint_max_segment_ratio=backprojection_superpoint_max_segment_ratio,
                superpoint_large_segment_min_coverage=backprojection_superpoint_large_segment_min_coverage,
                superpoint_min_seed_retention=backprojection_superpoint_min_seed_retention,
                superpoint_min_support_views=backprojection_superpoint_min_support_views,
                superpoint_min_support_ratio=backprojection_superpoint_min_support_ratio,
                superpoint_min_view_siou=backprojection_superpoint_min_view_siou,
                superpoint_view_siou_min_views=backprojection_superpoint_view_siou_min_views,
                superpoint_view_siou_min_visible_points=backprojection_superpoint_view_siou_min_visible_points,
                superpoint_box_context=superpoint_box_context,
                superpoint_min_box_positive_ratio=backprojection_superpoint_min_box_positive_ratio,
                superpoint_max_box_negative_ratio=backprojection_superpoint_max_box_negative_ratio,
                superpoint_box_min_visible_points=backprojection_superpoint_box_min_visible_points,
                superpoint_box_min_views=backprojection_superpoint_box_min_views,
                superpoint_box_padding_ratio=backprojection_superpoint_box_padding_ratio,
                local_superpoint_refine=backprojection_local_superpoint_refine,
                local_superpoint_knn=backprojection_local_superpoint_knn,
                local_superpoint_merge_k=backprojection_local_superpoint_merge_k,
                local_superpoint_min_size=backprojection_local_superpoint_min_size,
                local_superpoint_min_coverage=backprojection_local_superpoint_min_coverage,
                local_superpoint_max_expansion_ratio=backprojection_local_superpoint_max_expansion_ratio,
                local_superpoint_min_seed_retention=backprojection_local_superpoint_min_seed_retention,
                local_superpoint_max_points=backprojection_local_superpoint_max_points,
                merge_iou=backprojection_merge_iou,
                inclusion_threshold=backprojection_inclusion_threshold,
                postprocess_same_class_only=backprojection_postprocess_same_class_only,
                containment_action=backprojection_containment_action,
                containment_threshold=backprojection_containment_threshold,
                containment_min_area_ratio=backprojection_containment_min_area_ratio,
                containment_score_ratio=backprojection_containment_score_ratio,
                containment_quality_margin=backprojection_containment_quality_margin,
                containment_score_factor=backprojection_containment_score_factor,
                containment_min_points=backprojection_containment_min_points,
            )
            bpr_report = scene_prediction[3]
            bpr_added_counts[scene_name] = int(scene_prediction[0].shape[1] - original_num_scene_predictions)
            if backprojection_report_path is not None:
                bpr_reports[scene_name] = bpr_report
            scene_prediction = scene_prediction[:3]
        scene_prediction = tuple(
            item.detach().cpu().numpy() if torch.is_tensor(item) else item
            for item in scene_prediction
        )
        if stream_eval_predictions:
            eval_prediction = build_eval_prediction(scene_name, scene_prediction)
            if eval_prediction_cache_dir is not None:
                cached_eval_prediction_paths[scene_name] = save_eval_prediction(scene_name, eval_prediction)
                del eval_prediction
            else:
                preds[scene_name] = eval_prediction
        else:
            predictions[scene_name] = scene_prediction

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

        if path_to_3d_masks is None:
            openyolo3d.network_3d = None

        if points_xyz is not None:
            del points_xyz
        del prediction
        del scene_prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not stream_eval_predictions:
        print("Evaluation ...")
        for scene_name in tqdm(scene_names):
            pred_classes = predictions[scene_name][1]
            pred_scores = predictions[scene_name][2]
            if object_query_rescore:
                pred_classes, pred_scores, object_report = rescore_with_object_queries(
                    scene_name,
                    predictions[scene_name][0],
                    pred_classes,
                    pred_scores,
                    object_candidates,
                    labels=labels,
                    min_candidate_score=object_query_min_candidate_score,
                    min_seed_points=object_query_min_seed_points,
                    min_seed_overlap=object_query_min_seed_overlap,
                    min_support_views=object_query_min_support_views,
                    support_scale=object_query_support_scale,
                    overlap_power=object_query_overlap_power,
                    min_evidence=object_query_min_evidence,
                    min_margin=object_query_min_margin,
                    max_base_score=object_query_max_base_score,
                    score_alpha=object_query_score_alpha,
                    use_candidate_fusion_score=object_query_use_candidate_fusion_score,
                    allowed_classes=object_query_allowed_classes,
                    blocked_classes=object_query_blocked_classes,
                )
                object_query_reports[scene_name] = object_report
            if clip_object_rescore:
                pred_classes, pred_scores, clip_report = rescore_with_clip_object_features(
                    scene_name,
                    predictions[scene_name][0],
                    pred_classes,
                    pred_scores,
                    clip_features,
                    labels=labels,
                    min_seed_points=clip_object_min_seed_points,
                    min_seed_overlap=clip_object_min_seed_overlap,
                    min_support_views=clip_object_min_support_views,
                    support_scale=clip_object_support_scale,
                    topk_classes=clip_object_topk_classes,
                    min_clip_prob=clip_object_min_clip_prob,
                    min_evidence=clip_object_min_evidence,
                    min_margin=clip_object_min_margin,
                    max_base_score=clip_object_max_base_score,
                    score_alpha=clip_object_score_alpha,
                    allowed_classes=clip_object_allowed_classes,
                    blocked_classes=clip_object_blocked_classes,
                )
                clip_object_reports[scene_name] = clip_report
            if context_corrections is not None:
                pred_classes, pred_scores, correction_report = apply_context_corrections(
                    scene_name,
                    pred_classes,
                    pred_scores,
                    corrections,
                    score_policy=correction_score_policy,
                    score_blend_alpha=correction_score_blend_alpha,
                    apply_decisions=correction_apply_decisions.split(",") if correction_apply_decisions else None,
                    apply_min_confidence=correction_apply_min_confidence,
                    apply_min_score=correction_apply_min_score,
                    apply_max_score=correction_apply_max_score,
                    bad_mask_policy=correction_bad_mask_policy,
                    bad_mask_score=correction_bad_mask_score,
                    score_boost=correction_score_boost,
                    allowed_classes=correction_allowed_classes,
                    blocked_classes=correction_blocked_classes,
                )
                correction_reports[scene_name] = correction_report

            preds[scene_name] = build_eval_prediction(
                scene_name,
                predictions[scene_name],
                pred_classes=pred_classes,
                pred_scores=pred_scores,
            )

    if cached_eval_prediction_paths:
        preds = {
            scene_name: load_eval_prediction(cached_eval_prediction_paths[scene_name])
            for scene_name in scene_names
        }

    if correction_report_path is not None:
        report_dir = osp.dirname(correction_report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(correction_report_path, "w") as f:
            json.dump(
                {
                    "correction_summary": correction_summary,
                    "scene_reports": correction_reports,
                    "score_policy": correction_score_policy,
                    "score_blend_alpha": correction_score_blend_alpha,
                    "apply_decisions": correction_apply_decisions,
                    "apply_min_confidence": correction_apply_min_confidence,
                    "apply_min_score": correction_apply_min_score,
                    "apply_max_score": correction_apply_max_score,
                    "bad_mask_policy": correction_bad_mask_policy,
                    "bad_mask_score": correction_bad_mask_score,
                    "score_boost": correction_score_boost,
                    "allowed_classes": correction_allowed_classes,
                    "blocked_classes": correction_blocked_classes,
                    "min_confidence": correction_min_confidence,
                },
                f,
                indent=2,
            )
        print(f"[INFO] Saved context correction report to {correction_report_path}")

    if object_query_report_path is not None:
        save_object_query_report(
            object_query_report_path,
            object_summary,
            object_query_reports,
            {
                "min_candidate_score": object_query_min_candidate_score,
                "min_seed_points": object_query_min_seed_points,
                "min_seed_overlap": object_query_min_seed_overlap,
                "min_support_views": object_query_min_support_views,
                "support_scale": object_query_support_scale,
                "overlap_power": object_query_overlap_power,
                "min_evidence": object_query_min_evidence,
                "min_margin": object_query_min_margin,
                "max_base_score": object_query_max_base_score,
                "score_alpha": object_query_score_alpha,
                "use_candidate_fusion_score": object_query_use_candidate_fusion_score,
                "allowed_classes": object_query_allowed_classes,
                "blocked_classes": object_query_blocked_classes,
            },
        )
        print(f"[INFO] Saved object-query rescore report to {object_query_report_path}")

    if clip_object_report_path is not None:
        save_clip_object_report(
            clip_object_report_path,
            clip_summary,
            clip_object_reports,
            {
                "min_seed_points": clip_object_min_seed_points,
                "min_seed_overlap": clip_object_min_seed_overlap,
                "min_support_views": clip_object_min_support_views,
                "support_scale": clip_object_support_scale,
                "topk_classes": clip_object_topk_classes,
                "min_clip_prob": clip_object_min_clip_prob,
                "min_evidence": clip_object_min_evidence,
                "min_margin": clip_object_min_margin,
                "max_base_score": clip_object_max_base_score,
                "score_alpha": clip_object_score_alpha,
                "allowed_classes": clip_object_allowed_classes,
                "blocked_classes": clip_object_blocked_classes,
            },
        )
        print(f"[INFO] Saved CLIP object rescore report to {clip_object_report_path}")

    if backprojection_report_path is not None:
        report_dir = osp.dirname(backprojection_report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(backprojection_report_path, "w") as f:
            json.dump(
                {
                    "candidate_summary": bpr_summary,
                    "scene_reports": bpr_reports,
                    "params": {
                        "min_score": backprojection_min_score,
                        "min_seed_points": backprojection_min_seed_points,
                        "max_existing_iou": backprojection_max_existing_iou,
                        "max_seed_in_existing_mask_ratio": backprojection_max_seed_in_existing_mask_ratio,
                        "max_proposal_iou": backprojection_max_proposal_iou,
                        "max_candidates_per_scene": backprojection_max_candidates_per_scene,
                        "score_scale": backprojection_score_scale,
                        "use_candidate_fusion_score": backprojection_use_candidate_fusion_score,
                        "allowed_classes": backprojection_allowed_classes,
                        "blocked_classes": backprojection_blocked_classes,
                        "min_support_views": backprojection_min_support_views,
                        "min_support_mean_iou": backprojection_min_support_mean_iou,
                        "min_support_best_iou": backprojection_min_support_best_iou,
                        "min_fusion_score": backprojection_min_fusion_score,
                        "max_box_area_ratio": backprojection_max_box_area_ratio,
                        "min_quality_score": backprojection_min_quality_score,
                        "min_scene_source_quality_z": backprojection_min_scene_source_quality_z,
                        "quality_sort": backprojection_quality_sort,
                        "verifications": backprojection_verifications,
                        "verification_summary": bpr_verification_summary,
                        "verifier_min_confidence": backprojection_verifier_min_confidence,
                        "verifier_suppress_decisions": backprojection_verifier_suppress_decisions,
                        "grow_radius": backprojection_grow_radius,
                        "max_growth_ratio": backprojection_max_growth_ratio,
                        "seed_cc_cleanup": backprojection_seed_cc_cleanup,
                        "seed_cc_radius": backprojection_seed_cc_radius,
                        "seed_cc_min_component_points": backprojection_seed_cc_min_component_points,
                        "seed_cc_keep_topk": backprojection_seed_cc_keep_topk,
                        "seed_cc_max_points": backprojection_seed_cc_max_points,
                        "seed_cc_min_keep_ratio": backprojection_seed_cc_min_keep_ratio,
                        "cc_cleanup": backprojection_cc_cleanup,
                        "cc_radius": backprojection_cc_radius,
                        "cc_min_component_points": backprojection_cc_min_component_points,
                        "cc_keep_topk": backprojection_cc_keep_topk,
                        "cc_max_points": backprojection_cc_max_points,
                        "cc_split_components": backprojection_cc_split_components,
                        "cc_source_kinds": backprojection_cc_source_kinds,
                        "cc_min_keep_ratio": backprojection_cc_min_keep_ratio,
                        "cc_keep_ratio_score_weight": backprojection_cc_keep_ratio_score_weight,
                        "source_priorities": backprojection_source_priorities,
                        "source_max_candidates": backprojection_source_max_candidates,
                        "source_score_scales": backprojection_source_score_scales,
                        "source_min_scores": backprojection_source_min_scores,
                        "max_candidates_per_class": backprojection_max_candidates_per_class,
                        "class_max_candidates": backprojection_class_max_candidates,
                        "quality_calibration_weight": backprojection_quality_calibration_weight,
                        "novelty_calibration_weight": backprojection_novelty_calibration_weight,
                        "label_consensus_calibration_weight": backprojection_label_consensus_calibration_weight,
                        "score_calibration_min": backprojection_score_calibration_min,
                        "score_calibration_max": backprojection_score_calibration_max,
                        "max_proposal_score": backprojection_max_proposal_score,
                        "min_label_consensus_score": backprojection_min_label_consensus_score,
                        "max_label_conflict_score": backprojection_max_label_conflict_score,
                        "label_consensus_iou_threshold": backprojection_label_consensus_iou_threshold,
                        "label_consensus_min_visible_points": backprojection_label_consensus_min_visible_points,
                        "label_consensus_frame_mode": backprojection_label_consensus_frame_mode,
                        "projection_consistency_min_box_iou": backprojection_projection_consistency_min_box_iou,
                        "projection_consistency_min_point_ratio": backprojection_projection_consistency_min_point_ratio,
                        "projection_consistency_min_views": backprojection_projection_consistency_min_views,
                        "projection_consistency_min_visible_points": backprojection_projection_consistency_min_visible_points,
                        "projection_consistency_frame_mode": backprojection_projection_consistency_frame_mode,
                        "projection_consistency_box_padding_ratio": backprojection_projection_consistency_box_padding_ratio,
                        "projection_consistency_score_weight": backprojection_projection_consistency_score_weight,
                        "superpoint_refine": backprojection_superpoint_refine,
                        "superpoint_min_coverage": backprojection_superpoint_min_coverage,
                        "superpoint_max_expansion_ratio": backprojection_superpoint_max_expansion_ratio,
                        "superpoint_max_segment_ratio": backprojection_superpoint_max_segment_ratio,
                        "superpoint_large_segment_min_coverage": backprojection_superpoint_large_segment_min_coverage,
                        "superpoint_min_seed_retention": backprojection_superpoint_min_seed_retention,
                        "superpoint_min_support_views": backprojection_superpoint_min_support_views,
                        "superpoint_min_support_ratio": backprojection_superpoint_min_support_ratio,
                        "superpoint_min_view_siou": backprojection_superpoint_min_view_siou,
                        "superpoint_view_siou_min_views": backprojection_superpoint_view_siou_min_views,
                        "superpoint_view_siou_min_visible_points": backprojection_superpoint_view_siou_min_visible_points,
                        "superpoint_min_box_positive_ratio": backprojection_superpoint_min_box_positive_ratio,
                        "superpoint_max_box_negative_ratio": backprojection_superpoint_max_box_negative_ratio,
                        "superpoint_box_min_visible_points": backprojection_superpoint_box_min_visible_points,
                        "superpoint_box_min_views": backprojection_superpoint_box_min_views,
                        "superpoint_box_padding_ratio": backprojection_superpoint_box_padding_ratio,
                        "local_superpoint_refine": backprojection_local_superpoint_refine,
                        "local_superpoint_knn": backprojection_local_superpoint_knn,
                        "local_superpoint_merge_k": backprojection_local_superpoint_merge_k,
                        "local_superpoint_min_size": backprojection_local_superpoint_min_size,
                        "local_superpoint_min_coverage": backprojection_local_superpoint_min_coverage,
                        "local_superpoint_max_expansion_ratio": backprojection_local_superpoint_max_expansion_ratio,
                        "local_superpoint_min_seed_retention": backprojection_local_superpoint_min_seed_retention,
                        "local_superpoint_max_points": backprojection_local_superpoint_max_points,
                        "merge_iou": backprojection_merge_iou,
                        "inclusion_threshold": backprojection_inclusion_threshold,
                        "postprocess_same_class_only": backprojection_postprocess_same_class_only,
                        "containment_action": backprojection_containment_action,
                        "containment_threshold": backprojection_containment_threshold,
                        "containment_min_area_ratio": backprojection_containment_min_area_ratio,
                        "containment_score_ratio": backprojection_containment_score_ratio,
                        "containment_quality_margin": backprojection_containment_quality_margin,
                        "containment_score_factor": backprojection_containment_score_factor,
                        "containment_min_points": backprojection_containment_min_points,
                    },
                },
                f,
                indent=2,
            )
        print(f"[INFO] Saved back-projection report to {backprojection_report_path}")

    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=dataset_type, output_file=eval_output_file)
    if eval_cleanup_prediction_cache and eval_prediction_cache_dir is not None:
        preds = {}
        cached_eval_prediction_paths = {}
        gc.collect()
        shutil.rmtree(eval_prediction_cache_dir, ignore_errors=True)
        print(f"[INFO] Removed eval prediction cache dir: {eval_prediction_cache_dir}")
    return inst_AP

def load_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='scannet200', type=str, help='Name of the dataset [replica, scannet200]')
    parser.add_argument('--path_to_3d_masks', default=None, type=str, help='Path to pre computed 3d masks')
    parser.add_argument('--is_gt', default=False, action=argparse.BooleanOptionalAction, help='If pre computed 3d masks are ground truth masks')
    parser.add_argument('--use_pred_scores', default=False, action=argparse.BooleanOptionalAction, help='Use OpenYOLO3D prediction confidence scores during evaluation')
    parser.add_argument('--score_threshold', default=0.0, type=float, help='Filter predictions below this OpenYOLO3D confidence score before evaluation')
    parser.add_argument('--path_to_2d_preds', default=None, type=str, help='Optional directory or .pt file for cached YOLO-World 2D detections')
    parser.add_argument('--save_2d_preds', default=False, action=argparse.BooleanOptionalAction, help='Save YOLO-World 2D detections to --path_to_2d_preds after inference')
    parser.add_argument('--reuse_2d_preds', default=True, action=argparse.BooleanOptionalAction, help='Reuse cached YOLO-World 2D detections when available')
    parser.add_argument('--processed_scene_root', default=None, type=str, help='Optional root containing ScanNet200 processed .npy files, used to swap point superpoints without moving RGB-D data')
    parser.add_argument('--backprojection_candidates', default=None, type=str, help='Path to ESAM-style backprojection candidates directory or JSON')
    parser.add_argument('--backprojection_min_score', default=0.35, type=float, help='Minimum 2D detection score for adding a backprojection proposal')
    parser.add_argument('--backprojection_min_seed_points', default=80, type=int, help='Minimum seed points for adding a backprojection proposal')
    parser.add_argument('--backprojection_max_existing_iou', default=0.30, type=float, help='Skip backprojection proposals whose masks overlap existing 3D masks above this IoU')
    parser.add_argument('--backprojection_max_seed_in_existing_mask_ratio', default=0.70, type=float, help='Skip candidates whose seed points are already mostly covered by existing 3D masks')
    parser.add_argument('--backprojection_max_proposal_iou', default=0.50, type=float, help='NMS IoU threshold among newly added backprojection proposals')
    parser.add_argument('--backprojection_max_candidates_per_scene', default=None, type=int, help='Maximum added backprojection proposals per scene')
    parser.add_argument('--backprojection_score_scale', default=0.50, type=float, help='Scale 2D detector score before assigning new proposal confidence')
    parser.add_argument('--backprojection_use_candidate_fusion_score', default=True, action=argparse.BooleanOptionalAction, help='Use candidate fusion_score from BPR export when available')
    parser.add_argument('--backprojection_allowed_classes', default=None, type=str, help='Comma-separated class names allowed for BPR fusion')
    parser.add_argument('--backprojection_blocked_classes', default=None, type=str, help='Comma-separated class names blocked from BPR fusion')
    parser.add_argument('--backprojection_min_support_views', default=0, type=int, help='Minimum multi-view support count for adding a BPR proposal')
    parser.add_argument('--backprojection_min_support_mean_iou', default=0.0, type=float, help='Minimum mean 2D bbox agreement across support views for adding a BPR proposal')
    parser.add_argument('--backprojection_min_support_best_iou', default=0.0, type=float, help='Minimum best-view 2D bbox agreement for adding a BPR proposal')
    parser.add_argument('--backprojection_min_fusion_score', default=0.0, type=float, help='Minimum exported candidate fusion_score for adding a BPR proposal')
    parser.add_argument('--backprojection_max_box_area_ratio', default=None, type=float, help='Maximum 2D bbox image-area ratio for adding a BPR proposal')
    parser.add_argument('--backprojection_min_quality_score', default=0.0, type=float, help='Minimum quality-aware BPR score for adding a proposal')
    parser.add_argument('--backprojection_min_scene_source_quality_z', default=None, type=float, help='Minimum per-scene/per-source standardized BPR quality score; disabled when omitted')
    parser.add_argument('--backprojection_quality_sort', default=False, action=argparse.BooleanOptionalAction, help='Rank BPR candidates by quality-aware score before proposal priority')
    parser.add_argument('--backprojection_verifications', default=None, type=str, help='Optional MLLM/VLM verifier JSON/JSONL for BPR candidate keep/suppress decisions')
    parser.add_argument('--backprojection_verifier_min_confidence', default=0.0, type=float, help='Minimum verifier confidence used when loading BPR verifier decisions')
    parser.add_argument('--backprojection_verifier_suppress_decisions', default='suppress,bad_mask,invalid,reject', type=str, help='Comma-separated verifier decisions that suppress a BPR proposal')
    parser.add_argument('--backprojection_verifier_strict', default=False, action=argparse.BooleanOptionalAction, help='Raise on malformed BPR verifier records')
    parser.add_argument('--backprojection_grow_radius', default=0.0, type=float, help='Optional 3D bbox expansion radius in scene units for seed proposal growth')
    parser.add_argument('--backprojection_max_growth_ratio', default=4.0, type=float, help='Maximum grown-mask/seed-size ratio; larger growth falls back to seed-only')
    parser.add_argument('--backprojection_seed_cc_cleanup', default=False, action=argparse.BooleanOptionalAction, help='Clean raw BPR seed masks with 3D connected components before superpoint refinement')
    parser.add_argument('--backprojection_seed_cc_radius', default=0.03, type=float, help='3D radius used to connect neighboring raw BPR seed points before superpoint refinement')
    parser.add_argument('--backprojection_seed_cc_min_component_points', default=30, type=int, help='Minimum connected-component size kept by raw seed cleanup')
    parser.add_argument('--backprojection_seed_cc_keep_topk', default=1, type=int, help='Number of largest raw seed connected components kept before superpoint refinement')
    parser.add_argument('--backprojection_seed_cc_max_points', default=30000, type=int, help='Skip raw seed connected-component cleanup above this seed point count')
    parser.add_argument('--backprojection_seed_cc_min_keep_ratio', default=0.0, type=float, help='Skip proposals if raw seed CC kept/input point ratio is below this value; 0 disables')
    parser.add_argument('--backprojection_cc_cleanup', default=False, action=argparse.BooleanOptionalAction, help='Clean BPR proposal masks with 3D connected components before fusion')
    parser.add_argument('--backprojection_cc_radius', default=0.03, type=float, help='3D radius used to connect neighboring BPR proposal points')
    parser.add_argument('--backprojection_cc_min_component_points', default=50, type=int, help='Minimum connected-component size kept by BPR cleanup')
    parser.add_argument('--backprojection_cc_keep_topk', default=1, type=int, help='Number of largest connected components kept per BPR proposal')
    parser.add_argument('--backprojection_cc_max_points', default=30000, type=int, help='Skip connected-component cleanup above this proposal point count')
    parser.add_argument('--backprojection_cc_split_components', default=False, action=argparse.BooleanOptionalAction, help='Append kept BPR connected components as separate proposals')
    parser.add_argument('--backprojection_cc_source_kinds', default=None, type=str, help='Comma-separated source kinds/names cleaned by CC, e.g. bpr or sam_fused; disabled when omitted')
    parser.add_argument('--backprojection_cc_min_keep_ratio', default=0.0, type=float, help='Skip CC-cleaned proposals if kept component points/input points is below this ratio; 0 disables')
    parser.add_argument('--backprojection_cc_keep_ratio_score_weight', default=0.0, type=float, help='Downweight CC-cleaned proposal scores by kept/input point ratio; 0 disables, 1 uses the ratio directly')
    parser.add_argument('--backprojection_source_priorities', default=None, type=str, help='Comma-separated source priority rules, e.g. sam_fused=2.0,bpr=1.0')
    parser.add_argument('--backprojection_source_max_candidates', default=None, type=str, help='Comma-separated per-source proposal limits, e.g. sam_fused=20,bpr=10')
    parser.add_argument('--backprojection_source_score_scales', default=None, type=str, help='Comma-separated per-source score multipliers, e.g. sam_fused=1.2,bpr=0.9')
    parser.add_argument('--backprojection_source_min_scores', default=None, type=str, help='Comma-separated per-source minimum raw candidate scores, e.g. sam_fused=0.65,bpr=0.50')
    parser.add_argument('--backprojection_max_candidates_per_class', default=None, type=int, help='Maximum added BPR proposals per predicted class in each scene')
    parser.add_argument('--backprojection_class_max_candidates', default=None, type=str, help='Comma-separated per-class proposal limits, e.g. "office chair=1,mat=2"')
    parser.add_argument('--backprojection_quality_calibration_weight', default=0.0, type=float, help='SQS z-score calibration weight for BPR proposal scoring; 0 disables')
    parser.add_argument('--backprojection_novelty_calibration_weight', default=0.0, type=float, help='Novelty calibration weight using overlap with existing 3D masks; 0 disables')
    parser.add_argument('--backprojection_label_consensus_calibration_weight', default=0.0, type=float, help='Downweight BPR proposal scores when overlapping 2D evidence favors another class; 0 disables')
    parser.add_argument('--backprojection_score_calibration_min', default=0.2, type=float, help='Minimum multiplicative BPR score calibration factor')
    parser.add_argument('--backprojection_score_calibration_max', default=1.2, type=float, help='Maximum multiplicative BPR score calibration factor')
    parser.add_argument('--backprojection_max_proposal_score', default=None, type=float, help='Optional upper bound on appended BPR proposal confidence')
    parser.add_argument('--backprojection_min_label_consensus_score', default=0.0, type=float, help='Minimum candidate label-consensus score before adding a BPR proposal')
    parser.add_argument('--backprojection_max_label_conflict_score', default=1.0, type=float, help='Maximum candidate conflicting-label evidence ratio before adding a BPR proposal')
    parser.add_argument('--backprojection_label_consensus_iou_threshold', default=0.25, type=float, help='2D box IoU threshold used for on-the-fly BPR label-consensus evidence')
    parser.add_argument('--backprojection_label_consensus_min_visible_points', default=30, type=int, help='Minimum visible candidate points in a frame used for on-the-fly label consensus')
    parser.add_argument('--backprojection_label_consensus_frame_mode', default='support', choices=['support', 'all'], help='Frames used for on-the-fly BPR label consensus')
    parser.add_argument('--backprojection_projection_consistency_min_box_iou', default=0.0, type=float, help='Minimum mean IoU between projected BPR proposal boxes and same-class 2D detections; 0 disables')
    parser.add_argument('--backprojection_projection_consistency_min_point_ratio', default=0.0, type=float, help='Minimum mean ratio of projected proposal points inside matched same-class 2D boxes; 0 disables')
    parser.add_argument('--backprojection_projection_consistency_min_views', default=1, type=int, help='Minimum usable views before applying projected-box consistency filters')
    parser.add_argument('--backprojection_projection_consistency_min_visible_points', default=30, type=int, help='Minimum visible proposal points per view for projected-box consistency')
    parser.add_argument('--backprojection_projection_consistency_frame_mode', default='support', choices=['support', 'all'], help='Frames used for projected-box consistency')
    parser.add_argument('--backprojection_projection_consistency_box_padding_ratio', default=0.05, type=float, help='Padding ratio applied to matched 2D boxes when counting projected proposal points inside boxes')
    parser.add_argument('--backprojection_projection_consistency_score_weight', default=0.0, type=float, help='Downweight BPR proposal scores by projected-box consistency; 0 disables, 1 uses the consistency score directly')
    parser.add_argument('--backprojection_superpoint_refine', default=False, action=argparse.BooleanOptionalAction, help='Refine appended BPR masks by voting over ScanNet200 superpoint/segment ids')
    parser.add_argument('--backprojection_superpoint_min_coverage', default=0.30, type=float, help='Minimum fraction of a touched superpoint covered by seed points before filling that superpoint')
    parser.add_argument('--backprojection_superpoint_max_expansion_ratio', default=2.0, type=float, help='Fallback to seed mask if superpoint refinement expands a proposal above this ratio')
    parser.add_argument('--backprojection_superpoint_max_segment_ratio', default=None, type=float, help='Treat touched superpoints larger than this multiple of seed points as large segments')
    parser.add_argument('--backprojection_superpoint_large_segment_min_coverage', default=None, type=float, help='Minimum coverage required for large superpoints; if omitted large segments are skipped')
    parser.add_argument('--backprojection_superpoint_min_seed_retention', default=0.0, type=float, help='Fallback to seed mask if superpoint refinement keeps less than this fraction of original seed points')
    parser.add_argument('--backprojection_superpoint_min_support_views', default=0, type=int, help='Minimum candidate support views that must see a selected superpoint; 0 disables this constraint')
    parser.add_argument('--backprojection_superpoint_min_support_ratio', default=0.0, type=float, help='Minimum fraction of candidate support views that must see a selected superpoint; 0 disables this constraint')
    parser.add_argument('--backprojection_superpoint_min_view_siou', default=0.0, type=float, help='Minimum mean pairwise superpoint IoU across candidate support views before adding a proposal; 0 disables')
    parser.add_argument('--backprojection_superpoint_view_siou_min_views', default=2, type=int, help='Apply --backprojection_superpoint_min_view_siou only when at least this many support views are usable')
    parser.add_argument('--backprojection_superpoint_view_siou_min_visible_points', default=1, type=int, help='Minimum visible proposal points required for a support view to enter superpoint sIoU')
    parser.add_argument('--backprojection_superpoint_min_box_positive_ratio', default=0.0, type=float, help='Minimum fraction of selected superpoint visible points that project inside candidate support boxes; 0 disables')
    parser.add_argument('--backprojection_superpoint_max_box_negative_ratio', default=1.0, type=float, help='Maximum fraction of selected superpoint visible points that project outside candidate support boxes')
    parser.add_argument('--backprojection_superpoint_box_min_visible_points', default=5, type=int, help='Minimum visible segment points in a support view for superpoint box support filtering')
    parser.add_argument('--backprojection_superpoint_box_min_views', default=1, type=int, help='Minimum support-box views required before filtering a superpoint segment')
    parser.add_argument('--backprojection_superpoint_box_padding_ratio', default=0.05, type=float, help='Padding ratio applied to support boxes for superpoint box support filtering')
    parser.add_argument('--backprojection_local_superpoint_refine', default=False, action=argparse.BooleanOptionalAction, help='Locally re-segment each appended BPR proposal before connected-component cleanup')
    parser.add_argument('--backprojection_local_superpoint_knn', default=10, type=int, help='Number of local nearest neighbors used for BPR local superpoint refinement')
    parser.add_argument('--backprojection_local_superpoint_merge_k', default=0.25, type=float, help='Felzenszwalb-style merge threshold for BPR local superpoint refinement')
    parser.add_argument('--backprojection_local_superpoint_min_size', default=10, type=int, help='Minimum local component size after BPR local superpoint refinement')
    parser.add_argument('--backprojection_local_superpoint_min_coverage', default=0.25, type=float, help='Minimum fraction of a local segment covered by the pre-refinement seed reference')
    parser.add_argument('--backprojection_local_superpoint_max_expansion_ratio', default=1.0, type=float, help='Fallback if local superpoint refinement expands above this ratio of the input proposal')
    parser.add_argument('--backprojection_local_superpoint_min_seed_retention', default=0.80, type=float, help='Fallback if local superpoint refinement keeps less than this fraction of seed reference points')
    parser.add_argument('--backprojection_local_superpoint_max_points', default=30000, type=int, help='Skip local superpoint refinement above this proposal point count')
    parser.add_argument('--backprojection_merge_iou', default=0.0, type=float, help='Iteratively merge newly appended same-class BPR proposals whose 3D IoU is at least this value; 0 disables')
    parser.add_argument('--backprojection_inclusion_threshold', default=0.0, type=float, help='Remove newly appended same-class BPR proposals included in a larger appended proposal above this ratio; 0 disables')
    parser.add_argument('--backprojection_postprocess_same_class_only', default=True, action=argparse.BooleanOptionalAction, help='Restrict BPR merge/inclusion postprocessing to proposals with the same predicted class')
    parser.add_argument('--backprojection_containment_action', default='none', choices=['none', 'downweight', 'carve', 'remove_large'], help='Handle appended proposals that contain smaller protected masks: none, downweight, carve, or remove_large')
    parser.add_argument('--backprojection_containment_threshold', default=0.85, type=float, help='Minimum fraction of a smaller mask covered before containment handling triggers')
    parser.add_argument('--backprojection_containment_min_area_ratio', default=1.5, type=float, help='Minimum containing/smaller mask area ratio before containment handling triggers')
    parser.add_argument('--backprojection_containment_score_ratio', default=0.75, type=float, help='Smaller mask score must be at least this fraction of the containing proposal score')
    parser.add_argument('--backprojection_containment_quality_margin', default=0.0, type=float, help='When both are appended proposals, smaller quality plus this margin must be at least containing quality')
    parser.add_argument('--backprojection_containment_score_factor', default=0.5, type=float, help='Score multiplier used by --backprojection_containment_action downweight')
    parser.add_argument('--backprojection_containment_min_points', default=50, type=int, help='Minimum remaining points for containing proposal after carve; smaller outputs are removed')
    parser.add_argument('--backprojection_report_path', default=None, type=str, help='Optional JSON report for added/skipped backprojection proposals')
    parser.add_argument('--object_query_rescore', default=False, action=argparse.BooleanOptionalAction, help='Use SegDINO3D-inspired object-query evidence to rescore existing 3D masks')
    parser.add_argument('--object_query_candidates', default=None, type=str, help='Path to backprojection candidate JSON/directory used as 2D object-query evidence; defaults to --backprojection_candidates')
    parser.add_argument('--object_query_min_candidate_score', default=0.30, type=float, help='Minimum 2D object-query score used for rescoring')
    parser.add_argument('--object_query_min_seed_points', default=80, type=int, help='Minimum backprojected seed points for an object query')
    parser.add_argument('--object_query_min_seed_overlap', default=0.35, type=float, help='Minimum fraction of object-query seed points inside a 3D mask')
    parser.add_argument('--object_query_min_support_views', default=2, type=int, help='Minimum multi-view support count for an object query')
    parser.add_argument('--object_query_support_scale', default=8.0, type=float, help='Support count that maps to full object-query weight')
    parser.add_argument('--object_query_overlap_power', default=1.0, type=float, help='Exponent applied to seed-overlap evidence weight')
    parser.add_argument('--object_query_min_evidence', default=0.45, type=float, help='Minimum aggregated object-query evidence before changing a class')
    parser.add_argument('--object_query_min_margin', default=0.10, type=float, help='Minimum evidence margin over the current/second class before changing a class')
    parser.add_argument('--object_query_max_base_score', default=1.01, type=float, help='Only rescore instances whose original score is at or below this value')
    parser.add_argument('--object_query_score_alpha', default=0.50, type=float, help='Original-score weight when updating scores after object-query rescoring')
    parser.add_argument('--object_query_use_candidate_fusion_score', default=True, action=argparse.BooleanOptionalAction, help='Use candidate fusion_score as object-query score when available')
    parser.add_argument('--object_query_allowed_classes', default=None, type=str, help='Comma-separated class names allowed for object-query rescoring')
    parser.add_argument('--object_query_blocked_classes', default=None, type=str, help='Comma-separated class names blocked from object-query rescoring')
    parser.add_argument('--object_query_report_path', default=None, type=str, help='Optional JSON report for object-query rescoring decisions')
    parser.add_argument('--clip_object_rescore', default=False, action=argparse.BooleanOptionalAction, help='Use cached CLIP crop-text similarities to rescore existing 3D masks')
    parser.add_argument('--clip_object_features', default=None, type=str, help='Path to CLIP object feature cache directory or JSON')
    parser.add_argument('--clip_object_min_seed_points', default=80, type=int, help='Minimum backprojected seed points for a CLIP object record')
    parser.add_argument('--clip_object_min_seed_overlap', default=0.45, type=float, help='Minimum fraction of CLIP object seed points inside a 3D mask')
    parser.add_argument('--clip_object_min_support_views', default=2, type=int, help='Minimum multi-view support count for a CLIP object record')
    parser.add_argument('--clip_object_support_scale', default=8.0, type=float, help='Support count that maps to full CLIP object weight')
    parser.add_argument('--clip_object_topk_classes', default=5, type=int, help='Number of top CLIP classes allowed to contribute evidence')
    parser.add_argument('--clip_object_min_clip_prob', default=0.10, type=float, help='Minimum CLIP class probability used as evidence')
    parser.add_argument('--clip_object_min_evidence', default=0.35, type=float, help='Minimum aggregated CLIP evidence before changing a class')
    parser.add_argument('--clip_object_min_margin', default=0.12, type=float, help='Minimum CLIP evidence margin over the current/second class before changing a class')
    parser.add_argument('--clip_object_max_base_score', default=1.01, type=float, help='Only rescore instances whose original score is at or below this value')
    parser.add_argument('--clip_object_score_alpha', default=0.50, type=float, help='Original-score weight when updating scores after CLIP rescoring')
    parser.add_argument('--clip_object_allowed_classes', default=None, type=str, help='Comma-separated class names allowed for CLIP object rescoring')
    parser.add_argument('--clip_object_blocked_classes', default=None, type=str, help='Comma-separated class names blocked from CLIP object rescoring')
    parser.add_argument('--clip_object_report_path', default=None, type=str, help='Optional JSON report for CLIP object rescoring decisions')
    parser.add_argument('--context_corrections', default=None, type=str, help='Path to offline MLLM/VLM semantic corrections JSON, JSONL, or directory')
    parser.add_argument('--correction_min_confidence', default=0.0, type=float, help='Ignore context corrections below this confidence when confidence is provided')
    parser.add_argument('--correction_score_policy', default='keep', choices=['keep', 'replace', 'max', 'blend', 'boost'], help='How correction confidence changes prediction scores')
    parser.add_argument('--correction_score_blend_alpha', default=0.5, type=float, help='Base-score weight used when --correction_score_policy blend')
    parser.add_argument('--correction_apply_decisions', default='change,keep', type=str, help='Comma-separated MLLM decisions to apply, e.g. change or change,keep')
    parser.add_argument('--correction_apply_min_confidence', default=None, type=float, help='Minimum confidence for applying change/keep decisions after loading corrections')
    parser.add_argument('--correction_apply_min_score', default=None, type=float, help='Only apply change/keep decisions to predictions at or above this original score')
    parser.add_argument('--correction_apply_max_score', default=None, type=float, help='Only apply change/keep decisions to predictions at or below this original score')
    parser.add_argument('--correction_bad_mask_policy', default='skip', choices=['skip', 'suppress'], help='How to handle MLLM bad_mask decisions')
    parser.add_argument('--correction_bad_mask_score', default=0.0, type=float, help='Score assigned when --correction_bad_mask_policy suppress')
    parser.add_argument('--correction_score_boost', default=0.0, type=float, help='Maximum score used by --correction_score_policy boost')
    parser.add_argument('--correction_allowed_classes', default=None, type=str, help='Comma-separated corrected class names allowed for context correction')
    parser.add_argument('--correction_blocked_classes', default=None, type=str, help='Comma-separated corrected class names blocked from context correction')
    parser.add_argument('--correction_strict', default=False, action=argparse.BooleanOptionalAction, help='Raise on malformed or unknown correction records')
    parser.add_argument('--correction_report_path', default=None, type=str, help='Optional JSON report for applied/skipped context corrections')
    parser.add_argument('--eval_output_file', default='temp_output.txt', type=str, help='CSV file written by the dataset evaluator')
    parser.add_argument('--scene_list', default=None, type=str, help='Optional comma-separated scene names or a file with one scene per line')
    parser.add_argument('--max_scenes', default=None, type=int, help='Evaluate only the first N scenes after optional --scene_list filtering')
    parser.add_argument('--eval_prediction_cache_dir', default=None, type=str, help='Optional directory for memory-mapped evaluator inputs')
    parser.add_argument('--eval_cleanup_prediction_cache', default=False, action=argparse.BooleanOptionalAction, help='Remove --eval_prediction_cache_dir after a successful evaluation')
    opt = parser.parse_args() 
    test_pipeline_full(
        opt.dataset_name,
        opt.path_to_3d_masks,
        opt.is_gt,
        opt.use_pred_scores,
        opt.score_threshold,
        opt.path_to_2d_preds,
        opt.save_2d_preds,
        opt.reuse_2d_preds,
        opt.backprojection_candidates,
        opt.backprojection_min_score,
        opt.backprojection_min_seed_points,
        opt.backprojection_max_existing_iou,
        opt.backprojection_max_seed_in_existing_mask_ratio,
        opt.backprojection_max_proposal_iou,
        opt.backprojection_max_candidates_per_scene,
        opt.backprojection_score_scale,
        opt.backprojection_use_candidate_fusion_score,
        opt.backprojection_allowed_classes,
        opt.backprojection_blocked_classes,
        opt.backprojection_min_support_views,
        opt.backprojection_min_support_mean_iou,
        opt.backprojection_min_support_best_iou,
        opt.backprojection_min_fusion_score,
        opt.backprojection_max_box_area_ratio,
        opt.backprojection_min_quality_score,
        opt.backprojection_min_scene_source_quality_z,
        opt.backprojection_quality_sort,
        opt.backprojection_verifications,
        opt.backprojection_verifier_min_confidence,
        opt.backprojection_verifier_suppress_decisions,
        opt.backprojection_verifier_strict,
        opt.backprojection_grow_radius,
        opt.backprojection_max_growth_ratio,
        opt.backprojection_seed_cc_cleanup,
        opt.backprojection_seed_cc_radius,
        opt.backprojection_seed_cc_min_component_points,
        opt.backprojection_seed_cc_keep_topk,
        opt.backprojection_seed_cc_max_points,
        opt.backprojection_seed_cc_min_keep_ratio,
        opt.backprojection_cc_cleanup,
        opt.backprojection_cc_radius,
        opt.backprojection_cc_min_component_points,
        opt.backprojection_cc_keep_topk,
        opt.backprojection_cc_max_points,
        opt.backprojection_cc_split_components,
        opt.backprojection_cc_source_kinds,
        opt.backprojection_cc_min_keep_ratio,
        opt.backprojection_cc_keep_ratio_score_weight,
        opt.backprojection_source_priorities,
        opt.backprojection_source_max_candidates,
        opt.backprojection_source_score_scales,
        opt.backprojection_source_min_scores,
        opt.backprojection_max_candidates_per_class,
        opt.backprojection_class_max_candidates,
        opt.backprojection_quality_calibration_weight,
        opt.backprojection_novelty_calibration_weight,
        opt.backprojection_label_consensus_calibration_weight,
        opt.backprojection_score_calibration_min,
        opt.backprojection_score_calibration_max,
        opt.backprojection_max_proposal_score,
        opt.backprojection_min_label_consensus_score,
        opt.backprojection_max_label_conflict_score,
        opt.backprojection_label_consensus_iou_threshold,
        opt.backprojection_label_consensus_min_visible_points,
        opt.backprojection_label_consensus_frame_mode,
        opt.backprojection_projection_consistency_min_box_iou,
        opt.backprojection_projection_consistency_min_point_ratio,
        opt.backprojection_projection_consistency_min_views,
        opt.backprojection_projection_consistency_min_visible_points,
        opt.backprojection_projection_consistency_frame_mode,
        opt.backprojection_projection_consistency_box_padding_ratio,
        opt.backprojection_projection_consistency_score_weight,
        opt.backprojection_superpoint_refine,
        opt.backprojection_superpoint_min_coverage,
        opt.backprojection_superpoint_max_expansion_ratio,
        opt.backprojection_superpoint_max_segment_ratio,
        opt.backprojection_superpoint_large_segment_min_coverage,
        opt.backprojection_superpoint_min_seed_retention,
        opt.backprojection_superpoint_min_support_views,
        opt.backprojection_superpoint_min_support_ratio,
        opt.backprojection_superpoint_min_view_siou,
        opt.backprojection_superpoint_view_siou_min_views,
        opt.backprojection_superpoint_view_siou_min_visible_points,
        opt.backprojection_superpoint_min_box_positive_ratio,
        opt.backprojection_superpoint_max_box_negative_ratio,
        opt.backprojection_superpoint_box_min_visible_points,
        opt.backprojection_superpoint_box_min_views,
        opt.backprojection_superpoint_box_padding_ratio,
        opt.backprojection_local_superpoint_refine,
        opt.backprojection_local_superpoint_knn,
        opt.backprojection_local_superpoint_merge_k,
        opt.backprojection_local_superpoint_min_size,
        opt.backprojection_local_superpoint_min_coverage,
        opt.backprojection_local_superpoint_max_expansion_ratio,
        opt.backprojection_local_superpoint_min_seed_retention,
        opt.backprojection_local_superpoint_max_points,
        opt.backprojection_merge_iou,
        opt.backprojection_inclusion_threshold,
        opt.backprojection_postprocess_same_class_only,
        opt.backprojection_containment_action,
        opt.backprojection_containment_threshold,
        opt.backprojection_containment_min_area_ratio,
        opt.backprojection_containment_score_ratio,
        opt.backprojection_containment_quality_margin,
        opt.backprojection_containment_score_factor,
        opt.backprojection_containment_min_points,
        opt.backprojection_report_path,
        opt.object_query_rescore,
        opt.object_query_candidates,
        opt.object_query_min_candidate_score,
        opt.object_query_min_seed_points,
        opt.object_query_min_seed_overlap,
        opt.object_query_min_support_views,
        opt.object_query_support_scale,
        opt.object_query_overlap_power,
        opt.object_query_min_evidence,
        opt.object_query_min_margin,
        opt.object_query_max_base_score,
        opt.object_query_score_alpha,
        opt.object_query_use_candidate_fusion_score,
        opt.object_query_allowed_classes,
        opt.object_query_blocked_classes,
        opt.object_query_report_path,
        opt.clip_object_rescore,
        opt.clip_object_features,
        opt.clip_object_min_seed_points,
        opt.clip_object_min_seed_overlap,
        opt.clip_object_min_support_views,
        opt.clip_object_support_scale,
        opt.clip_object_topk_classes,
        opt.clip_object_min_clip_prob,
        opt.clip_object_min_evidence,
        opt.clip_object_min_margin,
        opt.clip_object_max_base_score,
        opt.clip_object_score_alpha,
        opt.clip_object_allowed_classes,
        opt.clip_object_blocked_classes,
        opt.clip_object_report_path,
        opt.context_corrections,
        opt.correction_min_confidence,
        opt.correction_score_policy,
        opt.correction_score_blend_alpha,
        opt.correction_apply_decisions,
        opt.correction_apply_min_confidence,
        opt.correction_apply_min_score,
        opt.correction_apply_max_score,
        opt.correction_bad_mask_policy,
        opt.correction_bad_mask_score,
        opt.correction_score_boost,
        opt.correction_allowed_classes,
        opt.correction_blocked_classes,
        opt.correction_strict,
        opt.correction_report_path,
        opt.eval_output_file,
        opt.scene_list,
        opt.max_scenes,
        opt.eval_prediction_cache_dir,
        opt.eval_cleanup_prediction_cache,
        opt.processed_scene_root,
        )
       
