import argparse
import gc
import json
import os.path as osp
import sys

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from tqdm import tqdm

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200
from run_evaluation import load_yaml
from utils import OpenYolo3D
from utils.context_export import export_context_candidates


def parse_longtail_classes(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def export_dataset_candidates(
    dataset_name,
    path_to_3d_masks,
    output_dir,
    scene_name=None,
    top_views=3,
    uncertain_score_th=0.35,
    uncertain_margin_th=0.15,
    longtail_classes=None,
    max_candidates_per_scene=None,
    max_mask_point_ratio=0.20,
    max_bbox_area_ratio=0.65,
    min_visible_points=50,
    min_label_votes=10,
    include_bad_quality=False,
    path_to_2d_preds=None,
    save_2d_preds=False,
    reuse_2d_preds=True,
):
    config = load_yaml(osp.join(f"./pretrained/config_{dataset_name}.yaml"))
    path_2_dataset = osp.join("./data", dataset_name)
    depth_scale = config["openyolo3d"]["depth_scale"]

    if dataset_name == "replica":
        scene_names = SCENE_NAMES_REPLICA
        datatype = "point cloud"
    elif dataset_name == "scannet200":
        scene_names = SCENE_NAMES_SCANNET200
        datatype = "mesh"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if scene_name is not None:
        scene_names = [scene_name]

    openyolo3d = OpenYolo3D(f"./pretrained/config_{dataset_name}.yaml")
    summaries = []
    for current_scene in tqdm(scene_names):
        scene_id = current_scene.replace("scene", "")
        processed_file = (
            osp.join(path_2_dataset, current_scene, f"{scene_id}.npy")
            if dataset_name == "scannet200"
            else None
        )

        openyolo3d.predict(
            path_2_scene_data=osp.join(path_2_dataset, current_scene),
            depth_scale=depth_scale,
            datatype=datatype,
            processed_scene=processed_file,
            path_to_3d_masks=path_to_3d_masks,
            is_gt=False,
            path_to_2d_preds=path_to_2d_preds,
            save_2d_preds=save_2d_preds,
            reuse_2d_preds=reuse_2d_preds,
        )

        json_path, candidates = export_context_candidates(
            openyolo3d=openyolo3d,
            scene_name=current_scene,
            output_dir=output_dir,
            top_views=top_views,
            uncertain_score_th=uncertain_score_th,
            uncertain_margin_th=uncertain_margin_th,
            longtail_classes=longtail_classes,
            max_candidates=max_candidates_per_scene,
            max_mask_point_ratio=max_mask_point_ratio,
            max_bbox_area_ratio=max_bbox_area_ratio,
            min_visible_points=min_visible_points,
            min_label_votes=min_label_votes,
            include_bad_quality=include_bad_quality,
        )
        summaries.append(
            {
                "scene_name": current_scene,
                "num_candidates": len(candidates),
                "json_path": json_path,
            }
        )

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

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = osp.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"dataset_name": dataset_name, "scenes": summaries}, f, indent=2)

    print(f"Saved context candidate summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--output_dir", default="./output/context_candidates")
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--top_views", default=3, type=int)
    parser.add_argument("--uncertain_score_th", default=0.35, type=float)
    parser.add_argument("--uncertain_margin_th", default=0.15, type=float)
    parser.add_argument("--longtail_classes", default="")
    parser.add_argument("--max_candidates_per_scene", default=None, type=int)
    parser.add_argument("--max_mask_point_ratio", default=0.20, type=float)
    parser.add_argument("--max_bbox_area_ratio", default=0.65, type=float)
    parser.add_argument("--min_visible_points", default=50, type=int)
    parser.add_argument("--min_label_votes", default=10, type=int)
    parser.add_argument("--include_bad_quality", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None, help="Optional directory or .pt file for cached YOLO-World 2D detections")
    parser.add_argument("--save_2d_preds", default=False, action=argparse.BooleanOptionalAction, help="Save YOLO-World 2D detections to --path_to_2d_preds after inference")
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction, help="Reuse cached YOLO-World 2D detections when available")
    args = parser.parse_args()

    export_dataset_candidates(
        dataset_name=args.dataset_name,
        path_to_3d_masks=args.path_to_3d_masks,
        output_dir=args.output_dir,
        scene_name=args.scene_name,
        top_views=args.top_views,
        uncertain_score_th=args.uncertain_score_th,
        uncertain_margin_th=args.uncertain_margin_th,
        longtail_classes=parse_longtail_classes(args.longtail_classes),
        max_candidates_per_scene=args.max_candidates_per_scene,
        max_mask_point_ratio=args.max_mask_point_ratio,
        max_bbox_area_ratio=args.max_bbox_area_ratio,
        min_visible_points=args.min_visible_points,
        min_label_votes=args.min_label_votes,
        include_bad_quality=args.include_bad_quality,
        path_to_2d_preds=args.path_to_2d_preds,
        save_2d_preds=args.save_2d_preds,
        reuse_2d_preds=args.reuse_2d_preds,
    )


if __name__ == "__main__":
    main()
