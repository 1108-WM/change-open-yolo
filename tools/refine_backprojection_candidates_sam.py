import argparse
import gc
import json
import os
import os.path as osp
import sys
import time

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200
from run_evaluation import load_yaml
from utils import OpenYolo3D


def _add_sam_to_path(sam_source):
    sam_source = osp.abspath(sam_source)
    if sam_source not in sys.path:
        sys.path.insert(0, sam_source)


def _load_sam_predictor(checkpoint, model_type, device, sam_source):
    _add_sam_to_path(sam_source)
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def _iter_candidate_json_paths(path):
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "backprojection_candidates.json":
                yield osp.join(root, filename)


def _candidate_priority(candidate):
    return (
        -float(candidate.get("proposal_priority", candidate.get("score", 0.0))),
        -float(candidate.get("score", 0.0)),
        -int(candidate.get("support_view_count", 0)),
        int(candidate.get("candidate_id", 0)),
    )


def _select_candidates(candidates, max_per_scene, require_sam_route):
    selected = []
    for candidate in sorted(candidates, key=_candidate_priority):
        refinement = candidate.get("refinement", {})
        if require_sam_route and not refinement.get("needs_sam", False):
            continue
        selected.append(candidate)
        if max_per_scene is not None and len(selected) >= max_per_scene:
            break
    return selected


def _clamp_box(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + 1.0, min(width * 1.0, x2))
    y2 = max(y1 + 1.0, min(height * 1.0, y2))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _mask_to_seed_indices(openyolo3d, frame_idx, sam_mask):
    projections, keep_visible_points = openyolo3d.mesh_projections
    projections_np = projections.detach().cpu().numpy() if torch.is_tensor(projections) else np.asarray(projections)
    visible_np = (
        keep_visible_points.detach().cpu().numpy()
        if torch.is_tensor(keep_visible_points)
        else np.asarray(keep_visible_points)
    )

    image_height, image_width = openyolo3d.world2cam.image_resolution
    visible_indices = np.flatnonzero(visible_np[frame_idx].astype(bool))
    if len(visible_indices) == 0:
        return np.asarray([], dtype=np.int64)

    coords = projections_np[frame_idx, visible_indices].astype(np.float32)
    xs = np.round(coords[:, 0] / openyolo3d.scaling_params[1]).astype(np.int64)
    ys = np.round(coords[:, 1] / openyolo3d.scaling_params[0]).astype(np.int64)
    valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    if not valid.any():
        return np.asarray([], dtype=np.int64)

    visible_indices = visible_indices[valid]
    xs = xs[valid]
    ys = ys[valid]
    inside = sam_mask[ys, xs].astype(bool)
    return np.unique(visible_indices[inside]).astype(np.int64)


def _save_overlay(image, sam_mask, box, output_prefix):
    overlay = image.copy()
    red = np.asarray([255, 0, 0], dtype=np.float32)
    current = overlay[sam_mask].astype(np.float32)
    if len(current) > 0:
        overlay[sam_mask] = (0.35 * current + 0.65 * red).astype(np.uint8)

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(image.shape[1] - 1, x1))
    y1 = max(0, min(image.shape[0] - 1, y1))
    x2 = max(x1 + 1, min(image.shape[1], x2))
    y2 = max(y1 + 1, min(image.shape[0], y2))
    overlay[max(0, y1 - 2):min(image.shape[0], y1 + 3), x1:x2] = [0, 255, 0]
    overlay[max(0, y2 - 3):min(image.shape[0], y2 + 2), x1:x2] = [0, 255, 0]
    overlay[y1:y2, max(0, x1 - 2):min(image.shape[1], x1 + 3)] = [0, 255, 0]
    overlay[y1:y2, max(0, x2 - 3):min(image.shape[1], x2 + 2)] = [0, 255, 0]

    mask_path = f"{output_prefix}_sam_mask.png"
    overlay_path = f"{output_prefix}_sam_overlay.jpg"
    imageio.imwrite(mask_path, (sam_mask.astype(np.uint8) * 255))
    imageio.imwrite(overlay_path, overlay)
    return mask_path, overlay_path


def refine_scene_candidates(
    openyolo3d,
    predictor,
    candidate_json,
    output_dir,
    max_per_scene=10,
    require_sam_route=True,
    min_refined_seed_points=30,
):
    with open(candidate_json) as f:
        payload = json.load(f)

    scene_name = payload["scene_name"]
    candidates = payload.get("candidates", [])
    selected = _select_candidates(candidates, max_per_scene, require_sam_route)
    selected_ids = {int(item["candidate_id"]) for item in selected}

    scene_dir = osp.join(output_dir, scene_name)
    seed_dir = osp.join(scene_dir, "sam_seed_points")
    image_dir = osp.join(scene_dir, "sam_images")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    image_cache = {}
    refined = 0
    skipped = []
    current_frame_idx = None

    for candidate in candidates:
        if int(candidate.get("candidate_id", -1)) not in selected_ids:
            candidate.setdefault("sam_refine", {"status": "not_selected"})
            continue

        frame_idx = int(candidate["frame_index"])
        image_path = candidate.get("evidence", {}).get("color_path")
        if image_path is None:
            skipped.append({"candidate_id": candidate.get("candidate_id"), "reason": "missing_image_path"})
            candidate["sam_refine"] = {"status": "skipped", "reason": "missing_image_path"}
            continue

        if frame_idx not in image_cache:
            image = imageio.imread(image_path)
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            if image.shape[-1] == 4:
                image = image[..., :3]
            image_cache[frame_idx] = image.astype(np.uint8)
        image = image_cache[frame_idx]

        if current_frame_idx != frame_idx:
            predictor.set_image(image)
            current_frame_idx = frame_idx

        box = _clamp_box(candidate["bbox_xyxy"], image.shape[1], image.shape[0])
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        best = int(np.argmax(scores))
        sam_mask = masks[best].astype(bool)
        seed_indices = _mask_to_seed_indices(openyolo3d, frame_idx, sam_mask)
        if len(seed_indices) < min_refined_seed_points:
            skipped.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": "few_refined_seed_points",
                    "num_refined_seed_points": int(len(seed_indices)),
                }
            )
            candidate["sam_refine"] = {
                "status": "skipped",
                "reason": "few_refined_seed_points",
                "sam_score": float(scores[best]),
                "num_refined_seed_points": int(len(seed_indices)),
            }
            continue

        prefix = osp.join(image_dir, f"candidate{int(candidate['candidate_id']):04d}_frame{candidate['frame_id']}")
        mask_path, overlay_path = _save_overlay(image, sam_mask, box, prefix)
        seed_path = osp.join(seed_dir, f"candidate{int(candidate['candidate_id']):04d}_sam_points.npz")
        np.savez_compressed(seed_path, point_indices=seed_indices)

        candidate["refined_seed_points_path"] = seed_path
        candidate["sam_refine"] = {
            "status": "applied",
            "model": "sam_vit_b",
            "sam_score": float(scores[best]),
            "num_refined_seed_points": int(len(seed_indices)),
            "seed_retention_ratio": float(len(seed_indices) / max(1, int(candidate.get("num_seed_points", 0)))),
            "sam_mask_path": mask_path,
            "sam_overlay_path": overlay_path,
        }
        refined += 1

    payload["sam_refine_summary"] = {
        "selected": len(selected),
        "refined": refined,
        "skipped": skipped,
        "max_per_scene": max_per_scene,
        "require_sam_route": require_sam_route,
        "min_refined_seed_points": min_refined_seed_points,
    }
    output_json = osp.join(scene_dir, "backprojection_candidates.json")
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    return output_json, payload["sam_refine_summary"]


def refine_dataset_candidates(
    dataset_name,
    path_to_3d_masks,
    candidates_dir,
    output_dir,
    sam_checkpoint,
    sam_source,
    sam_model_type="vit_b",
    scene_name=None,
    max_per_scene=10,
    require_sam_route=True,
    min_refined_seed_points=30,
    path_to_2d_preds=None,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = _load_sam_predictor(sam_checkpoint, sam_model_type, device, sam_source)
    openyolo3d = OpenYolo3D(f"./pretrained/config_{dataset_name}.yaml")

    candidate_jsons = {osp.basename(osp.dirname(path)): path for path in _iter_candidate_json_paths(candidates_dir)}
    summaries = []
    start_all = time.time()
    for current_scene in tqdm(scene_names):
        if current_scene not in candidate_jsons:
            continue
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
            save_2d_preds=False,
            reuse_2d_preds=reuse_2d_preds,
        )
        output_json, summary = refine_scene_candidates(
            openyolo3d,
            predictor,
            candidate_jsons[current_scene],
            output_dir,
            max_per_scene=max_per_scene,
            require_sam_route=require_sam_route,
            min_refined_seed_points=min_refined_seed_points,
        )
        summaries.append({"scene_name": current_scene, "json_path": output_json, **summary})

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

    os.makedirs(output_dir, exist_ok=True)
    summary_path = osp.join(output_dir, "sam_refine_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "candidates_dir": candidates_dir,
                "sam_checkpoint": sam_checkpoint,
                "sam_model_type": sam_model_type,
                "elapsed_seconds": time.time() - start_all,
                "scenes": summaries,
            },
            f,
            indent=2,
        )
    print(f"Saved SAM refinement summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--candidates_dir", default="./output/backprojection_candidates_replica_mv_routing_m20")
    parser.add_argument("--output_dir", default="./output/backprojection_candidates_replica_mv_sam_top10")
    parser.add_argument("--sam_checkpoint", default="./pretrained/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_source", default="./_external/segment-anything/segment-anything-main")
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--max_per_scene", default=10, type=int)
    parser.add_argument("--require_sam_route", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--min_refined_seed_points", default=30, type=int)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    refine_dataset_candidates(
        dataset_name=args.dataset_name,
        path_to_3d_masks=args.path_to_3d_masks,
        candidates_dir=args.candidates_dir,
        output_dir=args.output_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_source=args.sam_source,
        sam_model_type=args.sam_model_type,
        scene_name=args.scene_name,
        max_per_scene=args.max_per_scene,
        require_sam_route=args.require_sam_route,
        min_refined_seed_points=args.min_refined_seed_points,
        path_to_2d_preds=args.path_to_2d_preds,
        reuse_2d_preds=args.reuse_2d_preds,
    )


if __name__ == "__main__":
    main()
