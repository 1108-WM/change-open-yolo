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


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _iter_candidate_json_paths(path):
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "backprojection_candidates.json":
                yield osp.join(root, filename)


def _read_scene_list(scene_list):
    if scene_list is None:
        return None
    raw = str(scene_list).strip()
    if osp.exists(raw):
        with open(raw) as f:
            return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_seed_path(candidate, candidate_json):
    seed_path = candidate.get("refined_seed_points_path") or candidate.get("seed_points_path")
    if seed_path is None:
        return None
    if osp.exists(seed_path):
        return seed_path
    local_path = osp.join(osp.dirname(candidate_json), seed_path)
    if osp.exists(local_path):
        return local_path
    return seed_path


def _load_seed_indices(candidate, candidate_json, num_points):
    seed_path = _resolve_seed_path(candidate, candidate_json)
    if seed_path is None or not osp.exists(seed_path):
        return None
    payload = np.load(seed_path)
    seed_indices = payload["point_indices"].astype(np.int64)
    seed_indices = seed_indices[(seed_indices >= 0) & (seed_indices < num_points)]
    if len(seed_indices) == 0:
        return None
    return np.unique(seed_indices)


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


def _clamp_box(box, width, height, min_size=2.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + min_size, min(width * 1.0, x2))
    y2 = max(y1 + min_size, min(height * 1.0, y2))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _prepare_image(path):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def _project_seed_box(openyolo3d, frame_idx, seed_indices, projections_np, visible_np, box_padding_ratio):
    image_height, image_width = openyolo3d.world2cam.image_resolution
    visible_seed = seed_indices[visible_np[frame_idx, seed_indices].astype(bool)]
    if len(visible_seed) == 0:
        return None, 0

    coords = projections_np[frame_idx, visible_seed].astype(np.float32)
    xs = coords[:, 0] / openyolo3d.scaling_params[1]
    ys = coords[:, 1] / openyolo3d.scaling_params[0]
    valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    if not valid.any():
        return None, 0

    xs = xs[valid]
    ys = ys[valid]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1.0, ys.max() + 1.0
    pad = box_padding_ratio * max(x2 - x1, y2 - y1)
    box = _clamp_box([x1 - pad, y1 - pad, x2 + pad, y2 + pad], image_width, image_height)
    return box, int(valid.sum())


def _candidate_view_plan(candidate, openyolo3d, seed_indices, projections_np, visible_np, top_views, min_visible_points, box_padding_ratio):
    view_scores = {}
    frame_idx = int(candidate["frame_index"])
    view_scores[frame_idx] = {
        "frame_index": frame_idx,
        "frame_id": str(candidate.get("frame_id", frame_idx)),
        "source": "primary",
        "rank_score": float(candidate.get("score", 0.0)) + 1.0,
    }
    for support in candidate.get("support_views", []):
        support_idx = int(support["frame_index"])
        rank_score = float(support.get("iou", 0.0)) * max(0.01, float(support.get("score", 0.0)))
        rank_score *= np.log1p(float(support.get("visible_seed_points", 0)))
        current = view_scores.get(support_idx)
        if current is None or rank_score > current["rank_score"]:
            view_scores[support_idx] = {
                "frame_index": support_idx,
                "frame_id": str(support.get("frame_id", support_idx)),
                "source": "support",
                "rank_score": float(rank_score),
                "support_iou": float(support.get("iou", 0.0)),
                "support_score": float(support.get("score", 0.0)),
            }

    planned = []
    for view in sorted(view_scores.values(), key=lambda item: -item["rank_score"]):
        box, visible_seed_points = _project_seed_box(
            openyolo3d,
            view["frame_index"],
            seed_indices,
            projections_np,
            visible_np,
            box_padding_ratio,
        )
        if box is None or visible_seed_points < min_visible_points:
            continue
        view = dict(view)
        view["bbox_xyxy"] = [float(v) for v in box.tolist()]
        view["visible_seed_points"] = int(visible_seed_points)
        planned.append(view)
        if len(planned) >= top_views:
            break
    return planned


def _sam_mask_to_counts(openyolo3d, frame_idx, sam_mask, projections_np, visible_np, hit_counts, visible_counts):
    image_height, image_width = openyolo3d.world2cam.image_resolution
    visible_indices = np.flatnonzero(visible_np[frame_idx].astype(bool))
    if len(visible_indices) == 0:
        return 0

    coords = projections_np[frame_idx, visible_indices].astype(np.float32)
    xs = np.round(coords[:, 0] / openyolo3d.scaling_params[1]).astype(np.int64)
    ys = np.round(coords[:, 1] / openyolo3d.scaling_params[0]).astype(np.int64)
    valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    if not valid.any():
        return 0

    visible_indices = visible_indices[valid]
    xs = xs[valid]
    ys = ys[valid]
    visible_counts[visible_indices] += 1
    inside = sam_mask[ys, xs].astype(bool)
    hit_counts[visible_indices[inside]] += 1
    return int(inside.sum())


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


def refine_scene_candidates_multiview(
    openyolo3d,
    predictor,
    candidate_json,
    output_dir,
    max_per_scene=5,
    top_views=3,
    require_sam_route=True,
    min_visible_seed_points=30,
    min_refined_seed_points=30,
    min_vote_count=2,
    min_vote_ratio=0.50,
    box_padding_ratio=0.05,
    fallback_to_best_view=True,
):
    with open(candidate_json) as f:
        payload = json.load(f)

    scene_name = payload["scene_name"]
    candidates = payload.get("candidates", [])
    selected = _select_candidates(candidates, max_per_scene, require_sam_route)
    selected_ids = {int(item["candidate_id"]) for item in selected}

    projections, keep_visible_points = openyolo3d.mesh_projections
    projections_np = _to_numpy(projections).astype(np.int64)
    visible_np = _to_numpy(keep_visible_points).astype(bool)
    num_points = projections_np.shape[1]

    scene_dir = osp.join(output_dir, scene_name)
    seed_dir = osp.join(scene_dir, "sam_multiview_seed_points")
    image_dir = osp.join(scene_dir, "sam_multiview_images")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    image_cache = {}
    current_frame_idx = None
    refined = 0
    skipped = []

    for candidate in candidates:
        candidate_id = int(candidate.get("candidate_id", -1))
        if candidate_id not in selected_ids:
            candidate.setdefault("sam_multiview_refine", {"status": "not_selected"})
            continue

        seed_indices = _load_seed_indices(candidate, candidate_json, num_points)
        if seed_indices is None:
            skipped.append({"candidate_id": candidate_id, "reason": "missing_seed_points"})
            candidate["sam_multiview_refine"] = {"status": "skipped", "reason": "missing_seed_points"}
            continue

        view_plan = _candidate_view_plan(
            candidate,
            openyolo3d,
            seed_indices,
            projections_np,
            visible_np,
            top_views,
            min_visible_seed_points,
            box_padding_ratio,
        )
        if len(view_plan) == 0:
            skipped.append({"candidate_id": candidate_id, "reason": "no_valid_views"})
            candidate["sam_multiview_refine"] = {"status": "skipped", "reason": "no_valid_views"}
            continue

        hit_counts = np.zeros(num_points, dtype=np.uint16)
        visible_counts = np.zeros(num_points, dtype=np.uint16)
        view_records = []
        best_view_indices = None
        best_view_score = -1.0

        for view in view_plan:
            frame_idx = int(view["frame_index"])
            image_path = openyolo3d.world2cam.color_paths[frame_idx]
            if frame_idx not in image_cache:
                image_cache[frame_idx] = _prepare_image(image_path)
            image = image_cache[frame_idx]

            if current_frame_idx != frame_idx:
                predictor.set_image(image)
                current_frame_idx = frame_idx

            box = _clamp_box(view["bbox_xyxy"], image.shape[1], image.shape[0])
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            best = int(np.argmax(scores))
            sam_mask = masks[best].astype(bool)
            num_hits = _sam_mask_to_counts(
                openyolo3d,
                frame_idx,
                sam_mask,
                projections_np,
                visible_np,
                hit_counts,
                visible_counts,
            )
            single_hit_counts = np.zeros(num_points, dtype=np.uint8)
            single_visible_counts = np.zeros(num_points, dtype=np.uint8)
            _sam_mask_to_counts(
                openyolo3d,
                frame_idx,
                sam_mask,
                projections_np,
                visible_np,
                single_hit_counts,
                single_visible_counts,
            )
            single_indices = np.flatnonzero(single_hit_counts > 0).astype(np.int64)
            if float(scores[best]) > best_view_score and len(single_indices) >= min_refined_seed_points:
                best_view_score = float(scores[best])
                best_view_indices = single_indices

            prefix = osp.join(
                image_dir,
                f"candidate{candidate_id:04d}_frame{view.get('frame_id', frame_idx)}",
            )
            mask_path, overlay_path = _save_overlay(image, sam_mask, box, prefix)
            view_records.append(
                {
                    **view,
                    "sam_score": float(scores[best]),
                    "sam_hits": int(num_hits),
                    "sam_mask_path": mask_path,
                    "sam_overlay_path": overlay_path,
                }
            )

        vote_ratio = np.divide(hit_counts, np.maximum(visible_counts, 1), dtype=np.float32)
        refined_indices = np.flatnonzero(
            (hit_counts >= int(min_vote_count))
            & (visible_counts > 0)
            & (vote_ratio >= float(min_vote_ratio))
        ).astype(np.int64)
        aggregation = "visibility_aware_vote"

        if len(refined_indices) < min_refined_seed_points and fallback_to_best_view and best_view_indices is not None:
            refined_indices = best_view_indices
            aggregation = "best_view_fallback"

        if len(refined_indices) < min_refined_seed_points:
            skipped.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "few_multiview_refined_seed_points",
                    "num_refined_seed_points": int(len(refined_indices)),
                    "num_views": int(len(view_records)),
                }
            )
            candidate["sam_multiview_refine"] = {
                "status": "skipped",
                "reason": "few_multiview_refined_seed_points",
                "num_refined_seed_points": int(len(refined_indices)),
                "views": view_records,
            }
            continue

        seed_path = osp.join(seed_dir, f"candidate{candidate_id:04d}_sam_multiview_points.npz")
        np.savez_compressed(
            seed_path,
            point_indices=refined_indices,
            hit_counts=hit_counts[refined_indices],
            visible_counts=visible_counts[refined_indices],
        )

        candidate["refined_seed_points_path"] = seed_path
        candidate["sam_multiview_refine"] = {
            "status": "applied",
            "model": "sam_vit_b",
            "aggregation": aggregation,
            "num_views": int(len(view_records)),
            "min_vote_count": int(min_vote_count),
            "min_vote_ratio": float(min_vote_ratio),
            "num_refined_seed_points": int(len(refined_indices)),
            "seed_retention_ratio": float(len(refined_indices) / max(1, len(seed_indices))),
            "views": view_records,
        }
        refined += 1

    payload["sam_multiview_refine_summary"] = {
        "selected": len(selected),
        "refined": refined,
        "skipped": skipped,
        "max_per_scene": max_per_scene,
        "top_views": top_views,
        "require_sam_route": require_sam_route,
        "min_visible_seed_points": min_visible_seed_points,
        "min_refined_seed_points": min_refined_seed_points,
        "min_vote_count": min_vote_count,
        "min_vote_ratio": min_vote_ratio,
        "box_padding_ratio": box_padding_ratio,
        "fallback_to_best_view": fallback_to_best_view,
    }
    output_json = osp.join(scene_dir, "backprojection_candidates.json")
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    return output_json, payload["sam_multiview_refine_summary"]


def refine_dataset_candidates_multiview(
    dataset_name,
    path_to_3d_masks,
    candidates_dir,
    output_dir,
    sam_checkpoint,
    sam_source,
    sam_model_type="vit_b",
    scene_name=None,
    max_per_scene=5,
    top_views=3,
    require_sam_route=True,
    min_visible_seed_points=30,
    min_refined_seed_points=30,
    min_vote_count=2,
    min_vote_ratio=0.50,
    box_padding_ratio=0.05,
    fallback_to_best_view=True,
    path_to_2d_preds=None,
    reuse_2d_preds=True,
    scene_list=None,
    max_scenes=None,
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
    selected_scene_names = _read_scene_list(scene_list)
    if selected_scene_names is not None:
        allowed = set(scene_names)
        scene_names = [scene for scene in selected_scene_names if scene in allowed]
    if max_scenes is not None:
        scene_names = scene_names[: int(max_scenes)]

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
        output_json, summary = refine_scene_candidates_multiview(
            openyolo3d,
            predictor,
            candidate_jsons[current_scene],
            output_dir,
            max_per_scene=max_per_scene,
            top_views=top_views,
            require_sam_route=require_sam_route,
            min_visible_seed_points=min_visible_seed_points,
            min_refined_seed_points=min_refined_seed_points,
            min_vote_count=min_vote_count,
            min_vote_ratio=min_vote_ratio,
            box_padding_ratio=box_padding_ratio,
            fallback_to_best_view=fallback_to_best_view,
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
    summary_path = osp.join(output_dir, "sam_multiview_refine_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "candidates_dir": candidates_dir,
                "sam_checkpoint": sam_checkpoint,
                "sam_model_type": sam_model_type,
                "elapsed_seconds": time.time() - start_all,
                "params": {
                    "max_per_scene": max_per_scene,
                    "top_views": top_views,
                    "require_sam_route": require_sam_route,
                    "min_visible_seed_points": min_visible_seed_points,
                    "min_refined_seed_points": min_refined_seed_points,
                    "min_vote_count": min_vote_count,
                    "min_vote_ratio": min_vote_ratio,
                    "box_padding_ratio": box_padding_ratio,
                    "fallback_to_best_view": fallback_to_best_view,
                },
                "scenes": summaries,
            },
            f,
            indent=2,
        )
    print(f"Saved multi-view SAM refinement summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--candidates_dir", default="./output/backprojection_candidates_replica_mv_m20")
    parser.add_argument("--output_dir", default="./output/backprojection_candidates_replica_mv_sam_multiview_top5")
    parser.add_argument("--sam_checkpoint", default="./pretrained/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_source", default="./_external/segment-anything/segment-anything-main")
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--max_per_scene", default=5, type=int)
    parser.add_argument("--top_views", default=3, type=int)
    parser.add_argument("--require_sam_route", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--min_visible_seed_points", default=30, type=int)
    parser.add_argument("--min_refined_seed_points", default=30, type=int)
    parser.add_argument("--min_vote_count", default=2, type=int)
    parser.add_argument("--min_vote_ratio", default=0.50, type=float)
    parser.add_argument("--box_padding_ratio", default=0.05, type=float)
    parser.add_argument("--fallback_to_best_view", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--scene_list", default=None)
    parser.add_argument("--max_scenes", default=None, type=int)
    args = parser.parse_args()

    refine_dataset_candidates_multiview(
        dataset_name=args.dataset_name,
        path_to_3d_masks=args.path_to_3d_masks,
        candidates_dir=args.candidates_dir,
        output_dir=args.output_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_source=args.sam_source,
        sam_model_type=args.sam_model_type,
        scene_name=args.scene_name,
        max_per_scene=args.max_per_scene,
        top_views=args.top_views,
        require_sam_route=args.require_sam_route,
        min_visible_seed_points=args.min_visible_seed_points,
        min_refined_seed_points=args.min_refined_seed_points,
        min_vote_count=args.min_vote_count,
        min_vote_ratio=args.min_vote_ratio,
        box_padding_ratio=args.box_padding_ratio,
        fallback_to_best_view=args.fallback_to_best_view,
        path_to_2d_preds=args.path_to_2d_preds,
        reuse_2d_preds=args.reuse_2d_preds,
        scene_list=args.scene_list,
        max_scenes=args.max_scenes,
    )


if __name__ == "__main__":
    main()
