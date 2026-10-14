#!/usr/bin/env python3
"""Export independent YOLO/Alpha evidence for unique FI1 refined geometries.

--prepare-only validates geometry bytes and writes requests without GPU/GT/AP.
Full export reprojects each refined mask and encodes its own visible crops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_official100_fi1_b2_multisemantic import (
    EVIDENCE_NAME, digest_record, output_directory, read_jsonl, sha256_file,
    unit_score, validate_b1, validate_evidence, write_json, write_jsonl,
)

REQUESTS_NAME = "refined_geometry_requests.jsonl"
VERSION = "fi1_refined_geometry_own_evidence_v3_1"


def resolve_points(locator):
    import numpy as np
    path = Path(locator["points_path"])
    if not path.is_absolute() or locator.get("kind") != "point_indices_npz":
        raise ValueError("refined geometry requires an absolute point_indices_npz locator")
    with np.load(path, allow_pickle=False) as payload:
        raw = payload[locator.get("array_key", "point_indices")]
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("point indices must be a one-dimensional integer array")
        points = np.unique(raw.astype(np.int64))
    if not len(points) or points[0] < 0:
        raise ValueError("empty/negative point indices")
    return points


def verified_geometry(row):
    points = resolve_points(row["geometry_locator_read_only"])
    hashes = {hashlib.sha1(points.tobytes()).hexdigest(), hashlib.sha256(points.tobytes()).hexdigest()}
    if row["geometry_hash"] not in hashes or len(points) != row["point_count"]:
        raise ValueError("geometry locator/hash/point count mismatch")
    return points, hashes


def resolve_b1_points(locator):
    """Resolve a purported duplicate from actual B1 bytes, not a summary flag."""
    if locator.get("kind") == "point_indices_npz":
        return resolve_points(locator)
    import numpy as np
    if locator.get("kind") != "native_mask_column":
        raise ValueError("unsupported B1 duplicate locator")
    path = Path(locator["masks_path"])
    if not path.is_absolute():
        raise ValueError("B1 masks path must be absolute")
    masks = np.load(path, mmap_mode="r", allow_pickle=False)
    column = locator["column_index"]
    if masks.ndim != 2 or type(column) is not int or not 0 <= column < masks.shape[1]:
        raise ValueError("invalid B1 mask column")
    return np.flatnonzero(masks[:, column]).astype(np.int64)


def validate_parameters(args):
    for name in ("top_views", "min_visible_points"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"invalid {name}")
    if not math.isfinite(args.crop_padding_ratio) or args.crop_padding_ratio < 0:
        raise ValueError("invalid crop padding")
    if type(args.alpha_mask_dilation_iters) is not int or args.alpha_mask_dilation_iters < 0:
        raise ValueError("invalid mask dilation")
    for name in ("geometry_root", "b1_root", "config_path", "output_root", "stage_c_root",
                 "prepared_root", "alpha_clip_source", "alpha_clip_base_model", "alpha_clip_checkpoint"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    if not args.prepare_only:
        if not str(args.device).startswith("cuda"):
            raise ValueError("full Alpha export requires an explicit CUDA device")
        template = args.yolo_cache_template
        if not template or "{scene}" not in template or not Path(template).is_absolute():
            raise ValueError("YOLO cache template must be absolute and contain {scene}")
        if args.prepared_root is None:
            raise ValueError("full export requires prepared-root")
        for name in ("alpha_clip_source", "alpha_clip_base_model", "alpha_clip_checkpoint"):
            if not getattr(args, name).exists():
                raise FileNotFoundError(getattr(args, name))


def prepare_requests(geometry_rows, b1, class_count):
    scenes = validate_b1(b1, class_count)
    legacy_hashes = {scene: {r["geometry_evidence_key"].rsplit(":", 1)[-1] for r in rows}
                     for scene, rows in scenes.items()}
    requests, excluded, seen = [], [], set()
    all_plan_keys = set()
    for row in geometry_rows:
        refined = [m for m in row["fi1_plan_members"] if m.get("append_only") is True]
        if not refined:
            continue
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            raise ValueError("geometry ledger is not no-GT/no-AP")
        scene = row["scene_name"]
        if scene not in scenes:
            raise ValueError("refined geometry has unknown B1 scene")
        points, hashes = verified_geometry(row)
        canonical = hashlib.sha256(points.tobytes()).hexdigest()
        identity = (scene, canonical)
        if identity in seen:
            raise ValueError("duplicate refined geometry; group plan aliases before export")
        seen.add(identity)
        for member in refined:
            match = re.fullmatch(re.escape(scene) + r":union:(\d+):refined", str(member.get("plan_key", "")))
            if match is None or type(member.get("candidate_id")) is not int or int(match.group(1)) != member["candidate_id"]:
                raise ValueError("plan key does not match scene/candidate identity")
            if member["plan_key"] in all_plan_keys:
                raise ValueError("duplicate plan key across geometries")
            all_plan_keys.add(member["plan_key"])
            for flag in ("candidate_deletion", "geometry_mutation", "class_mutation", "ground_truth_read", "ap_computed"):
                if member.get(flag) is not False:
                    raise ValueError(f"refined plan missing or forbidden flag: {flag}")
            if (member.get("candidate_source") != "refined_union"
                    or member["geometry_hash"] not in hashes or member["point_count"] != len(points)):
                raise ValueError("refined plan member identity differs")
            member_points, _ = verified_geometry(member)
            if member_points.tobytes() != points.tobytes():
                raise ValueError("plan aliases resolve to different geometry")
            unit_score(member["fi1_geometry_score"], "FI1 final score")
        keys = sorted(m["plan_key"] for m in refined)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate refined plan alias")
        overlap = hashes & legacy_hashes[scene]
        if row.get("legacy_semantic_member_count", 0) and not overlap:
            raise ValueError("claimed legacy membership has no matching B1 geometry")
        if overlap:
            old = next(r for r in scenes[scene] if r["geometry_evidence_key"].rsplit(":", 1)[-1] in overlap)
            old_points = resolve_b1_points(old["geometry_locator_read_only"])
            if old_points.tobytes() != points.tobytes():
                raise ValueError("B1 hash identity does not match actual duplicate mask")
            excluded.append({"scene_name": scene, "fi1_plan_keys": keys,
                             "reason": "exact_geometry_already_in_b1",
                             "b1_geometry_evidence_key": old["geometry_evidence_key"],
                             "b1_locator": old["geometry_locator_read_only"],
                             "pointwise_verified": True})
            continue
        first = min(refined, key=lambda m: m["plan_key"])
        requests.append({
            "scene_name": scene, "fi1_plan_key": first["plan_key"], "fi1_plan_keys": keys,
            "candidate_id": first["candidate_id"], "geometry_hash": row["geometry_hash"],
            "geometry_evidence_key": row["geometry_evidence_key"],
            "point_count": len(points), "geometry_locator_read_only": row["geometry_locator_read_only"],
            "geometry_file_sha256": sha256_file(row["geometry_locator_read_only"]["points_path"]),
            "fi1_geometry_score": first["fi1_geometry_score"],
            "fi1_final_scores_by_plan": {m["plan_key"]: m["fi1_geometry_score"] for m in refined},
            "fi1_geometry_files_by_plan": {m["plan_key"]: {
                "locator": m["geometry_locator_read_only"],
                "sha256": sha256_file(m["geometry_locator_read_only"]["points_path"])} for m in refined},
            "geometry_quality": None, "geometry_quality_provenance": None,
            "ground_truth_read": False, "ap_computed": False,
        })
    return sorted(requests, key=lambda r: (r["scene_name"], r["fi1_plan_key"])), excluded


def attach_quality(requests, stage_c_root):
    if stage_c_root is None:
        return
    path = stage_c_root / "stage_c_v2_refined_union_plan.jsonl"
    rows = read_jsonl(path)
    if any(any(k.startswith("label_") for k in r) for r in rows):
        raise ValueError("Stage C training diagnostics are not a no-GT inference quality input")
    if any(r.get("ground_truth_usage") != "none" or r.get("ap_computed") is not False for r in rows):
        raise ValueError("Stage C quality requires no-GT inference provenance")
    index = {(r["scene_name"], int(r["union_candidate_id"])): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate Stage C quality record")
    digest = sha256_file(path)
    for request in requests:
        row = index[(request["scene_name"], request["candidate_id"])]
        points, _ = verified_geometry(request)
        if row["temporary_refined_sha256"] != hashlib.sha256(points.tobytes()).hexdigest():
            raise ValueError("Stage C quality belongs to a different refined geometry")
        request["geometry_quality"] = unit_score(row["corrected_oof_temporary_refined_quality"], "Stage C quality")
        request["geometry_quality_provenance"] = {"stage_c_sha256": digest,
            "field": "corrected_oof_temporary_refined_quality", "supervised_feature": True}


def load_scene_context(scene, args, config, labels):
    """Load one projection/cache context, shared by whole-mask and paired-region exports."""
    import numpy as np
    import torch
    from utils import WORLD_2_CAM
    cache = Path(str(args.yolo_cache_template).format(scene=scene))
    # Check vocabulary before loading detections; class indices must mean the same thing.
    payload = torch.load(cache, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("YOLO cache lacks metadata")
    if payload["metadata"].get("scene_name") != scene:
        raise ValueError("YOLO cache scene differs")
    if payload.get("metadata", {}).get("text_prompts") != labels:
        raise ValueError(f"{scene}: YOLO cache vocabulary differs")
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("YOLO predictions must be a frame mapping")
    cache_metadata = payload["metadata"]
    del payload
    cache_hash = sha256_file(cache)
    world = WORLD_2_CAM(str(args.prepared_root / scene), float(config["openyolo3d"]["depth_scale"]), config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    if cache_metadata.get("image_resolution") != list(world.image_resolution):
        raise ValueError("YOLO cache image resolution differs from projected images")
    lookup = {Path(p).stem: i for i, p in enumerate(world.color_paths)}
    scaling = (world.depth_resolution[0] / world.image_resolution[0],
               world.depth_resolution[1] / world.image_resolution[1])
    projection_inputs = {str(Path(path).resolve()): sha256_file(path)
                         for path in dict.fromkeys([world.mesh, *world.poses, *world.intrinsics, *world.depth_maps_paths])}
    projection_hash = digest_record(projection_inputs)

    return dict(world=world, projections=projections, visibility=visibility, lookup=lookup,
                scaling=scaling, predictions=predictions, cache_hash=cache_hash,
                projection_inputs=projection_inputs, projection_hash=projection_hash)


def export_scene(requests, args, config, labels, alpha_state, *, scene_context=None):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from utils import WORLD_2_CAM
    from tools.build_z1_yoloworld_multiview_distribution_ledger import _frame_evidence
    from tools.export_automatic_track_alphaclip_semantics import _select_track_views
    from tools.export_multiview_object_clip_features import (
        _encode_alpha_clip_images, _make_crop_alpha_mask, _make_crop_image,
    )
    context = scene_context if scene_context is not None else load_scene_context(requests[0]["scene_name"], args, config, labels)
    world, projections, visibility = (context[k] for k in ("world", "projections", "visibility"))
    lookup, scaling, predictions = (context[k] for k in ("lookup", "scaling", "predictions"))
    cache_hash, projection_inputs, projection_hash = (context[k] for k in ("cache_hash", "projection_inputs", "projection_hash"))
    scene = requests[0]["scene_name"]

    records = []
    for request in requests:
        points, _ = verified_geometry(request)
        if points[-1] >= visibility.shape[1]:
            raise ValueError("geometry and prepared scene point spaces differ")
        selected = _select_track_views(points, list(lookup), lookup, projections, visibility,
            scaling, world.image_resolution, args.top_views, args.min_visible_points, args.crop_padding_ratio)
        yolo = np.zeros(len(labels), dtype=np.float64)
        alpha = np.zeros(len(labels), dtype=np.float64)
        alpha_weight = 0.0
        views = []
        for view in selected:
            frame = view["frame_id"]
            weight = view["visible_points"] / len(points)
            yview = None
            if frame in predictions:
                yview, values = _frame_evidence(points, view["frame_index"], projections, visibility,
                                                scaling, predictions[frame], len(labels))
                for c, value in values.items():
                    yolo[c] += weight * value
            image_path = Path(world.color_paths[view["frame_index"]])
            rgb = np.asarray(imageio.imread(image_path))
            coords = projections[view["frame_index"], view["visible_point_ids"]]
            coords = np.stack([np.round(coords[:, 0] / scaling[1]).astype(np.int64),
                               np.round(coords[:, 1] / scaling[0]).astype(np.int64)], axis=1)
            bbox = tuple(view["bbox_xyxy"])
            crop = _make_crop_image(rgb, bbox, coords=coords, mask_background=False)
            mask = _make_crop_alpha_mask(rgb.shape[:2], bbox, coords=coords, dilation_iters=args.alpha_mask_dilation_iters)
            mask_name = f"{scene}_union{request['candidate_id']}_frame{frame}_alpha.png"
            mask_path = args.artifact_staging / mask_name
            mask.save(mask_path)
            with torch.inference_mode():
                encoded = _encode_alpha_clip_images(alpha_state, [crop], [mask], args.device)
            probs = encoded["probs"][0].astype(np.float64)
            alpha += weight * probs
            alpha_weight += weight
            views.append({"frame_id": frame, "visible_point_count": view["visible_points"],
                "visible_ratio": weight, "bbox_xyxy": list(bbox), "image_path": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "alpha_mask_sha256": hashlib.sha256(np.asarray(mask).tobytes()).hexdigest(),
                "alpha_mask_path": mask_name, "alpha_mask_file_sha256": sha256_file(mask_path),
                "yolo_frame_evidence": yview, "alpha_probabilities": probs.tolist(),
                "alpha_logits": encoded["logits"][0].tolist()})
        if alpha_weight:
            alpha /= alpha_weight
        binding = {"geometry_hash": request["geometry_hash"], "evidence_scope": "geometry_own"}
        record = {**request, "yolo_scores": yolo.tolist(), "alpha_scores": alpha.tolist(), "views": views,
            "vocabulary_sha256": digest_record(labels), "producer_sha256": sha256_file(Path(__file__)),
            "projection_inputs": projection_inputs, "projection_inputs_sha256": projection_hash,
            "yolo_provenance": {**binding, "input_sha256": cache_hash, "available": bool(yolo.sum())},
            "alpha_provenance": {**binding, "input_sha256": digest_record({"views": views,
                "checkpoint": args.checkpoint_hash, "base": args.base_hash,
                "prompt_template": args.prompt_template}), "available": bool(alpha_weight)}}
        validate_evidence(record, len(labels))
        records.append(record)
    del world, projections, visibility
    return records


def run(args):
    validate_parameters(args)
    import yaml
    geometry_path = args.geometry_root / "geometry_evidence_ledger.jsonl"
    b1_path = args.b1_root / "b1_prediction_column_manifest.jsonl"
    config = yaml.safe_load(args.config_path.read_text())
    labels = config["network2d"]["text_prompts"]
    if len(labels) != 198 or len(set(labels)) != len(labels):
        raise ValueError("expected unique ScanNet vocabulary of 198 classes")
    geometry = read_jsonl(geometry_path)
    b1 = read_jsonl(b1_path)
    requests, excluded = prepare_requests(geometry, b1, len(labels))
    attach_quality(requests, args.stage_c_root)
    provenance = {"geometry_ledger": sha256_file(geometry_path), "b1_manifest": sha256_file(b1_path),
                  "config": sha256_file(args.config_path), "producer": sha256_file(Path(__file__))}
    with output_directory(args.output_root) as out:
        write_jsonl(out / REQUESTS_NAME, requests)
        hashes = {REQUESTS_NAME: sha256_file(out / REQUESTS_NAME)}
        if not args.prepare_only:
            if args.prepared_root is None or args.yolo_cache_template is None:
                raise ValueError("full export requires prepared-root and yolo-cache-template")
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for independent Alpha evidence export")
            from tools.export_multiview_object_clip_features import _load_alpha_clip
            args.checkpoint_hash = sha256_file(args.alpha_clip_checkpoint)
            args.base_hash = sha256_file(args.alpha_clip_base_model)
            with torch.inference_mode():
                state = _load_alpha_clip(args, labels, args.device)
            by_scene = defaultdict(list)
            for request in requests:
                by_scene[request["scene_name"]].append(request)
            evidence = []
            for scene, rows in by_scene.items():
                args.artifact_staging = out
                current = export_scene(rows, args, config, labels, state)
                write_jsonl(out / f"{scene}.jsonl", current)
                evidence.extend(current)
                print(f"[refined evidence] {scene}: {len(current)} geometries", flush=True)
            write_jsonl(out / EVIDENCE_NAME, evidence)
            hashes[EVIDENCE_NAME] = sha256_file(out / EVIDENCE_NAME)
        result = {"version": VERSION, "geometry_count": len(requests),
            "plan_member_count": sum(len(r["fi1_plan_keys"]) for r in requests),
            "excluded_existing_geometry": excluded, "prepare_only": args.prepare_only,
            "class_count": len(labels), "vocabulary_sha256": digest_record(labels),
            "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "artifact_staging"},
            "input_provenance": provenance, "hashes": hashes, "ground_truth_read": False, "ap_computed": False}
        write_json(out / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry-root", type=Path, required=True)
    p.add_argument("--b1-root", type=Path, required=True)
    p.add_argument("--config-path", type=Path, default=ROOT / "pretrained/config_scannet200.yaml")
    p.add_argument("--stage-c-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--prepared-root", type=Path)
    p.add_argument("--yolo-cache-template", help="absolute cache filename containing {scene}")
    p.add_argument("--alpha-clip-source", type=Path, default=ROOT / "_external/AlphaCLIP/AlphaCLIP-main")
    p.add_argument("--alpha-clip-base-model", type=Path, default=ROOT / "pretrained/alpha_clip/checkpoints/ViT-L-14.pt")
    p.add_argument("--alpha-clip-checkpoint", type=Path, default=ROOT / "pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth")
    p.add_argument("--prompt-template", default="a photo of a {label}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-views", type=int, default=3)
    p.add_argument("--min-visible-points", type=int, default=20)
    p.add_argument("--crop-padding-ratio", type=float, default=0.5)
    p.add_argument("--alpha-mask-dilation-iters", type=int, default=5)
    args = p.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
