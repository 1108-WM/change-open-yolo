#!/usr/bin/env python3
"""Prepare/export the FI1 A/B/C/D frontend comparison in an isolated directory.

Selection uses scene names only. Preparation may succeed as an inventory while
reporting missing assets; export requires all selected frame manifests.
No GT matching, semantic inference, learned scoring or AP evaluation is done here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.fi1_frontend_protocol import (
    ARMS, PARAMETERS, build_scene_manifest, choose_scenes, digest, freeze_json,
    sha256_file, validate_observation, verify_scene_inputs, write_json,
)

SAM_SOURCE = ROOT / "_external/segment-anything/segment-anything-main"
SAM_WEIGHT = ROOT / "pretrained/checkpoints/sam_vit_b_01ec64.pth"
GD_SOURCE = ROOT / "_external/GroundingDINO/GroundingDINO-main"
GD_WEIGHT = ROOT / "pretrained/groundingdino/groundingdino_swint_ogc.pth"
GD_CONFIG = GD_SOURCE / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
BERT = ROOT / "pretrained/bert-base-uncased"


def assets():
    files = [SAM_WEIGHT, GD_WEIGHT, GD_CONFIG, ROOT / "pretrained/config_scannet200.yaml",
             ROOT / "utils/__init__.py", Path(__file__),
             ROOT / "tools/fi1_frontend_protocol.py",
             ROOT / "tools/fi1_mask_projection.py",
             ROOT / "tools/export_groundingdino_f30_boxes.py"]
    files += [ROOT / "tools" / name for name in (
        "build_details_same_frame_hierarchy_preprocessor.py", "build_details_frame_siou_tracks.py",
        "refine_details_automatic_tracks_consensus.py", "build_details_iterative_proposal_merges.py")]
    files += sorted((SAM_SOURCE / "segment_anything").rglob("*.py"))
    files += sorted((GD_SOURCE / "groundingdino").rglob("*.py"))
    files += [BERT / name for name in ("config.json", "pytorch_model.bin", "model.safetensors",
                                       "tokenizer.json", "tokenizer_config.json", "vocab.txt")]
    return {str(p): sha256_file(p) for p in files}


def fixed_selection(root):
    """Consume a previously frozen val48 selection; never select by asset availability."""
    root = Path(root).resolve()
    receipt_path = root / 'selection_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get('complete'):
        raise ValueError('external selection incomplete')
    for name, expected in receipt['inputs'].items():
        if sha256_file(name) != expected:
            raise ValueError('external selection input changed: ' + name)
    for name, expected in receipt['outputs'].items():
        if sha256_file(root / name) != expected:
            raise ValueError('external selection output changed: ' + name)
    selected = json.loads((root / 'selection.json').read_text())
    if selected.get('dataset_role') == 'additional_train_g10_pilot30':
        master_path = ROOT / '资料/scannet_new397_20260914_master397.txt'
        split_path = ROOT / '.local/fi1_pairwise_small_20260918_v5/split.json'
        master = master_path.read_text().split()
        split = json.loads(split_path.read_text())
        excluded = set(split['holdout_main_scenes'])
        representatives = {}
        for scan in sorted(master):
            representatives.setdefault(scan.split('_')[0], scan)
        seed = selected['seed']
        ranked = sorted(
            (main, scan) for main, scan in representatives.items() if main not in excluded
        )
        expected = [scan for main, scan in sorted(
            ranked,
            key=lambda item: (hashlib.sha256((seed + item[0]).encode()).hexdigest(), item[0]),
        )[:30]]
        scenes = (root / 'scenes_execution_order.txt').read_text().split()
        folds = json.loads((root / 'folds.json').read_text())
        heldout = [scene for fold in folds['folds'] for scene in fold['holdout_scans']]
        if (scenes != expected or len(scenes) != 30 or len(set(scenes)) != 30
                or len({scene.split('_')[0] for scene in scenes}) != 30
                or set(heldout) != set(scenes) or len(heldout) != 30
                or len(folds['folds']) != 5
                or any(len(fold['holdout_scans']) != 6 for fold in folds['folds'])
                or any(set(fold['fit_scans']) | set(fold['holdout_scans']) != set(scenes)
                       or set(fold['fit_scans']) & set(fold['holdout_scans'])
                       for fold in folds['folds'])
                or selected.get('pilot_scenes') != scenes[:1]
                or selected.get('active_arms') != ['D']
                or selected.get('stride') != 10):
            raise ValueError('G10 pilot30 selection/folds differ from frozen name-only rule')
        pilot_arms = dict(ARMS)
        pilot_arms['D'] = {**ARMS['D'], 'sampling': 'original_stride10'}
        return {'version': 4, 'seed': seed, 'scenes': scenes,
                'selection_root': str(root), 'selection_receipt_sha256': sha256_file(receipt_path),
                'selection_inputs': 'new397 names only; exclude prior pairwise holdout20; SHA-256 rank',
                'dataset_role': selected['dataset_role'], 'arms': pilot_arms,
                'parameters': {**PARAMETERS, 'pilot_stride': 10},
                'variant': 'grounded_sam_stride10_matched_pilot30', 'stride': 10,
                'semantic_evaluation': 'own-mask YOLO/Alpha acquisition; five main-scene-isolated folds',
                'blind_test': False}
    if selected.get('dataset_role') in ('additional_train199', 'additional_train198'):
        role = selected['dataset_role']
        partition, count = ('first199', 199) if role == 'additional_train199' else ('remaining198', 198)
        master = (ROOT / ('资料/scannet_new397_20260916_' + partition + '.txt')).read_text().split()
        scenes = (root / 'scenes_execution_order.txt').read_text().split()
        if (scenes != master or len(scenes) != count or len(set(scenes)) != count
                or len({s.split('_')[0] for s in scenes}) != count
                or selected.get('pilot_scenes') != master[:3]):
            raise ValueError('additional training selection differs from frozen additional partition/pilot3')
        return {'version': 3, 'seed': selected['seed'], 'scenes': scenes,
                'selection_root': str(root), 'selection_receipt_sha256': sha256_file(receipt_path),
                'selection_inputs': 'original additional partition order; pilot first3; no GT/AP selection',
                'dataset_role': role, 'arms': ARMS, 'parameters': PARAMETERS,
                'semantic_evaluation': 'own-mask YOLO/Alpha acquisition only; no training/AP',
                'blind_test': False}
    scenes = (root / 'scenes48_execution_order.txt').read_text().split()
    if (len(scenes) != 48 or len(set(scenes)) != 48
            or len({s.split('_')[0] for s in scenes}) != 48
            or sorted(scenes) != (root / 'scenes48.txt').read_text().split()
            or scenes != [s for g in selected['groups'] for s in g['scans']]):
        raise ValueError('external selection scene/group coverage differs')
    return {'version': 2, 'seed': selected['seed'], 'scenes': scenes,
            'selection_root': str(root), 'selection_receipt_sha256': sha256_file(receipt_path),
            'selection_inputs': 'frozen val48 name-only selection; no reselection',
            'arms': ARMS, 'parameters': PARAMETERS,
            'semantic_evaluation': 'separate full D-v3/R3b backend; export only generates geometry',
            'blind_test': False}


def prepare(args):
    selection_root = getattr(args, 'selection_root', None)
    if selection_root:
        selection = fixed_selection(selection_root)
    else:
        selection = legacy_selection(args)
    scenes = selection['scenes']
    freeze_json(args.output / "selection.json", selection)
    (args.output / "scenes.txt").write_text("\n".join(scenes) + "\n")
    inventory = []
    for scene in scenes:
        try:
            manifest = build_scene_manifest(scene, args.dataset_root, args.processed_root or args.dataset_root)
            freeze_json(args.output / "manifests" / f"{scene}.json", manifest)
            inventory.append({"scene": scene, "ready": True,
                              "frame_counts": {k: len(v) for k, v in manifest["arms"].items()},
                              "invalid_pose_frame_ids": [f["frame_id"] for f in manifest["frames"] if not f["pose_valid"]]})
        except (FileNotFoundError, ValueError) as exc:
            inventory.append({"scene": scene, "ready": False, "reason": str(exc)})
        print('prepared', scene, inventory[-1]['ready'], flush=True)
    receipt = {"selection_sha256": digest(selection), "assets": assets()}
    freeze_json(args.output / "contract.json", receipt)
    ready = sum(row["ready"] for row in inventory)
    report = {"state": "ready_for_export" if ready == len(scenes) else "blocked_missing_inputs",
              "ready_scene_count": ready, "required_scene_count": len(scenes), "inventory": inventory,
              "dataset_root": str(args.dataset_root), "ground_truth_usage": "none",
              "real_inference_complete": False, "AP_evaluated": False}
    write_json(args.output / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def legacy_selection(args):
    reference = json.loads(args.reference.read_text())
    # The outer-fold manifest contains all 100 scan IDs; no GT fields are used.
    training = sorted({s for fold in reference["split"]["heldout_folds"] for s in fold})
    excluded = args.exclude_list.read_text().split()
    scenes = choose_scenes(training, excluded)
    return {
        "version": 1, "seed": "fi1-frontends-v1", "scenes": scenes,
        "training_pool": training, "excluded_main_scenes": sorted({s.split('_')[0] for s in excluded}),
        "selection_inputs": "scan names only; no asset/performance/GT-based replacement",
        "arms": ARMS, "parameters": PARAMETERS,
        "semantic_evaluation": "pending separate common-protocol integration; GD labels provenance only",
    }


class Frontends:
    def __init__(self, frontend, receipt_path):
        if frontend != "grounded_sam":
            raise ValueError("Only the retained GroundingDINO/SAM frontend is supported")
        import torch
        import yaml
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; do not silently fall back to CPU")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.manual_seed(0)
        np.random.seed(0)
        sys.path.insert(0, str(SAM_SOURCE))
        from segment_anything import SamPredictor, sam_model_registry
        self.frontend = frontend
        self.sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHT)).eval().cuda()
        self.predictor = SamPredictor(self.sam)
        receipt = {"frontend": frontend, "torch": torch.__version__,
                   "device": torch.cuda.get_device_name(), "sam_strict_load": True}
        if frontend == "grounded_sam":
            sys.path.insert(0, str(GD_SOURCE))
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.misc import clean_state_dict
            from tools.export_groundingdino_f30_boxes import _build_prompt_groups
            cfg = SLConfig.fromfile(str(GD_CONFIG))
            cfg.device = "cuda"
            cfg.text_encoder_type = str(BERT)
            self.gd = build_model(cfg)
            state = clean_state_dict(torch.load(GD_WEIGHT, map_location="cpu")["model"])
            incompatible = self.gd.load_state_dict(state, strict=False)
            # Missing trainable weights are never accepted as a valid pretrained model.
            missing_parameters = set(incompatible.missing_keys) & set(dict(self.gd.named_parameters()))
            if missing_parameters:
                raise RuntimeError(f"Grounding DINO parameter mismatch: {sorted(missing_parameters)}")
            self.gd.eval().cuda()
            self.labels = yaml.safe_load((ROOT / "pretrained/config_scannet200.yaml").read_text())["network2d"]["text_prompts"]
            self.groups = _build_prompt_groups(self.labels, self.gd.tokenizer, int(cfg.max_text_len))
            self.gd_args = SimpleNamespace(box_threshold=0.35, nms_iou=0.30, max_detections=100)
            receipt.update({"gd_missing_keys": incompatible.missing_keys,
                            "gd_unexpected_keys": incompatible.unexpected_keys,
                            "prompt_groups": self.groups, "labels": self.labels})
        write_json(receipt_path, receipt)

    def box_masks(self, image, boxes, labels, scores):
        import torch
        from segment_anything.utils.amg import calculate_stability_score, batched_mask_to_box
        if len(boxes) == 0:
            return []
        self.predictor.set_image(image)
        output = []
        # Small batches bound full-resolution mask-logit memory.
        for start in range(0, len(boxes), 8):
            batch = torch.as_tensor(boxes[start:start + 8], dtype=torch.float32, device="cuda")
            batch = self.predictor.transform.apply_boxes_torch(batch, image.shape[:2])
            with torch.no_grad():
                logits, quality, _ = self.predictor.predict_torch(
                    point_coords=None, point_labels=None, boxes=batch,
                    multimask_output=False, return_logits=True,
                )
            logits, quality = logits[:, 0], quality[:, 0]
            stability = calculate_stability_score(logits, self.sam.mask_threshold, 1.0)
            masks = logits > self.sam.mask_threshold
            mask_boxes = batched_mask_to_box(masks).cpu().numpy()
            for j, (mask, q, stable, bounds) in enumerate(zip(masks.cpu().numpy(), quality.cpu().numpy(),
                                                            stability.cpu().numpy(), mask_boxes)):
                index = start + j
                x1, y1, x2, y2 = bounds.tolist()
                output.append({"segmentation": mask, "predicted_iou": float(q),
                               "stability_score": float(stable), "bbox": [x1, y1, x2-x1, y2-y1],
                               "crop_box": [0, 0, image.shape[1], image.shape[0]],
                               "provenance": {"gd_class_id": int(labels[index]),
                                              "gd_label": self.labels[int(labels[index])],
                                              "gd_score": float(scores[index]),
                                              "gd_box_xyxy": boxes[index].tolist()}})
        self.predictor.reset_image()
        return output

    def generate(self, image, image_path):
        from groundingdino.util.inference import load_image
        from tools.export_groundingdino_f30_boxes import _predict_frame
        _, tensor = load_image(str(image_path))
        boxes, labels, scores = _predict_frame(self.gd, tensor.cuda(), image.shape, self.groups,
                                               self.labels, self.gd.tokenizer, self.gd_args)
        return self.box_masks(image, boxes, labels, scores)


def project_frame(manifest, frame):
    """Use the unchanged baseline projection implementation one frame at a time."""
    import imageio.v2 as imageio
    import torch
    from utils import WORLD_2_CAM
    world = WORLD_2_CAM.__new__(WORLD_2_CAM)
    world.poses = [frame["pose"]["path"]]
    world.intrinsics = [manifest["files"]["intrinsics"]["path"]]
    world.mesh = manifest["files"]["mesh"]["path"]
    world.depth_maps_paths = [frame["depth"]["path"]]
    world.color_paths = [frame["color"]["path"]]
    world.image_resolution = imageio.imread(world.color_paths[0]).shape[:2]
    world.depth_resolution = imageio.imread(world.depth_maps_paths[0]).shape
    world.height, world.width = world.depth_resolution
    world.depth_scale, world.vis_depth_threshold = 1000.0, 0.05
    world.device = torch.device("cuda")
    projections, visibility = world.get_mesh_projections()
    scaling = tuple(d / c for d, c in zip(world.depth_resolution, world.image_resolution))
    return projections.numpy(), visibility.numpy(), scaling


def check_point_order(manifest):
    from utils import WORLD_2_CAM
    points, _ = WORLD_2_CAM.load_ply(manifest["files"]["mesh"]["path"])
    processed = np.load(manifest["files"]["processed"]["path"], mmap_mode="r", allow_pickle=False)
    if processed.ndim != 2 or processed.shape[1] < 10 or not np.array_equal(processed[:, :3], points[:, :3]):
        raise ValueError("processed superpoints and mesh do not share exact XYZ/point order")
    return len(points)


def export_frame(model, manifest, frame, target, identity, point_count):
    import imageio.v2 as imageio
    import torch
    from tools.fi1_mask_projection import (
        encode_binary_mask_rle, exact_mask_relation, mask_to_visible_points, _select_masks,
    )
    completed = target / "receipt.json"
    if completed.exists():
        receipt = json.loads(completed.read_text())
        if receipt["identity"] != identity:
            raise ValueError(f"stale frame cache: {target}")
        for name, sha in receipt["files"].items():
            if sha256_file(target / name) != sha:
                raise ValueError(f"damaged frame cache: {target / name}")
        return receipt
    # A crash leaves an isolated staging folder; never accept it as complete.
    staging = target.with_name(target.name + f".writing.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    image = np.ascontiguousarray(imageio.imread(frame["color"]["path"])[:, :, :3])
    torch.cuda.synchronize()
    start = time.perf_counter()
    if frame.get("pose_valid", True):
        projections, visibility, scaling = project_frame(manifest, frame)
    else:
        projections = np.zeros((1, point_count, 2), dtype=np.int16)
        visibility = np.zeros((1, point_count), dtype=bool)
        scaling = (1.0, 1.0)  # No projection/mask lookup is performed for an invalid pose.
    torch.cuda.synchronize()
    projection_seconds = time.perf_counter() - start
    start = time.perf_counter()
    generated = model.generate(image, frame["color"]["path"]) if frame.get("pose_valid", True) else []
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    candidates = []
    for item in generated:
        # Match automatic SAM's strict predicted-IoU and inclusive stability gate.
        if item["predicted_iou"] <= 0.88 or item["stability_score"] < 0.95:
            continue
        mask = np.asarray(item["segmentation"], dtype=bool)
        if mask.sum() < 64:
            continue
        points = mask_to_visible_points(mask, 0, projections, visibility, scaling)
        if len(points) < 20:
            continue
        candidates.append({"mask": mask, "points": points, "area": int(mask.sum()),
                           "predicted_iou": float(item["predicted_iou"]),
                           "stability_score": float(item["stability_score"]),
                           "bbox_xywh": list(map(float, item["bbox"])),
                           "crop_box_xywh": list(map(float, item["crop_box"])),
                           "provenance": item.get("provenance", {})})
    kept = _select_masks(candidates, 40)
    rows, relations = [], []
    for i, item in enumerate(kept):
        filename = f"obs{i:03d}_points.npz"
        np.savez_compressed(staging / filename, point_indices=item["points"])
        row = {k: v for k, v in item.items() if k not in ("points", "mask")}
        row.update({"observation_id": i, "scene_name": manifest["scene_name"],
                    "frame_id": frame["frame_id"], "frame_index": frame["frame_index"],
                    "point_indices_path": str(target / filename),
                    "mask_rle": encode_binary_mask_rle(item["mask"])})
        validate_observation({**row, "point_indices_path": str(staging / filename)}, frame, point_count)
        rows.append(row)
        for j in range(i):
            rel = exact_mask_relation(kept[j]["mask"], item["mask"])
            if rel["intersection_pixel_count"]:
                relations.append({"left_observation_id": j, "right_observation_id": i,
                                  "left_area": kept[j]["area"], "right_area": item["area"], **rel})
    np.save(staging / "visibility.npy", visibility[0], allow_pickle=False)
    write_json(staging / "observations.json", rows)
    write_json(staging / "relations.json", relations)
    receipt = {"identity": identity, "generated_mask_count": len(generated),
               "inference_skipped_invalid_pose": not frame.get("pose_valid", True),
               "eligible_mask_count": len(candidates), "kept_observation_count": len(rows),
               "projection_seconds": projection_seconds, "inference_seconds": inference_seconds,
               "files": {p.name: sha256_file(p) for p in sorted(staging.iterdir()) if p.is_file()}}
    write_json(staging / "receipt.json", receipt)
    staging.rename(target)
    return receipt


def assemble_arm(output, arm, manifest, frame_root, identity):
    scene = manifest["scene_name"]
    target = output / "arms" / arm / "observations" / scene
    if target.exists():
        old = json.loads((target / "summary.json").read_text())
        if old["identity"] != identity:
            raise ValueError("arm output identity differs")
        for name, sha in old["output_sha256"].items():
            if sha256_file(target / name) != sha:
                raise ValueError("arm output checksum differs")
        return
    rows, relations, frames = [], [], []
    chosen = set(map(str, manifest["arms"][arm]))
    for frame in manifest["frames"]:
        if frame["frame_id"] not in chosen:
            continue
        cache = frame_root / f"frame{int(frame['frame_id']):06d}"
        local = json.loads((cache / "observations.json").read_text())
        offset = len(rows)
        rows.extend({**r, "observation_id": offset + r["observation_id"]} for r in local)
        for relation in json.loads((cache / "relations.json").read_text()):
            relations.append({**relation, "scene_name": scene, "frame_id": frame["frame_id"],
                              "frame_index": frame["frame_index"],
                              "left_observation_id": offset + relation["left_observation_id"],
                              "right_observation_id": offset + relation["right_observation_id"]})
        receipt = json.loads((cache / "receipt.json").read_text())
        frames.append({"frame_id": frame["frame_id"], "frame_index": frame["frame_index"],
                       **{k: receipt[k] for k in ("generated_mask_count", "eligible_mask_count",
                                                "kept_observation_count", "inference_seconds")}})
        frames[-1]["inference_skipped_invalid_pose"] = receipt["inference_skipped_invalid_pose"]
    staging = target.with_name(target.name + f".writing.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    for name, records in (("automatic_observations.jsonl", rows), ("same_frame_mask_relations.jsonl", relations)):
        (staging / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write_json(staging / "summary.json", {
        "identity": identity, "scene_name": scene, "arm": arm, "frame_count": len(frames),
        "observation_count": len(rows), "same_frame_mask_relation_count": len(relations),
        "mask_rle_saved": True, "exact_same_frame_relations_saved": True, "frames": frames,
        "ground_truth_usage": "none", "frame_index_domain": "manifest.frames",
        "output_sha256": {p.name: sha256_file(p) for p in staging.iterdir()},
        "timing_note": "per-arm sum of cached frame inference times; model load excluded; A/B and C/D share frames",
    })
    staging.rename(target)


class FrameVisibility:
    """Lazy visibility rows in the explicit original-frame manifest domain."""
    def __init__(self, manifest, frame_root, point_count):
        self.frames = manifest["frames"]
        self.root = frame_root
        self.point_count = point_count

    def __getitem__(self, index):
        if not 0 <= index < len(self.frames):
            raise IndexError(index)
        frame = self.frames[index]
        if frame["frame_index"] != index:
            raise ValueError("noncanonical frame-index domain")
        path = self.root / f"frame{int(frame['frame_id']):06d}" / "visibility.npy"
        row = np.load(path, mmap_mode="r", allow_pickle=False)
        if row.shape != (self.point_count,) or row.dtype != np.bool_:
            raise ValueError("visibility does not match mesh point domain")
        return row


def track_arm(output, arm, manifest, frame_root, point_count, identity):
    """Reuse the frozen four geometry stages, passing matching visibility rows."""
    from tools import build_details_same_frame_hierarchy_preprocessor as hierarchy
    from tools import build_details_frame_siou_tracks as tracking
    from tools import refine_details_automatic_tracks_consensus as consensus
    from tools import build_details_iterative_proposal_merges as merge
    scene = manifest["scene_name"]
    root = output / "arms" / arm
    processed_root = Path(manifest["files"]["processed"]["path"]).parent.parent
    common = dict(processed_scene_root=processed_root, dataset_root=processed_root,
                  min_superpoint_visible_ratio=0.10, min_superpoint_mask_support=0.30,
                  min_tracking_siou=0.30, min_track_views=2,
                  min_initial_mask_support=0.30, min_visible_points_per_superpoint=3,
                  frame_superpoint_coverage=0.50, min_support_frames=2,
                  min_consensus_rate=0.30, mean_superpoint_coverage=0.55,
                  policy="hierarchy_safe", duplicate_min_coverage=0.95)
    visibility = FrameVisibility(manifest, frame_root, point_count)
    stages = [
        ("hierarchy", hierarchy._build_scene, {"automatic_root": root / "observations"}, False),
        ("tracks", tracking._build_scene, {"automatic_root": root / "hierarchy"}, True),
        ("consensus", consensus._refine_scene, {"automatic_root": root / "hierarchy", "track_root": root / "tracks"}, True),
        ("merged", merge._build_and_publish_scene, {"automatic_root": root / "hierarchy", "track_root": root / "consensus"}, True),
    ]
    for stage, function, paths, needs_visibility in stages:
        target = root / stage / scene
        receipt_path = root / "stage_receipts" / f"{scene}_{stage}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["identity"] != identity:
                raise ValueError("tracking stage identity changed")
            for name, sha in receipt["files"].items():
                if sha256_file(target / name) != sha:
                    raise ValueError("tracking stage output changed")
            continue
        kwargs = SimpleNamespace(**common, **paths, output_root=target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        summary = function(scene, kwargs, visibility=visibility) if needs_visibility else function(scene, kwargs)
        write_json(receipt_path, {"identity": identity, "seconds": time.perf_counter() - start,
                                 "files": {str(p.relative_to(target)): sha256_file(p)
                                           for p in sorted(target.rglob('*')) if p.is_file()}})
        print("geometry_stage", arm, scene, stage, summary.get("track_count", summary.get("final_proposal_count")), flush=True)


def export_scope(selection, requested_scenes=None, requested_arms=None):
    scenes = requested_scenes if requested_scenes is not None else selection['scenes']
    arms = requested_arms if requested_arms is not None else ['D']
    if not scenes or len(set(scenes)) != len(scenes) or not set(scenes) <= set(selection['scenes']):
        raise ValueError('export scans must be a unique nonempty subset of frozen selection')
    if arms != ['D']:
        raise ValueError('Only retained Full arm D can be exported')
    return [s for s in selection['scenes'] if s in scenes], [a for a in ARMS if a in arms]


def export_status(output, selection, contract):
    """Partial runs must never publish a full-scene/full-arm completion marker."""
    completed = []
    for scene in selection['scenes']:
        manifest = json.loads((output / 'manifests' / f'{scene}.json').read_text())
        for arm in ['D']:
            spec = ARMS[arm]
            identity = digest({'contract': digest(contract), 'manifest': digest(manifest),
                               'frontend': spec['frontend'], 'arm': arm})
            present = True
            for stage in ['hierarchy', 'tracks', 'consensus', 'merged']:
                rp = output / 'arms' / arm / 'stage_receipts' / f'{scene}_{stage}.json'
                if not rp.exists():
                    present = False
                    continue
                receipt = json.loads(rp.read_text())
                if receipt['identity'] != identity:
                    raise ValueError('completed stage identity changed')
                target = output / 'arms' / arm / stage / scene
                for name, expected in receipt['files'].items():
                    if sha256_file(target / name) != expected:
                        raise ValueError('completed stage output changed')
            if present:
                completed.append({'scene': scene, 'arm': arm})
    full = len(completed) == len(selection['scenes'])
    report = {'scene_count': len(selection['scenes']), 'arm_count': 1,
              'completed_scene_arms': completed, 'completed_scene_arm_count': len(completed),
              'required_scene_arm_count': len(selection['scenes']),
              'tracking_complete': full, 'AP_evaluated': False,
              'contract_sha256': digest(contract)}
    write_json(output / 'export_status.json', report)
    if full:
        write_json(output / 'observations_complete.json', report)
    elif (output / 'observations_complete.json').exists():
        raise ValueError('full completion marker contradicts available stages')
    return report


def export(args):
    selection = json.loads((args.output / "selection.json").read_text())
    contract = json.loads((args.output / "contract.json").read_text())
    if contract["selection_sha256"] != digest(selection) or contract["assets"] != assets():
        raise ValueError("frozen selection/code/model identity changed")
    if selection.get('selection_root') and fixed_selection(selection['selection_root']) != selection:
        raise ValueError('external frozen selection changed')
    scenes, arms = export_scope(selection, getattr(args, 'scenes', None), getattr(args, 'arms', None))
    preflight = json.loads((args.output / "preflight.json").read_text())
    if preflight["state"] != "ready_for_export":
        raise ValueError("selected inputs incomplete; inspect preflight.json, then rerun prepare")
    for frontend in ("grounded_sam",):
        active_arms = [a for a in arms if ARMS[a]['frontend'] == frontend]
        if not active_arms:
            continue
        model = Frontends(frontend, args.output / f"{frontend}_model_receipt.json")
        for scene in scenes:
            manifest = json.loads((args.output / "manifests" / f"{scene}.json").read_text())
            verify_scene_inputs(manifest)
            point_count = check_point_order(manifest)
            base_identity = {"contract": digest(contract), "manifest": digest(manifest), "frontend": frontend}
            frame_root = args.output / "frame_cache" / frontend / scene
            required_frames = {str(i) for a in active_arms for i in manifest['arms'][a]}
            for frame in manifest["frames"]:
                if frame['frame_id'] not in required_frames:
                    continue
                identity = digest({**base_identity, "frame": frame})
                receipt = export_frame(model, manifest, frame,
                                       frame_root / f"frame{int(frame['frame_id']):06d}", identity, point_count)
                write_json(args.output / "progress.json", {"frontend": frontend, "scene": scene,
                           "frame_id": frame["frame_id"], "frame_index": frame["frame_index"],
                           "frames_in_scene": len(manifest["frames"]), "receipt": receipt})
                print(frontend, scene, frame["frame_id"], receipt["kept_observation_count"], flush=True)
            for arm in active_arms:
                arm_identity = digest({**base_identity, "arm": arm})
                assemble_arm(args.output, arm, manifest, frame_root, arm_identity)
                track_arm(args.output, arm, manifest, frame_root, point_count, arm_identity)
        del model
        import torch
        torch.cuda.empty_cache()
    export_status(args.output, selection, contract)


def smoke(args):
    """Exercise GD and box-prompted SAM on a synthetic image."""
    import imageio.v2 as imageio
    image = np.full((240, 320, 3), 180, dtype=np.uint8)
    image[50:180, 80:240] = [40, 90, 160]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "synthetic_smoke.png"
    imageio.imwrite(path, image)
    result = {"input": "synthetic; no ScanNet images/GT; not experimental evidence", "frontends": {}}
    for frontend in ("grounded_sam",):
        model = Frontends(frontend, args.output / f"{frontend}_model_receipt.json")
        start = time.perf_counter()
        masks = model.generate(image, path)
        row = {"returned_masks": len(masks), "seconds": time.perf_counter() - start}
        if frontend == "grounded_sam":
            forced = model.box_masks(image, np.array([[80., 50., 240., 180.]], np.float32), [0], [1.])
            row["synthetic_box_sam_quality"] = [{k: x[k] for k in ("predicted_iou", "stability_score")} for x in forced]
        result["frontends"][frontend] = row
        del model
        import torch
        torch.cuda.empty_cache()
    write_json(args.output / "smoke.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "export", "smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared"))
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument('--selection-root', type=Path, help='Previously frozen val48 selection; avoids legacy train20 selection')
    parser.add_argument('--scenes', nargs='+', help='Export only these frozen scans; preserve global completion coverage')
    parser.add_argument('--arms', nargs='+', choices=list(ARMS), help='Export selected arms, reusing shared frame caches on later runs')
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/fi1_scene_reference_20260911/references/fold_0/fit/reference.json")
    parser.add_argument("--exclude-list", type=Path, default=ROOT / "output/scannet200/scene_splits/ncs_independent_20260813/ncs_validation60.txt")
    args = parser.parse_args()
    args.output = args.output.resolve()
    {"prepare": prepare, "export": export, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
