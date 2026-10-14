#!/usr/bin/env python3
"""Export frozen Mask3D/YOLO-World native assets for an NCS split.

The exporter consumes only prepared RGB-D/pose/mesh arrays.  It does not open
ScanNet labels, aggregation files or prepared ground truth, and never runs AP.
One OpenYOLO3D model instance is reused across scenes; outputs are committed
per scene and may be resumed only when a complete compatible manifest exists.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MASK3D_ROOT = PROJECT_ROOT / "models" / "Mask3D"
YOLO_WORLD_ROOT = PROJECT_ROOT / "models" / "YOLO-World"
for root in (MASK3D_ROOT, YOLO_WORLD_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "output/scannet200/scene_splits/ncs_independent_20260813"
DEFAULT_PREPARED_ROOT = Path("/media/jia/软件1/scannet_train_stream/prepared_ncs")
DEFAULT_RECORDS_ROOT = Path("/media/jia/软件1/scannet_train_stream/records_ncs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError(f"scene list is empty or contains duplicates: {path}")
    return scenes


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _validate_prepared(scene: str, prepared_root: Path, split: str) -> tuple[Path, dict]:
    root = prepared_root / scene
    manifest = _read_json(root / "stream_prepare_manifest.json")
    expected_policy = "train_allowed" if split in {"NCS-train", "additional_train199", "additional_train198"} else "validation_forbidden"
    if manifest.get("scene_name") != scene or manifest.get("ncs_split") != split:
        raise ValueError(f"{scene}: prepared split manifest mismatch")
    if manifest.get("gt_policy") != expected_policy:
        raise ValueError(f"{scene}: prepared GT policy mismatch")
    processed = root / f"{scene.removeprefix('scene')}.npy"
    required = [
        processed,
        root / f"{scene}_vh_clean_2.ply",
        root / "color",
        root / "depth",
        root / "poses",
        root / "intrinsics.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{scene}: incomplete prepared assets: {missing}")
    return processed, manifest


def _save_native(output_root: Path, scene: str, prediction) -> dict:
    values = [value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value) for value in prediction[:3]]
    masks, classes, scores = values
    if masks.ndim != 2 or classes.ndim != 1 or scores.ndim != 1:
        raise ValueError(f"{scene}: invalid native shapes {masks.shape}, {classes.shape}, {scores.shape}")
    if masks.shape[1] != classes.shape[0] or classes.shape[0] != scores.shape[0]:
        raise ValueError(f"{scene}: native candidate dimensions disagree")
    keep = np.asarray(scores >= 0.0, dtype=bool)
    np.save(output_root / f"{scene}_pred_masks.npy", np.asarray(masks[:, keep], dtype=bool))
    np.save(output_root / f"{scene}_pred_classes.npy", np.asarray(classes[keep], dtype=np.int64))
    np.save(output_root / f"{scene}_pred_scores.npy", np.asarray(scores[keep], dtype=np.float32))
    return {"point_count": int(masks.shape[0]), "native_candidate_count": int(keep.sum())}


def _compatible_complete(scene_root: Path, scene: str, split: str) -> bool:
    manifest_path = scene_root / "native_export_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    required = [
        scene_root / "mask3d_masks" / f"{scene}.pt",
        scene_root / "yoloworld_bboxes_2d" / f"{scene}.pt",
        scene_root / "native_cache" / f"{scene}_pred_masks.npy",
        scene_root / "native_cache" / f"{scene}_pred_classes.npy",
        scene_root / "native_cache" / f"{scene}_pred_scores.npy",
    ]
    return (
        manifest.get("scene_name") == scene
        and manifest.get("ncs_split") == split
        and manifest.get("ground_truth_usage") == "none"
        and all(path.is_file() for path in required)
    )


def _clear_model_scene_state(model) -> None:
    for attr in (
        "world2cam", "mesh_projections", "preds_3d", "preds_2d",
        "predicted_masks", "predicated_scores", "predicated_classes",
    ):
        if hasattr(model, attr):
            setattr(model, attr, None)
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--ncs-train", action="store_true")
    split_group.add_argument("--ncs-validation", action="store_true")
    split_group.add_argument("--additional-train", action="store_true")
    parser.add_argument("--prepared-split", choices=["additional_train199", "additional_train198"], default="additional_train199")
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "pretrained/config_scannet200.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--min-free-gib", type=float, default=30.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing CPU fallback")
    split = args.prepared_split if args.additional_train else ("NCS-train" if args.ncs_train else "NCS-validation")
    if args.additional_train:
        if args.scene_list is None or args.prepared_root == DEFAULT_PREPARED_ROOT or args.records_root == DEFAULT_RECORDS_ROOT:
            parser.error("additional training requires explicit scene-list, prepared-root and records-root")
    default_list = DEFAULT_SPLIT_DIR / ("ncs_train100.txt" if args.ncs_train else "ncs_validation60.txt")
    scene_list = (args.scene_list or default_list).resolve()
    prepared_root = args.prepared_root.resolve()
    records_root = args.records_root.resolve()
    config_path = args.config_path.resolve()
    scenes = _scenes(scene_list)
    if args.additional_train:
        partition = "first199" if split == "additional_train199" else "remaining198"
        authorized = _scenes(PROJECT_ROOT / f"资料/scannet_new397_20260916_{partition}.txt")
        if scenes != authorized:
            raise ValueError("Additional scene list must match the frozen partition in order")
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error("max-scenes must be positive")
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    from utils import OpenYolo3D

    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    depth_scale = float(config["openyolo3d"]["depth_scale"])
    os.environ.setdefault("MPLCONFIGDIR", str(records_root / ".mplconfig"))
    records_root.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        _validate_prepared(scene, prepared_root, split)
    state_path = records_root / "native_export_execution.json"
    state = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
             "pid": os.getpid(), "split": split, "total": len(scenes), "completed": [],
             "scene_list_sha256": _sha256(scene_list), "config_sha256": _sha256(config_path),
             "source_sha256": _sha256(Path(__file__)), "ground_truth_usage": "none", "training": False,
             "checkpoints": {key: {"path": config[key]["pretrained_path"],
                                      "sha256": _sha256(PROJECT_ROOT / config[key]["pretrained_path"])}
                             for key in ("network2d", "network3d")}}

    def save_state():
        state["updated_at"] = datetime.now().astimezone().isoformat()
        temp = state_path.with_suffix(".tmp.json")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temp.replace(state_path)

    save_state()
    model = OpenYolo3D(str(config_path))
    prepared_count = 0
    skipped_count = 0
    summaries = []
    try:
        for index, scene in enumerate(scenes, 1):
            started = time.monotonic()
            state["current"] = scene
            save_state()
            print(f"[{index}/{len(scenes)}] {split} {scene}", flush=True)
            if shutil.disk_usage(records_root).free < (args.min_free_gib + 4) * 1024**3:
                raise RuntimeError("Insufficient disk reserve for the next native scene")
            processed_path, prepared_manifest = _validate_prepared(scene, prepared_root, split)
            final_root = records_root / scene
            if final_root.exists():
                if args.resume and _compatible_complete(final_root, scene, split):
                    if args.additional_train:
                        existing = _read_json(final_root / "native_export_manifest.json")
                        if (existing.get("prepared_manifest_sha256") != _sha256(prepared_root / scene / "stream_prepare_manifest.json")
                                or existing.get("config_sha256") != state["config_sha256"]
                                or existing.get("checkpoints") != state["checkpoints"]):
                            raise ValueError(f"{scene}: cached input/model identity changed")
                    skipped_count += 1
                    summaries.append({"scene_name": scene, "state": "already_exported"})
                    state["completed"].append({"scene": scene, "state": "already_exported"})
                    save_state()
                    continue
                raise FileExistsError(f"{scene}: incomplete or incompatible output exists: {final_root}")
            staging = records_root / f".{scene}.tmp.{os.getpid()}"
            if staging.exists():
                raise FileExistsError(f"{scene}: staging output already exists: {staging}")
            staging.mkdir(parents=True)
            mask_root = staging / "mask3d_masks"
            bbox_root = staging / "yoloworld_bboxes_2d"
            native_root = staging / "native_cache"
            for root in (mask_root, bbox_root, native_root):
                root.mkdir()
            try:
                result = model.predict(
                    path_2_scene_data=str(prepared_root / scene),
                    depth_scale=depth_scale,
                    datatype="mesh",
                    processed_scene=str(processed_path),
                    path_to_3d_masks=None,
                    is_gt=False,
                    path_to_2d_preds=str(bbox_root),
                    save_2d_preds=True,
                    reuse_2d_preds=False,
                )
                raw_masks, raw_scores = model.preds_3d
                torch.save((raw_masks.detach().cpu(), raw_scores.detach().cpu()), mask_root / f"{scene}.pt")
                native_summary = _save_native(native_root, scene, result[scene])
                bbox_payload = torch.load(bbox_root / f"{scene}.pt", map_location="cpu")
                if not isinstance(bbox_payload, dict) or "metadata" not in bbox_payload:
                    raise ValueError(f"{scene}: YOLO-World cache lacks signed metadata")
                manifest = {
                    "scene_name": scene,
                    "ncs_split": split,
                    "scene_list_sha256": _sha256(scene_list),
                    "prepared_manifest_sha256": _sha256(prepared_root / scene / "stream_prepare_manifest.json"),
                    "prepared_gt_policy": prepared_manifest["gt_policy"],
                    "dataset_root": str(prepared_root),
                    "config_path": str(config_path),
                    "config_sha256": state["config_sha256"],
                    "checkpoints": state["checkpoints"],
                    "cache_contract": "frozen Mask3D + YOLO-World only",
                    "raw_mask3d_candidate_count": int(raw_scores.numel()),
                    "yoloworld_frame_count": int(len(bbox_payload["predictions"])),
                    **native_summary,
                    "ground_truth_usage": "none",
                    "ap_evaluation": False,
                    "sam_inference": False,
                    "training_label_generation": False,
                }
                (staging / "native_export_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                )
                staging.rename(final_root)
                prepared_count += 1
                summaries.append({"scene_name": scene, "state": "exported", **native_summary})
                elapsed = time.monotonic() - started
                state["completed"].append({"scene": scene, "elapsed_seconds": elapsed,
                                           "frames": manifest["yoloworld_frame_count"], **native_summary})
                save_state()
                print(f"DONE {index}/{len(scenes)} {scene} seconds={elapsed:.2f}", flush=True)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            finally:
                _clear_model_scene_state(model)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save_state()
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "dataset": "ScanNet v2 additional training" if args.additional_train else "ScanNet v2 independent NCS",
        "split": split,
        "scene_list": str(scene_list),
        "scene_list_sha256": _sha256(scene_list),
        "scene_count": len(scenes),
        "exported_this_run": prepared_count,
        "already_exported": skipped_count,
        "records_root": str(records_root),
        "ground_truth_usage": "none",
        "ap_evaluation": False,
        "scenes": summaries,
    }
    summary_path = records_root / f"{split.lower().replace('-', '_')}_native_export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    state.update(status="complete", current=None, finished_at=datetime.now().astimezone().isoformat())
    save_state()
    print(json.dumps({"split": split, "scene_count": len(scenes), "exported": prepared_count, "skipped": skipped_count, "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
