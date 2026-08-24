#!/usr/bin/env python3
"""Independently audit the no-GT DM-SMS-1 Stage D Alpha/SMS ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_dm_sms1_alpha_embedding_ledger import (
    PROMPT_TEMPLATE,
    SCALE_COUNT,
    _array_sha256,
)
from tools.dm_sms_core import compute_sms, sms_keep_mask, visible_ratio_multiscale_feature


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scenes(path: Path) -> list[str]:
    result = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or contains duplicates")
    return result


def _close(left, right, atol=1e-6) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=atol, equal_nan=True))


def _audit_scene(
    scene: str, source_rows: list[dict], ledger_root: Path,
    text_features: np.ndarray, threshold: float,
) -> dict:
    scene_root = ledger_root / "scenes" / scene
    summary_path = scene_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    records_path = scene_root / "records.jsonl"
    records = _read_jsonl(records_path)
    scale_features = np.load(scene_root / "scale_features.npy", allow_pickle=False)
    aggregate = np.load(scene_root / "aggregate_features.npy", allow_pickle=False)
    similarities = np.load(scene_root / "similarities.npy", allow_pickle=False)
    class_stats = np.load(scene_root / "sms_class_stats.npy", allow_pickle=False)
    if summary.get("scene_complete") is not True:
        raise ValueError(f"{scene}: scene output is incomplete")
    if summary.get("ground_truth_read") is not False or summary.get("ap_computed") is not False:
        raise ValueError(f"{scene}: Stage D scene violates no-GT/no-AP contract")
    if len(records) != len(source_rows):
        raise ValueError(f"{scene}: record count differs from Stage C manifest")
    feature_dim = int(text_features.shape[1])
    expected_scale_count = SCALE_COUNT * sum(len(row["views"]) for row in source_rows)
    if scale_features.shape != (expected_scale_count, feature_dim):
        raise ValueError(f"{scene}: scale feature shape mismatch")
    if aggregate.shape != (len(records), feature_dim):
        raise ValueError(f"{scene}: aggregate feature shape mismatch")
    if similarities.shape != (len(records), len(text_features)):
        raise ValueError(f"{scene}: similarity shape mismatch")
    if class_stats.shape != (2, len(text_features)):
        raise ValueError(f"{scene}: SMS class statistics shape mismatch")
    feature_indices = []
    valid_geometry = np.zeros(len(records), dtype=bool)
    recomputed_aggregate = np.full_like(aggregate, np.nan)
    recomputed_similarities = np.full_like(similarities, np.nan)
    sam_mask_count = 0
    scale_hash_count = 0
    for geometry_index, (source, row) in enumerate(zip(source_rows, records)):
        if (
            int(row["geometry_index"]) != geometry_index
            or str(row["geometry_hash"]) != str(source["geometry_hash"])
            or str(row["geometry_key"]) != str(source["geometry_key"])
            or int(row["point_count"]) != int(source["point_count"])
            or len(row["views"]) != len(source["views"])
        ):
            raise ValueError(f"{scene}: Stage C join mismatch at geometry {geometry_index}")
        incomplete_scale = False
        per_view = []
        ratios = []
        for source_view, view in zip(source["views"], row["views"]):
            if (
                int(view["view_rank"]) != int(source_view["view_rank"])
                or str(view["frame_id"]) != str(source_view["frame_id"])
                or not _close(float(view["visible_ratio"]), float(source_view["visible_ratio"]), 1e-12)
                or not _close(view["sam_box_prompt_xyxy"], source_view["sam_box_prompt_xyxy"], 1e-9)
            ):
                raise ValueError(f"{scene}/{row['geometry_hash']}: view manifest mismatch")
            scales = sorted(view["scales"], key=lambda item: int(item["scale_index"]))
            if len(scales) != SCALE_COUNT or [int(item["scale_index"]) for item in scales] != list(range(SCALE_COUNT)):
                complete = False
                continue
            for source_scale, scale in zip(source_view["crop_scales"], scales):
                feature_indices.append(int(scale["feature_index"]))
                if list(scale["crop_xyxy_integer_exclusive"]) != list(
                    source_scale["crop_xyxy_integer_exclusive"]
                ):
                    raise ValueError(f"{scene}: scale crop differs from Stage C")
            if view.get("sam_mask_valid") is not True:
                for scale in scales:
                    feature = scale_features[int(scale["feature_index"])]
                    if scale.get("feature_valid") is not False or np.isfinite(feature).any():
                        raise ValueError(f"{scene}: invalid SAM view has materialized features")
                continue
            if (
                int(view.get("sam_selected_mask_index", -1)) not in (0, 1, 2)
                or not math.isfinite(float(view.get("sam_predicted_iou", float("nan"))))
                or int(view.get("sam_mask_area", 0)) <= 0
                or len(str(view.get("sam_mask_sha256", ""))) != 64
            ):
                raise ValueError(f"{scene}/{row['geometry_hash']}: invalid SAM provenance")
            sam_mask_count += 1
            features = []
            for source_scale, scale in zip(source_view["crop_scales"], scales):
                feature_index = int(scale["feature_index"])
                if (
                    scale.get("feature_valid") is not True
                    or len(str(scale.get("alpha_crop_mask_sha256", ""))) != 64
                ):
                    incomplete_scale = True
                    break
                feature = scale_features[feature_index]
                if (
                    not np.isfinite(feature).all()
                    or not np.isclose(np.linalg.norm(feature), 1.0, rtol=0.0, atol=2e-4)
                    or _array_sha256(feature) != scale.get("feature_sha256")
                ):
                    raise ValueError(f"{scene}: scale feature/hash/norm mismatch")
                scale_hash_count += 1
                features.append(feature)
            if len(features) != SCALE_COUNT:
                incomplete_scale = True
                break
            per_view.append(np.stack(features))
            ratios.append(float(view["visible_ratio"]))
        if per_view and not incomplete_scale:
            valid_geometry[geometry_index] = True
            recomputed_aggregate[geometry_index] = visible_ratio_multiscale_feature(
                np.stack(per_view), np.asarray(ratios, dtype=np.float32)
            )
            recomputed_similarities[geometry_index] = (
                recomputed_aggregate[geometry_index] @ text_features.T
            ).astype(np.float32)
            if not _close(aggregate[geometry_index], recomputed_aggregate[geometry_index], 2e-6):
                raise ValueError(f"{scene}: aggregate feature mismatch")
            if not _close(similarities[geometry_index], recomputed_similarities[geometry_index], 2e-6):
                raise ValueError(f"{scene}: cosine similarity mismatch")
            top = int(np.argmax(recomputed_similarities[geometry_index]))
            if (
                row.get("alpha_feature_valid") is not True
                or int(row["alpha_class_index"]) != top
                or not np.isclose(
                    float(row["alpha_top_similarity"]),
                    float(recomputed_similarities[geometry_index, top]),
                    rtol=0.0, atol=2e-6,
                )
            ):
                raise ValueError(f"{scene}: Alpha top1 mismatch")
        elif row.get("alpha_feature_valid") is not False:
            raise ValueError(f"{scene}: invalid feature was not marked invalid")
    if sorted(feature_indices) != list(range(expected_scale_count)):
        raise ValueError(f"{scene}: feature indexes are not a complete unique range")
    population_complete = bool(valid_geometry.all())
    deleted = 0
    if population_complete:
        sms = compute_sms(recomputed_similarities)
        keep = sms_keep_mask(sms, threshold)
        if not _close(class_stats[0], sms.class_means, 2e-6) or not _close(
            class_stats[1], sms.class_stds, 2e-6
        ):
            raise ValueError(f"{scene}: SMS class statistics mismatch")
        for index, row in enumerate(records):
            if (
                not np.isclose(float(row["sms_score"]), float(sms.scores[index]), rtol=0.0, atol=2e-5)
                or bool(row["sms_valid"]) != bool(sms.valid[index])
                or bool(row["sms_keep"]) != bool(keep[index])
            ):
                raise ValueError(f"{scene}: SMS row mismatch")
        deleted = int((~keep).sum())
    else:
        if np.isfinite(class_stats).any():
            raise ValueError(f"{scene}: incomplete population has nonmissing SMS statistics")
        if any(row.get("sms_valid") is not False or row.get("sms_keep") is not True for row in records):
            raise ValueError(f"{scene}: incomplete population was not conservatively retained")
    derived = {
        "geometry_count": len(records),
        "selected_view_count": sum(len(row["views"]) for row in records),
        "scale_feature_count": expected_scale_count,
        "sam_missing_view_count": sum(
            int(view.get("sam_mask_valid") is not True)
            for row in records for view in row["views"]
        ),
        "alpha_valid_count": int(valid_geometry.sum()),
        "alpha_invalid_count": int((~valid_geometry).sum()),
        "population_complete": population_complete,
        "sms_deleted_count": deleted,
        "sms_kept_count": len(records) - deleted,
    }
    for key, value in derived.items():
        if summary.get(key) != value:
            raise ValueError(f"{scene}: summary mismatch for {key}")
    expected_hashes = {
        "records_sha256": records_path,
        "scale_features_sha256": scene_root / "scale_features.npy",
        "aggregate_features_sha256": scene_root / "aggregate_features.npy",
        "similarities_sha256": scene_root / "similarities.npy",
        "sms_class_stats_sha256": scene_root / "sms_class_stats.npy",
    }
    for key, path in expected_hashes.items():
        if summary.get(key) != _sha256(path):
            raise ValueError(f"{scene}: file provenance mismatch for {key}")
    return {
        "scene_name": scene,
        **derived,
        "sam_mask_count": sam_mask_count,
        "scale_feature_hash_count": scale_hash_count,
        "source_geometry_counts": dict(sorted(Counter(
            row["canonical_candidate_source"] for row in records
        ).items())),
        "deleted_predicted_class_counts": dict(sorted(Counter(
            int(row["alpha_class_index"]) for row in records if not row["sms_keep"]
        ).items())),
    }


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "manifest_root", "ledger_root", "output_root",
        "asset_provenance",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.sms_threshold != 0.0:
        raise ValueError("DM-SMS-1A audit only accepts frozen tau_SMS=0")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    all_scenes = _scenes(args.scene_list)
    scenes = all_scenes[:args.max_scenes] if args.max_scenes is not None else all_scenes
    source_manifest_path = args.manifest_root / "alpha_view_manifest.jsonl"
    source_rows = _read_jsonl(source_manifest_path)
    source_by_scene = defaultdict(list)
    for row in source_rows:
        source_by_scene[str(row["scene_name"])].append(row)
    if not set(scenes).issubset(source_by_scene):
        raise ValueError("Stage C manifest scene coverage differs")
    for scene in scenes:
        source_by_scene[scene].sort(key=lambda row: row["geometry_hash"])
    summary_path = args.ledger_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    model_provenance_path = args.ledger_root / "model_provenance.json"
    model_provenance = json.loads(model_provenance_path.read_text())
    prompts_path = args.ledger_root / "text_prompts.json"
    prompts = json.loads(prompts_path.read_text())
    text_features_path = args.ledger_root / "text_features.npy"
    text_features = np.load(text_features_path, allow_pickle=False).astype(np.float32)
    if (
        summary.get("stage_complete") is not True
        or summary.get("ground_truth_read") is not False
        or summary.get("ap_computed") is not False
        or float(summary.get("sms_threshold", float("nan"))) != 0.0
        or int(summary.get("class_count", -1)) != 198
        or summary.get("prompt_template") != PROMPT_TEMPLATE
    ):
        raise ValueError("Stage D summary contract is invalid")
    if summary["input_provenance"]["alpha_view_manifest_sha256"] != _sha256(source_manifest_path):
        raise ValueError("Stage D input manifest hash mismatch")
    if summary["input_provenance"]["model_provenance_sha256"] != _sha256(model_provenance_path):
        raise ValueError("Stage D model provenance hash mismatch")
    if model_provenance.get("parameters_frozen") is not True:
        raise ValueError("Alpha/SAM parameters are not declared frozen")
    if model_provenance.get("text_features_sha256") != _sha256(text_features_path):
        raise ValueError("text feature file hash mismatch")
    if model_provenance.get("text_prompts_sha256") != _sha256(prompts_path):
        raise ValueError("text prompt file hash mismatch")
    if (
        len(prompts.get("class_names", [])) != 198
        or prompts.get("prompts") != [
            PROMPT_TEMPLATE.format(CLASS_NAME=name) for name in prompts["class_names"]
        ]
        or text_features.shape[0] != 198
        or not np.isfinite(text_features).all()
        or not np.allclose(np.linalg.norm(text_features, axis=1), 1.0, rtol=0.0, atol=2e-4)
    ):
        raise ValueError("198-class text feature/prompt contract is invalid")
    provenance = json.loads(args.asset_provenance.read_text())
    for key in (
        "alpha_clip_base_sha256", "alpha_clip_checkpoint_sha256", "sam_checkpoint_sha256"
    ):
        if model_provenance.get(key) != provenance["models"].get(key):
            raise ValueError(f"model provenance differs for {key}")

    scene_summaries = []
    for index, scene in enumerate(scenes, 1):
        audited = _audit_scene(
            scene, source_by_scene[scene], args.ledger_root,
            text_features, args.sms_threshold,
        )
        scene_summaries.append(audited)
        print(
            f"[DM-SMS-1 Stage D audit] {index}/{len(scenes)} {scene}: "
            f"geometry={audited['geometry_count']} deleted={audited['sms_deleted_count']}",
            flush=True,
        )
    derived = {
        "scene_count": len(scenes),
        "geometry_count": sum(row["geometry_count"] for row in scene_summaries),
        "selected_view_count": sum(row["selected_view_count"] for row in scene_summaries),
        "scale_feature_count": sum(row["scale_feature_count"] for row in scene_summaries),
        "sam_missing_view_count": sum(row["sam_missing_view_count"] for row in scene_summaries),
        "alpha_valid_count": sum(row["alpha_valid_count"] for row in scene_summaries),
        "alpha_invalid_count": sum(row["alpha_invalid_count"] for row in scene_summaries),
        "sms_population_complete_scene_count": sum(
            int(row["population_complete"]) for row in scene_summaries
        ),
        "sms_deleted_count": sum(row["sms_deleted_count"] for row in scene_summaries),
        "sms_kept_count": sum(row["sms_kept_count"] for row in scene_summaries),
        "sam_mask_count": sum(row["sam_mask_count"] for row in scene_summaries),
        "scale_feature_hash_count": sum(
            row["scale_feature_hash_count"] for row in scene_summaries
        ),
    }
    for key in (
        "scene_count", "geometry_count", "selected_view_count", "scale_feature_count",
        "sam_missing_view_count",
        "alpha_valid_count", "alpha_invalid_count",
        "sms_population_complete_scene_count", "sms_deleted_count", "sms_kept_count",
    ):
        if int(summary.get(key, -1)) != int(derived[key]):
            raise ValueError(f"Stage D aggregate summary mismatch for {key}")
    args.output_root.mkdir(parents=True)
    output = {
        "version": "dm_sms1_alpha_embedding_audit_v1",
        "audit_valid": True,
        **derived,
        "class_count": 198,
        "sms_threshold": 0.0,
        "prompt_error_count": 0,
        "model_provenance_error_count": 0,
        "manifest_join_error_count": 0,
        "sam_provenance_error_count": 0,
        "feature_hash_error_count": 0,
        "feature_norm_error_count": 0,
        "aggregation_error_count": 0,
        "similarity_error_count": 0,
        "sms_population_error_count": 0,
        "sms_decision_error_count": 0,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "candidate_geometry_mutation": False,
        "input_provenance": {
            "scene_list": str(args.scene_list),
            "scene_list_sha256": _sha256(args.scene_list),
            "stage_c_manifest_sha256": _sha256(source_manifest_path),
            "stage_d_summary_sha256": _sha256(summary_path),
            "model_provenance_sha256": _sha256(model_provenance_path),
            "text_features_sha256": _sha256(text_features_path),
            "text_prompts_sha256": _sha256(prompts_path),
        },
        "scene_summaries": scene_summaries,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"
    ))
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-provenance", type=Path, default=Path(
        "docs/DM_SMS1_ASSET_PROVENANCE_20260817.json"
    ))
    parser.add_argument("--sms-threshold", type=float, default=0.0)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
