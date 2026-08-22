#!/usr/bin/env python3
"""Independently audit the read-only official100 FI1-D-v3 input adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "official100_fi1_d_v3_input_adapters_audit_v1"
FORBIDDEN_TOKENS = ("ncs_train100", "ncs-validation60", "validation60", "val312")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {path}:{line_number}") from error
    return rows


def contains_forbidden_path(value: object) -> bool:
    lowered = str(value).lower().replace("-", "_")
    return any(token.replace("-", "_") in lowered for token in FORBIDDEN_TOKENS)


def geometry_sha256(path: Path) -> tuple[str, int]:
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(len(points).to_bytes(8, "little"))
    digest.update(np.ascontiguousarray(points, dtype=np.int64).tobytes())
    return digest.hexdigest(), len(points)


def add_error(errors: Counter, name: str, count: int | bool = 1) -> None:
    errors[f"{name}_error_count"] += int(count)


def audit(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "fold_manifest", "records_root", "adapter_root",
        "legacy_plan_root", "quality_oof_predictions", "z1_root",
        "pair_union_oof_summary", "relation_source_root", "preregistration", "output_root",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    scenes = [line.strip() for line in args.scene_list.read_text().splitlines() if line.strip()]
    scene_set = set(scenes)
    errors = Counter()
    if len(scenes) != 100 or len(scene_set) != 100:
        add_error(errors, "scene_list_contract")
    if sha256(args.scene_list) != "dfa9017e206190eb2973b247c78e4bf1b2d9c01bb8468a15775c30335e44fb68":
        add_error(errors, "scene_list_sha256")

    validation_fold = {}
    manifest = json.loads(args.fold_manifest.read_text())
    for fold in manifest.get("folds", []):
        fold_index = int(fold["fold_index"])
        train = set(fold["train_scenes"])
        validation = set(fold["validation_scenes"])
        if train & validation or train | validation != scene_set or len(validation) != 20:
            add_error(errors, "fold_partition")
        for scene in validation:
            if scene in validation_fold:
                add_error(errors, "fold_duplicate_validation_scene")
            validation_fold[scene] = fold_index
    if set(validation_fold) != scene_set:
        add_error(errors, "fold_coverage")

    adapter_summary = json.loads((args.adapter_root / "summary.json").read_text())
    if adapter_summary.get("ground_truth_read") is not False:
        add_error(errors, "adapter_ground_truth_contract")
    if adapter_summary.get("ap_evaluation_run") is not False:
        add_error(errors, "adapter_ap_contract")
    if any(adapter_summary.get(key) is not False for key in (
        "candidate_mutation", "geometry_mutation", "class_mutation", "historical_score_mutation"
    )):
        add_error(errors, "adapter_mutation_contract")
    expected_provenance = {
        "preregistration_sha256": sha256(args.preregistration),
        "scene_list_sha256": sha256(args.scene_list),
        "fold_manifest_sha256": sha256(args.fold_manifest),
        "legacy_track_overrides_sha256": sha256(args.legacy_plan_root / "champion_track_score_overrides.jsonl"),
        "legacy_pair_unions_sha256": sha256(args.legacy_plan_root / "pair_union_append_candidates.jsonl"),
        "quality_oof_predictions_sha256": sha256(args.quality_oof_predictions),
        "pair_union_oof_summary_sha256": sha256(args.pair_union_oof_summary),
        "z1_candidate_bindings_sha256": sha256(args.z1_root / "candidate_bindings.jsonl"),
        "relation_source_summary_sha256": sha256(args.relation_source_root / "summary.json"),
    }
    if adapter_summary.get("input_provenance") != expected_provenance:
        add_error(errors, "input_provenance")

    quality = {}
    for row in read_jsonl(args.quality_oof_predictions):
        if row.get("candidate_source") == "d2b_track":
            key = (str(row["scene_name"]), int(row["candidate_id"]))
            if key in quality:
                add_error(errors, "quality_duplicate")
            quality[key] = float(np.float32(row["predictions"]["D_plus_gvc"]["q"]))
    overrides = {
        (str(row["scene_name"]), int(row["candidate_id"])): row
        for row in read_jsonl(args.legacy_plan_root / "champion_track_score_overrides.jsonl")
    }
    source_unions = {
        (str(row["scene_name"]), int(row["candidate_id"])): row
        for row in read_jsonl(args.legacy_plan_root / "pair_union_append_candidates.jsonl")
    }
    bindings = {}
    for row in read_jsonl(args.z1_root / "candidate_bindings.jsonl"):
        if row.get("candidate_source") == "track" and str(row["scene_name"]) in scene_set:
            bindings[(str(row["scene_name"]), int(row["track_id"]))] = row
    pair_oof_summary = json.loads(args.pair_union_oof_summary.read_text())
    prior_by_fold = {
        int(row["fold_index"]): float(row["training_component_balanced_natural_positive_rate"])
        for row in pair_oof_summary["fold_details"]
    }

    global_score_rows = read_jsonl(args.adapter_root / "champion_plan" / "frozen_score_plan.jsonl")
    global_union_rows = read_jsonl(args.adapter_root / "champion_plan" / "pair_union_append_candidates.jsonl")
    global_score_keys = set()
    global_union_keys = set()
    native_total = track_total = union_total = relation_total = link_total = 0
    changed_track_total = 0
    for scene in scenes:
        record = args.records_root / scene
        for suffix in ("classes.npy", "masks.npy", "scores.npy"):
            link = args.adapter_root / "native_cache" / f"{scene}_pred_{suffix}"
            source = record / "native_cache" / f"{scene}_pred_{suffix}"
            if not link.is_symlink() or link.resolve() != source.resolve():
                add_error(errors, "native_symlink")
            if contains_forbidden_path(link.resolve()):
                add_error(errors, "forbidden_dataset_path")
            link_total += 1
        for kind in (
            "gvc_quality_ledger", "d2b_tracks_filtered", "d1_hierarchy_safe",
            "sam_automatic_uniform30", "yoloworld_sam_uniform30",
        ):
            link = args.adapter_root / kind / scene
            source = record / kind / scene
            if not link.is_symlink() or link.resolve() != source.resolve():
                add_error(errors, "scene_asset_symlink")
            if contains_forbidden_path(link.resolve()):
                add_error(errors, "forbidden_dataset_path")
            link_total += 1

        masks = np.load(record / "native_cache" / f"{scene}_pred_masks.npy", mmap_mode="r")
        native_scores = np.asarray(np.load(record / "native_cache" / f"{scene}_pred_scores.npy"), dtype=np.float32)
        tracks = json.loads(
            (record / "d2b_tracks_filtered" / scene / "automatic_tracks.json").read_text()
        )["tracks"]
        track_ids = {int(row["track_id"]) for row in tracks}
        native_total += len(native_scores); track_total += len(track_ids)
        if masks.shape[1] != len(native_scores):
            add_error(errors, "native_dimension")

        score_rows = read_jsonl(args.adapter_root / "champion_plan" / scene / "frozen_score_plan.jsonl")
        score_index = {(str(row["candidate_source"]), int(row["candidate_id"])): row for row in score_rows}
        if len(score_index) != len(score_rows):
            add_error(errors, "score_duplicate")
        expected_keys = {
            *(("native_mask3d_yoloworld", i) for i in range(len(native_scores))),
            *(("d2b_track", i) for i in track_ids),
        }
        if set(score_index) != expected_keys:
            add_error(errors, "score_coverage")
        for candidate_id, expected in enumerate(native_scores):
            row = score_index.get(("native_mask3d_yoloworld", candidate_id), {})
            if row and (
                float(row["planned_score"]) != float(expected)
                or float(row["frozen_coexist_score"]) != float(expected)
                or row.get("score_changed_vs_frozen_coexist") is not False
            ):
                add_error(errors, "native_score_freeze")
        for track_id in track_ids:
            key = (scene, track_id)
            row = score_index.get(("d2b_track", track_id), {})
            expected_original = quality.get(key)
            expected_planned = float(overrides[key]["new_score"]) if key in overrides else expected_original
            if expected_original is None or not row:
                add_error(errors, "track_score_source")
            elif abs(float(row["frozen_coexist_score"]) - expected_original) > 1e-12 or abs(float(row["planned_score"]) - expected_planned) > 1e-12:
                add_error(errors, "track_score_freeze")
            changed_track_total += int(key in overrides)
        for row in score_rows:
            key = (scene, str(row["candidate_source"]), int(row["candidate_id"]))
            if key in global_score_keys:
                add_error(errors, "global_score_duplicate")
            global_score_keys.add(key)
            if row.get("ground_truth_usage") != "none" or row.get("ap_evaluation_run") is not False:
                add_error(errors, "score_no_gt_no_ap_contract")
            if row.get("candidate_removed") is not False or row.get("geometry_modified") is not False or row.get("class_modified") is not False:
                add_error(errors, "score_mutation_contract")

        semantic_rows = json.loads(
            (args.adapter_root / "track_yoloworld_semantics" / scene / "automatic_track_yoloworld_semantics.json").read_text()
        )
        semantic_index = {int(row["track_id"]): row for row in semantic_rows}
        if set(semantic_index) != track_ids or len(semantic_index) != len(semantic_rows):
            add_error(errors, "semantic_coverage")
        for track_id, row in semantic_index.items():
            source = bindings.get((scene, track_id))
            if source is None or int(row["voted_class_index"]) != int(source["current_voted_class_index"]):
                add_error(errors, "semantic_inheritance")
            if row.get("ground_truth_usage") != "none" or row.get("ap_evaluation_run") is not False:
                add_error(errors, "semantic_contract")

        source_relation_rows = read_jsonl(args.relation_source_root / scene / "relation_features.jsonl")
        relation_rows = read_jsonl(args.adapter_root / "relation_ledger" / scene / "relation_features_no_gt.jsonl")
        source_relation_index = {
            (int(row["track_id"]), str(row["native_exact_geometry_group_id"])): row
            for row in source_relation_rows
        }
        relation_index = {}
        for row in relation_rows:
            key = (int(row["track_id"]), str(row["native_exact_geometry_group_id"]))
            relation_index[key] = row
            source = source_relation_index.get(key)
            if source is None or row["features"] != source["features"]:
                add_error(errors, "relation_feature_identity")
            if "labels" in row or "label" in row:
                add_error(errors, "relation_label_field")
            contracts = row.get("contracts", {})
            if contracts.get("feature_ground_truth_usage") != "none" or contracts.get("label_ground_truth_usage") != "none":
                add_error(errors, "relation_no_gt_contract")
            if contracts.get("ap_evaluation_run") is not False or contracts.get("candidate_geometry_modified") is not False:
                add_error(errors, "relation_mutation_ap_contract")
        if set(relation_index) != set(source_relation_index):
            add_error(errors, "relation_coverage")
        relation_total += len(relation_rows)

        union_rows = read_jsonl(args.adapter_root / "champion_plan" / scene / "pair_union_append_candidates.jsonl")
        union_total += len(union_rows)
        for row in union_rows:
            key = (scene, int(row["candidate_id"]))
            global_union_keys.add(key)
            source = source_unions.get(key)
            if source is None:
                add_error(errors, "union_source")
                continue
            for field in ("new_score", "points_path", "selected_track_id", "selected_native_exact_geometry_group_id", "selected_native_member_candidate_ids"):
                if row.get(field) != source.get(field):
                    add_error(errors, "union_frozen_field")
            corrected = min(1.0 - 1e-6, max(1e-6, float(source["threshold_cross_probability"])))
            prior = prior_by_fold[validation_fold[scene]]
            corrected_odds = corrected / (1.0 - corrected)
            balanced_odds = corrected_odds * (1.0 - prior) / prior
            expected_raw = balanced_odds / (1.0 + balanced_odds)
            if abs(float(row.get("balanced_fit_raw_probability", -1.0)) - expected_raw) > 1e-15:
                add_error(errors, "union_balanced_probability_reconstruction")
            if row.get("eligible_novel_min_region") is not True:
                add_error(errors, "union_eligibility_contract")
            digest, point_count = geometry_sha256(resolve(Path(row["points_path"])))
            if digest != row.get("proposal_geometry_sha256") or point_count != int(row.get("proposal_point_count", -1)):
                add_error(errors, "union_geometry_hash")
            relation = relation_index.get((int(row["track_id"]), str(row["native_exact_geometry_group_id"])))
            if relation is None or int(row["relation_component_id"]) != int(relation["relation_component_id"]):
                add_error(errors, "union_relation_join")
            if row.get("ground_truth_usage") != "none" or row.get("ap_evaluation_run") is not False:
                add_error(errors, "union_contract")
            if row.get("candidate_removed") is not False or row.get("geometry_modified") is not False or row.get("class_modified") is not False:
                add_error(errors, "union_mutation_contract")

    expected_global_score_keys = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        for row in global_score_rows
    }
    if global_score_keys != expected_global_score_keys or len(global_score_rows) != len(global_score_keys):
        add_error(errors, "aggregate_score_identity")
    expected_global_union_keys = {(str(row["scene_name"]), int(row["candidate_id"])) for row in global_union_rows}
    if global_union_keys != expected_global_union_keys or len(global_union_rows) != len(global_union_keys):
        add_error(errors, "aggregate_union_identity")

    fixed_expectations = {
        "native_total": (native_total, 59997), "track_total": (track_total, 5266),
        "track_quality_total": (len(quality), 5266), "track_override_total": (changed_track_total, 3380),
        "union_total": (union_total, 1501), "relation_total": (relation_total, 4786),
        "link_total": (link_total, 800),
    }
    for name, (actual, expected) in fixed_expectations.items():
        if actual != expected:
            add_error(errors, name)

    error_count = sum(errors.values())
    summary = {
        "version": VERSION, "audit_valid": error_count == 0, "error_count": error_count,
        **dict(sorted(errors.items())), "scene_count": len(scenes),
        "fold_scene_counts": dict(sorted(Counter(validation_fold.values()).items())),
        "native_candidate_count": native_total, "track_candidate_count": track_total,
        "track_override_count": changed_track_total, "pair_union_append_candidate_count": union_total,
        "relation_count": relation_total, "read_only_link_count": link_total,
        "ground_truth_read": False, "ap_evaluation_run": False, "gpu_used": False,
        "candidate_mutation": False, "geometry_mutation": False, "class_mutation": False,
        "adapter_summary_sha256": sha256(args.adapter_root / "summary.json"),
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging already exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if error_count:
        raise RuntimeError(f"official100 FI1-D-v3 input adapter audit failed with {error_count} errors")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=Path("output/scannet200/scene_splits/official_train100_20260808/official_train100.txt"))
    parser.add_argument("--fold-manifest", type=Path, default=Path("output/scannet200/scene_splits/official_train100_20260808/oof_5fold_manifest.json"))
    parser.add_argument("--records-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/records"))
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--legacy-plan-root", type=Path, default=Path("output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1"))
    parser.add_argument("--quality-oof-predictions", type=Path, default=Path("output/train_candidate_quality_oof_official100_v2/oof_predictions.jsonl"))
    parser.add_argument("--pair-union-oof-summary", type=Path, default=Path("output/diagnose_candidate_pair_union_threshold_cross_oof_official100_v1/summary.json"))
    parser.add_argument("--z1-root", type=Path, default=Path("docs/diagnostics/z1_yoloworld_multiview_distribution_official100_20260811_v3_frozen_support_vote"))
    parser.add_argument("--relation-source-root", type=Path, default=Path("output/train_candidate_relation_feature_ledger_official100_v2_exclusive_pair"))
    parser.add_argument("--preregistration", type=Path, default=Path("docs/OFFICIAL100_FI1_D_V3_MIGRATION_PREREGISTRATION_20260822.md"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
