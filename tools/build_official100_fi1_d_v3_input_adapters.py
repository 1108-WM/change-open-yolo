#!/usr/bin/env python3
"""Build read-only official100 adapters for the frozen FI1-D-v3 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "official100_fi1_d_v3_input_adapters_v1"
FORBIDDEN_DATASET_TOKENS = ("ncs_train100", "ncs-validation60", "validation60", "val312")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise ValueError("official100 scene list must contain exactly 100 unique scenes")
    return scenes


def read_jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL row {path}:{line_number}") from error


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def safe_symlink(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    lowered = str(source).lower().replace("-", "_")
    if any(token.replace("-", "_") in lowered for token in FORBIDDEN_DATASET_TOKENS):
        raise ValueError(f"forbidden dataset path in adapter source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def fold_by_scene(path: Path, scenes: list[str]) -> dict[str, int]:
    payload = json.loads(path.read_text())
    result = {}
    for fold in payload["folds"]:
        index = int(fold["fold_index"])
        for scene in fold["validation_scenes"]:
            if scene in result:
                raise ValueError(f"scene occurs in multiple validation folds: {scene}")
            result[str(scene)] = index
    if set(result) != set(scenes) or Counter(result.values()) != Counter({i: 20 for i in range(5)}):
        raise ValueError("fold manifest is not the fixed official100 20-scene five-fold partition")
    return result


def track_quality_predictions(path: Path, scenes: set[str]) -> dict[tuple[str, int], float]:
    result = {}
    for row in read_jsonl(path):
        if row.get("candidate_source") != "d2b_track":
            continue
        scene = str(row["scene_name"])
        if scene not in scenes:
            raise ValueError(f"quality OOF file contains non-official100 track scene: {scene}")
        key = (scene, int(row["candidate_id"]))
        if key in result:
            raise ValueError(f"duplicate OOF track quality prediction: {key}")
        # Only the already-frozen OOF prediction is extracted. Co-located label fields
        # are deliberately neither copied nor consulted.
        # Historical official100 AP materialized evaluator confidences through
        # numpy float32 before the frozen Legacy score heads consumed them.
        result[key] = float(np.float32(row["predictions"]["D_plus_gvc"]["q"]))
    return result


def build(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "fold_manifest", "records_root", "legacy_plan_root",
        "quality_oof_predictions", "pair_union_oof_summary", "z1_root", "relation_source_root",
        "preregistration", "output_root",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    scenes = read_scenes(args.scene_list)
    scene_set = set(scenes)
    folds = fold_by_scene(args.fold_manifest, scenes)
    if sha256(args.scene_list) != "dfa9017e206190eb2973b247c78e4bf1b2d9c01bb8468a15775c30335e44fb68":
        raise ValueError("official100 scene-list SHA-256 differs from the preregistered value")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging already exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        track_quality = track_quality_predictions(args.quality_oof_predictions, scene_set)
        pair_oof_summary = json.loads(args.pair_union_oof_summary.read_text())
        prior_by_fold = {
            int(row["fold_index"]): float(row["training_component_balanced_natural_positive_rate"])
            for row in pair_oof_summary["fold_details"]
        }
        if set(prior_by_fold) != set(range(5)) or not all(0.0 < value < 1.0 for value in prior_by_fold.values()):
            raise ValueError("pair-union OOF summary lacks five valid frozen natural positive rates")

        overrides = {}
        override_path = args.legacy_plan_root / "champion_track_score_overrides.jsonl"
        for row in read_jsonl(override_path):
            key = (str(row["scene_name"]), int(row["candidate_id"]))
            if key in overrides or key[0] not in scene_set:
                raise ValueError(f"invalid or duplicate legacy track override: {key}")
            if abs(float(row["original_score"]) - track_quality[key]) > 1e-12:
                raise ValueError(f"legacy override original score differs from frozen OOF quality: {key}")
            overrides[key] = row

        union_source_path = args.legacy_plan_root / "pair_union_append_candidates.jsonl"
        unions_by_scene: dict[str, list[dict]] = defaultdict(list)
        for row in read_jsonl(union_source_path):
            scene = str(row["scene_name"])
            if scene not in scene_set:
                raise ValueError(f"legacy union has non-official100 scene: {scene}")
            unions_by_scene[scene].append(row)

        semantics_by_scene: dict[str, dict[int, dict]] = defaultdict(dict)
        binding_path = args.z1_root / "candidate_bindings.jsonl"
        for row in read_jsonl(binding_path):
            if row.get("candidate_source") != "track":
                continue
            scene = str(row["scene_name"])
            if scene not in scene_set:
                continue
            track_id = int(row["track_id"])
            if track_id in semantics_by_scene[scene]:
                raise ValueError(f"duplicate Z1 track binding: {(scene, track_id)}")
            semantics_by_scene[scene][track_id] = {
                "scene_name": scene,
                "track_id": track_id,
                "voted_class_index": int(row["current_voted_class_index"]),
                "vote_probability": float(row.get("current_vote_probability", 0.0)),
                "semantic_evidence_node_key": row["semantic_evidence_node_key"],
                "semantic_source": "frozen_official100_z1_track_yoloworld_vote",
                "ground_truth_usage": "none",
                "ap_evaluation_run": False,
            }

        relation_by_scene: dict[str, list[dict]] = {}
        relation_index: dict[tuple[str, int, str], dict] = {}
        for scene in scenes:
            rows = []
            source = args.relation_source_root / scene / "relation_features.jsonl"
            for raw in read_jsonl(source):
                if str(raw["scene_name"]) != scene:
                    raise ValueError(f"relation scene mismatch in {source}")
                contracts = raw.get("contracts", {})
                if contracts.get("feature_ground_truth_usage") != "none":
                    raise ValueError(f"relation feature contract is not no-GT: {scene}")
                row = {
                    "scene_name": scene,
                    "track_id": int(raw["track_id"]),
                    "native_exact_geometry_group_id": str(raw["native_exact_geometry_group_id"]),
                    "native_member_candidate_ids": [int(x) for x in raw["native_member_candidate_ids"]],
                    "relation_component_id": int(raw["relation_component_id"]),
                    "features": raw["features"],
                    "contracts": {
                        "feature_ground_truth_usage": "none",
                        "label_ground_truth_usage": "none",
                        "ap_evaluation_run": False,
                        "candidate_geometry_modified": False,
                        "relation_model_trained": False,
                        "replacement_action_generated": False,
                        "track_source_frames_excluded_from_public_view_features": bool(
                            contracts.get("track_source_frames_excluded_from_public_view_features", True)
                        ),
                    },
                }
                key = (scene, row["track_id"], row["native_exact_geometry_group_id"])
                if key in relation_index:
                    raise ValueError(f"duplicate relation identity: {key}")
                relation_index[key] = row
                rows.append(row)
            relation_by_scene[scene] = rows

        all_score_rows, all_union_rows = [], []
        scene_summaries = []
        native_total = track_total = union_total = relation_total = 0
        link_kinds = (
            "gvc_quality_ledger", "d2b_tracks_filtered", "d1_hierarchy_safe",
            "sam_automatic_uniform30", "yoloworld_sam_uniform30",
        )
        for index, scene in enumerate(scenes, 1):
            record = args.records_root / scene
            for suffix in ("classes.npy", "masks.npy", "scores.npy"):
                source = record / "native_cache" / f"{scene}_pred_{suffix}"
                safe_symlink(source, staging / "native_cache" / source.name)
            for kind in link_kinds:
                source = record / kind / scene
                safe_symlink(source, staging / kind / scene)

            track_path = record / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
            tracks = json.loads(track_path.read_text()).get("tracks", [])
            track_ids = {int(row["track_id"]) for row in tracks}
            if set(semantics_by_scene[scene]) != track_ids:
                raise ValueError(f"Z1 semantic coverage differs from filtered tracks: {scene}")
            semantic_dir = staging / "track_yoloworld_semantics" / scene
            semantic_dir.mkdir(parents=True)
            (semantic_dir / "automatic_track_yoloworld_semantics.json").write_text(
                json.dumps(
                    [semantics_by_scene[scene][track_id] for track_id in sorted(track_ids)],
                    ensure_ascii=False, indent=2, sort_keys=True,
                ) + "\n"
            )

            masks_path = record / "native_cache" / f"{scene}_pred_masks.npy"
            scores_path = record / "native_cache" / f"{scene}_pred_scores.npy"
            masks = np.load(masks_path, mmap_mode="r")
            native_scores = np.asarray(np.load(scores_path), dtype=np.float32)
            if masks.ndim != 2 or masks.shape[1] != len(native_scores):
                raise ValueError(f"native cache dimensions disagree: {scene}")
            score_rows = []
            for candidate_id, score in enumerate(native_scores):
                score_rows.append({
                    "scene_name": scene, "candidate_source": "native_mask3d_yoloworld",
                    "candidate_id": candidate_id, "original_source_score": float(score),
                    "frozen_coexist_score": float(score), "planned_score": float(score),
                    "reason": "native_score_bit_for_bit_frozen", "candidate_removed": False,
                    "geometry_modified": False, "class_modified": False,
                    "score_changed_vs_frozen_coexist": False,
                    "ground_truth_usage": "none", "ap_evaluation_run": False,
                })
            for track_id in sorted(track_ids):
                key = (scene, track_id)
                original = track_quality.get(key)
                if original is None:
                    raise ValueError(f"missing frozen OOF quality for filtered track: {key}")
                override = overrides.get(key)
                planned = float(override["new_score"]) if override else original
                score_rows.append({
                    "scene_name": scene, "candidate_source": "d2b_track",
                    "candidate_id": track_id, "original_source_score": original,
                    "frozen_coexist_score": original, "planned_score": planned,
                    "reason": "track_harm_focal_suppression" if override else "track_without_relation_component_frozen",
                    "candidate_removed": False, "geometry_modified": False,
                    "class_modified": False, "score_changed_vs_frozen_coexist": planned != original,
                    "ground_truth_usage": "none", "ap_evaluation_run": False,
                })

            union_rows = []
            for raw in sorted(unions_by_scene[scene], key=lambda row: int(row["candidate_id"])):
                track_id = int(raw["selected_track_id"])
                native_group = str(raw["selected_native_exact_geometry_group_id"])
                relation = relation_index.get((scene, track_id, native_group))
                if relation is None:
                    raise ValueError(f"union parent relation is missing: {(scene, track_id, native_group)}")
                native_ids = [int(x) for x in raw["selected_native_member_candidate_ids"]]
                if native_ids != relation["native_member_candidate_ids"]:
                    raise ValueError(f"union native members differ from relation ledger: {scene}")
                corrected = float(raw["threshold_cross_probability"])
                corrected = min(1.0 - 1e-6, max(1e-6, corrected))
                prior = prior_by_fold[folds[scene]]
                corrected_odds = corrected / (1.0 - corrected)
                balanced_odds = corrected_odds * (1.0 - prior) / prior
                balanced_raw = balanced_odds / (1.0 + balanced_odds)
                union_rows.append({
                    **raw,
                    "track_id": track_id,
                    "native_exact_geometry_group_id": native_group,
                    "native_member_candidate_ids": native_ids,
                    "relation_component_id": int(relation["relation_component_id"]),
                    "proposal_geometry_sha256": str(raw["geometry_sha256"]),
                    "proposal_point_count": int(raw["point_count"]),
                    "eligible_novel_min_region": True,
                    "balanced_fit_raw_probability": balanced_raw,
                    "probability_prior_correction_natural_positive_rate": prior,
                    "candidate_removed": False, "geometry_modified": False,
                    "class_modified": False, "ground_truth_usage": "none",
                    "ap_evaluation_run": False,
                })

            plan_scene = staging / "champion_plan" / scene
            plan_scene.mkdir(parents=True)
            write_jsonl(plan_scene / "frozen_score_plan.jsonl", score_rows)
            write_jsonl(plan_scene / "pair_union_append_candidates.jsonl", union_rows)
            relation_scene = staging / "relation_ledger" / scene
            relation_scene.mkdir(parents=True)
            write_jsonl(relation_scene / "relation_features_no_gt.jsonl", relation_by_scene[scene])
            relation_scene_summary = {
                "scene_name": scene, "relation_count": len(relation_by_scene[scene]),
                "feature_ground_truth_usage": "none", "ground_truth_read": False,
                "candidate_mutation": False, "ap_evaluation_run": False,
            }
            (relation_scene / "summary.json").write_text(
                json.dumps(relation_scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            scene_summary = {
                "scene_name": scene, "fold_index": folds[scene],
                "native_candidate_count": len(native_scores), "track_candidate_count": len(track_ids),
                "track_override_count": sum((scene, t) in overrides for t in track_ids),
                "pair_union_append_candidate_count": len(union_rows),
                "relation_count": len(relation_by_scene[scene]), "ground_truth_usage": "none",
                "ap_evaluation_run": False, "candidate_mutation": False,
            }
            (plan_scene / "summary.json").write_text(
                json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            all_score_rows.extend(score_rows)
            all_union_rows.extend(union_rows)
            scene_summaries.append(scene_summary)
            native_total += len(native_scores); track_total += len(track_ids)
            union_total += len(union_rows); relation_total += len(relation_by_scene[scene])
            print(f"[official100 adapter] {index}/100 {scene}: native={len(native_scores)} track={len(track_ids)} union={len(union_rows)}", flush=True)

        if set(track_quality) != {
            (row["scene_name"], int(row["candidate_id"]))
            for row in all_score_rows if row["candidate_source"] == "d2b_track"
        }:
            raise ValueError("historical OOF track-quality coverage differs from filtered-track coverage")
        if len(overrides) != 3380 or union_total != 1501 or relation_total != 4786:
            raise ValueError("historical official100 legacy/relation counts differ from frozen contract")

        write_jsonl(staging / "champion_plan" / "frozen_score_plan.jsonl", all_score_rows)
        write_jsonl(staging / "champion_plan" / "pair_union_append_candidates.jsonl", all_union_rows)
        plan_summary = {
            "version": VERSION, "scene_count": 100, "candidate_count": len(all_score_rows),
            "native_candidate_count": native_total, "track_candidate_count": track_total,
            "track_score_changed_count": len(overrides),
            "pair_union_append_candidate_count": union_total, "ground_truth_usage": "none",
            "ground_truth_read": False, "ap_evaluation_run": False,
            "candidate_count_modification_count": 0, "candidate_geometry_modification_count": 0,
            "candidate_class_modification_count": 0, "candidate_removal_count": 0,
            "native_scores_bit_for_bit_frozen": True, "pair_union_append_only": True,
        }
        (staging / "champion_plan" / "summary.json").write_text(
            json.dumps(plan_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        relation_summary = {
            "version": VERSION, "scene_count": 100, "relation_count": relation_total,
            "relation_component_count": len({
                (scene, int(row["relation_component_id"]))
                for scene, rows in relation_by_scene.items() for row in rows
            }),
            "feature_ground_truth_usage": "none", "ground_truth_usage": "none",
            "ground_truth_read": False, "candidate_mutation": False,
            "ap_evaluation_run": False, "label_fields_removed": True,
        }
        (staging / "relation_ledger" / "summary.json").write_text(
            json.dumps(relation_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        gvc_summary = {
            "version": VERSION, "scene_count": 100, "native_candidate_count": native_total,
            "track_candidate_count": track_total, "ground_truth_usage": "none",
            "ground_truth_read": False, "candidate_mutation": False, "ap_evaluation_run": False,
            "adapter_kind": "scene-directory read-only symlinks to official100 GVC ledgers",
        }
        (staging / "gvc_quality_ledger" / "summary.json").write_text(
            json.dumps(gvc_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "version": VERSION, "status": "built_pending_independent_audit", "scene_count": 100,
            "fold_scene_counts": {str(i): 20 for i in range(5)},
            "native_candidate_count": native_total, "track_candidate_count": track_total,
            "track_override_count": len(overrides), "pair_union_append_candidate_count": union_total,
            "relation_count": relation_total, "feature_ground_truth_usage": "none",
            "ground_truth_read": False, "ap_evaluation_run": False, "gpu_used": False,
            "candidate_mutation": False, "geometry_mutation": False,
            "class_mutation": False, "historical_score_mutation": False,
            "input_provenance": {
                "preregistration_sha256": sha256(args.preregistration),
                "scene_list_sha256": sha256(args.scene_list),
                "fold_manifest_sha256": sha256(args.fold_manifest),
                "legacy_track_overrides_sha256": sha256(override_path),
                "legacy_pair_unions_sha256": sha256(union_source_path),
                "quality_oof_predictions_sha256": sha256(args.quality_oof_predictions),
                "pair_union_oof_summary_sha256": sha256(args.pair_union_oof_summary),
                "z1_candidate_bindings_sha256": sha256(binding_path),
                "relation_source_summary_sha256": sha256(args.relation_source_root / "summary.json"),
            },
            "scene_summaries": scene_summaries,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=Path("output/scannet200/scene_splits/official_train100_20260808/official_train100.txt"))
    parser.add_argument("--fold-manifest", type=Path, default=Path("output/scannet200/scene_splits/official_train100_20260808/oof_5fold_manifest.json"))
    parser.add_argument("--records-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/records"))
    parser.add_argument("--legacy-plan-root", type=Path, default=Path("output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1"))
    parser.add_argument("--quality-oof-predictions", type=Path, default=Path("output/train_candidate_quality_oof_official100_v2/oof_predictions.jsonl"))
    parser.add_argument("--pair-union-oof-summary", type=Path, default=Path("output/diagnose_candidate_pair_union_threshold_cross_oof_official100_v1/summary.json"))
    parser.add_argument("--z1-root", type=Path, default=Path("docs/diagnostics/z1_yoloworld_multiview_distribution_official100_20260811_v3_frozen_support_vote"))
    parser.add_argument("--relation-source-root", type=Path, default=Path("output/train_candidate_relation_feature_ledger_official100_v2_exclusive_pair"))
    parser.add_argument("--preregistration", type=Path, default=Path("docs/OFFICIAL100_FI1_D_V3_MIGRATION_PREREGISTRATION_20260822.md"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
