#!/usr/bin/env python3
"""Run the one allowed AP for a frozen track-harm score plan.

Preflight is entirely no-GT: it verifies scene isolation, model and plan
digests, official100 five-fold eligibility, current candidate inputs, and
every planned score.  With explicit GT authorization, the tool then builds
the frozen candidate set in memory and makes exactly one AP aggregation call.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    read_jsonl,
    read_scene_list,
)
from tools.build_frozen_track_harm_score_plan import (  # noqa: E402
    POLICY,
    VERSION as PLAN_VERSION,
    fixed_track_score,
)
from tools.build_train_candidate_component_action_utility_ledger import _sha256  # noqa: E402
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    FEATURE_GROUP,
    PROTOCOL_NAME,
    _evaluate,
    _load_track_points,
    _prediction,
    load_safety_scene,
)
from tools.evaluate_official100_geometry_group_ranking_oof_ap import (  # noqa: E402
    geometry_groups_and_audit,
)
from tools.train_candidate_quality_head_oof import canonicalize_predictions, feature_matrix  # noqa: E402


VERSION = "frozen_track_harm_score_plan_ap_v1"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_score_row(row: dict) -> None:
    coexist = float(row["frozen_coexist_score"])
    planned = float(row["planned_score"])
    multiplier = float(row["suppression_multiplier"])
    source = str(row["candidate_source"])
    if not 0.0 <= coexist <= 1.0 or not 0.0 <= planned <= 1.0:
        raise ValueError("score plan contains an out-of-range score")
    if row.get("candidate_removed") or row.get("geometry_modified") or row.get("class_modified"):
        raise ValueError("score plan changes a forbidden candidate property")
    if row.get("ground_truth_usage") != "none" or row.get("ap_evaluation_run") is not False:
        raise ValueError("score plan violates no-GT/no-AP contract")
    if source == NATIVE_SOURCE:
        if planned != coexist or multiplier != 1.0 or row.get("keep_probability") is not None:
            raise ValueError("native score is not exactly frozen")
    elif source == TRACK_SOURCE:
        keep = row.get("keep_probability")
        if keep is None:
            if planned != coexist or multiplier != 1.0:
                raise ValueError("unrelated track score is not frozen")
        else:
            expected_multiplier, expected_score = fixed_track_score(coexist, float(keep))
            if multiplier != expected_multiplier or planned != expected_score:
                raise ValueError("related track score differs from frozen cubic formula")
    else:
        raise ValueError(f"unknown candidate source: {source}")


def _official_eligibility(path: Path) -> dict:
    summary = json.loads(path.read_text())
    selected = summary["policy_metrics"][POLICY]
    baseline = summary["policy_metrics"]["frozen_coexist"]
    deltas = {
        key: float(selected[key] - baseline[key]) for key in ("ap", "ap50", "ap25")
    }
    if not all(value > 0.0 for value in deltas.values()):
        raise ValueError(f"official100 selected policy is not all-positive: {deltas}")
    fold_deltas = []
    for fold in summary["fold_policy_metrics"]["folds"]:
        metrics = fold["policy_metrics"]
        fold_deltas.append(float(
            metrics[POLICY]["official_ap"] - metrics["frozen_coexist"]["official_ap"]
        ))
    if len(fold_deltas) != 5 or not all(value > 0.0 for value in fold_deltas):
        raise ValueError(f"official100 main AP is not positive in every fold: {fold_deltas}")
    return {"global_deltas": deltas, "fold_main_ap_deltas": fold_deltas}


def preflight(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    official = read_scene_list(args.official_scene_list)
    if len(scenes) != 60 or len(official) != 100:
        raise ValueError("frozen protocol requires safety60 and official100")
    overlap = sorted(set(scenes) & set(official))
    if overlap:
        raise ValueError(f"official100/safety60 overlap: {overlap}")

    plan_summary_path = args.plan_root / "summary.json"
    plan_summary = json.loads(plan_summary_path.read_text())
    required = {
        "version": PLAN_VERSION,
        "policy": POLICY,
        "scene_count": 60,
        "official100_target_scene_overlap_count": 0,
        "feature_schema_verified": True,
        "plan_complete": True,
        "native_scores_bit_for_bit_frozen": True,
        "candidate_count_modification_count": 0,
        "candidate_geometry_modification_count": 0,
        "candidate_class_modification_count": 0,
        "candidate_removal_count": 0,
        "threshold_scanning": False,
        "continuous_weight_scanning": False,
        "ground_truth_usage": "none",
        "ap_evaluation_run": False,
        "ap_evaluation_run_count": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": plan_summary.get(key)}
        for key, value in required.items() if plan_summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"frozen plan contract mismatch: {mismatches}")

    model_metadata_path = args.model_root / "metadata.json"
    model_metadata = json.loads(model_metadata_path.read_text())
    package_path = args.model_root / "model_package.pkl"
    package_sha = _sha256(package_path)
    if package_sha != model_metadata["model_package_sha256"]:
        raise ValueError("model package SHA-256 mismatch")
    if plan_summary["input_provenance"]["model_package_sha256"] != package_sha:
        raise ValueError("plan/model package SHA-256 mismatch")
    if model_metadata.get("frozen_policy") != POLICY:
        raise ValueError("model metadata policy mismatch")
    with package_path.open("rb") as handle:
        package = pickle.load(handle)
    expected_names = package["metadata"]["feature_contract"]["candidate_quality_feature_names"]

    eligibility = _official_eligibility(args.official_oof_summary)
    baseline_summary = json.loads(args.frozen_coexist_summary.read_text())
    baseline = baseline_summary["AP结果"]["基线分组原始分数加轨迹质量分数"]

    aggregate = read_jsonl(args.plan_root / "frozen_score_plan.jsonl")
    identities = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        for row in aggregate
    }
    if len(identities) != len(aggregate) or len(aggregate) != plan_summary["candidate_count"]:
        raise ValueError("aggregate score plan is incomplete or duplicated")
    aggregate_by_scene = {}
    for row in aggregate:
        validate_score_row(row)
        aggregate_by_scene.setdefault(str(row["scene_name"]), []).append(row)
    if set(aggregate_by_scene) != set(scenes):
        raise ValueError("aggregate score plan scene coverage mismatch")

    candidate_audit_count = 0
    score_row_count = 0
    for index, scene in enumerate(scenes, start=1):
        masks, native_scores, _classes, feature_rows, tracks, candidate_audit = load_safety_scene(
            scene, args.native_cache, args.quality_ledger_root, args.track_root,
        )
        scene_summary = json.loads((args.plan_root / scene / "summary.json").read_text())
        if candidate_audit != scene_summary["candidate_audit"]:
            raise ValueError(f"{scene}: current candidate inputs differ from plan")
        scene_rows = read_jsonl(args.plan_root / scene / "frozen_score_plan.jsonl")
        if scene_rows != aggregate_by_scene[scene]:
            raise ValueError(f"{scene}: aggregate and scene score plans differ")
        matrix, names = feature_matrix(feature_rows, FEATURE_GROUP, PROTOCOL_NAME)
        if names != expected_names:
            raise ValueError(f"{scene}: candidate-quality feature schema mismatch")
        q_model = package["quality_models"]["q"]
        q_values = canonicalize_predictions(
            "q", q_model.predict_proba(matrix)[:, 1]
            if hasattr(q_model, "predict_proba") else q_model.predict(matrix)
        )
        by_key = {
            (str(row["candidate_source"]), int(row["candidate_id"])): row
            for row in scene_rows
        }
        native_count = masks.shape[1]
        if len(by_key) != native_count + len(tracks):
            raise ValueError(f"{scene}: score plan candidate coverage mismatch")
        for position, feature in enumerate(feature_rows):
            key = (str(feature["candidate_source"]), int(feature["candidate_id"]))
            row = by_key[key]
            expected_coexist = (
                float(native_scores[key[1]]) if key[0] == NATIVE_SOURCE
                else float(q_values[position])
            )
            if float(row["frozen_coexist_score"]) != expected_coexist:
                raise ValueError(f"{scene}: frozen coexist score replay mismatch for {key}")
            validate_score_row(row)
        candidate_audit_count += 1
        score_row_count += len(scene_rows)
        print(f"[track-harm preflight] {index}/60 {scene}", flush=True)
    return {
        "preflight_passed": True,
        "scene_count": 60,
        "official100_safety60_overlap_count": 0,
        "official100_eligibility": eligibility,
        "frozen_coexist_baseline": baseline,
        "candidate_audit_count": candidate_audit_count,
        "score_row_count": score_row_count,
        "model_package_sha256": package_sha,
        "plan_summary_sha256": _sha256(plan_summary_path),
        "score_plan_sha256": _sha256(args.plan_root / "frozen_score_plan.jsonl"),
        "official_oof_summary_sha256": _sha256(args.official_oof_summary),
        "frozen_coexist_summary_sha256": _sha256(args.frozen_coexist_summary),
        "ground_truth_read": False,
        "ap_evaluation_run_count": 0,
    }


def run_once(args: argparse.Namespace, audit: dict) -> dict:
    scenes = read_scene_list(args.scene_list)
    matches = {}
    scene_rows = []
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original_load_ids(path))
    try:
        for index, scene in enumerate(scenes, start=1):
            masks, native_scores, _classes, feature_rows, tracks, _audit = load_safety_scene(
                scene, args.native_cache, args.quality_ledger_root, args.track_root,
            )
            plan = {
                (str(row["candidate_source"]), int(row["candidate_id"])): row
                for row in read_jsonl(args.plan_root / scene / "frozen_score_plan.jsonl")
            }
            groups, group_audit = geometry_groups_and_audit(
                masks,
                [int(row["point_count"]) for row in feature_rows[:masks.shape[1]]],
                None,
            )
            native_ids = sorted(min(
                members, key=lambda candidate_id: (-float(native_scores[candidate_id]), candidate_id)
            ) for members in groups)
            tracks = sorted(tracks, key=lambda row: int(row["track_id"]))
            track_ids = [int(row["track_id"]) for row in tracks]
            track_masks = np.zeros((masks.shape[0], len(tracks)), dtype=bool)
            for column, track in enumerate(tracks):
                points, _ = _load_track_points(track, masks.shape[0])
                track_masks[points, column] = True
            selected_masks = np.concatenate([
                np.asarray(masks[:, native_ids], dtype=bool), track_masks,
            ], axis=1)
            selected_scores = np.asarray([
                *(float(plan[(NATIVE_SOURCE, candidate_id)]["planned_score"])
                  for candidate_id in native_ids),
                *(float(plan[(TRACK_SOURCE, track_id)]["planned_score"])
                  for track_id in track_ids),
            ], dtype=np.float64)
            gt_file = str(args.gt_dir / f"{scene}.txt")
            gt, pred = instance_eval.assign_instances_for_scan(
                _prediction(selected_masks, selected_scores, len(selected_scores)), gt_file,
            )
            matches[os.path.abspath(gt_file)] = {"gt": gt, "pred": pred}
            scene_rows.append({
                "scene_name": scene,
                "native_geometry_representative_count": len(native_ids),
                "track_candidate_count": len(track_ids),
                "candidate_count": len(selected_scores),
                "score_changed_track_count": sum(
                    bool(plan[(TRACK_SOURCE, track_id)]["score_changed_vs_frozen_coexist"])
                    for track_id in track_ids
                ),
                "canonical_mask_element_sha256": group_audit["canonical_mask_element_sha256"],
                "candidate_files_modified": False,
            })
            print(f"[track-harm AP matching] {index}/60 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    staging = args.output_dir.parent / f".{args.output_dir.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    receipt = args.output_dir.parent / f".{args.output_dir.name}.ap_once_receipt.json"
    if receipt.exists():
        raise ValueError(f"one-shot AP receipt already exists: {receipt}")
    receipt.write_text(json.dumps({
        "version": VERSION,
        "policy": POLICY,
        "ap_call_started_utc": datetime.now(timezone.utc).isoformat(),
        "ap_evaluation_run_count": 1,
        "completed": False,
        "plan_summary_sha256": audit["plan_summary_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    try:
        # Deliberately the sole AP aggregation call in this entry point.
        selected = _evaluate(POLICY, matches, staging)
        baseline = audit["frozen_coexist_baseline"]
        summary = {
            "version": VERSION,
            "policy": POLICY,
            "experiment": "one-shot safety60 class-agnostic AP for frozen track-harm score plan",
            "preflight": audit,
            "frozen_coexist_baseline": baseline,
            "selected_policy_ap": selected,
            "delta_vs_frozen_coexist": {
                key: float(selected[key] - baseline[key]) for key in ("ap", "ap50", "ap25")
            },
            "scene_candidate_counts": scene_rows,
            "score_changed_track_count": sum(
                row["score_changed_track_count"] for row in scene_rows
            ),
            "candidate_count_modified": False,
            "candidate_geometry_modified": False,
            "candidate_class_modified": False,
            "candidate_files_modified": False,
            "candidate_file_modification_count": 0,
            "ground_truth_usage": "one frozen AP evaluation only",
            "threshold_scanning": False,
            "continuous_weight_scanning": False,
            "evaluated_policy_count": 1,
            "ap_evaluation_run_count": 1,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_dir)
        receipt.write_text(json.dumps({
            "version": VERSION,
            "policy": POLICY,
            "ap_call_started": True,
            "ap_evaluation_run_count": 1,
            "completed": True,
            "output_summary_sha256": _sha256(args.output_dir / "summary.json"),
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--official-scene-list", type=Path, required=True)
    parser.add_argument("--official-oof-summary", type=Path, required=True)
    parser.add_argument("--frozen-coexist-summary", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list", "official_scene_list", "official_oof_summary",
        "frozen_coexist_summary", "model_root", "plan_root", "native_cache",
        "track_root", "quality_ledger_root", "gt_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"output already exists; one-shot AP will not rerun: {args.output_dir}")
    audit = preflight(args)
    if args.preflight_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not args.allow_gt_evaluation:
        raise SystemExit("must pass --allow-gt-evaluation after preflight")
    print(json.dumps(run_once(args, audit), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
