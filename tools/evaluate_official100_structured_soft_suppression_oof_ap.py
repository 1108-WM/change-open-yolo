#!/usr/bin/env python3
"""Evaluate confidence-scaled soft suppression for official100 OOF actions.

The selected structured action is never materialized as candidate deletion.
Candidates that the frozen action would remove retain their mask and class,
but their score is multiplied by ``1 - max(0, P(positive)-P(harmful))``.
This has no fitted coefficient or threshold and uses only official100 OOF
predictions.  It never reads or evaluates safety60.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    _scene_inputs,
    _scene_records,
    _sha256,
    configure_track_score_context,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    _evaluate,
    _set_match_scores,
)
from tools.train_candidate_component_action_head_oof import _metrics_from_records  # noqa: E402
from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


POLICY = "structured_lower_relation_veto"
VERSION = "official100_structured_soft_suppression_oof_v1"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def suppression_strength(prediction: dict) -> float:
    probability = prediction["state_probability"]
    value = float(probability["positive"] - probability["harmful"])
    return min(1.0, max(0.0, value))


def _load_action_context(args, scenes: list[str]) -> tuple[dict, dict]:
    predictions = {}
    for row in read_jsonl(args.structured_oof_root / "oof_action_predictions.jsonl"):
        key = (str(row["scene_name"]), int(row["relation_component_id"]), str(row["action_name"]))
        if key in predictions:
            raise ValueError(f"duplicate OOF action prediction: {key}")
        predictions[key] = row
    selected = {}
    for row in read_jsonl(
        args.structured_oof_root / "policy_selected_actions_diagnostic_only.jsonl"
    ):
        if row["policy"] != POLICY:
            continue
        key = (str(row["scene_name"]), int(row["relation_component_id"]))
        if key in selected:
            raise ValueError(f"duplicate selected action: {key}")
        selected[key] = row
    actions = {}
    for scene in scenes:
        for row in read_jsonl(
            args.action_ledger_root / scene / "component_action_utilities.jsonl"
        ):
            key = (scene, int(row["relation_component_id"]), str(row["action_name"]))
            actions[key] = row
    if set(selected) != {
        (scene, int(row["relation_component_id"]))
        for scene in scenes
        for row in read_jsonl(
            args.relation_feature_ledger_root / scene / "relation_components.jsonl"
        )
    }:
        raise ValueError("selected policy does not cover every official100 component")
    return predictions, {key: actions[(key[0], key[1], row["selected_action_name"])] for key, row in selected.items()}


def _suppressed_candidate_keys(action: dict) -> set[tuple[str, object]]:
    kind = str(action["action_kind"])
    if kind == "coexist":
        return set()
    if kind == "baseline_only":
        return {("track", int(value)) for value in action["component_track_ids"]}
    if kind == "track_only_one":
        selected_track = int(action["selected_track_id"])
        keys = {
            ("native_group", str(value))
            for value in action["component_native_exact_geometry_group_ids"]
        }
        keys.update(
            ("track", int(value)) for value in action["component_track_ids"]
            if int(value) != selected_track
        )
        return keys
    raise ValueError(f"unknown action kind: {kind}")


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("soft suppression diagnostic requires official100")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    oof_summary = json.loads((args.structured_oof_root / "summary.json").read_text())
    if oof_summary.get("scene_count") != 100 or oof_summary.get("safety60_evaluated"):
        raise ValueError("structured OOF source violates official-only contract")
    score_context = configure_track_score_context(args, scenes)
    predictions, selected = _load_action_context(args, scenes)
    matches = {}
    caches = {}
    score_rows = []
    action_counts = Counter()
    suppressed_count = 0
    original_load_ids = instance_eval.util_3d.load_ids
    _configure_scannet200_instance_eval()
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original_load_ids(path))
    try:
        for scene_index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            caches[scene] = cache
            scores = {
                row["uuid"]: float(row["confidence"])
                for row in cache["pred"]["chair"]
            }
            candidate_by_uuid = {
                uuid: key for key, uuid in cache["uuid_by_candidate"].items()
            }
            factor_by_candidate = {
                key: 1.0 for key in cache["uuid_by_candidate"]
            }
            for component in cache["components"]:
                component_id = int(component["relation_component_id"])
                action = selected[(scene, component_id)]
                action_counts[str(action["action_kind"])] += 1
                if action["action_kind"] == "coexist":
                    continue
                prediction_key = (scene, component_id, str(action["action_name"]))
                strength = suppression_strength(predictions[prediction_key])
                factor = 1.0 - strength
                for kind, value in _suppressed_candidate_keys(action):
                    if kind == "track":
                        candidate_key = ("track", int(value))
                    else:
                        candidate_key = (
                            "native",
                            int(cache["groups"][str(value)]["representative_candidate_id"]),
                        )
                    if candidate_key in factor_by_candidate:
                        factor_by_candidate[candidate_key] *= factor
            for uuid, candidate_key in candidate_by_uuid.items():
                original = scores[uuid]
                factor = float(factor_by_candidate[candidate_key])
                scores[uuid] = original * factor
                if factor < 1.0:
                    suppressed_count += 1
                score_rows.append({
                    "scene_name": scene,
                    "candidate_kind": candidate_key[0],
                    "candidate_id": candidate_key[1],
                    "original_score": original,
                    "suppression_factor": factor,
                    "selected_score": scores[uuid],
                    "candidate_retained": True,
                })
            matches[os.path.abspath(str(args.gt_dir / f"{scene}.txt"))] = {
                "gt": copy.deepcopy(cache["gt"]),
                "pred": copy.deepcopy(cache["pred"]),
            }
            _set_match_scores(
                {os.path.abspath(str(args.gt_dir / f"{scene}.txt")): matches[os.path.abspath(str(args.gt_dir / f"{scene}.txt"))]},
                scores,
            )
            print(f"[soft suppression OOF] {scene_index}/100 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    scene_records = {}
    for scene in scenes:
        key = os.path.abspath(str(args.gt_dir / f"{scene}.txt"))
        cache = {
            **caches[scene],
            "gt": matches[key]["gt"],
            "pred": matches[key]["pred"],
        }
        scene_records[scene] = _scene_records(
            cache, cache["all_native_representatives"], cache["all_track_ids"],
        )
    overall_records = _metrics_from_records(scene_records)
    fold_metrics = []
    for fold in manifest["folds"]:
        validation = set(fold["validation_scenes"])
        fold_metrics.append({
            "fold_index": int(fold["fold_index"]),
            "global_metrics": _metrics_from_records({
                scene: scene_records[scene] for scene in validation
            }),
        })

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        ap = _evaluate("soft_suppression", matches, staging)
        baseline = oof_summary["policy_ap_results"]["always_coexist"]["global_metrics"]
        hard = oof_summary["policy_ap_results"][POLICY]["global_metrics"]
        summary = {
            "version": VERSION,
            "scene_count": 100,
            "source_policy": POLICY,
            "score_rule": "score *= 1 - max(0, P(positive)-P(harmful)) for candidates the hard action removes",
            "fitted_soft_suppression_parameter_count": 0,
            "candidate_count_modified": False,
            "candidate_geometry_modified": False,
            "candidate_class_modified": False,
            "suppressed_candidate_count": suppressed_count,
            "selected_action_kind_counts": dict(sorted(action_counts.items())),
            "soft_ap": ap,
            "soft_global_record_metrics": overall_records,
            "hard_policy_global_metrics": hard,
            "coexist_global_metrics": baseline,
            "delta_vs_coexist": {
                "official_ap": overall_records["official_ap"] - baseline["official_ap"],
                "ap50": overall_records["threshold_metrics"]["50"]["ap"] - baseline["threshold_metrics"]["50"]["ap"],
                "ap25": overall_records["threshold_metrics"]["25"]["ap"] - baseline["threshold_metrics"]["25"]["ap"],
            },
            "fold_metrics": fold_metrics,
            "score_context": score_context,
            "safety60_read": False,
            "safety60_evaluated": False,
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "split_manifest_sha256": _sha256(args.split_manifest),
                "structured_oof_summary_sha256": _sha256(args.structured_oof_root / "summary.json"),
            },
        }
        (staging / "candidate_scores.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in score_rows
        ))
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--structured-oof-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--track-score-mode", choices=("original", "oof_quality"), default="oof_quality")
    parser.add_argument("--oof-predictions", type=Path)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must explicitly pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "relation_feature_ledger_root",
        "action_ledger_root", "structured_oof_root", "gt_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.oof_predictions is not None:
        args.oof_predictions = _resolve(args.oof_predictions)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root exists and is non-empty: {args.output_root}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
