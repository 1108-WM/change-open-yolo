#!/usr/bin/env python3
"""Apply the frozen official100 champion to a GT-free DM-SMS-1 candidate pool."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.fi1_shared.evaluate_candidate_quality_reranking_class_agnostic_ap import load_safety_scene
from tools.fi1_shared.run_frozen_safety60_champion_pair_union_ap import (
    POLICY,
    _load_combined_package,
    _track_score_rows_for_scene,
    _union_rows_for_scene,
)
from tools.fi1_shared.build_train_candidate_component_action_utility_ledger import _sha256
from tools.audit_train_candidate_quality_dataset import read_jsonl


VERSION = "dm_sms1_frozen_champion_plan_v1_prior_corrected"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path: Path) -> list[str]:
    result = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or contains duplicates")
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _install_sklearn_pickle_compat() -> None:
    """Restore two stateless Cython unpicklers removed by newer sklearn.

    The frozen official100 package was serialized by a sklearn release whose
    generated module exposed these helpers.  Current local sklearn retains the
    same loss classes but pickles them through their constructors instead.
    Both affected classes are stateless, so construction is an exact adapter,
    not a model or parameter change.
    """
    import sklearn._loss._loss as loss_module

    def stateless_unpickle(pyx_type, _checksum, state):
        if state not in (None, (), {}):
            raise ValueError(f"unexpected non-empty sklearn loss state: {state!r}")
        return pyx_type()

    for class_name in ("CyHalfSquaredError", "CyHalfBinomialLoss"):
        helper = f"__pyx_unpickle_{class_name}"
        if not hasattr(loss_module, helper):
            setattr(loss_module, helper, stateless_unpickle)


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "native_cache", "quality_ledger_root", "track_root",
        "relation_root", "model_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError(
            f"scene count {len(scenes)} differs from expected {args.expected_scene_count}"
        )
    relation_summary = json.loads((args.relation_root / "summary.json").read_text())
    if (
        int(relation_summary.get("scene_count", -1)) != len(scenes)
        or relation_summary.get("ground_truth_usage") != "none"
        or relation_summary.get("ground_truth_read") is not False
        or relation_summary.get("ap_evaluation_run") is not False
    ):
        raise ValueError("relation root does not satisfy the frozen no-GT contract")
    _install_sklearn_pickle_compat()
    metadata, track_package, union_package = _load_combined_package(args.model_root)
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    all_scores, all_unions, all_audits = [], [], []
    scene_summaries = []
    try:
        for index, scene in enumerate(scenes, 1):
            masks, _native_scores, _classes, feature_rows, tracks, candidate_audit = load_safety_scene(
                scene, args.native_cache, args.quality_ledger_root, args.track_root,
            )
            relation_scene_summary = json.loads(
                (args.relation_root / scene / "summary.json").read_text()
            )
            if relation_scene_summary.get("candidate_audit") != candidate_audit:
                raise ValueError(f"{scene}: relation candidate audit differs from current inputs")
            relation_rows = read_jsonl(
                args.relation_root / scene / "relation_features_no_gt.jsonl"
            )
            score_rows, score_summary = _track_score_rows_for_scene(
                scene, masks, feature_rows, tracks, relation_rows, candidate_audit,
                track_package,
            )
            union_audits, union_rows, union_summary = _union_rows_for_scene(
                scene, relation_rows, masks, feature_rows, tracks,
                track_package, union_package,
            )
            scene_root = staging / scene
            candidate_root = scene_root / "pair_union_candidates"
            candidate_root.mkdir(parents=True)
            persisted_unions = []
            for row in union_rows:
                points = row.pop("_points")
                path = candidate_root / f"union{int(row['candidate_id']):04d}_points.npz"
                np.savez_compressed(path, point_indices=points)
                persisted_unions.append({
                    **row,
                    "points_path": str(
                        (args.output_root / scene / "pair_union_candidates" / path.name).resolve()
                    ),
                })
            _write_jsonl(scene_root / "frozen_score_plan.jsonl", score_rows)
            _write_jsonl(scene_root / "pair_union_relation_audit_no_gt.jsonl", union_audits)
            _write_jsonl(scene_root / "pair_union_append_candidates.jsonl", persisted_unions)
            scene_summary = {
                "scene_name": scene,
                "candidate_audit": candidate_audit,
                "track_score_plan": score_summary,
                "pair_union": union_summary,
                "ground_truth_usage": "none",
                "ground_truth_read": False,
                "ap_evaluation_run": False,
                "candidate_file_modification_count": 0,
                "candidate_files_modified": False,
            }
            (scene_root / "summary.json").write_text(
                json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            all_scores.extend(score_rows)
            all_unions.extend(persisted_unions)
            all_audits.extend(union_audits)
            scene_summaries.append(scene_summary)
            print(
                f"[DM-SMS-1 champion] {index}/{len(scenes)} {scene}: "
                f"scores={len(score_rows)} unions={len(persisted_unions)}",
                flush=True,
            )
        _write_jsonl(staging / "frozen_score_plan.jsonl", all_scores)
        _write_jsonl(staging / "pair_union_relation_audit_no_gt.jsonl", all_audits)
        _write_jsonl(staging / "pair_union_append_candidates.jsonl", all_unions)
        output = {
            "version": VERSION,
            "policy": POLICY,
            "scene_count": len(scenes),
            "candidate_count": len(all_scores),
            "track_score_changed_count": sum(
                bool(row["score_changed_vs_frozen_coexist"]) for row in all_scores
            ),
            "pair_union_relation_count": len(all_audits),
            "pair_union_eligible_relation_proposal_count": sum(
                bool(row["eligible_novel_min_region"]) for row in all_audits
            ),
            "pair_union_append_candidate_count": len(all_unions),
            "candidate_count_modification_count": 0,
            "candidate_geometry_modification_count": 0,
            "candidate_class_modification_count": 0,
            "candidate_removal_count": 0,
            "native_scores_bit_for_bit_frozen": all(
                row["planned_score"] == row["frozen_coexist_score"]
                for row in all_scores
                if row["candidate_source"] == "native_mask3d_yoloworld"
            ),
            "track_suppression_is_continuous_score_only": True,
            "pair_union_append_only": True,
            "pair_union_probability_prior_correction": {
                "applied": True,
                "natural_positive_rate": float(union_package["metadata"][
                    "training_component_balanced_natural_positive_rate"
                ]),
            },
            "threshold_scanning": False,
            "continuous_weight_scanning": False,
            "ground_truth_usage": "none",
            "ground_truth_read": False,
            "ap_evaluation_run": False,
            "ap_evaluation_run_count": 0,
            "input_provenance": {
                "combined_model_package_sha256": metadata["model_package_sha256"],
                "combined_model_metadata_sha256": _sha256(args.model_root / "metadata.json"),
                "scene_list_sha256": _sha256(args.scene_list),
                "relation_root_summary_sha256": _sha256(args.relation_root / "summary.json"),
                "scene_candidate_audits": {
                    row["scene_name"]: row["candidate_audit"] for row in scene_summaries
                },
            },
        }
        if not output["native_scores_bit_for_bit_frozen"]:
            raise AssertionError("native score freeze contract failed")
        (staging / "summary.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"),
    )
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--relation-root", type=Path, required=True)
    parser.add_argument(
        "--model-root", type=Path,
        default=Path("output/train_candidate_champion_pair_union_combined_full_official100_v2_prior_corrected"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
