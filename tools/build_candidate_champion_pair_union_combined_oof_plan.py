#!/usr/bin/env python3
"""Build the frozen no-GT combined champion plus pair-union OOF plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.build_candidate_pair_union_oof_plan import (  # noqa: E402
    POLICY as UNION_POLICY,
    VERSION as UNION_PLAN_VERSION,
)
from tools.train_candidate_component_union_list_head_oof import (  # noqa: E402
    PRIMARY_POLICY as CHAMPION_POLICY,
    VERSION as CHAMPION_VERSION,
)


VERSION = "official100_champion_pair_union_combined_oof_plan_v1"
POLICY = "champion_track_suppression_plus_pair_union_append"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("combined OOF plan requires official100")
    champion_summary = json.loads(args.champion_summary.read_text())
    if champion_summary.get("version") != CHAMPION_VERSION:
        raise ValueError("unexpected champion OOF summary version")
    if champion_summary.get("primary_policy") != CHAMPION_POLICY:
        raise ValueError("unexpected champion primary policy")
    union_summary_path = args.pair_union_plan_root / "summary.json"
    union_summary = json.loads(union_summary_path.read_text())
    if (
        union_summary.get("version") != UNION_PLAN_VERSION
        or union_summary.get("policy") != UNION_POLICY
    ):
        raise ValueError("unexpected pair-union plan")
    if union_summary.get("ap_computed") is not False:
        raise ValueError("pair-union plan must be frozen before combined AP")

    overrides = []
    identities = set()
    inherited_zero_score_clip_count = 0
    for line in args.champion_oof_scores.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("policy") != CHAMPION_POLICY:
            continue
        if row["candidate_source"] != "d2b_track":
            raise ValueError("champion primary policy must only rewrite tracks")
        key = (str(row["scene_name"]), int(row["candidate_id"]))
        if key in identities:
            raise ValueError(f"duplicate champion track override: {key}")
        identities.add(key)
        original = float(row["original_score"])
        new_score = float(row["new_score"])
        inherited_zero_clip = original == 0.0 and new_score <= 1e-6 + 1e-12
        if not 0.0 <= new_score <= original + 1e-12 and not inherited_zero_clip:
            raise ValueError(f"champion track override is not suppression: {key}")
        inherited_zero_score_clip_count += int(inherited_zero_clip)
        overrides.append({
            "scene_name": key[0],
            "candidate_source": "d2b_track",
            "candidate_id": key[1],
            "policy": CHAMPION_POLICY,
            "original_score": original,
            "new_score": new_score,
            "candidate_retained": True,
        })
    if not overrides:
        raise ValueError("champion score plan has no primary-policy track overrides")

    unions = [
        json.loads(line)
        for line in (
            args.pair_union_plan_root / "pair_union_append_plan.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    if len(unions) != int(union_summary["unique_materialized_candidate_count"]):
        raise ValueError("pair-union plan count differs from summary")
    override_lookup = {
        (row["scene_name"], row["candidate_id"]): float(row["new_score"])
        for row in overrides
    }
    source_track_scores = []
    union_scores = []
    union_above_source = 0
    exact_ties = 0
    for row in unions:
        if not Path(row["points_path"]).is_file():
            raise FileNotFoundError(row["points_path"])
        key = (str(row["scene_name"]), int(row["selected_track_id"]))
        source_score = override_lookup.get(key, float(row["track_quality_q"]))
        union_score = float(row["new_score"])
        source_track_scores.append(source_score)
        union_scores.append(union_score)
        union_above_source += int(union_score > source_score)
        exact_ties += int(union_score == source_score)

    observed_scenes = {row["scene_name"] for row in overrides} | {
        row["scene_name"] for row in unions
    }
    if observed_scenes != set(scenes):
        raise ValueError("combined plan scene coverage differs from official100")
    summary = {
        "version": VERSION,
        "policy": POLICY,
        "scene_count": len(scenes),
        "champion_track_override_count": len(overrides),
        "pair_union_append_candidate_count": len(unions),
        "score_collision_audit": {
            "champion_track_new_score_distribution": _distribution([
                float(row["new_score"]) for row in overrides
            ]),
            "pair_union_score_distribution": _distribution(union_scores),
            "selected_source_track_score_after_champion_distribution": _distribution(
                source_track_scores
            ),
            "pair_union_score_above_selected_source_track_count": union_above_source,
            "pair_union_score_equal_selected_source_track_count": exact_ties,
            "inherited_champion_zero_to_1e6_clip_count": inherited_zero_score_clip_count,
            "nonfinite_score_count": sum(
                not np.isfinite(value)
                for value in [*union_scores, *source_track_scores]
            ),
        },
        "combination_contract": {
            "champion_formula_unchanged": True,
            "pair_union_formula_unchanged": True,
            "cross_module_weight_added": False,
            "threshold_selected": False,
            "candidate_removed": False,
            "native_geometry_score_class_modified": False,
            "uncontrolled_track_modified": False,
            "pair_union_append_only": True,
        },
        "ground_truth_usage_for_plan_generation": "none; joins two frozen no-GT OOF plans",
        "ap_computed": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "champion_summary_sha256": _sha256(args.champion_summary),
            "champion_oof_scores_sha256": _sha256(args.champion_oof_scores),
            "pair_union_plan_summary_sha256": _sha256(union_summary_path),
            "pair_union_plan_rows_sha256": _sha256(
                args.pair_union_plan_root / "pair_union_append_plan.jsonl"
            ),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "champion_track_score_overrides.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in sorted(overrides, key=lambda row: (
                row["scene_name"], row["candidate_id"]
            ))
        ))
        (staging / "pair_union_append_candidates.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in unions
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
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--champion-summary", type=Path, required=True)
    parser.add_argument("--champion-oof-scores", type=Path, required=True)
    parser.add_argument("--pair-union-plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "champion_summary", "champion_oof_scores",
        "pair_union_plan_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "champion_track_override_count": summary["champion_track_override_count"],
        "pair_union_append_candidate_count": summary["pair_union_append_candidate_count"],
        "score_collision_audit": summary["score_collision_audit"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
