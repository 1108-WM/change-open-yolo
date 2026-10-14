#!/usr/bin/env python3
"""Materialize a frozen no-GT official100 OOF pair-proposal append plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
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
from tools.build_train_candidate_component_union_feature_ledger import _track_points  # noqa: E402
from tools.build_train_candidate_pair_union_utility_ledger import proposal_points  # noqa: E402
from tools.diagnose_candidate_pair_union_threshold_cross_oof import (  # noqa: E402
    VERSIONS as OOF_VERSIONS,
)
from tools.fit_candidate_pair_union_threshold_cross_full import (  # noqa: E402
    VERSIONS as FULL_VERSIONS,
)
from tools.evaluate_official100_geometry_group_ranking_oof_ap import (  # noqa: E402
    EXPECTED_OOF_SHA256,
)


VERSIONS = {
    "pair_union": "official100_pair_union_oof_append_plan_v1",
    "pair_intersection": "official100_pair_intersection_oof_append_plan_v1",
}
POLICIES = {
    "pair_union": "pair_union_threshold_cross_focal_append",
    "pair_intersection": "pair_intersection_threshold_cross_focal_append",
}
VERSION = VERSIONS["pair_union"]
POLICY = POLICIES["pair_union"]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_score(base_quality: float, crossing_probability: float) -> float:
    base = min(1.0, max(0.0, float(base_quality)))
    probability = min(1.0, max(0.0, float(crossing_probability)))
    return base * (1.0 - (1.0 - probability) ** 3)


def _geometry_digest(points: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(len(points).to_bytes(8, "little"))
    digest.update(np.ascontiguousarray(points, dtype=np.int64).tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("pair-proposal OOF plan requires official100")
    diagnostic = json.loads(args.oof_diagnostic_summary.read_text())
    if diagnostic.get("version") != OOF_VERSIONS[args.proposal_kind]:
        raise ValueError(f"unexpected {args.proposal_kind} OOF diagnostic version")
    if not diagnostic["gates"].get("proposal_materialization_allowed"):
        raise ValueError(f"{args.proposal_kind} OOF diagnostic did not allow materialization")
    if _sha256(args.quality_oof_predictions) != EXPECTED_OOF_SHA256:
        raise ValueError("candidate-quality OOF prediction SHA-256 mismatch")
    full_metadata = json.loads(args.full_model_metadata.read_text())
    if full_metadata.get("version") != FULL_VERSIONS[args.proposal_kind]:
        raise ValueError(f"unexpected full {args.proposal_kind} model metadata")

    quality = {}
    for line in args.quality_oof_predictions.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        quality[key] = float(row["predictions"]["D_plus_gvc"]["q"])
    predictions = [
        json.loads(line)
        for line in args.pair_proposal_oof_predictions.read_text().splitlines()
        if line.strip()
    ]
    by_scene = defaultdict(list)
    for row in predictions:
        by_scene[str(row["scene_name"])].append(row)
    if set(by_scene) != set(scenes):
        raise ValueError(f"{args.proposal_kind} OOF prediction scene coverage mismatch")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    all_plan_rows = []
    raw_count = 0
    try:
        for scene_index, scene in enumerate(scenes, start=1):
            scene_root = staging / scene
            candidates_root = scene_root / "candidates"
            candidates_root.mkdir(parents=True)
            relation_rows = read_jsonl(
                args.relation_feature_ledger_root / scene / "relation_features.jsonl"
            )
            relation_lookup = {
                (int(row["track_id"]), str(row["native_exact_geometry_group_id"])): row
                for row in relation_rows
            }
            cache_root = args.records_root / scene / "native_cache"
            masks = np.load(cache_root / f"{scene}_pred_masks.npy", mmap_mode="r")
            track_payload = json.loads((
                args.records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
            ).read_text())
            track_by_id = {int(row["track_id"]): row for row in track_payload["tracks"]}
            track_points = {
                track_id: _track_points(track, masks.shape[0])
                for track_id, track in track_by_id.items()
            }
            proposed = defaultdict(list)
            for prediction in by_scene[scene]:
                track_id = int(prediction["track_id"])
                group_id = str(prediction["native_exact_geometry_group_id"])
                relation = relation_lookup[(track_id, group_id)]
                members = [int(value) for value in relation["native_member_candidate_ids"]]
                native = np.flatnonzero(np.asarray(masks[:, members[0]], dtype=bool)).astype(np.int64)
                for member in members[1:]:
                    if not np.array_equal(
                        np.asarray(masks[:, member], dtype=bool),
                        np.asarray(masks[:, members[0]], dtype=bool),
                    ):
                        raise ValueError(f"{scene}/{group_id}: exact native group geometry differs")
                proposal = proposal_points(args.proposal_kind, track_points[track_id], native)
                digest = _geometry_digest(proposal)
                track_q = quality[(scene, TRACK_SOURCE, track_id)]
                native_q = float(np.median([
                    quality[(scene, NATIVE_SOURCE, member)] for member in members
                ]))
                probability = float(prediction["threshold_cross_probability"])
                base = min(track_q, native_q)
                proposed[digest].append({
                    "points": proposal,
                    "track_id": track_id,
                    "native_exact_geometry_group_id": group_id,
                    "native_member_candidate_ids": members,
                    "threshold_cross_probability": probability,
                    "track_quality_q": track_q,
                    "native_group_median_quality_q": native_q,
                    "base_quality": base,
                    "new_score": append_score(base, probability),
                })
                raw_count += 1
            scene_rows = []
            for candidate_id, (digest, supports) in enumerate(sorted(proposed.items())):
                selected = max(
                    supports,
                    key=lambda row: (
                        row["threshold_cross_probability"], row["base_quality"],
                        -row["track_id"], row["native_exact_geometry_group_id"],
                    ),
                )
                prefix = "union" if args.proposal_kind == "pair_union" else "intersection"
                path = candidates_root / f"{prefix}{candidate_id:04d}_points.npz"
                np.savez_compressed(path, point_indices=selected["points"])
                row = {
                    "scene_name": scene,
                    "candidate_source": f"{args.proposal_kind}_append",
                    "candidate_id": candidate_id,
                    "policy": POLICIES[args.proposal_kind],
                    # Persist the post-commit path.  ``staging`` is atomically
                    # renamed to ``output_root`` after all scene contracts pass.
                    "points_path": str((
                        args.output_root / scene / "candidates" / path.name
                    ).resolve()),
                    "point_count": len(selected["points"]),
                    "geometry_sha256": digest,
                    "support_relation_count": len(supports),
                    "selected_track_id": selected["track_id"],
                    "selected_native_exact_geometry_group_id": selected[
                        "native_exact_geometry_group_id"
                    ],
                    "selected_native_member_candidate_ids": selected[
                        "native_member_candidate_ids"
                    ],
                    "threshold_cross_probability": selected[
                        "threshold_cross_probability"
                    ],
                    "track_quality_q": selected["track_quality_q"],
                    "native_group_median_quality_q": selected[
                        "native_group_median_quality_q"
                    ],
                    "base_quality": selected["base_quality"],
                    "new_score": selected["new_score"],
                    "candidate_retained": True,
                }
                scene_rows.append(row)
                all_plan_rows.append(row)
            plan_name = f"{args.proposal_kind}_append_plan.jsonl"
            (scene_root / plan_name).write_text("".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in scene_rows
            ))
            print(
                f"[{args.proposal_kind} no-GT plan] {scene_index}/100 {scene}: "
                f"raw={len(by_scene[scene])}, unique={len(scene_rows)}",
                flush=True,
            )

        summary = {
            "version": VERSIONS[args.proposal_kind],
            "policy": POLICIES[args.proposal_kind],
            "proposal_kind": args.proposal_kind,
            "scene_count": len(scenes),
            "raw_novel_relation_proposal_count": raw_count,
            "unique_materialized_candidate_count": len(all_plan_rows),
            "duplicate_relation_proposal_count": raw_count - len(all_plan_rows),
            "score_contract": full_metadata["score_contract"],
            "contracts": {
                "append_only": True,
                "existing_candidate_count_modified": False,
                "existing_candidate_geometry_modified": False,
                "existing_candidate_score_modified": False,
                "existing_candidate_class_modified": False,
                "ground_truth_fields_written_to_plan": False,
                "threshold_selected": False,
            },
            "ground_truth_usage_for_plan_generation": "none; only frozen OOF probabilities, quality scores, relation identities, and geometry",
            "ap_computed": False,
            "safety60_read": False,
            "even48_read": False,
            "test60_read": False,
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "quality_oof_predictions_sha256": _sha256(args.quality_oof_predictions),
                "pair_proposal_oof_predictions_sha256": _sha256(
                    args.pair_proposal_oof_predictions
                ),
                "pair_proposal_oof_summary_sha256": _sha256(args.oof_diagnostic_summary),
                "full_model_metadata_sha256": _sha256(args.full_model_metadata),
            },
        }
        (staging / f"{args.proposal_kind}_append_plan.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in all_plan_rows
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
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--quality-oof-predictions", type=Path, required=True)
    parser.add_argument(
        "--pair-union-oof-predictions", "--pair-proposal-oof-predictions",
        dest="pair_proposal_oof_predictions", type=Path, required=True,
    )
    parser.add_argument("--oof-diagnostic-summary", type=Path, required=True)
    parser.add_argument("--full-model-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--proposal-kind", choices=tuple(VERSIONS), default="pair_union"
    )
    args = parser.parse_args()
    for name in (
        "scene_list", "records_root", "relation_feature_ledger_root",
        "quality_oof_predictions", "pair_proposal_oof_predictions",
        "oof_diagnostic_summary", "full_model_metadata", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "raw_novel_relation_proposal_count": summary["raw_novel_relation_proposal_count"],
        "unique_materialized_candidate_count": summary["unique_materialized_candidate_count"],
        "duplicate_relation_proposal_count": summary["duplicate_relation_proposal_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
