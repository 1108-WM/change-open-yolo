#!/usr/bin/env python3
"""Independently audit semantic candidates against Stage D and joint geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member_value(member: dict, primary: str, legacy: str) -> object:
    return member[primary] if primary in member else member.get(legacy)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=65536)
def _camera_center(pose_path: str) -> np.ndarray:
    matrix = np.asarray(np.loadtxt(pose_path), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid pose matrix: {pose_path}")
    return matrix[:3, 3]


def _view_paths(view: dict) -> tuple[str, str, str, str]:
    rgb_path = Path(str(view["rgb_path"]))
    scene_root = rgb_path.parent.parent
    frame_id = str(view["frame_id"])
    return (
        str(rgb_path),
        str(view.get("depth_path") or (scene_root / "depth" / f"{frame_id}.png")),
        str(view.get("pose_path") or (scene_root / "poses" / f"{frame_id}.txt")),
        str(view.get("intrinsics_path") or (scene_root / "intrinsics.txt")),
    )


def _select_complementary_views(
    views: list[dict], target_count: int, max_input_views: int,
) -> list[int]:
    candidates = [
        (index, view) for index, view in enumerate(views[:max_input_views])
        if _finite(view.get("visible_ratio", 0.0))
        and float(view.get("visible_ratio", 0.0)) > 0.0
        and bool(view.get("sam_mask_valid", False))
    ]
    candidates.sort(key=lambda item: (
        -float(item[1].get("visible_ratio", 0.0)),
        int(item[1].get("frame_index", 0)),
        str(item[1].get("frame_id", "")),
    ))
    if not candidates:
        return []
    selected = [candidates[0][0]]
    centers = {
        str(view["frame_id"]): _camera_center(_view_paths(view)[2])
        for view in views[:max_input_views]
    }
    while len(selected) < min(target_count, len(candidates)):
        scored = []
        for index, view in candidates:
            if index in selected:
                continue
            center = centers[str(view["frame_id"])]
            min_distance = min(
                float(np.linalg.norm(center - centers[str(views[prior]["frame_id"])]))
                for prior in selected
            )
            scored.append((
                min_distance, float(view.get("visible_ratio", 0.0)),
                -int(view.get("frame_index", index)), str(view.get("frame_id", "")), index,
            ))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        selected.append(scored[0][-1])
    return selected


def _expected_selected_views(alpha: dict, target_count: int, max_input_views: int) -> list[dict]:
    views = list(alpha.get("views", []))
    selected = []
    for rank, index in enumerate(_select_complementary_views(views, target_count, max_input_views)):
        view = views[index]
        rgb_path, depth_path, pose_path, intrinsics_path = _view_paths(view)
        selected.append({
            "selection_rank": rank,
            "source_view_index": index,
            "frame_id": str(view["frame_id"]),
            "frame_index": int(view["frame_index"]),
            "visible_ratio": float(view["visible_ratio"]),
            "visible_point_count": int(view["visible_point_count"]),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "pose_path": pose_path,
            "intrinsics_path": intrinsics_path,
            "sam_box_prompt_xyxy": list(view["sam_box_prompt_xyxy"]),
            "sam_mask_sha256": str(view["sam_mask_sha256"]),
            "sam_mask_valid": bool(view["sam_mask_valid"]),
            "sam_mask_area": int(view["sam_mask_area"]),
            "view_selection_reason": "highest_visible_then_farthest_camera_center",
        })
    return selected


def _finite_hypotheses(frozen_class: int, alpha_class: object) -> list[dict]:
    expected = []
    if 0 <= frozen_class < 198:
        expected.append({"class_index": frozen_class, "sources": ["frozen_control"]})
    if alpha_class is not None and 0 <= int(alpha_class) < 198:
        alpha = int(alpha_class)
        if expected and alpha == frozen_class:
            expected[0]["sources"].append("alpha_main")
        else:
            expected.append({"class_index": alpha, "sources": ["alpha_main"]})
    if not expected:
        raise ValueError("candidate has no valid frozen finite hypothesis")
    return expected


def audit(
    root: Path,
    alpha_ledger_root: Path,
    joint_geometry_root: Path,
    expected_candidate_count: int = 39304,
    expected_unique_geometry_count: int = 39250,
) -> dict:
    summary_path = root / "summary.json"
    records_path = root / "semantic_arbitration_manifest.jsonl"
    joint_path = joint_geometry_root / "unique_geometry_ledger.jsonl"
    summary = json.loads(summary_path.read_text())
    rows = _rows(records_path)
    joint_rows = _rows(joint_path)
    errors: list[str] = []
    target_views = int(summary.get("target_views", -1))
    max_input_views = int(summary.get("max_input_views", -1))
    if target_views != 3 or max_input_views != 20:
        errors.append("semantic summary complementary-view parameters differ from frozen contract")

    joint_by_geometry: dict[tuple[str, str], dict] = {}
    expected_by_plan: dict[tuple[str, str], dict] = {}
    for geometry_index, geometry in enumerate(joint_rows):
        identity = (str(geometry.get("scene_name", "")), str(geometry.get("geometry_hash", "")))
        if not all(identity) or identity in joint_by_geometry:
            errors.append(f"joint[{geometry_index}]: empty or duplicate visual geometry identity")
            continue
        joint_by_geometry[identity] = geometry
        members = geometry.get("members", [])
        if not isinstance(members, list) or not members or len(members) != int(geometry.get("member_count", -1)):
            errors.append(f"joint[{geometry_index}]: invalid member contract")
            continue
        for member in members:
            if (
                int(member.get("point_count", -1)) != int(geometry.get("point_count", -2))
                or str(member.get("geometry_hash", identity[1])) != identity[1]
            ):
                errors.append(f"joint[{geometry_index}]: member geometry metadata differs")
            plan_key = str(member.get("plan_key") or member.get("fi1_d_v3_plan_key") or "")
            candidate_identity = (identity[0], plan_key)
            if not plan_key or candidate_identity in expected_by_plan:
                errors.append(f"joint[{geometry_index}]: empty or duplicate plan_key")
                continue
            expected_by_plan[candidate_identity] = {
                "plan_index": int(member.get("plan_index", -1)),
                "geometry_hash": identity[1],
                "visual_geometry_key": str(geometry.get("geometry_key", "")),
                "point_count": int(geometry.get("point_count", -1)),
                "canonical_candidate_id": int(member.get("candidate_id", -1)),
                "member_count": int(geometry.get("member_count", -1)),
                "geometry_locator_read_only": _member_value(
                    member, "geometry_locator_read_only", "geometry_locator"
                ),
                "frozen_class_index": int(member.get("frozen_class_index", -999)),
                "challenger_score": float(_member_value(member, "challenger_score", "frozen_score")),
                "candidate_source": str(member.get("candidate_source", "")),
                "append_only": bool(_member_value(member, "append_only", "fi1_d_v3_append_only")),
            }
    expected_order = [
        identity for identity, _row in sorted(
            expected_by_plan.items(), key=lambda item: int(item[1]["plan_index"])
        )
    ]
    plan_indices = [int(row["plan_index"]) for row in expected_by_plan.values()]
    if len(plan_indices) != len(set(plan_indices)):
        errors.append("joint geometry ledger has duplicate plan_index values")

    alpha_by_geometry: dict[tuple[str, str], dict] = {}
    for scene in sorted({identity[0] for identity in joint_by_geometry}):
        scene_path = alpha_ledger_root / "scenes" / scene / "records.jsonl"
        if not scene_path.is_file():
            errors.append(f"alpha: missing records for {scene}")
            continue
        for alpha_index, alpha in enumerate(_rows(scene_path)):
            alpha_identity = (
                str(alpha.get("scene_name", "")), str(alpha.get("geometry_hash", ""))
            )
            if alpha_identity in alpha_by_geometry:
                errors.append(f"alpha[{scene}/{alpha_index}]: duplicate visual geometry identity")
            else:
                alpha_by_geometry[alpha_identity] = alpha
    if set(alpha_by_geometry) != set(joint_by_geometry):
        errors.append("Alpha/joint visual geometry coverage mismatch")
    for identity, geometry in joint_by_geometry.items():
        alpha = alpha_by_geometry.get(identity)
        if alpha is None:
            continue
        if (
            str(alpha.get("geometry_key", "")) != str(geometry.get("geometry_key", ""))
            or int(alpha.get("point_count", -1)) != int(geometry.get("point_count", -2))
            or int(alpha.get("member_count", -1)) != int(geometry.get("member_count", -2))
            or alpha.get("members") != geometry.get("members")
        ):
            errors.append(f"Alpha/joint member mismatch: {identity}")

    identities = []
    selected_view_count = 0
    hypotheses = 0
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        identity = (str(row.get("scene_name", "")), str(row.get("plan_key", "")))
        identities.append(identity)
        expected = expected_by_plan.get(identity)
        alpha = alpha_by_geometry.get((identity[0], str(row.get("geometry_hash", ""))))
        if expected is None:
            errors.append(f"{prefix}: plan_key is absent from joint geometry ledger")
        else:
            checks = {
                "plan_index": row.get("plan_index") == expected["plan_index"],
                "geometry_hash": row.get("geometry_hash") == expected["geometry_hash"],
                "geometry_locator_read_only": (
                    row.get("geometry_locator_read_only") == expected["geometry_locator_read_only"]
                ),
                "frozen_class_index": row.get("frozen_class_index") == expected["frozen_class_index"],
                "challenger_score": row.get("challenger_score") == expected["challenger_score"],
                "candidate_source": row.get("candidate_source") == expected["candidate_source"],
                "append_only": row.get("append_only") == expected["append_only"],
                "visual_geometry_key": row.get("visual_geometry_key") == expected["visual_geometry_key"],
                "geometry_key": row.get("geometry_key") == row.get("plan_key"),
                "point_count": row.get("point_count") == expected["point_count"],
                "canonical_candidate_id": (
                    row.get("canonical_candidate_id") == expected["canonical_candidate_id"]
                ),
                "visual_evidence_shared_member_count": (
                    row.get("visual_evidence_shared_member_count") == expected["member_count"]
                ),
            }
            for name, valid in checks.items():
                if not valid:
                    errors.append(f"{prefix}: frozen {name} differs from joint ledger")
        if alpha is None:
            errors.append(f"{prefix}: visual evidence is absent from Alpha ledger")
        else:
            if (
                row.get("alpha_class_index") != alpha.get("alpha_class_index")
                or row.get("alpha_top_similarity") != alpha.get("alpha_top_similarity")
                or row.get("sms_keep") != alpha.get("sms_keep")
            ):
                errors.append(f"{prefix}: Alpha/SMS evidence differs from Stage D")
            try:
                expected_views = _expected_selected_views(alpha, target_views, max_input_views)
            except (KeyError, TypeError, ValueError, OSError) as error:
                errors.append(f"{prefix}: complementary-view reconstruction failed: {error}")
            else:
                if row.get("selected_views") != expected_views:
                    errors.append(f"{prefix}: selected views differ from frozen complementary selection")
        if row.get("fi1_d_v3_plan_key") != row.get("plan_key") or not row.get("plan_key"):
            errors.append(f"{prefix}: invalid plan_key identity")
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            errors.append(f"{prefix}: GT/AP provenance is not false")
        if (
            row.get("candidate_source") != row.get("canonical_candidate_source")
            or row.get("frozen_class_index") != row.get("canonical_frozen_class_index")
            or row.get("challenger_score") != row.get("canonical_frozen_score")
            or row.get("append_only") != row.get("fi1_d_v3_append_only")
            or not isinstance(row.get("geometry_locator_read_only"), dict)
            or row.get("candidate_retained") is not True
            or row.get("candidate_deletion") is not False
        ):
            errors.append(f"{prefix}: frozen candidate aliases differ")
        for key in (
            "candidate_mutation", "geometry_mutation", "class_mutation",
            "score_mutation", "class_decision_made",
        ):
            if row.get(key) is not False:
                errors.append(f"{prefix}: {key} is true")
        candidates = row.get("finite_class_hypotheses", [])
        if expected is not None and alpha is not None:
            try:
                expected_candidates = _finite_hypotheses(
                    expected["frozen_class_index"], alpha.get("alpha_class_index")
                )
            except (TypeError, ValueError) as error:
                errors.append(f"{prefix}: finite hypothesis reconstruction failed: {error}")
            else:
                if candidates != expected_candidates:
                    errors.append(f"{prefix}: finite hypotheses differ from frozen class union")
        views = row.get("selected_views", [])
        if len(views) > 3 or len({view.get("frame_id") for view in views}) != len(views):
            errors.append(f"{prefix}: selected views are not unique or exceed three")
        for view in views:
            for key in ("rgb_path", "depth_path", "pose_path", "intrinsics_path"):
                if not Path(view.get(key, "")).is_file():
                    errors.append(f"{prefix}: missing {key}")
            if not view.get("sam_mask_valid"):
                errors.append(f"{prefix}: selected view has invalid SAM mask")
        selected_view_count += len(views)
        hypotheses += len(candidates)

    unique_geometries = len({(row.get("scene_name"), row.get("geometry_hash")) for row in rows})
    if identities != expected_order:
        errors.append("semantic plan_key coverage or plan_index order mismatch")
    if len(identities) != len(set(identities)):
        errors.append("duplicate scene/plan identities")
    if len(rows) != expected_candidate_count:
        errors.append("frozen candidate_count mismatch")
    if len(expected_by_plan) != expected_candidate_count:
        errors.append("joint candidate_count mismatch")
    if unique_geometries != expected_unique_geometry_count:
        errors.append("frozen unique_geometry_count mismatch")
    if len(joint_by_geometry) != expected_unique_geometry_count:
        errors.append("joint unique_geometry_count mismatch")
    if int(summary.get("candidate_count", -1)) != expected_candidate_count:
        errors.append("summary candidate_count mismatch")
    if int(summary.get("unique_geometry_count", -1)) != expected_unique_geometry_count:
        errors.append("summary unique_geometry_count mismatch")
    if int(summary.get("candidate_deletion_count", -1)) != 0:
        errors.append("summary candidate_deletion_count mismatch")
    if int(summary.get("selected_view_count", -1)) != selected_view_count:
        errors.append("summary selected view count mismatch")
    if int(summary.get("candidate_hypothesis_count", -1)) != hypotheses:
        errors.append("summary candidate hypothesis count mismatch")
    if summary.get("ground_truth_read") is not False or summary.get("ap_computed") is not False:
        errors.append("summary GT/AP provenance is not false")
    result = {
        "version": "dm_sms1_semantic_arbitration_manifest_audit_v3",
        "manifest_root": str(root),
        "candidate_count": len(rows),
        "unique_geometry_count": unique_geometries,
        "candidate_deletion_count": len(set(expected_by_plan) - set(identities)),
        "plan_key_coverage_complete": identities == expected_order,
        "frozen_field_error_count": sum("frozen" in error for error in errors),
        "error_count": len(errors),
        "errors": errors,
        "audit_valid": not errors,
        "ground_truth_read": False,
        "ap_computed": False,
        "input_provenance": {
            "semantic_manifest_sha256": _sha256(records_path),
            "alpha_summary_sha256": _sha256(alpha_ledger_root / "summary.json"),
            "joint_geometry_ledger_sha256": _sha256(joint_path),
        },
    }
    (root / "audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("--alpha-ledger-root", type=Path, required=True)
    parser.add_argument("--joint-geometry-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=39304)
    parser.add_argument("--expected-unique-geometry-count", type=int, default=39250)
    args = parser.parse_args()
    result = audit(
        args.manifest_root, args.alpha_ledger_root, args.joint_geometry_root,
        args.expected_candidate_count, args.expected_unique_geometry_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
