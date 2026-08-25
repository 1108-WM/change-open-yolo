#!/usr/bin/env python3
"""Audit category-blind attribute inputs against the semantic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dm_sms1_terminal_safe_keep import (  # noqa: E402
    TERMINAL_KEEP_REASON,
    terminal_expected_identities,
)


FROZEN_ATTRIBUTE_PROMPT = (
    "你将看到同一个三维候选物体的三张互补视角，以及对应的深度、位姿和候选掩码证据。"
    "不要猜测类别，不要输出任何类别名称，也不要把它和候选类别列表联系起来。"
    "只根据可观察证据记录：外观颜色和纹理、材质、形状和结构、可能的功能线索、"
    "与周围环境的空间关系。每项都要给出观察内容、支持它的视角编号、0到1的证据把握度、"
    "以及看不清或相互矛盾的地方；无法判断时填写 unknown。"
)

FROZEN_RESPONSE_SCHEMA = {
    "appearance": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "material": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "shape_structure": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "function_cues": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "spatial_context": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "cross_view_consistency": {"observation": "string", "confidence": "number_0_to_1", "counterevidence": "string"},
    "missing_or_unclear_evidence": ["string"],
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_task_id(semantic: dict) -> str:
    payload = {
        "scene_name": semantic["scene_name"],
        "plan_key": semantic["plan_key"],
        "geometry_hash": semantic["geometry_hash"],
        "frames": [view["frame_id"] for view in semantic.get("selected_views", [])],
        "mask_hashes": [view["sam_mask_sha256"] for view in semantic.get("selected_views", [])],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{semantic['scene_name']}::{semantic['plan_key']}::{digest}"


def audit(
    root: Path,
    semantic_root: Path,
    expected_candidate_count: int = 39304,
    expected_unique_geometry_count: int = 39250,
    expected_terminal_identities: set[tuple[int, str]] | None = None,
) -> dict:
    summary_path = root / "summary.json"
    records_path = root / "attribute_extraction_manifest.jsonl"
    semantic_path = semantic_root / "semantic_arbitration_manifest.jsonl"
    summary = json.loads(summary_path.read_text())
    semantic_summary = json.loads((semantic_root / "summary.json").read_text())
    records = _rows(records_path)
    semantic_rows = _rows(semantic_path)
    errors: list[str] = []

    semantic_ids = [
        (str(row.get("scene_name", "")), str(row.get("plan_key", "")))
        for row in semantic_rows
    ]
    semantic_by_id = dict(zip(semantic_ids, semantic_rows))
    if any(not scene or not key for scene, key in semantic_ids):
        errors.append("semantic manifest has empty scene/plan identity")
    if len(semantic_ids) != len(set(semantic_ids)):
        errors.append("semantic manifest has duplicate scene/plan identities")

    task_ids = []
    candidate_ids = []
    view_count = 0
    terminal_hits: set[tuple[int, str]] = set()
    for index, row in enumerate(records):
        prefix = f"row[{index}]"
        task_ids.append(str(row.get("task_id", "")))
        identity = (str(row.get("scene_name", "")), str(row.get("plan_key", "")))
        candidate_ids.append(identity)
        semantic = semantic_by_id.get(identity)
        if semantic is None:
            errors.append(f"{prefix}: plan_key is absent from semantic manifest")
        else:
            expected_views = [
                {
                    "view_rank": view.get("selection_rank"),
                    "frame_id": view.get("frame_id"),
                    "frame_index": view.get("frame_index"),
                    "rgb_path": view.get("rgb_path"),
                    "depth_path": view.get("depth_path"),
                    "pose_path": view.get("pose_path"),
                    "intrinsics_path": view.get("intrinsics_path"),
                    "sam_box_prompt_xyxy": view.get("sam_box_prompt_xyxy"),
                    "sam_mask_sha256": view.get("sam_mask_sha256"),
                    "visible_ratio": view.get("visible_ratio"),
                    "visible_point_count": view.get("visible_point_count"),
                }
                for view in semantic.get("selected_views", [])
            ]
            checks = {
                "task_id": row.get("task_id") == _expected_task_id(semantic),
                "plan_index": row.get("plan_index") == semantic.get("plan_index"),
                "geometry_key": row.get("geometry_key") == semantic.get("geometry_key"),
                "geometry_hash": row.get("geometry_hash") == semantic.get("geometry_hash"),
                "visual_geometry_key": (
                    row.get("visual_geometry_key") == semantic.get("visual_geometry_key")
                ),
                "point_count": row.get("point_count") == semantic.get("point_count"),
                "candidate_hypothesis_count": (
                    row.get("candidate_hypothesis_count")
                    == len(semantic.get("finite_class_hypotheses", []))
                ),
                "view_inputs": row.get("view_inputs") == expected_views,
            }
            for name, valid in checks.items():
                if not valid:
                    errors.append(f"{prefix}: semantic {name} differs")
            if semantic.get("terminal_safe_keep") is True:
                terminal_identity = (int(semantic.get("plan_index", -1)), identity[1])
                terminal_hits.add(terminal_identity)
                for key, value in {
                    "attribute_execution_required": False,
                    "view_inputs": [],
                    "attribute_extraction_completed": False,
                    "terminal_safe_keep": True,
                    "terminal_keep_reason": TERMINAL_KEEP_REASON,
                    "candidate_hypothesis_count": 0,
                    "attribute_prompt": None,
                    "response_schema": None,
                }.items():
                    if row.get(key) != value:
                        errors.append(f"{prefix}: terminal field {key} differs")
            elif row.get("terminal_safe_keep") is not False:
                errors.append(f"{prefix}: ordinary row has terminal-safe-keep state")
            elif row.get("attribute_execution_required") is not True:
                errors.append(f"{prefix}: ordinary row disables attribute execution")
        if row.get("fi1_d_v3_plan_key") != row.get("plan_key") or not row.get("plan_key"):
            errors.append(f"{prefix}: invalid plan_key identity")
        if str(row.get("plan_key", "")) not in str(row.get("task_id", "")):
            errors.append(f"{prefix}: task_id does not contain plan_key")
        if row.get("candidate_labels_hidden") is not True:
            errors.append(f"{prefix}: candidate labels are not hidden")
        if row.get("attribute_extraction_completed") is not False or row.get("class_decision_made") is not False:
            errors.append(f"{prefix}: attribute/class decision already completed")
        for key in ("candidate_mutation", "geometry_mutation", "score_mutation"):
            if row.get(key) is not False:
                errors.append(f"{prefix}: {key} is true")
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            errors.append(f"{prefix}: GT/AP provenance is not false")
        terminal = bool(row.get("terminal_safe_keep", False))
        if not terminal and row.get("attribute_prompt") != FROZEN_ATTRIBUTE_PROMPT:
            errors.append(f"{prefix}: fixed category-blind prompt differs")
        if not terminal and row.get("response_schema") != FROZEN_RESPONSE_SCHEMA:
            errors.append(f"{prefix}: fixed response schema differs")
        forbidden_keys = {
            "finite_class_hypotheses", "canonical_frozen_class_index", "alpha_class_index",
            "class_names", "candidate_labels",
        }
        if forbidden_keys.intersection(row):
            errors.append(f"{prefix}: candidate label field leaked into model input")
        views = row.get("view_inputs", [])
        if (terminal and views) or (not terminal and (not views or len(views) > 3)) or len({view.get("frame_id") for view in views}) != len(views):
            errors.append(f"{prefix}: invalid view input set")
        for view in views:
            for key in ("rgb_path", "depth_path", "pose_path", "intrinsics_path"):
                if not Path(view.get(key, "")).is_file():
                    errors.append(f"{prefix}: missing {key}")
        view_count += len(views)

    unique_geometries = len({(row.get("scene_name"), row.get("geometry_hash")) for row in records})
    if candidate_ids != semantic_ids:
        errors.append("attribute plan_key coverage or order differs from semantic manifest")
    if len(task_ids) != len(set(task_ids)) or any(not value for value in task_ids):
        errors.append("empty or duplicate task ids")
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("duplicate scene/plan identities")
    if len(records) != expected_candidate_count:
        errors.append("frozen candidate_count mismatch")
    if unique_geometries != expected_unique_geometry_count:
        errors.append("frozen unique_geometry_count mismatch")
    if int(summary.get("task_count", -1)) != expected_candidate_count:
        errors.append("summary task_count mismatch")
    if int(summary.get("candidate_count", -1)) != expected_candidate_count:
        errors.append("summary candidate_count mismatch")
    if int(summary.get("unique_geometry_count", -1)) != expected_unique_geometry_count:
        errors.append("summary unique_geometry_count mismatch")
    if int(summary.get("candidate_deletion_count", -1)) != 0:
        errors.append("summary candidate_deletion_count mismatch")
    if (
        int(semantic_summary.get("candidate_count", -1)) != expected_candidate_count
        or int(semantic_summary.get("unique_geometry_count", -1))
        != expected_unique_geometry_count
        or int(semantic_summary.get("candidate_deletion_count", -1)) != 0
    ):
        errors.append("semantic summary frozen counts mismatch")
    if int(summary.get("view_input_count", -1)) != view_count:
        errors.append("summary view count mismatch")
    if (
        summary.get("candidate_labels_hidden") is not True
        or summary.get("ground_truth_read") is not False
        or summary.get("ap_computed") is not False
    ):
        errors.append("summary contract mismatch")
    required_terminal = (
        terminal_expected_identities()
        if expected_terminal_identities is None and expected_candidate_count == 39304
        and expected_unique_geometry_count == 39250
        else set(expected_terminal_identities or ())
    )
    if terminal_hits != required_terminal:
        errors.append("terminal-safe-keep identity coverage is not exactly the frozen four")
    summary_terminal_count = summary.get(
        "terminal_safe_keep_count", 0 if not required_terminal else -1
    )
    if int(summary_terminal_count) != len(terminal_hits) or terminal_hits != required_terminal:
        errors.append("terminal-safe-keep count or identity set differs")
    result = {
        "version": "dm_sms1_attribute_extraction_manifest_audit_v3",
        "candidate_count": len(records),
        "unique_geometry_count": unique_geometries,
        "candidate_deletion_count": len(set(semantic_ids) - set(candidate_ids)),
        "plan_key_coverage_complete": candidate_ids == semantic_ids,
        "error_count": len(errors),
        "errors": errors,
        "audit_valid": not errors,
        "candidate_labels_hidden": True,
        "ground_truth_read": False,
        "ap_computed": False,
        "terminal_safe_keep_count": len(terminal_hits),
        "input_provenance": {
            "attribute_manifest_sha256": _sha256(records_path),
            "semantic_manifest_sha256": _sha256(semantic_path),
            "semantic_summary_sha256": _sha256(semantic_root / "summary.json"),
        },
    }
    (root / "audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=39304)
    parser.add_argument("--expected-unique-geometry-count", type=int, default=39250)
    args = parser.parse_args()
    result = audit(
        args.manifest_root, args.semantic_root,
        args.expected_candidate_count, args.expected_unique_geometry_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
