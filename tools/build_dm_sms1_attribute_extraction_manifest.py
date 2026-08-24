#!/usr/bin/env python3
"""Build a category-blind attribute-extraction input ledger.

The ledger is an input contract for a later multimodal model call.  It does
not call a model, expose candidate names, read GT or make a class decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ATTRIBUTE_PROMPT = (
    "你将看到同一个三维候选物体的三张互补视角，以及对应的深度、位姿和候选掩码证据。"
    "不要猜测类别，不要输出任何类别名称，也不要把它和候选类别列表联系起来。"
    "只根据可观察证据记录：外观颜色和纹理、材质、形状和结构、可能的功能线索、"
    "与周围环境的空间关系。每项都要给出观察内容、支持它的视角编号、0到1的证据把握度、"
    "以及看不清或相互矛盾的地方；无法判断时填写 unknown。"
)

RESPONSE_SCHEMA = {
    "appearance": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "material": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "shape_structure": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "function_cues": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "spatial_context": {"observation": "string", "supporting_view_ranks": ["integer"], "confidence": "number_0_to_1", "counterevidence": "string"},
    "cross_view_consistency": {"observation": "string", "confidence": "number_0_to_1", "counterevidence": "string"},
    "missing_or_unclear_evidence": ["string"],
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_id(row: dict) -> str:
    payload = {
        "scene_name": row["scene_name"],
        "plan_key": row["plan_key"],
        "geometry_hash": row["geometry_hash"],
        "frames": [view["frame_id"] for view in row["selected_views"]],
        "mask_hashes": [view["sam_mask_sha256"] for view in row["selected_views"]],
    }
    evidence_digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{row['scene_name']}::{row['plan_key']}::{evidence_digest}"


def build_attribute_row(row: dict) -> dict:
    views = list(row.get("selected_views", []))
    if not views:
        raise ValueError(f"{row.get('geometry_key')}: no selected views")
    view_inputs = []
    for view in views:
        view_inputs.append({
            "view_rank": int(view["selection_rank"]),
            "frame_id": str(view["frame_id"]),
            "frame_index": int(view["frame_index"]),
            "rgb_path": str(view["rgb_path"]),
            "depth_path": str(view["depth_path"]),
            "pose_path": str(view["pose_path"]),
            "intrinsics_path": str(view["intrinsics_path"]),
            "sam_box_prompt_xyxy": list(view["sam_box_prompt_xyxy"]),
            "sam_mask_sha256": str(view["sam_mask_sha256"]),
            "visible_ratio": float(view["visible_ratio"]),
            "visible_point_count": int(view["visible_point_count"]),
        })
    result = {
        "task_id": _task_id(row),
        "scene_name": str(row["scene_name"]),
        "plan_index": int(row["plan_index"]),
        "plan_key": str(row["plan_key"]),
        "fi1_d_v3_plan_key": str(row["plan_key"]),
        "geometry_key": str(row["geometry_key"]),
        "visual_geometry_key": str(row["visual_geometry_key"]),
        "geometry_hash": str(row["geometry_hash"]),
        "point_count": int(row["point_count"]),
        "view_inputs": view_inputs,
        "candidate_labels_hidden": True,
        "candidate_hypothesis_count": len(row.get("finite_class_hypotheses", [])),
        "attribute_prompt": ATTRIBUTE_PROMPT,
        "response_schema": RESPONSE_SCHEMA,
        "attribute_extraction_completed": False,
        "class_decision_made": False,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
    }
    return result


def run(args: argparse.Namespace) -> dict:
    args.input_root = _resolve(args.input_root)
    args.output_root = _resolve(args.output_root)
    input_summary = json.loads((args.input_root / "summary.json").read_text())
    if input_summary.get("ground_truth_read") is not False or input_summary.get("ap_computed") is not False:
        raise ValueError("input manifest is not no-GT")
    records_path = args.input_root / "semantic_arbitration_manifest.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"output root is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    built = [build_attribute_row(row) for row in rows]
    task_ids = [row["task_id"] for row in built]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate attribute task id")
    with (args.output_root / "attribute_extraction_manifest.jsonl").open("w") as handle:
        for row in built:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "version": "dm_sms1_attribute_extraction_manifest_v1",
        "scene_count": len({row["scene_name"] for row in built}),
        "candidate_count": len(built),
        "geometry_count": len(built),
        "unique_geometry_count": len({(row["scene_name"], row["geometry_hash"]) for row in built}),
        "candidate_deletion_count": 0,
        "task_count": len(built),
        "view_input_count": sum(len(row["view_inputs"]) for row in built),
        "candidate_labels_hidden": True,
        "attribute_extraction_completed": False,
        "class_decision_made": False,
        "mutation_contract": {
            "candidate_mutation": False,
            "geometry_mutation": False,
            "score_mutation": False,
            "proposal_deletion": False,
        },
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "input_provenance": {
            "input_root": str(args.input_root),
            "input_summary_sha256": _sha256(args.input_root / "summary.json"),
            "input_manifest_sha256": _sha256(records_path),
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
