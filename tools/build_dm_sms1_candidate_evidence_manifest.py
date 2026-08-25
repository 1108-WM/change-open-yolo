#!/usr/bin/env python3
"""Build the finite-candidate positive/counter-evidence input ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dm_sms1_terminal_safe_keep import (  # noqa: E402
    TERMINAL_KEEP_REASON,
    terminal_identity,
)

def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidate_row(attribute_row: dict, semantic_row: dict, class_names: list[str]) -> dict:
    if (
        attribute_row["scene_name"] != semantic_row["scene_name"]
        or attribute_row["plan_key"] != semantic_row["plan_key"]
        or attribute_row["geometry_hash"] != semantic_row["geometry_hash"]
    ):
        raise ValueError("attribute/semantic scene and geometry join mismatch")
    candidates = list(semantic_row.get("finite_class_hypotheses", []))
    terminal = bool(semantic_row.get("terminal_safe_keep", False))
    if terminal:
        if (
            candidates
            or not terminal_identity(
                int(semantic_row.get("plan_index", -1)), str(semantic_row.get("plan_key", ""))
            )
            or semantic_row.get("qwen_execution_required") is not False
            or semantic_row.get("attribute_execution_required") is not False
            or semantic_row.get("canonical_frozen_class_index") != 198
            or attribute_row.get("terminal_safe_keep") is not True
            or attribute_row.get("attribute_execution_required") is not False
            or attribute_row.get("view_inputs") != []
        ):
            raise ValueError(f"{semantic_row['geometry_key']}: terminal row has finite hypotheses")
        return {
            "task_id": str(attribute_row["task_id"]), "scene_name": str(attribute_row["scene_name"]),
            "plan_index": int(attribute_row["plan_index"]), "plan_key": str(attribute_row["plan_key"]),
            "fi1_d_v3_plan_key": str(attribute_row["plan_key"]), "geometry_key": str(attribute_row["geometry_key"]),
            "visual_geometry_key": str(attribute_row["visual_geometry_key"]), "geometry_hash": str(attribute_row["geometry_hash"]),
            "geometry_locator_read_only": dict(semantic_row["geometry_locator_read_only"]),
            "candidate_source": str(semantic_row["candidate_source"]), "frozen_class_index": int(semantic_row["frozen_class_index"]),
            "challenger_score": float(semantic_row["challenger_score"]), "append_only": bool(semantic_row["append_only"]),
            "canonical_frozen_class_index": int(semantic_row["canonical_frozen_class_index"]),
            "canonical_frozen_score": float(semantic_row["canonical_frozen_score"]),
            "attribute_task_id": str(attribute_row["task_id"]), "candidate_hypotheses": [],
            "candidate_order_ab": [], "candidate_order_ba": [],
            "evidence_prompt_ab": None, "evidence_prompt_ba": None,
            "attribute_evidence_required": False, "swap_order_required": False,
            "qwen_execution_required": False, "terminal_safe_keep": True,
            "terminal_keep_reason": TERMINAL_KEEP_REASON,
            "decision_rule": {
                "only_alternative_supported_in_both_orders_may_be_considered": True,
                "otherwise_keep_frozen_control_class": True,
                "all_geometry_nodes_decided_simultaneously": True,
                "no_proposal_deletion": True, "no_score_change": True,
            },
            "class_decision_made": False, "selected_class_index": None,
            "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False,
            "ground_truth_usage": "none", "ground_truth_read": False, "ap_computed": False,
        }
    if not candidates or len(candidates) > 2:
        raise ValueError(f"{semantic_row['geometry_key']}: finite candidate count must be 1 or 2")
    hypotheses = []
    for candidate in candidates:
        class_index = int(candidate["class_index"])
        if not 0 <= class_index < len(class_names):
            raise ValueError("candidate class index outside configured class space")
        hypotheses.append({
            "class_index": class_index,
            "class_name": str(class_names[class_index]),
            "sources": list(candidate.get("sources", [])),
        })
    names_ab = [item["class_name"] for item in hypotheses]
    names_ba = list(reversed(names_ab))
    base = (
        "你已经得到同一个三维物体的无类别属性证据。现在只比较下面给出的有限候选，"
        "不要提出候选列表之外的新类别。请分别记录每个候选的支持证据、反对证据、"
        "证据来自哪些视角，以及证据把握度。不要修改几何、候选成员或排序分数。候选顺序为："
    )
    return {
        "task_id": str(attribute_row["task_id"]),
        "scene_name": str(attribute_row["scene_name"]),
        "plan_index": int(attribute_row["plan_index"]),
        "plan_key": str(attribute_row["plan_key"]),
        "fi1_d_v3_plan_key": str(attribute_row["plan_key"]),
        "geometry_key": str(attribute_row["geometry_key"]),
        "visual_geometry_key": str(attribute_row["visual_geometry_key"]),
        "geometry_hash": str(attribute_row["geometry_hash"]),
        "geometry_locator_read_only": dict(semantic_row["geometry_locator_read_only"]),
        "candidate_source": str(semantic_row["candidate_source"]),
        "frozen_class_index": int(semantic_row["frozen_class_index"]),
        "challenger_score": float(semantic_row["challenger_score"]),
        "append_only": bool(semantic_row["append_only"]),
        "canonical_frozen_class_index": int(semantic_row["canonical_frozen_class_index"]),
        "canonical_frozen_score": float(semantic_row["canonical_frozen_score"]),
        "attribute_task_id": str(attribute_row["task_id"]),
        "candidate_hypotheses": hypotheses,
        "candidate_order_ab": names_ab,
        "candidate_order_ba": names_ba,
        "evidence_prompt_ab": base + "、".join(names_ab),
        "evidence_prompt_ba": base + "、".join(names_ba),
        "attribute_evidence_required": True,
        "swap_order_required": len(hypotheses) == 2,
        "qwen_execution_required": True,
        "terminal_safe_keep": False,
        "terminal_keep_reason": None,
        "decision_rule": {
            "only_alternative_supported_in_both_orders_may_be_considered": True,
            "otherwise_keep_frozen_control_class": True,
            "all_geometry_nodes_decided_simultaneously": True,
            "no_proposal_deletion": True,
            "no_score_change": True,
        },
        "class_decision_made": False,
        "selected_class_index": None,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
    }


def run(args: argparse.Namespace) -> dict:
    args.attribute_root = _resolve(args.attribute_root)
    args.semantic_root = _resolve(args.semantic_root)
    args.output_root = _resolve(args.output_root)
    args.config_path = _resolve(args.config_path)
    config = yaml.safe_load(args.config_path.read_text())
    class_names = [str(value) for value in config["network2d"]["text_prompts"]]
    attribute_path = args.attribute_root / "attribute_extraction_manifest.jsonl"
    semantic_path = args.semantic_root / "semantic_arbitration_manifest.jsonl"
    attribute_rows = [json.loads(line) for line in attribute_path.read_text().splitlines() if line.strip()]
    semantic_list = [
        json.loads(line) for line in semantic_path.read_text().splitlines() if line.strip()
    ]
    attribute_ids = [
        (str(row.get("scene_name", "")), str(row.get("plan_key", "")))
        for row in attribute_rows
    ]
    semantic_ids = [
        (str(row.get("scene_name", "")), str(row.get("plan_key", "")))
        for row in semantic_list
    ]
    if any(not scene or not digest for scene, digest in attribute_ids):
        raise ValueError("attribute manifest contains an empty scene/plan identity")
    if any(not scene or not digest for scene, digest in semantic_ids):
        raise ValueError("semantic manifest contains an empty scene/plan identity")
    if len(attribute_ids) != len(set(attribute_ids)):
        raise ValueError("attribute manifest contains duplicate scene/plan identities")
    if len(semantic_ids) != len(set(semantic_ids)):
        raise ValueError("semantic manifest contains duplicate scene/plan identities")
    if set(attribute_ids) != set(semantic_ids):
        missing = sorted(set(attribute_ids) - set(semantic_ids))[:3]
        extra = sorted(set(semantic_ids) - set(attribute_ids))[:3]
        raise ValueError(
            f"attribute/semantic identity coverage mismatch: missing={missing}, extra={extra}"
        )
    semantic_rows = dict(zip(semantic_ids, semantic_list))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"output root is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    built = []
    for row in attribute_rows:
        if row.get("attribute_extraction_completed") is not False:
            raise ValueError("attribute input unexpectedly contains completed evidence")
        identity = (str(row["scene_name"]), str(row["plan_key"]))
        built.append(build_candidate_row(row, semantic_rows[identity], class_names))
    with (args.output_root / "candidate_evidence_manifest.jsonl").open("w") as handle:
        for row in built:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "version": "dm_sms1_candidate_evidence_manifest_v1",
        "scene_count": len({row["scene_name"] for row in built}),
        "candidate_count": len(built),
        "geometry_count": len(built),
        "unique_geometry_count": len({(row["scene_name"], row["geometry_hash"]) for row in built}),
        "candidate_deletion_count": 0,
        "candidate_pair_count": sum(len(row["candidate_hypotheses"]) == 2 for row in built),
        "single_candidate_count": sum(len(row["candidate_hypotheses"]) == 1 for row in built),
        "terminal_safe_keep_count": sum(bool(row.get("terminal_safe_keep")) for row in built),
        "class_decision_made": False,
        "selected_class_count": 0,
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
            "attribute_manifest_sha256": _sha256(attribute_path),
            "semantic_manifest_sha256": _sha256(semantic_path),
            "config_sha256": _sha256(args.config_path),
        },
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribute-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
