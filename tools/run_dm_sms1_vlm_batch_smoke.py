#!/usr/bin/env python3
"""Run a deterministic small-batch no-GT Qwen semantic-arbitration smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.apply_dm_sms1_semantic_arbitration import decide_row
from tools.run_dm_sms1_single_scene_vlm_smoke import (
    _attribute_prompt,
    _candidate_prompt,
    _json_object,
    _read_rows,
    _target_images,
    _validate_candidate_output,
)


ATTRIBUTE_KEYS = {
    "appearance", "material", "shape_structure", "function_cues",
    "spatial_context", "cross_view_consistency", "missing_or_unclear_evidence",
}
ATTRIBUTE_ITEM_KEYS = (
    "appearance", "material", "shape_structure", "function_cues", "spatial_context",
)
ATTRIBUTE_PART_TERMS = {"cushion", "seat"}
AMBIGUOUS_ATTRIBUTE_TERMS = {"light", "mat"}
GENERIC_OUTSIDE_CLASS_TERMS = {
    "bar", "board", "case", "container", "furniture", "light", "machine",
    "mat", "object", "paper", "rack", "stand", "structure",
}
EXPECTED_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
PROMPT_CONTRACT_VERSION = "dm_sms1_attribute_candidate_prompt_v5_strict_object"


class JsonSyntaxFailure(ValueError):
    pass


class InferenceValidationFailure(ValueError):
    """Retain both model attempts when fixed structural repair also fails."""

    def __init__(self, first_raw: str, repaired_raw: str, repair_kind: str, final_error: Exception):
        super().__init__(str(final_error))
        self.first_raw = first_raw
        self.repaired_raw = repaired_raw
        self.repair_kind = repair_kind
        self.final_error = final_error


def _unique_rows_by_task(rows: list[dict], label: str) -> dict[str, dict]:
    task_ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in task_ids):
        raise ValueError(f"{label} contains an empty task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{label} contains duplicate task_id")
    return dict(zip(task_ids, rows))


def _verify_model_revision(model_dir: Path) -> None:
    metadata_root = model_dir / ".cache" / "huggingface" / "download"
    metadata_files = sorted(metadata_root.glob("*.metadata"))
    if not metadata_files:
        raise FileNotFoundError("local model revision metadata is missing")
    revisions = {
        path.read_text().splitlines()[0].strip()
        for path in metadata_files if path.read_text().splitlines()
    }
    if revisions != {EXPECTED_MODEL_REVISION}:
        raise ValueError(f"local model revision mismatch: {sorted(revisions)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_batch(
    candidate_rows: list[dict], scene_count: int, per_scene: int,
    per_scene_offset: int = 0,
) -> list[dict]:
    if scene_count <= 0 or per_scene <= 0:
        raise ValueError("scene_count and per_scene must be positive")
    if per_scene_offset < 0:
        raise ValueError("per_scene_offset must be non-negative")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        if row.get("terminal_safe_keep") is True:
            if row.get("qwen_execution_required") is not False or row.get("candidate_hypotheses") != []:
                raise ValueError("terminal-safe-keep row violates Qwen exclusion contract")
            continue
        if len(row.get("candidate_hypotheses", [])) == 2:
            if row.get("qwen_execution_required", True) is not True:
                raise ValueError("ordinary pair task unexpectedly disables Qwen")
            grouped[str(row["scene_name"])].append(row)
    selected = []
    for scene in sorted(grouped)[:scene_count]:
        rows = sorted(grouped[scene], key=lambda row: (row["geometry_hash"], row["task_id"]))
        selected.extend(rows[per_scene_offset:per_scene_offset + per_scene])
    return selected


def _validate_qwen_selection(selected: list[dict]) -> None:
    for row in selected:
        if row.get("terminal_safe_keep") is True:
            raise ValueError("Qwen selection contains a terminal-safe-keep task")
        if row.get("qwen_execution_required", True) is not True:
            raise ValueError("Qwen selection contains a task with execution disabled")
        if len(row.get("candidate_hypotheses", [])) != 2:
            raise ValueError("Qwen selection contains a non-pair task")


def _string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _string_values(nested)


def _term_present(value: object, name: str) -> bool:
    strings = "\n".join(_string_values(value)).lower()
    return bool(re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", strings))


TARGET_CUE_PATTERNS = (
    r"\bthis object\b", r"\bthe object\b", r"\bit is\b", r"\bpossibly\b",
    r"\bperhaps\b", r"\bcould be\b", r"\blikely\b", r"\bsuggest(?:s|ing)?\b",
    r"\bused for\b", r"\bfunction\b",
)
RELATION_PREFIX_PATTERNS = (
    r"\bpart of\s+(?:a|an|the)?(?:\s+\w+){0,3}\s*$", r"\bnear\s+$", r"\bnext to\s+$", r"\bin front of\s+$",
    r"\bbehind\s+$", r"\bon\s+$", r"\babove\s+$", r"\bbelow\s+$",
    r"\baround\s+$", r"\bsurrounded by\s+$", r"\bwith\s+(?:a|an|the)?\s*$",
    r"\balongside\s+$", r"\bbeside\s+$", r"\bnearby\s+$",
    r"\b(?:on|above|below|near|beside|behind|next to|in front of|resting on)\s+(?:a|an|the)?(?:\s+\w+){0,3}\s*$",
)


def _term_is_target_guess(value: object, name: str) -> bool:
    """Detect target classification while ignoring local-part/environment relations."""
    if not isinstance(value, str):
        return any(_term_is_target_guess(item, name) for item in _string_values(value))
    lowered = value.lower()
    term = name.lower()
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
    for match in pattern.finditer(lowered):
        clause_start = max(lowered.rfind(mark, 0, match.start()) for mark in (".", ";", ":", "\n")) + 1
        clause = lowered[clause_start:]
        before = clause[: match.start() - clause_start]
        if any(re.search(pattern_text, before) for pattern_text in RELATION_PREFIX_PATTERNS):
            continue
        if any(re.search(pattern_text, before) for pattern_text in TARGET_CUE_PATTERNS):
            return True
        # A noun at the start of a descriptive clause, such as "whiteboard
        # with writing" or "chair with wheels", is a target guess.  A noun
        # following a relation marker was handled above as context.
        if not before.strip() or re.search(r"\b(?:a|an|the)\s*$", before):
            return True
        if re.search(r"\b(?:chair|table|desk|object|item|target)\s*$", before):
            return True
    return False


def _contains_cjk(value: object) -> bool:
    return any(re.search(r"[\u3400-\u9fff]", text) for text in _string_values(value))


def normalize_attribute_view_ranks(
    value: dict, allowed_view_ranks: set[int] | None = None,
) -> tuple[dict, bool]:
    """Return a copied attribute object with deterministic 1-based repair.

    The old six-image prompt caused the model to use 1,2,3 for the three
    logical views.  A row is repaired only when every referenced rank is in
    1..3 and rank 0 is absent.  Any other rank remains a structural error.
    """
    normalized = copy.deepcopy(value)
    allowed = {0, 1, 2} if allowed_view_ranks is None else set(allowed_view_ranks)
    if not allowed or allowed != set(range(len(allowed))):
        raise ValueError("allowed view ranks must be contiguous from zero")
    observed = []
    for field in ATTRIBUTE_ITEM_KEYS:
        item = normalized.get(field)
        if not isinstance(item, dict):
            continue
        ranks = item.get("supporting_view_ranks")
        if not isinstance(ranks, list):
            continue
        converted = []
        for rank in ranks:
            if isinstance(rank, bool):
                raise ValueError(f"attribute field {field} has a non-integer view rank")
            try:
                converted_rank = int(rank)
            except (TypeError, ValueError) as error:
                raise ValueError(f"attribute field {field} has a non-integer view rank") from error
            if isinstance(rank, float) and not rank.is_integer():
                raise ValueError(f"attribute field {field} has a non-integer view rank")
            if isinstance(rank, str) and not re.fullmatch(r"[+-]?\d+", rank.strip()):
                raise ValueError(f"attribute field {field} has a non-integer view rank")
            converted.append(converted_rank)
        item["supporting_view_ranks"] = converted
        observed.extend(converted)
    # Shift only when the observed ranks are impossible under zero-based
    # indexing but become valid after subtracting one.  A lone rank 1 is
    # ambiguous and must remain rank 1 rather than being silently changed.
    observed_set = set(observed)
    shifted = bool(observed) and not observed_set.issubset(allowed) and {
        rank - 1 for rank in observed_set
    }.issubset(allowed)
    if shifted:
        for field in ATTRIBUTE_ITEM_KEYS:
            item = normalized.get(field)
            if isinstance(item, dict) and isinstance(item.get("supporting_view_ranks"), list):
                item["supporting_view_ranks"] = [rank - 1 for rank in item["supporting_view_ranks"]]
    return normalized, shifted


def validate_attribute_structure(
    value: dict, allowed_view_ranks: set[int] | None = None,
) -> tuple[dict, bool]:
    if not isinstance(value, dict) or not ATTRIBUTE_KEYS.issubset(value):
        raise ValueError("attribute output misses required fields")
    allowed = {0, 1, 2} if allowed_view_ranks is None else set(allowed_view_ranks)
    normalized, ranks_normalized = normalize_attribute_view_ranks(value, allowed)
    if _contains_cjk(normalized):
        raise ValueError("attribute evidence text must use English for deterministic class-word auditing")
    for field in ATTRIBUTE_ITEM_KEYS:
        item = normalized[field]
        if not isinstance(item, dict):
            raise ValueError(f"attribute field {field} is not an object")
        if not isinstance(item.get("observation"), str) or not item["observation"].strip():
            raise ValueError(f"attribute field {field} has an empty observation")
        if not isinstance(item.get("counterevidence"), str):
            raise ValueError(f"attribute field {field} misses counterevidence")
        confidence = item.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0):
            raise ValueError(f"attribute field {field} has invalid confidence")
        ranks = item.get("supporting_view_ranks")
        if not isinstance(ranks, list) or any(rank not in allowed for rank in ranks):
            raise ValueError(f"attribute field {field} references an invalid view rank")
    consistency = normalized["cross_view_consistency"]
    if not isinstance(consistency, dict):
        raise ValueError("cross_view_consistency is not an object")
    if not isinstance(consistency.get("observation"), str) or not consistency["observation"].strip():
        raise ValueError("cross_view_consistency has an empty observation")
    if not isinstance(consistency.get("counterevidence"), str):
        raise ValueError("cross_view_consistency misses counterevidence")
    confidence = consistency.get("confidence")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0):
        raise ValueError("cross_view_consistency has invalid confidence")
    unclear = normalized["missing_or_unclear_evidence"]
    if not isinstance(unclear, list) or any(not isinstance(item, str) for item in unclear):
        raise ValueError("missing_or_unclear_evidence is not a string array")
    return normalized, ranks_normalized


def attribute_term_findings(value: dict, candidate_names: list[str]) -> tuple[list[str], list[str]]:
    """Separate harmless contextual/part words from target-class guesses."""
    warnings = []
    leaked = []
    strict_fields = (
        value.get("appearance"), value.get("material"), value.get("function_cues"),
        value.get("cross_view_consistency"), value.get("missing_or_unclear_evidence"),
    )
    shape = value.get("shape_structure")
    for name in candidate_names:
        lowered = name.lower()
        if lowered in ATTRIBUTE_PART_TERMS:
            if any(_term_present(field, name) for field in (*strict_fields, shape)):
                warnings.append(name)
            continue
        if lowered in AMBIGUOUS_ATTRIBUTE_TERMS:
            if any(_term_present(field, name) for field in strict_fields) or _term_present(shape, name):
                warnings.append(name)
            continue
        if any(_term_is_target_guess(field, name) for field in strict_fields):
            leaked.append(name)
        elif _term_is_target_guess(shape, name):
            leaked.append(name)
        # spatial_context is deliberately excluded: "on a desk", "above a
        # counter" and similar phrases describe the surroundings, not the
        # red-box target.
    return sorted(set(warnings)), sorted(set(leaked))


def validate_attribute_output(value: dict, candidate_names: list[str]) -> list[str]:
    warnings, leaked = attribute_term_findings(value, candidate_names)
    if leaked:
        raise ValueError(f"attribute output leaked candidate names: {leaked}")
    return warnings


def outside_candidate_class_terms(
    value: dict, candidate_names: list[str], all_class_names: set[str],
) -> list[str]:
    """Conservatively find explicit target-category guesses outside the finite set."""
    candidates = {name.lower() for name in candidate_names}
    strict_fields = (
        value.get("appearance"), value.get("material"), value.get("function_cues"),
        value.get("cross_view_consistency"), value.get("missing_or_unclear_evidence"),
    )
    return sorted({
        name for name in all_class_names
        if name not in candidates
        and name not in GENERIC_OUTSIDE_CLASS_TERMS
        and name not in ATTRIBUTE_PART_TERMS
        and any(_term_is_target_guess(field, name) for field in strict_fields)
    })


def candidate_set_insufficient_or_unclear(output_ab: dict, output_ba: dict) -> bool:
    """Flag cases where no finite candidate has stable positive support."""
    maps = []
    for output in (output_ab, output_ba):
        maps.append({int(item["class_index"]): item for item in output["candidate_results"]})
    common = set(maps[0]).intersection(maps[1])
    return not any(
        maps[0][index]["supported"] and maps[1][index]["supported"]
        and not maps[0][index]["strong_counterevidence"]
        and not maps[1][index]["strong_counterevidence"]
        for index in common
    )


def validate_completed_model_record(
    record: dict, candidate_row: dict, attribute_row: dict,
    all_class_names: set[str],
) -> dict:
    """Strictly replay a completed valid record from its saved raw evidence."""
    task_id = str(candidate_row["task_id"])
    if record.get("valid") is not True:
        raise ValueError(f"{task_id}: completed model record is not valid")
    if str(record.get("task_id", "")) != task_id:
        raise ValueError(f"{task_id}: completed model record task join mismatch")
    if (record.get("scene_name") != candidate_row.get("scene_name")
            or record.get("plan_key") != candidate_row.get("plan_key")
            or record.get("geometry_hash") != candidate_row.get("geometry_hash")):
        raise ValueError(f"{task_id}: completed model record identity mismatch")
    if (attribute_row.get("scene_name") != candidate_row.get("scene_name")
            or attribute_row.get("plan_key") != candidate_row.get("plan_key")
            or attribute_row.get("geometry_hash") != candidate_row.get("geometry_hash")):
        raise ValueError(f"{task_id}: attribute/candidate identity mismatch")
    reparsed = {}
    for raw_key, parsed_key in (
        ("attribute_raw_output", "attribute_output"),
        ("order_ab_raw_output", "order_ab"),
        ("order_ba_raw_output", "order_ba"),
    ):
        raw = record.get(raw_key)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{task_id}: valid evidence misses {raw_key}")
        try:
            reparsed[parsed_key] = _json_object(raw)
        except Exception as error:
            raise ValueError(f"{task_id}: {raw_key} is not exactly one complete JSON object") from error
    allowed_view_ranks = {
        int(view["view_rank"]) for view in attribute_row.get("view_inputs", [])
    }
    attribute_output, _normalized = validate_attribute_structure(
        reparsed["attribute_output"], allowed_view_ranks,
    )
    if attribute_output != record.get("attribute_output"):
        raise ValueError(f"{task_id}: raw attribute evidence does not reproduce the saved output")
    names = [item["class_name"] for item in candidate_row["candidate_hypotheses"]]
    validate_attribute_output(attribute_output, names)
    outside_terms = outside_candidate_class_terms(attribute_output, names, all_class_names)
    if outside_terms:
        raise ValueError(f"{task_id}: attribute output points outside finite candidates: {outside_terms}")
    by_name = {item["class_name"]: int(item["class_index"]) for item in candidate_row["candidate_hypotheses"]}
    expected_ab = [by_name[name] for name in candidate_row["candidate_order_ab"]]
    expected_ba = [by_name[name] for name in candidate_row["candidate_order_ba"]]
    for parsed_key, expected_order in (("order_ab", expected_ab), ("order_ba", expected_ba)):
        if reparsed[parsed_key] != record.get(parsed_key):
            raise ValueError(f"{task_id}: raw evidence does not reproduce {parsed_key}")
        _validate_candidate_output(record[parsed_key], expected_order)
    insufficient = candidate_set_insufficient_or_unclear(record["order_ab"], record["order_ba"])
    if bool(record.get("candidate_set_insufficient_or_unclear")) != insufficient:
        raise ValueError(f"{task_id}: candidate-set sufficiency flag mismatch")
    recomputed_decision = decide_row(candidate_row, record)
    if record.get("decision") != recomputed_decision:
        raise ValueError(f"{task_id}: stored decision does not match strict replay")
    return recomputed_decision


def _selection_records(selected: list[dict]) -> list[dict]:
    return [{
        "task_id": row["task_id"], "scene_name": row["scene_name"],
        "plan_key": row["plan_key"],
        "geometry_hash": row["geometry_hash"],
        "candidate_indices": [item["class_index"] for item in row["candidate_hypotheses"]],
        "candidate_names": [item["class_name"] for item in row["candidate_hypotheses"]],
    } for row in selected]


def summarize_batch_records(records: list[dict], selected: list[dict]) -> dict:
    """Recompute totals from the complete append-only ledger."""
    counts = Counter()
    for record in records:
        if record.get("valid"):
            counts["valid"] += 1
            counts["class_changed"] += int(record.get("decision", {}).get("class_changed", False))
            counts["kept_frozen_control"] += int(not record.get("decision", {}).get("class_changed", False))
        else:
            counts["invalid"] += 1
        counts["attribute_term_warning"] += len(record.get("attribute_term_warnings", []))
        counts["attribute_view_ranks_normalized"] += int(record.get("attribute_view_ranks_normalized", False))
        counts["candidate_set_insufficient_or_unclear"] += int(
            record.get("candidate_set_insufficient_or_unclear", False)
        )
        counts["json_retry"] += int(record.get("json_retried", False))
        kinds = record.get("repair_kinds", [])
        counts["json_syntax_repair"] += list(kinds).count("json_syntax")
        counts["output_structure_repair"] += list(kinds).count("output_structure")
    return {
        "version": "dm_sms1_vlm_batch_smoke_v5",
        "model_id": "Qwen2.5-VL-7B-Instruct",
        "model_revision": EXPECTED_MODEL_REVISION,
        "selected_scene_count": len({row["scene_name"] for row in selected}),
        "selected_candidate_count": len(selected),
        "selected_geometry_count": len(selected),
        "selected_unique_geometry_count": len({
            (row.get("scene_name"), row.get("geometry_hash")) for row in selected
            if row.get("scene_name") and row.get("geometry_hash")
        }),
        "processed_record_count": len(records),
        "valid_count": counts["valid"],
        "invalid_count": counts["invalid"],
        "class_change_count": counts["class_changed"],
        "kept_frozen_control_count": counts["kept_frozen_control"],
        "attribute_term_warning_count": counts["attribute_term_warning"],
        "attribute_view_ranks_normalized_count": counts["attribute_view_ranks_normalized"],
        "candidate_set_insufficient_or_unclear_count": counts["candidate_set_insufficient_or_unclear"],
        "json_retry_count": counts["json_retry"],
        "json_syntax_repair_count": counts["json_syntax_repair"],
        "output_structure_repair_count": counts["output_structure_repair"],
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "json_retry_contract": "one deterministic repair call after non-object/malformed JSON or invalid structure/view ranks; semantic candidate leakage is not retried",
        "resume_contract": "append-only exact-prefix task ledger with frozen input/config/model contract hashes",
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "ground_truth_read": False,
        "ap_computed": False,
        "scope": "small-batch interface and stability smoke; not accuracy evidence",
    }


def run(args: argparse.Namespace) -> dict:
    attribute_rows = _unique_rows_by_task(_read_rows(args.attribute_manifest), "attribute manifest")
    candidate_rows = _read_rows(args.candidate_manifest)
    _unique_rows_by_task(candidate_rows, "candidate manifest")
    config = yaml.safe_load(args.config_path.read_text())
    all_class_names = {
        str(name).strip().lower()
        for name in config["network2d"]["text_prompts"] if str(name).strip()
    }
    selected = select_batch(
        candidate_rows, args.scene_count, args.per_scene, args.per_scene_offset,
    )
    if args.task_id and args.failed_from_batch_output:
        raise ValueError("--task-id and --failed-from-batch-output cannot be combined")
    if args.failed_from_batch_output:
        failed_ids = [
            str(row["task_id"])
            for row in _read_rows(args.failed_from_batch_output)
            if row.get("valid") is False
        ]
        by_task = {row["task_id"]: row for row in candidate_rows}
        if not set(failed_ids).issubset(by_task):
            raise ValueError("failed batch output contains a task absent from candidate manifest")
        selected = [by_task[task_id] for task_id in failed_ids]
    elif args.task_id:
        requested = set(args.task_id)
        by_task = {row["task_id"]: row for row in candidate_rows}
        if not requested.issubset(by_task):
            raise ValueError("requested task_id is absent from candidate manifest")
        selected = [by_task[task_id] for task_id in args.task_id]
    _validate_qwen_selection(selected)
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    selected = selected[args.offset: args.offset + args.limit if args.limit is not None else None]
    if not selected:
        raise ValueError("small-batch selection is empty")
    selected_ids_in_order = [str(row["task_id"]) for row in selected]
    if len(selected_ids_in_order) != len(set(selected_ids_in_order)):
        raise ValueError("small-batch selection contains duplicate task_id")
    missing_attribute_ids = set(selected_ids_in_order) - set(attribute_rows)
    if missing_attribute_ids:
        raise ValueError(f"attribute manifest misses selected tasks: {sorted(missing_attribute_ids)[:3]}")
    expected_selection = _selection_records(selected)
    selection_path = args.output_root / "selection.jsonl"
    run_contract = {
        "version": "dm_sms1_vlm_run_contract_v1",
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "attribute_manifest": str(args.attribute_manifest),
        "attribute_manifest_sha256": _sha256(args.attribute_manifest),
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": _sha256(args.candidate_manifest),
        "config_path": str(args.config_path),
        "config_sha256": _sha256(args.config_path),
        "model_dir": str(args.model_dir),
        "model_revision": EXPECTED_MODEL_REVISION,
        "scene_count": int(args.scene_count),
        "per_scene": int(args.per_scene),
        "per_scene_offset": int(args.per_scene_offset),
        "offset": int(args.offset),
        "limit": args.limit,
        "task_ids": selected_ids_in_order,
        "attribute_max_tokens": int(args.attribute_max_tokens),
        "candidate_max_tokens": int(args.candidate_max_tokens),
        "failed_from_batch_output": str(args.failed_from_batch_output) if args.failed_from_batch_output else None,
    }
    contract_path = args.output_root / "run_contract.json"
    if args.output_root.exists():
        if not args.resume:
            raise FileExistsError(f"output root exists; pass --resume to continue: {args.output_root}")
        if not selection_path.exists():
            raise FileNotFoundError("resume output root is missing selection.jsonl")
        if not contract_path.exists():
            raise FileNotFoundError("resume output root is missing frozen run_contract.json")
        existing_contract = json.loads(contract_path.read_text())
        if existing_contract != run_contract:
            raise ValueError("resume run contract does not exactly match the frozen inputs/config/model")
        existing_selection = [json.loads(line) for line in selection_path.read_text().splitlines() if line.strip()]
        if existing_selection != expected_selection:
            raise ValueError("resume selection does not exactly match the requested task slice")
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)
        selection_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in expected_selection
        ))
        contract_path.write_text(
            json.dumps(run_contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    output_path = args.output_root / "batch_outputs.jsonl"
    decision_path = args.output_root / "valid_decisions.jsonl"
    existing_records = (
        [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
        if output_path.exists() else []
    )
    existing_ids = [str(row.get("task_id", "")) for row in existing_records]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("resume batch output contains duplicate task_id")
    if existing_ids != selected_ids_in_order[:len(existing_ids)]:
        raise ValueError("resume batch output is not the exact append-only selection prefix")
    for record, candidate_row in zip(existing_records, selected):
        task_id = str(record["task_id"])
        if (record.get("scene_name") != candidate_row.get("scene_name")
                or record.get("plan_key") != candidate_row.get("plan_key")
                or record.get("geometry_hash") != candidate_row.get("geometry_hash")):
            raise ValueError(f"resume batch identity mismatch: {task_id}")
        if record.get("ground_truth_read") is not False or record.get("ap_computed") is not False:
            raise ValueError(f"resume batch violates no-GT/no-AP contract: {task_id}")
        if not isinstance(record.get("valid"), bool):
            raise ValueError(f"resume batch has non-boolean valid flag: {task_id}")
        if record["valid"]:
            validate_completed_model_record(
                record, candidate_row, attribute_rows[task_id], all_class_names,
            )
        elif "decision" in record:
            raise ValueError(f"resume invalid record unexpectedly contains a decision: {task_id}")
    existing_decisions = (
        [json.loads(line) for line in decision_path.read_text().splitlines() if line.strip()]
        if decision_path.exists() else []
    )
    expected_decisions = [row["decision"] for row in existing_records if row.get("valid") is True]
    if existing_decisions != expected_decisions:
        raise ValueError("resume valid_decisions ledger does not exactly match valid batch outputs")
    completed_ids = set(existing_ids)
    pending = [row for row in selected if row["task_id"] not in completed_ids]
    if not pending:
        summary = summarize_batch_records(existing_records, selected)
        (args.output_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return summary

    _verify_model_revision(args.model_dir)

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("batch VLM smoke requires CUDA; no CPU fallback is allowed")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=100352, max_pixels=200704, local_files_only=True,
    )

    def infer_raw(images, prompt: str, max_tokens: int) -> str:
        messages = [{"role": "user", "content": [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return raw

    def infer_json(
        images, prompt: str, max_tokens: int, validator=None,
        repair_view_ranks: set[int] | None = None,
    ) -> tuple[str, dict, bool, str | None, dict]:
        def decode_and_validate(raw_text: str) -> tuple[dict, dict]:
            try:
                parsed = _json_object(raw_text)
            except Exception as error:
                raise JsonSyntaxFailure("model output is not a JSON object") from error
            if validator is None:
                return parsed, {}
            return validator(parsed)

        raw = infer_raw(images, prompt, max_tokens)
        try:
            value, metadata = decode_and_validate(raw)
            return raw, value, False, None, metadata
        except Exception as first_error:
            repair_kind = "json_syntax" if isinstance(first_error, JsonSyntaxFailure) else "output_structure"
            rank_instruction = ""
            if repair_view_ranks is not None:
                rank_instruction = (
                    " View ranks are limited to "
                    + ", ".join(str(rank) for rank in sorted(repair_view_ranks)) + "."
                )
            repaired_raw = infer_raw(
                images,
                prompt + "\nThe previous answer violated the required JSON structure or view-rank contract. "
                "Rewrite it once with every required field, exact field types, short English evidence text, "
                + rank_instruction
                + " Return exactly one complete JSON object, not an array, with no explanation or Markdown.",
                max_tokens,
            )
            try:
                value, metadata = decode_and_validate(repaired_raw)
                return repaired_raw, value, True, repair_kind, metadata
            except Exception as repaired_error:
                raise InferenceValidationFailure(
                    raw, repaired_raw, repair_kind, repaired_error,
                ) from first_error

    counts = Counter()
    with output_path.open("a") as output_handle, decision_path.open("a") as decision_handle:
        for index, candidate_row in enumerate(pending, 1):
            attribute_row = attribute_rows[candidate_row["task_id"]]
            record = {
                "task_id": candidate_row["task_id"],
                "scene_name": candidate_row["scene_name"],
                "plan_key": candidate_row["plan_key"],
                "geometry_hash": candidate_row["geometry_hash"],
                "valid": False,
                "error": None,
                "candidate_set_insufficient_or_unclear": False,
                "json_retried": False,
                "repair_kinds": [],
                "attribute_term_warnings": [],
                "ground_truth_read": False,
                "ap_computed": False,
            }
            active_stage = "attribute"
            try:
                images = _target_images(attribute_row)
                allowed_view_ranks = {
                    int(view["view_rank"]) for view in attribute_row.get("view_inputs", [])
                }
                def prepare_attribute(value: dict) -> tuple[dict, dict]:
                    checked, ranks_normalized = validate_attribute_structure(
                        value, allowed_view_ranks,
                    )
                    return checked, {"attribute_view_ranks_normalized": ranks_normalized}

                raw_attribute, attribute_output, attribute_retried, attribute_repair_kind, attribute_metadata = infer_json(
                    images, _attribute_prompt(attribute_row), args.attribute_max_tokens,
                    validator=prepare_attribute, repair_view_ranks=allowed_view_ranks,
                )
                record["attribute_raw_output"] = raw_attribute
                record["attribute_output"] = attribute_output
                record["json_retried"] = bool(attribute_retried)
                if attribute_repair_kind is not None:
                    record["repair_kinds"].append(attribute_repair_kind)
                names = [item["class_name"] for item in candidate_row["candidate_hypotheses"]]
                warnings = validate_attribute_output(attribute_output, names)
                record["attribute_term_warnings"] = warnings
                record.update(attribute_metadata)
                counts["attribute_term_warning"] += len(warnings)
                counts["attribute_view_ranks_normalized"] += int(
                    attribute_metadata.get("attribute_view_ranks_normalized", False)
                )
                outside_terms = outside_candidate_class_terms(attribute_output, names, all_class_names)
                record["outside_candidate_class_terms"] = outside_terms
                if outside_terms:
                    record["candidate_set_insufficient_or_unclear"] = True
                    counts["candidate_set_insufficient_or_unclear"] += 1
                    raise ValueError(
                        f"attribute output points outside the finite candidate set: {outside_terms}"
                    )
                expected_ab = [
                    next(item["class_index"] for item in candidate_row["candidate_hypotheses"] if item["class_name"] == name)
                    for name in candidate_row["candidate_order_ab"]
                ]
                active_stage = "order_ab"
                raw_ab, output_ab, ab_retried, ab_repair_kind, _ = infer_json(images, _candidate_prompt(
                    candidate_row["evidence_prompt_ab"], attribute_output,
                    candidate_row["candidate_hypotheses"], candidate_row["candidate_order_ab"],
                ), args.candidate_max_tokens, validator=lambda value: (
                    _validate_candidate_output(value, expected_ab) or value, {}
                ))
                record["order_ab_raw_output"] = raw_ab
                record["order_ab"] = output_ab
                record["json_retried"] = bool(record["json_retried"] or ab_retried)
                if ab_repair_kind is not None:
                    record["repair_kinds"].append(ab_repair_kind)
                expected_ba = list(reversed(expected_ab))
                active_stage = "order_ba"
                raw_ba, output_ba, ba_retried, ba_repair_kind, _ = infer_json(images, _candidate_prompt(
                    candidate_row["evidence_prompt_ba"], attribute_output,
                    candidate_row["candidate_hypotheses"], candidate_row["candidate_order_ba"],
                ), args.candidate_max_tokens, validator=lambda value: (
                    _validate_candidate_output(value, expected_ba) or value, {}
                ))
                record["order_ba_raw_output"] = raw_ba
                record["order_ba"] = output_ba
                record["json_retried"] = bool(record["json_retried"] or ba_retried)
                if ba_repair_kind is not None:
                    record["repair_kinds"].append(ba_repair_kind)
                counts["json_retry"] += int(record["json_retried"])
                counts["json_syntax_repair"] += record["repair_kinds"].count("json_syntax")
                counts["output_structure_repair"] += record["repair_kinds"].count("output_structure")
                evidence = {"task_id": candidate_row["task_id"], "order_ab": output_ab, "order_ba": output_ba}
                decision = decide_row(candidate_row, evidence)
                insufficient = candidate_set_insufficient_or_unclear(output_ab, output_ba)
                record["candidate_set_insufficient_or_unclear"] = insufficient
                counts["candidate_set_insufficient_or_unclear"] += int(insufficient)
                if insufficient and decision["class_changed"]:
                    raise AssertionError("an insufficient candidate set attempted to change the frozen class")
                record.update({
                    "valid": True,
                    "decision": decision,
                })
                decision_handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
                decision_handle.flush()
                counts["valid"] += 1
                counts["class_changed"] += int(decision["class_changed"])
                counts["kept_frozen_control"] += int(not decision["class_changed"])
            except Exception as error:
                if isinstance(error, InferenceValidationFailure):
                    record[f"{active_stage}_first_raw_output"] = error.first_raw
                    record[f"{active_stage}_repair_raw_output"] = error.repaired_raw
                    record["json_retried"] = True
                    record["repair_kinds"].append(error.repair_kind)
                    record["error"] = f"{type(error.final_error).__name__}: {error.final_error}"
                else:
                    record["error"] = f"{type(error).__name__}: {error}"
                counts["invalid"] += 1
            output_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output_handle.flush()
            print(
                f"[DM-SMS-1 batch smoke] {len(existing_records) + index}/{len(selected)} {candidate_row['scene_name']} "
                f"valid={record['valid']} changed={record.get('decision', {}).get('class_changed')} "
                f"error={record['error']}", flush=True,
            )
    all_records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    if len(all_records) != len(selected):
        raise AssertionError("batch output does not cover the complete selected slice")
    summary = summarize_batch_records(all_records, selected)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribute-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("pretrained/checkpoints/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--per-scene", type=int, default=2)
    parser.add_argument("--per-scene-offset", type=int, default=0)
    parser.add_argument("--attribute-max-tokens", type=int, default=700)
    parser.add_argument("--candidate-max-tokens", type=int, default=450)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--failed-from-batch-output", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
