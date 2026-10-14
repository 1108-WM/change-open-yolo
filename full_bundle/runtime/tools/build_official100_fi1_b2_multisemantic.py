#!/usr/bin/env python3
"""Prepare geometry-own hypotheses; production calibration is not implemented.

This writes a hypothesis ledger, NOT an AP manifest.
--calibration-root is rejected until independent model replay is implemented.
Raw semantic fusion and FI1 marginal scores are never used as evaluator scores.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

VERSION = "fi1_b2_multisemantic_v3_1"
B1_NAME = "b1_prediction_column_manifest.jsonl"
B2_NAME = "b2_prediction_column_manifest.jsonl"
EVIDENCE_NAME = "refined_geometry_semantic_evidence.jsonl"
HYPOTHESES_NAME = "refined_semantic_hypotheses.jsonl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_record(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return [json.loads(line, parse_constant=reject_constant)
            for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


@contextmanager
def output_directory(output):
    output = Path(output)
    staging = output.parent / f".{output.name}.tmp.{os.getpid()}"
    failed = output.parent / f"{output.name}.failed.{os.getpid()}"
    if output.exists() or staging.exists() or failed.exists():
        raise FileExistsError(output)
    staging.mkdir(parents=True)
    try:
        yield staging
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    except Exception as error:
        write_json(staging / "failure.json", {"error": str(error), "completed": False})
        staging.rename(failed)
        raise


def unit_score(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name}: boolean is not a score")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name}: expected finite [0,1] score")
    return value


def distribution(values, count, name):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{name}: expected {count} nonnegative evidence values")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name}: boolean evidence")
    values = list(map(float, values))
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError(f"{name}: nonfinite/negative values; pass probabilities, not cosine logits")
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ValueError(f"{name}: nonfinite sum") from error
    if not math.isfinite(total):
        raise ValueError(f"{name}: nonfinite sum")
    return [x / total for x in values] if total else [0.0] * count


def validate_b1(rows, class_count):
    if not rows:
        raise ValueError("empty B1")
    by_scene = defaultdict(list)
    identities = set()
    closed_scenes, previous_scene = set(), None
    for row in rows:
        for field in ("candidate_deletion", "geometry_mutation", "class_mutation", "ground_truth_read", "ap_computed"):
            if row.get(field) is not False:
                raise ValueError(f"B1 forbidden/missing flag: {field}")
        if not row.get("geometry_evidence_key") or not row.get("geometry_locator_read_only"):
            raise ValueError("B1 missing geometry identity")
        key = row["semantic_hypothesis_key"]
        if not key or key in identities:
            raise ValueError("duplicate/empty B1 semantic identity")
        identities.add(key)
        scene = row["scene_name"]
        if not isinstance(scene, str) or not scene:
            raise ValueError("B1 missing scene identity")
        if scene != previous_scene:
            if scene in closed_scenes:
                raise ValueError("B1 scenes are interleaved")
            if previous_scene is not None:
                closed_scenes.add(previous_scene)
            previous_scene = scene
        if type(row["column_index"]) is not int or row["column_index"] != len(by_scene[scene]):
            raise ValueError("B1 columns must be consecutive per scene")
        if type(row["class_index"]) is not int or not 0 <= row["class_index"] <= class_count:
            raise ValueError("B1 class outside vocabulary/background boundary")
        unit_score(row["score"], "B1 score")
        by_scene[scene].append(row)
    return by_scene


def require_audit(path, expected):
    audit = json.loads(Path(path).read_text())
    if audit.get("audit_valid") is not True or audit.get("error_count") != 0:
        raise ValueError(f"audit did not pass: {path}")
    if any(audit.get("input_provenance", {}).get(key) != value for key, value in expected.items()):
        raise ValueError(f"audit is not bound to current files: {path}")
    return audit


def validate_evidence(row, class_count):
    for flag in ("ground_truth_read", "ap_computed"):
        if row.get(flag) is not False:
            raise ValueError(f"evidence missing no-GT/no-AP contract: {flag}")
    for key in ("scene_name", "fi1_plan_key", "geometry_evidence_key", "geometry_hash",
                "geometry_locator_read_only", "vocabulary_sha256", "producer_sha256"):
        if not row.get(key):
            raise ValueError(f"missing evidence field: {key}")
    if type(row.get("point_count")) is not int or row["point_count"] <= 0:
        raise ValueError("invalid point count")
    if type(row.get("candidate_id")) is not int or row["candidate_id"] < 0:
        raise ValueError("invalid candidate id")
    unit_score(row["fi1_geometry_score"], "FI1 final ranking score")
    if row.get("geometry_quality") is not None and not row.get("geometry_quality_provenance"):
        raise ValueError("independent quality lacks provenance")
    # Null means unavailable; never substitute the final marginal score here.
    if row.get("geometry_quality") is not None:
        unit_score(row["geometry_quality"], "independent geometry quality")
    for source in ("yolo", "alpha"):
        binding = row[f"{source}_provenance"]
        if binding.get("geometry_hash") != row["geometry_hash"] or binding.get("evidence_scope") != "geometry_own":
            raise ValueError(f"{source}: parent/inherited evidence cannot represent refined geometry")
        if not binding.get("input_sha256") or type(binding.get("available")) is not bool:
            raise ValueError(f"{source}: missing provenance/availability")
        values = distribution(row[f"{source}_scores"], class_count, source)
        if bool(sum(values)) != binding["available"]:
            raise ValueError(f"{source}: availability differs from distribution")


def build_refined_columns(evidence, *, class_count=198, top_k=5):
    """Return candidate hypotheses, not calibrated evaluator columns."""
    if type(class_count) is not int or class_count < 2 or type(top_k) is not int or not 1 <= top_k <= class_count:
        raise ValueError("invalid class_count/top_k")
    validate_evidence(evidence, class_count)
    sources = {s: distribution(evidence[f"{s}_scores"], class_count, s) for s in ("yolo", "alpha")}
    supported = {s: {i for i in sorted(range(class_count), key=lambda i: (-p[i], i))[:top_k] if p[i] > 0}
                 for s, p in sources.items()}
    available = sum(evidence[f"{s}_provenance"]["available"] for s in sources)
    records = []
    for c in sorted(set.union(*supported.values())):
        records.append({
            "scene_name": evidence["scene_name"], "candidate_source": "refined_union",
            "candidate_id": evidence["candidate_id"], "class_index": c,
            "semantic_hypothesis_key": f"{evidence['fi1_plan_key']}:semantic:v3:class:{c}",
            "geometry_evidence_key": evidence["geometry_evidence_key"],
            "geometry_hash": evidence["geometry_hash"], "point_count": evidence["point_count"],
            "geometry_locator_read_only": copy.deepcopy(evidence["geometry_locator_read_only"]),
            "fi1_plan_key": evidence["fi1_plan_key"],
            "fi1_plan_keys": evidence.get("fi1_plan_keys", [evidence["fi1_plan_key"]]),
            "fi1_geometry_score": evidence["fi1_geometry_score"],
            "geometry_quality": evidence.get("geometry_quality"),
            "geometry_quality_provenance": evidence.get("geometry_quality_provenance"),
            "yolo_semantic_score": sources["yolo"][c], "alpha_semantic_score": sources["alpha"][c],
            "semantic_evidence_score": sum(p[c] for p in sources.values()) / available,
            "hypothesis_sources": [s for s in sources if c in supported[s]],
            "yolo_available": evidence["yolo_provenance"]["available"],
            "alpha_available": evidence["alpha_provenance"]["available"],
            "evidence_sha256": digest_record(evidence),
            "score": None, "score_source": "awaiting_independent_calibration",
            "append_only": True, "candidate_deletion": False, "geometry_mutation": False,
            "class_mutation": False, "ground_truth_read": False, "ap_computed": False,
        })
    return records


def append_calibrated(b1, hypotheses, scores, class_count=198):
    """Structural assembly helper for tests; not a production audit or entrypoint."""
    by_scene = validate_b1(b1, class_count)
    indexed = {r["semantic_hypothesis_key"]: r for r in scores}
    if len(indexed) != len(scores) or set(indexed) != {r["semantic_hypothesis_key"] for r in hypotheses}:
        raise ValueError("calibration coverage/identity differs")
    additions = defaultdict(list)
    for h in hypotheses:
        s = indexed[h["semantic_hypothesis_key"]]
        if (s.get("evidence_sha256") != h["evidence_sha256"] or s.get("scene_name") != h["scene_name"]
                or s.get("class_index") != h["class_index"]):
            raise ValueError("calibration evidence/class identity differs")
        # This helper checks structure only; model inference and splits are unverified.
        if not s.get("model_id"):
            raise ValueError("missing calibration model identity")
        row = copy.deepcopy(h)
        row.update(score=unit_score(s["score"], "calibrated score"),
                   score_source="structural_test_only_unverified_calibration", calibration_model_id=s["model_id"],
                   production_ready=False)
        additions[h["scene_name"]].append(row)
    if not set(additions).issubset(by_scene):
        raise ValueError("new geometry references unknown scene")
    output = []
    for scene, original in by_scene.items():
        output.extend(copy.deepcopy(original))
        for index, row in enumerate(additions[scene], start=len(original)):
            row["column_index"] = index
            output.append(row)
    audit_prefix(b1, output, class_count)
    return output


def audit_prefix(b1, b2, class_count):
    original = validate_b1(b1, class_count)
    output = defaultdict(list)
    keys = set()
    for row in b2:
        if row["semantic_hypothesis_key"] in keys:
            raise ValueError("duplicate B2 hypothesis")
        keys.add(row["semantic_hypothesis_key"])
        output[row["scene_name"]].append(row)
    if set(output) != set(original):
        raise ValueError("B2 scene coverage differs")
    for scene, rows in output.items():
        if rows[:len(original[scene])] != original[scene]:
            raise ValueError("B1 scene prefix was mutated/deleted/reordered")
        if [r["column_index"] for r in rows] != list(range(len(rows))):
            raise ValueError("B2 scene column indices differ")


def run(args):
    if args.calibration_root is not None:
        raise ValueError("production B2 disabled: independent calibration model replay and upstream split audit are not implemented")
    if not 1 <= args.top_k <= args.class_count:
        raise ValueError("invalid top-k")
    b1_path = args.b1_root / B1_NAME
    evidence_path = args.evidence_root / EVIDENCE_NAME
    evidence_summary_path = args.evidence_root / "summary.json"
    b1, evidence = read_jsonl(b1_path), read_jsonl(evidence_path)
    scenes = validate_b1(b1, args.class_count)
    summary = json.loads(evidence_summary_path.read_text())
    if (summary.get("ground_truth_read") is not False or summary.get("ap_computed") is not False
            or summary.get("class_count") != args.class_count
            or summary.get("hashes", {}).get(EVIDENCE_NAME) != sha256_file(evidence_path)):
        raise ValueError("evidence summary/hash contract differs")
    identities, plan_keys, hypotheses, missing = set(), set(), [], []
    for row in evidence:
        identity = (row["scene_name"], row["geometry_hash"])
        if identity in identities or row["fi1_plan_key"] in plan_keys or row["scene_name"] not in scenes:
            raise ValueError("duplicate evidence geometry/plan or unknown scene")
        if row["vocabulary_sha256"] != summary["vocabulary_sha256"]:
            raise ValueError("vocabulary identity differs")
        if row["geometry_evidence_key"] in {r["geometry_evidence_key"] for r in scenes[row["scene_name"]]}:
            raise ValueError("refined geometry already exists in B1")
        identities.add(identity)
        plan_keys.add(row["fi1_plan_key"])
        new = build_refined_columns(row, class_count=args.class_count, top_k=args.top_k)
        hypotheses.extend(new)
        if not new:
            missing.append(row["fi1_plan_key"])
    if not evidence or summary.get("geometry_count") != len(evidence):
        raise ValueError("evidence geometry coverage differs")
    provenance = {"b1_manifest": sha256_file(b1_path), "evidence_manifest": sha256_file(evidence_path),
                  "evidence_summary": sha256_file(evidence_summary_path), "builder": sha256_file(Path(__file__))}
    with output_directory(args.output_root) as out:
        write_jsonl(out / HYPOTHESES_NAME, hypotheses)
        hashes = {HYPOTHESES_NAME: sha256_file(out / HYPOTHESES_NAME)}
        result = {"version": VERSION, "b1_candidate_count": len(b1), "geometry_count": len(evidence),
                  "new_hypothesis_count": len(hypotheses), "missing_evidence_plan_keys": missing,
                  "top_k_per_source": args.top_k, "class_count": args.class_count,
                  "prediction_manifest_materialized": False,
                  "production_ready": False, "calibration_status": "not_implemented",
                  "b2_candidate_count": None,
                  "input_provenance": provenance, "hashes": hashes,
                  "ground_truth_read": False, "ap_computed": False}
        write_json(out / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b1-root", type=Path, required=True)
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--class-count", type=int, default=198)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--calibration-root", type=Path, help="reserved; rejected until independent calibration is implemented")
    print(json.dumps(run(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
