#!/usr/bin/env python3
"""Pure helpers for the preregistered FI1 geometry/semantic separation ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


VERSION = "fi1_geometry_semantic_separation_v1"
GEOMETRY_LEDGER_NAME = "geometry_evidence_ledger.jsonl"
SEMANTIC_LEDGER_NAME = "semantic_hypothesis_ledger.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def geometry_evidence_key(scene_name: str, geometry_hash: str) -> str:
    return f"{scene_name}:geometry_evidence:{geometry_hash}"


def legacy_semantic_hypothesis_key(member: dict) -> str:
    return (
        f"{member['scene_name']}:semantic:legacy:"
        f"{member['candidate_source']}:{int(member['candidate_id'])}:"
        f"{member['geometry_hash']}"
    )


def refined_semantic_hypothesis_key(member: dict, scene_name: str) -> str:
    return f"{scene_name}:semantic:refined:{member['plan_key']}"


def frozen_flags() -> dict:
    return {
        "candidate_retained": True,
        "candidate_deletion": False,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "class_mutation": False,
        "score_mutation": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }


def plan_member_projection(member: dict) -> dict:
    return {
        "plan_index": int(member["plan_index"]),
        "plan_key": str(member["plan_key"]),
        "candidate_source": str(member["candidate_source"]),
        "candidate_id": int(member["candidate_id"]),
        "frozen_class_index": int(member["frozen_class_index"]),
        "frozen_class_valid": bool(member["frozen_class_valid"]),
        "fi1_geometry_score": float(member["challenger_score"]),
        "point_count": int(member["point_count"]),
        "geometry_hash": str(member["geometry_hash"]),
        "geometry_locator_read_only": dict(member["geometry_locator_read_only"]),
        "append_only": bool(member["append_only"]),
        **frozen_flags(),
    }
