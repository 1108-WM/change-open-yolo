"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE

from tools.fi1_shared.build_safety60_structured_action_plan import _relation_predictions

def fixed_track_score(original_score: float, keep_probability: float) -> tuple[float, float]:
    """Return the fixed cubic multiplier and suppressed track score."""
    original = float(original_score)
    keep = float(keep_probability)
    if not 0.0 <= original <= 1.0 or not 0.0 <= keep <= 1.0:
        raise ValueError("score probabilities must lie in [0, 1]")
    multiplier = 1.0 - (1.0 - keep) ** 3
    planned = original * multiplier
    if planned > original or planned < 0.0:
        raise AssertionError("fixed track suppression violated score bounds")
    return multiplier, planned

def _components(relation_rows: list[dict]) -> list[dict]:
    by_component: dict[int, list[dict]] = defaultdict(list)
    for row in relation_rows:
        if row.get("contracts", {}).get("feature_ground_truth_usage") != "none":
            raise ValueError("relation ledger violates no-GT feature contract")
        by_component[int(row["relation_component_id"])].append(row)
    ids = sorted(by_component)
    if ids != list(range(len(ids))):
        raise ValueError("relation component ids are not contiguous")
    return [{
        "relation_component_id": component_id,
        "native_exact_geometry_group_ids": sorted({
            str(row["native_exact_geometry_group_id"])
            for row in by_component[component_id]
        }),
        "track_ids": sorted({
            int(row["track_id"]) for row in by_component[component_id]
        }),
    } for component_id in ids]

def _component_candidates(
    scene: str,
    components: list[dict],
    relation_rows: list[dict],
    feature_rows: list[dict],
) -> list[dict]:
    lookup = {
        (str(row["candidate_source"]), int(row["candidate_id"])): row
        for row in feature_rows
    }
    groups: dict[str, list[int]] = {}
    for row in relation_rows:
        group_id = str(row["native_exact_geometry_group_id"])
        members = sorted(map(int, row["native_member_candidate_ids"]))
        previous = groups.setdefault(group_id, members)
        if previous != members:
            raise ValueError(f"{scene}: inconsistent native geometry group {group_id}")
    result = []
    seen = set()
    for component in components:
        component_id = int(component["relation_component_id"])
        for group_id in component["native_exact_geometry_group_ids"]:
            members = groups[str(group_id)]
            representative = min(members, key=lambda candidate_id: (
                -float(lookup[(NATIVE_SOURCE, candidate_id)]["original_source_score"]),
                candidate_id,
            ))
            key = (NATIVE_SOURCE, representative)
            if key in seen:
                raise ValueError(f"{scene}: native representative repeated across components")
            seen.add(key)
            result.append({
                **lookup[key],
                "relation_component_id": component_id,
                "native_exact_geometry_group_id": str(group_id),
            })
        for track_id in component["track_ids"]:
            key = (TRACK_SOURCE, int(track_id))
            if key in seen:
                raise ValueError(f"{scene}: track repeated across components")
            seen.add(key)
            result.append({
                **lookup[key],
                "relation_component_id": component_id,
            })
    return result

def _stacked_relation_rows(
    relation_rows: list[dict], feature_rows: list[dict], quality: dict, package: dict,
) -> list[dict]:
    candidate_index = {
        (str(row["candidate_source"]), int(row["candidate_id"])): index
        for index, row in enumerate(feature_rows)
    }
    relation_predictions = _relation_predictions(package, relation_rows)
    result = []
    for row, relation in zip(relation_rows, relation_predictions):
        track_index = candidate_index[(TRACK_SOURCE, int(row["track_id"]))]
        native_indexes = [
            candidate_index[(NATIVE_SOURCE, int(candidate_id))]
            for candidate_id in row["native_member_candidate_ids"]
        ]
        evidence = {}
        for target in ("q", "valid25", "valid50"):
            track_value = float(quality[target][track_index])
            native_value = float(np.median(quality[target][native_indexes]))
            evidence[f"nested_track_{target}"] = track_value
            evidence[f"nested_native_{target}_median"] = native_value
            evidence[f"nested_{target}_delta_track_minus_native"] = track_value - native_value
        result.append({**evidence, **relation})
    return result
