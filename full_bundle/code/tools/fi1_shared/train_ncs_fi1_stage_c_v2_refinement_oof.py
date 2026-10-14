"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

from collections import defaultdict

from tools.fi1_shared.build_ncs_fi1_stage_c_v2_member_dataset_gt import MEMBER_EVIDENCE_FEATURE_NAMES

EXCLUSIVE_ROLES = ("track_only", "native_only")

QUALITY_BASE_FEATURE_NAMES = (
    "stage_a_union_quality",
    "stage_a_track_quality",
    "stage_a_native_quality",
    "stage_b_union_score",
    "log1p_original_union_point_count",
    "log1p_temporary_refined_point_count",
    "temporary_refined_point_fraction",
    "atom_count",
    "shared_atom_fraction",
    "track_only_atom_fraction",
    "native_only_atom_fraction",
    "confident_removal_atom_fraction",
    "connectivity_removed_atom_fraction",
    "total_removed_atom_fraction",
    "track_only_predicted_delta_mean",
    "track_only_predicted_delta_max",
    "track_only_lower_bound_mean",
    "track_only_lower_bound_max",
    "native_only_predicted_delta_mean",
    "native_only_predicted_delta_max",
    "native_only_lower_bound_mean",
    "native_only_lower_bound_max",
)

QUALITY_FEATURE_NAMES = (
    *QUALITY_BASE_FEATURE_NAMES,
    *tuple(f"all_mean__{name}" for name in MEMBER_EVIDENCE_FEATURE_NAMES),
    *tuple(f"track_only_mean__{name}" for name in MEMBER_EVIDENCE_FEATURE_NAMES),
    *tuple(f"native_only_mean__{name}" for name in MEMBER_EVIDENCE_FEATURE_NAMES),
)

def conservative_kept_indexes(
    atoms: list[dict], lower_bounds: list[float | None], neighbors: dict[int, list[dict]],
) -> tuple[set[int], dict]:
    """Delete confident exclusive atoms, then retain components attached to shared."""
    if len(atoms) != len(lower_bounds):
        raise ValueError("atom/lower-bound lengths differ")
    shared = {index for index, atom in enumerate(atoms) if atom["role"] == "shared"}
    all_indexes = set(range(len(atoms)))
    if not shared:
        return all_indexes, {
            "fallback_no_shared": True,
            "confident_removed_indexes": set(),
            "connectivity_removed_indexes": set(),
        }
    confident_removed = {
        index for index, (atom, lower) in enumerate(zip(atoms, lower_bounds))
        if atom["role"] != "shared" and lower is not None and float(lower) > 0.0
    }
    selected = all_indexes - confident_removed
    by_raw = defaultdict(set)
    for index in selected:
        by_raw[int(atoms[index]["raw_superpoint_id"])].add(index)
    raw_neighbors = {
        raw_id: {int(row["neighbor_superpoint_id"]) for row in neighbors.get(raw_id, [])}
        for raw_id in by_raw
    }
    reached = set(shared)
    stack = sorted(shared)
    while stack:
        current = stack.pop()
        raw_id = int(atoms[current]["raw_superpoint_id"])
        linked = set(by_raw[raw_id])
        for other_raw in raw_neighbors.get(raw_id, set()):
            linked.update(by_raw.get(other_raw, set()))
        new = linked - reached
        reached.update(new)
        stack.extend(sorted(new))
    connectivity_removed = selected - reached
    if not shared <= reached:
        raise AssertionError("shared atom was removed by connectivity")
    return reached, {
        "fallback_no_shared": False,
        "confident_removed_indexes": confident_removed,
        "connectivity_removed_indexes": connectivity_removed,
    }
