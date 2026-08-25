"""Frozen, narrowly scoped terminal-safe-keep contract for val312."""

from __future__ import annotations

TERMINAL_KEEP_REASON = "no_visible_view_and_background_sentinel"
TERMINAL_CLASS_INDEX = 198
TERMINAL_PLAN_INDICES = frozenset({611, 6649, 21496, 33247})
TERMINAL_PLAN_KEYS = frozenset({
    "scene0019_01:geometry:ff479173f407ab38dc0c5b1b56f522a536f05dc3",
    "scene0153_00:geometry:3641d0238b6def23b8cc4b108acc5eaeeecbe737",
    "scene0496_00:geometry:5413dbcf0af35be526d05cde4d8b41842cd52f41",
    "scene0663_02:geometry:b9aefa68d6ee24f5c9f2db54ebd9906db920b44e",
})


def terminal_identity(plan_index: int, plan_key: str) -> bool:
    """Return whether an identity is one of the four frozen val312 blockers."""
    return (int(plan_index), str(plan_key)) in terminal_expected_identities()


def terminal_safe_keep_eligible(
    *, plan_index: int, plan_key: str, frozen_class_index: int,
    selected_views: list, selected_view_count: int, alpha_feature_valid: bool,
    alpha_class_index: object,
    candidate_retained: bool, candidate_deletion: bool,
) -> bool:
    """Check the complete contract; no inferred or broadened matching is allowed."""
    return (
        terminal_identity(plan_index, plan_key)
        and int(frozen_class_index) == TERMINAL_CLASS_INDEX
        and selected_views == []
        and int(selected_view_count) == 0
        and alpha_feature_valid is False
        and alpha_class_index is None
        and candidate_retained is True
        and candidate_deletion is False
    )


def terminal_expected_identities() -> set[tuple[int, str]]:
    return {
        (611, "scene0019_01:geometry:ff479173f407ab38dc0c5b1b56f522a536f05dc3"),
        (6649, "scene0153_00:geometry:3641d0238b6def23b8cc4b108acc5eaeeecbe737"),
        (21496, "scene0496_00:geometry:5413dbcf0af35be526d05cde4d8b41842cd52f41"),
        (33247, "scene0663_02:geometry:b9aefa68d6ee24f5c9f2db54ebd9906db920b44e"),
    }
