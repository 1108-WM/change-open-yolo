import argparse
import csv
import json
import os
import os.path as osp
from collections import Counter, defaultdict

import numpy as np


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _mean(values):
    values = [float(item) for item in values]
    if not values:
        return 0.0
    return float(np.mean(values))


def _load_scene_names(args):
    if args.scene:
        return [str(item) for item in args.scene]
    if args.scene_list:
        with open(args.scene_list) as f:
            return [line.strip() for line in f if line.strip()]
    return sorted(
        item
        for item in os.listdir(args.candidates_dir)
        if item.startswith("scene") and osp.isdir(osp.join(args.candidates_dir, item))
    )


def _connectivity(summary):
    return summary.get("superpoint_candidate_connectivity", {})


LARGE_PLANE_CLASSES = {
    "bulletin board",
    "whiteboard",
    "tv",
    "door",
    "curtain",
    "mat",
    "mattress",
    "mirror",
    "projector screen",
}
SMALL_PLANE_CLASSES = {
    "picture",
}
COMPACT_REVIEW_CLASSES = {
    "picture",
    "mini fridge",
    "toilet",
    "dishwasher",
}


def _mask_metrics(summary):
    metrics = summary.get("superpoint_candidate_existing_mask_metrics", {})
    return {
        "existing_mask_seed_ratio": _safe_float(metrics.get("seed_in_existing_mask_ratio", 0.0)),
        "existing_mask_seed_coverage": _safe_float(metrics.get("best_existing_seed_coverage", 0.0)),
        "existing_mask_iou": _safe_float(metrics.get("best_existing_iou", 0.0)),
        "existing_mask_id": _safe_int(metrics.get("best_existing_mask_id", -1), -1),
    }


def _append_reason(reasons, condition, text):
    if condition:
        reasons.append(text)


def _candidate_action(row):
    class_name = str(row.get("class_name") or "")
    ratio = row["largest_cc_to_point_ratio"]
    covered_by_point = row["largest_cc_covered_by_point_ratio"]
    point_coverage = row["point_covered_by_largest_cc_ratio"]
    conflict = row["largest_cc_point_conflict_overlap_ratio"]
    existing_iou = row["existing_mask_iou"]
    existing_coverage = row["existing_mask_seed_coverage"]
    core_boundary_to_core_only = row["core_boundary_to_core_only_ratio"]
    largest_keep = row["largest_cc_keep_ratio"]
    has_mask3d_support = existing_iou >= 0.50 or (existing_iou >= 0.35 and existing_coverage >= 0.90)
    has_strong_geometric_coverage = (
        point_coverage >= 0.85
        and covered_by_point >= 0.60
        and conflict < 0.10
    )
    reasons = []

    _append_reason(reasons, row["is_large_plane_class"], "large_plane_class")
    _append_reason(reasons, row["is_small_plane_class"], "small_plane_class")
    _append_reason(reasons, ratio >= 2.0, f"largest_cc_to_point={ratio:.2f}")
    _append_reason(reasons, covered_by_point < 0.45, f"largest_cc_covered_by_point={covered_by_point:.2f}")
    _append_reason(reasons, point_coverage >= 0.80, f"point_covered_by_largest_cc={point_coverage:.2f}")
    _append_reason(reasons, conflict >= 0.15, f"conflict_overlap={conflict:.2f}")
    _append_reason(reasons, has_mask3d_support, f"mask3d_iou={existing_iou:.2f}/coverage={existing_coverage:.2f}")
    _append_reason(
        reasons,
        core_boundary_to_core_only >= 1.35 and largest_keep >= 0.98,
        f"boundary_expands_without_cleanup={core_boundary_to_core_only:.2f}",
    )

    if row["largest_cc_count"] <= 0:
        return "reject_or_needs_mask3d_support", "missing_reliable_superpoint_core"

    if (
        row["is_large_plane_class"]
        and ratio >= 2.0
        and covered_by_point < 0.45
    ):
        if has_mask3d_support:
            return "reject_or_needs_mask3d_support", ";".join(
                reasons + ["large_plane_overexpanded_requires_visual_or_mask3d_review"]
            )
        return "reject_or_needs_mask3d_support", ";".join(
            reasons + ["large_plane_overexpanded_without_mask3d_support"]
        )

    if row["is_large_plane_class"] and ratio >= 1.5:
        return "manual_review", ";".join(reasons + ["large_plane_moderate_expansion"])

    if (
        class_name in COMPACT_REVIEW_CLASSES
        and ratio >= 2.0
        and conflict < 0.12
    ):
        if class_name in SMALL_PLANE_CLASSES:
            return "manual_review", ";".join(reasons + ["small_plane_large_expansion"])
        if has_mask3d_support and point_coverage >= 0.75:
            return "manual_review", ";".join(reasons + ["compact_object_large_but_supported"])
        return "manual_review", ";".join(reasons + ["compact_object_large_expansion"])

    if ratio >= 2.0:
        if has_mask3d_support and conflict < 0.10 and point_coverage >= 0.80:
            return "manual_review", ";".join(reasons + ["large_expansion_with_mask3d_support"])
        return "reject_or_needs_mask3d_support", ";".join(reasons + ["large_expansion"])

    if conflict >= 0.18:
        return "manual_review", ";".join(reasons + ["conflict_ge_0_18_requires_review"])

    if point_coverage < 0.70:
        return "manual_review", ";".join(reasons + ["low_point_coverage_not_auto_accept"])

    if existing_iou < 0.30 and not has_strong_geometric_coverage:
        return "manual_review", ";".join(reasons + ["low_mask3d_iou_without_strong_geometry"])

    if (
        ratio < 1.5
        and conflict < 0.18
        and point_coverage >= 0.70
        and covered_by_point >= 0.50
    ):
        return "accept_completion", ";".join(reasons + ["small_expansion_low_conflict"])

    if (
        class_name in {"dishwasher", "toilet"}
        and ratio < 1.5
        and (
            point_coverage >= 0.75
            or existing_coverage >= 0.90
        )
        and largest_keep >= 0.95
    ):
        if conflict >= 0.25:
            return "manual_review", ";".join(reasons + ["compact_object_small_expansion_high_conflict"])
        return "accept_completion", ";".join(reasons + ["compact_object_small_expansion"])

    if (
        core_boundary_to_core_only >= 1.35
        and largest_keep >= 0.98
        and ratio >= 1.5
    ):
        if conflict < 0.10 and has_mask3d_support and point_coverage >= 0.80:
            return "manual_review", ";".join(reasons + ["boundary_expansion_supported_but_large"])
        return "keep_core_only", ";".join(reasons + ["boundary_expansion_not_removed_by_largest_cc"])

    if conflict >= 0.20:
        return "manual_review", ";".join(reasons + ["high_conflict"])

    if point_coverage >= 0.75 and covered_by_point >= 0.45:
        return "accept_completion", ";".join(reasons + ["moderate_expansion_supported"])

    return "manual_review", ";".join(reasons + ["ambiguous_coverage"])


def _candidate_row(scene_name, candidate):
    diag = candidate.get("superpoint_diagnostics", {})
    core_comp = diag.get("core_only_point_level_comparison", {})
    plus_comp = diag.get("point_level_comparison", {})
    largest_cc_comp = diag.get("largest_cc_point_level_comparison", {})
    largest_cc_cleanup = diag.get("largest_cc_cleanup", {})
    proposal = diag.get("proposal", {})
    point_connectivity = plus_comp.get("point_candidate_connectivity", {})
    core_connectivity = _connectivity(core_comp)
    plus_connectivity = _connectivity(plus_comp)
    largest_cc_connectivity = _connectivity(largest_cc_comp)
    existing_metrics = _mask_metrics(largest_cc_comp)
    point_count = _safe_int(plus_comp.get("point_candidate_point_count", 0))
    core_only_count = _safe_int(core_comp.get("superpoint_candidate_point_count", 0))
    core_boundary_count = _safe_int(plus_comp.get("superpoint_candidate_point_count", 0))
    largest_cc_count = _safe_int(largest_cc_comp.get("superpoint_candidate_point_count", 0))
    class_name = candidate.get("class_name")
    row = {
        "scene_name": scene_name,
        "candidate_id": _safe_int(candidate.get("candidate_id", -1), -1),
        "class_name": class_name,
        "point_count": point_count,
        "core_only_count": core_only_count,
        "core_boundary_count": core_boundary_count,
        "largest_cc_count": largest_cc_count,
        "largest_cc_to_point_ratio": float(largest_cc_count / max(1, point_count)),
        "core_boundary_to_core_only_ratio": float(core_boundary_count / max(1, core_only_count)),
        "point_component_count": _safe_int(point_connectivity.get("component_count", 0)),
        "core_only_component_count": _safe_int(core_connectivity.get("component_count", 0)),
        "core_boundary_component_count": _safe_int(plus_connectivity.get("component_count", 0)),
        "largest_cc_component_count": _safe_int(largest_cc_connectivity.get("component_count", 0)),
        "point_largest_component_ratio": _safe_float(point_connectivity.get("largest_component_ratio", 0.0)),
        "core_only_largest_component_ratio": _safe_float(core_connectivity.get("largest_component_ratio", 0.0)),
        "core_boundary_largest_component_ratio": _safe_float(plus_connectivity.get("largest_component_ratio", 0.0)),
        "largest_cc_largest_component_ratio": _safe_float(largest_cc_connectivity.get("largest_component_ratio", 0.0)),
        "point_covered_by_core_only_ratio": _safe_float(core_comp.get("point_candidate_covered_by_superpoint_ratio", 0.0)),
        "point_covered_by_core_boundary_ratio": _safe_float(plus_comp.get("point_candidate_covered_by_superpoint_ratio", 0.0)),
        "point_covered_by_largest_cc_ratio": _safe_float(largest_cc_comp.get("point_candidate_covered_by_superpoint_ratio", 0.0)),
        "core_only_covered_by_point_ratio": _safe_float(core_comp.get("superpoint_candidate_covered_by_point_ratio", 0.0)),
        "core_boundary_covered_by_point_ratio": _safe_float(plus_comp.get("superpoint_candidate_covered_by_point_ratio", 0.0)),
        "largest_cc_covered_by_point_ratio": _safe_float(largest_cc_comp.get("superpoint_candidate_covered_by_point_ratio", 0.0)),
        "point_conflict_overlap_ratio": _safe_float(plus_comp.get("point_candidate_conflict_overlap_ratio", 0.0)),
        "largest_cc_point_conflict_overlap_ratio": _safe_float(largest_cc_comp.get("point_candidate_conflict_overlap_ratio", 0.0)),
        "largest_cc_keep_ratio": _safe_float(largest_cc_cleanup.get("largest_component_ratio", 0.0)),
        "largest_cc_cleanup_component_count": _safe_int(largest_cc_cleanup.get("component_count", 0)),
        "largest_cc_cleanup_radius": _safe_float(largest_cc_cleanup.get("component_radius", 0.0)),
        "largest_cc_report_component_radius": _safe_float(largest_cc_connectivity.get("component_radius", 0.0)),
        "boundary_superpoint_count": _safe_int(proposal.get("boundary_superpoint_count", 0)),
        "boundary_point_count": _safe_int(proposal.get("boundary_point_count", 0)),
        "bridge_boundary_superpoint_count": _safe_int(proposal.get("bridge_boundary_superpoint_count", 0)),
        "rejected_boundary_support_superpoint_count": _safe_int(
            proposal.get("rejected_boundary_support_superpoint_count", 0)
        ),
        "rejected_boundary_budget_superpoint_count": _safe_int(
            proposal.get("rejected_boundary_budget_superpoint_count", 0)
        ),
        "is_large_plane_class": str(class_name or "") in LARGE_PLANE_CLASSES,
        "is_small_plane_class": str(class_name or "") in SMALL_PLANE_CLASSES,
        **existing_metrics,
    }
    action, reason = _candidate_action(row)
    row["recommended_action"] = action
    row["action_reason"] = reason
    return row


def _summarize_rows(rows):
    if not rows:
        return {
            "candidate_count": 0,
            "class_counts": {},
        }
    summary = {
        "candidate_count": len(rows),
        "class_counts": dict(Counter(row["class_name"] for row in rows)),
        "point_count_mean": _mean(row["point_count"] for row in rows),
        "core_only_count_mean": _mean(row["core_only_count"] for row in rows),
        "core_boundary_count_mean": _mean(row["core_boundary_count"] for row in rows),
        "largest_cc_count_mean": _mean(row["largest_cc_count"] for row in rows),
        "core_only_to_point_ratio_mean": _mean(
            row["core_only_count"] / max(1, row["point_count"]) for row in rows
        ),
        "core_boundary_to_point_ratio_mean": _mean(
            row["core_boundary_count"] / max(1, row["point_count"]) for row in rows
        ),
        "core_boundary_to_core_only_ratio_mean": _mean(
            row["core_boundary_count"] / max(1, row["core_only_count"]) for row in rows
        ),
        "largest_cc_to_core_boundary_ratio_mean": _mean(
            row["largest_cc_count"] / max(1, row["core_boundary_count"]) for row in rows
        ),
        "point_component_count_mean": _mean(row["point_component_count"] for row in rows),
        "core_only_component_count_mean": _mean(row["core_only_component_count"] for row in rows),
        "core_boundary_component_count_mean": _mean(row["core_boundary_component_count"] for row in rows),
        "largest_cc_component_count_mean": _mean(row["largest_cc_component_count"] for row in rows),
        "point_single_component_count": sum(row["point_component_count"] == 1 for row in rows),
        "core_only_single_component_count": sum(row["core_only_component_count"] == 1 for row in rows),
        "core_boundary_single_component_count": sum(row["core_boundary_component_count"] == 1 for row in rows),
        "largest_cc_single_component_count": sum(row["largest_cc_component_count"] == 1 for row in rows),
        "core_only_missing_count": sum(row["core_only_component_count"] == 0 for row in rows),
        "core_boundary_missing_count": sum(row["core_boundary_component_count"] == 0 for row in rows),
        "largest_cc_missing_count": sum(row["largest_cc_component_count"] == 0 for row in rows),
        "core_only_nonzero_single_component_count": sum(
            row["core_only_component_count"] == 1 for row in rows if row["core_only_component_count"] > 0
        ),
        "core_boundary_nonzero_single_component_count": sum(
            row["core_boundary_component_count"] == 1
            for row in rows
            if row["core_boundary_component_count"] > 0
        ),
        "point_covered_by_core_only_ratio_mean": _mean(
            row["point_covered_by_core_only_ratio"] for row in rows
        ),
        "point_covered_by_core_boundary_ratio_mean": _mean(
            row["point_covered_by_core_boundary_ratio"] for row in rows
        ),
        "core_only_covered_by_point_ratio_mean": _mean(
            row["core_only_covered_by_point_ratio"] for row in rows
        ),
        "core_boundary_covered_by_point_ratio_mean": _mean(
            row["core_boundary_covered_by_point_ratio"] for row in rows
        ),
        "point_conflict_overlap_ratio_mean": _mean(row["point_conflict_overlap_ratio"] for row in rows),
        "largest_cc_point_conflict_overlap_ratio_mean": _mean(
            row["largest_cc_point_conflict_overlap_ratio"] for row in rows
        ),
        "largest_cc_keep_ratio_mean": _mean(row["largest_cc_keep_ratio"] for row in rows),
        "recommended_action_counts": dict(Counter(row["recommended_action"] for row in rows)),
        "boundary_superpoint_count_sum": sum(row["boundary_superpoint_count"] for row in rows),
        "bridge_boundary_superpoint_count_sum": sum(row["bridge_boundary_superpoint_count"] for row in rows),
        "rejected_boundary_support_superpoint_count_sum": sum(
            row["rejected_boundary_support_superpoint_count"] for row in rows
        ),
        "rejected_boundary_budget_superpoint_count_sum": sum(
            row["rejected_boundary_budget_superpoint_count"] for row in rows
        ),
    }
    return summary


ACTION_CSV_FIELDS = [
    "scene_name",
    "candidate_id",
    "class_name",
    "point_count",
    "core_only_count",
    "core_boundary_count",
    "largest_cc_count",
    "largest_cc_to_point_ratio",
    "largest_cc_covered_by_point_ratio",
    "point_covered_by_largest_cc_ratio",
    "existing_mask_iou",
    "existing_mask_seed_coverage",
    "existing_mask_seed_ratio",
    "conflict_overlap",
    "core_boundary_to_core_only_ratio",
    "largest_cc_keep_ratio",
    "is_large_plane_class",
    "is_small_plane_class",
    "recommended_action",
    "action_reason",
]


def _action_rows(rows):
    output = []
    for row in rows:
        output.append(
            {
                "scene_name": row["scene_name"],
                "candidate_id": row["candidate_id"],
                "class_name": row["class_name"],
                "point_count": row["point_count"],
                "core_only_count": row["core_only_count"],
                "core_boundary_count": row["core_boundary_count"],
                "largest_cc_count": row["largest_cc_count"],
                "largest_cc_to_point_ratio": row["largest_cc_to_point_ratio"],
                "largest_cc_covered_by_point_ratio": row["largest_cc_covered_by_point_ratio"],
                "point_covered_by_largest_cc_ratio": row["point_covered_by_largest_cc_ratio"],
                "existing_mask_iou": row["existing_mask_iou"],
                "existing_mask_seed_coverage": row["existing_mask_seed_coverage"],
                "existing_mask_seed_ratio": row["existing_mask_seed_ratio"],
                "conflict_overlap": row["largest_cc_point_conflict_overlap_ratio"],
                "core_boundary_to_core_only_ratio": row["core_boundary_to_core_only_ratio"],
                "largest_cc_keep_ratio": row["largest_cc_keep_ratio"],
                "is_large_plane_class": row["is_large_plane_class"],
                "is_small_plane_class": row["is_small_plane_class"],
                "recommended_action": row["recommended_action"],
                "action_reason": row["action_reason"],
            }
        )
    return output


def _write_actions_csv(path, rows):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACTION_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_review_lists(action_rows):
    return {
        "all_reject_or_needs_mask3d_support": [
            row for row in action_rows if row["recommended_action"] == "reject_or_needs_mask3d_support"
        ],
        "all_keep_core_only": [
            row for row in action_rows if row["recommended_action"] == "keep_core_only"
        ],
        "manual_review_largest_cc_to_point_ge_2": [
            row
            for row in action_rows
            if row["recommended_action"] == "manual_review"
            and _safe_float(row["largest_cc_to_point_ratio"]) >= 2.0
        ],
        "accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30": [
            row
            for row in action_rows
            if row["recommended_action"] == "accept_completion"
            and (
                _safe_float(row["conflict_overlap"]) >= 0.18
                or _safe_float(row["existing_mask_iou"]) < 0.30
            )
        ],
    }


def _write_review_lists_json(path, review_lists):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(review_lists, f, indent=2)


def analyze(args):
    scene_names = _load_scene_names(args)
    rows = []
    scene_summaries = {}
    missing = []
    for scene_name in scene_names:
        candidate_path = osp.join(args.candidates_dir, scene_name, "backprojection_candidates.json")
        if not osp.exists(candidate_path):
            missing.append(scene_name)
            continue
        with open(candidate_path) as f:
            payload = json.load(f)
        scene_rows = [_candidate_row(scene_name, candidate) for candidate in payload.get("candidates", [])]
        rows.extend(scene_rows)
        scene_summaries[scene_name] = _summarize_rows(scene_rows)

    non_single = [
        row
        for row in rows
        if row["core_boundary_component_count"] != 1
    ]
    non_single_with_core = [
        row
        for row in non_single
        if row["core_boundary_component_count"] > 0
    ]
    missing_core_boundary = [
        row
        for row in rows
        if row["core_boundary_component_count"] == 0
    ]
    focus = []
    focus_keys = set()
    invalid_focus = []
    for item in args.focus:
        if ":" not in item:
            invalid_focus.append(item)
            continue
        scene_name, class_name = item.split(":", 1)
        if not scene_name or not class_name:
            invalid_focus.append(item)
            continue
        focus_keys.add((scene_name, class_name))
    for row in rows:
        if (row["scene_name"], str(row["class_name"])) in focus_keys:
            focus.append(row)

    action_rows = _action_rows(rows)
    review_lists = _build_review_lists(action_rows)
    output = {
        "candidates_dir": args.candidates_dir,
        "scene_names": scene_names,
        "missing_scenes": missing,
        "summary": _summarize_rows(rows),
        "scene_summaries": scene_summaries,
        "candidate_actions": action_rows,
        "review_lists": review_lists,
        "non_single_core_boundary_candidates": non_single_with_core,
        "missing_core_boundary_candidates": missing_core_boundary,
        "focus_candidates": focus,
        "invalid_focus": invalid_focus,
    }
    if args.output_json:
        os.makedirs(osp.dirname(osp.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
    if args.output_actions_csv:
        _write_actions_csv(args.output_actions_csv, output["candidate_actions"])
    if args.output_review_lists_json:
        _write_review_lists_json(args.output_review_lists_json, review_lists)

    summary = output["summary"]
    print(
        "[SUPERPOINT_DIAG] "
        f"scenes={len(scene_names) - len(missing)}/{len(scene_names)} "
        f"candidates={summary.get('candidate_count', 0)} "
        f"point_mean={summary.get('point_count_mean', 0.0):.1f} "
        f"core_only_mean={summary.get('core_only_count_mean', 0.0):.1f} "
        f"core_boundary_mean={summary.get('core_boundary_count_mean', 0.0):.1f} "
        f"largest_cc_mean={summary.get('largest_cc_count_mean', 0.0):.1f} "
        f"point_cc_mean={summary.get('point_component_count_mean', 0.0):.2f} "
        f"core_only_cc_mean={summary.get('core_only_component_count_mean', 0.0):.2f} "
        f"core_boundary_cc_mean={summary.get('core_boundary_component_count_mean', 0.0):.2f} "
        f"largest_cc_cc_mean={summary.get('largest_cc_component_count_mean', 0.0):.2f}"
    )
    print(
        "[SUPERPOINT_DIAG] "
        f"single_component point/core_only/core_boundary/largest_cc="
        f"{summary.get('point_single_component_count', 0)}/"
        f"{summary.get('core_only_single_component_count', 0)}/"
        f"{summary.get('core_boundary_single_component_count', 0)}/"
        f"{summary.get('largest_cc_single_component_count', 0)}"
    )
    print(
        "[SUPERPOINT_DIAG] "
        f"missing_core_only/core_boundary/largest_cc="
        f"{summary.get('core_only_missing_count', 0)}/"
        f"{summary.get('core_boundary_missing_count', 0)}/"
        f"{summary.get('largest_cc_missing_count', 0)} "
        f"nonzero_single_core_only/core_boundary="
        f"{summary.get('core_only_nonzero_single_component_count', 0)}/"
        f"{summary.get('core_boundary_nonzero_single_component_count', 0)}"
    )
    print(
        "[SUPERPOINT_DIAG] "
        f"boundary accepted={summary.get('boundary_superpoint_count_sum', 0)} "
        f"bridge={summary.get('bridge_boundary_superpoint_count_sum', 0)} "
        f"rejected_support={summary.get('rejected_boundary_support_superpoint_count_sum', 0)} "
        f"rejected_budget={summary.get('rejected_boundary_budget_superpoint_count_sum', 0)}"
    )
    print(
        "[SUPERPOINT_DIAG] "
        f"recommended_actions={json.dumps(summary.get('recommended_action_counts', {}), sort_keys=True)}"
    )
    print(
        "[SUPERPOINT_DIAG] "
        f"review_lists={json.dumps({key: len(value) for key, value in review_lists.items()}, sort_keys=True)}"
    )
    if focus:
        print("[SUPERPOINT_DIAG] focus_candidates:")
        for row in focus:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    if invalid_focus:
        print(f"[SUPERPOINT_DIAG] ignored_invalid_focus={','.join(invalid_focus)}")
    if non_single_with_core:
        print("[SUPERPOINT_DIAG] non_single_core_boundary_candidates:")
        for row in non_single_with_core:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    if missing_core_boundary:
        print(f"[SUPERPOINT_DIAG] missing_core_boundary_candidates={len(missing_core_boundary)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize superpoint candidate diagnostic exports.")
    parser.add_argument("--candidates_dir", required=True)
    parser.add_argument("--scene_list", default=None)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_actions_csv", default=None)
    parser.add_argument("--output_review_lists_json", default=None)
    parser.add_argument(
        "--focus",
        action="append",
        default=[],
        help="Scene/class pair in the form scene0011_00:dishwasher. May be repeated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
