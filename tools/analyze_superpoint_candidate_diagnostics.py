import argparse
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
    return {
        "scene_name": scene_name,
        "candidate_id": _safe_int(candidate.get("candidate_id", -1), -1),
        "class_name": candidate.get("class_name"),
        "point_count": _safe_int(plus_comp.get("point_candidate_point_count", 0)),
        "core_only_count": _safe_int(core_comp.get("superpoint_candidate_point_count", 0)),
        "core_boundary_count": _safe_int(plus_comp.get("superpoint_candidate_point_count", 0)),
        "largest_cc_count": _safe_int(largest_cc_comp.get("superpoint_candidate_point_count", 0)),
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
        "boundary_superpoint_count": _safe_int(proposal.get("boundary_superpoint_count", 0)),
        "boundary_point_count": _safe_int(proposal.get("boundary_point_count", 0)),
        "bridge_boundary_superpoint_count": _safe_int(proposal.get("bridge_boundary_superpoint_count", 0)),
        "rejected_boundary_support_superpoint_count": _safe_int(
            proposal.get("rejected_boundary_support_superpoint_count", 0)
        ),
        "rejected_boundary_budget_superpoint_count": _safe_int(
            proposal.get("rejected_boundary_budget_superpoint_count", 0)
        ),
    }


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

    output = {
        "candidates_dir": args.candidates_dir,
        "scene_names": scene_names,
        "missing_scenes": missing,
        "summary": _summarize_rows(rows),
        "scene_summaries": scene_summaries,
        "non_single_core_boundary_candidates": [
            row
            for row in non_single
            if row["core_boundary_component_count"] > 0
        ],
        "missing_core_boundary_candidates": [
            row
            for row in rows
            if row["core_boundary_component_count"] == 0
        ],
        "focus_candidates": focus,
        "invalid_focus": invalid_focus,
    }
    if args.output_json:
        os.makedirs(osp.dirname(osp.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)

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
    if focus:
        print("[SUPERPOINT_DIAG] focus_candidates:")
        for row in focus:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    if invalid_focus:
        print(f"[SUPERPOINT_DIAG] ignored_invalid_focus={','.join(invalid_focus)}")
    if non_single:
        print("[SUPERPOINT_DIAG] non_single_core_boundary_candidates:")
        for row in non_single:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize superpoint candidate diagnostic exports.")
    parser.add_argument("--candidates_dir", required=True)
    parser.add_argument("--scene_list", default=None)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--focus",
        action="append",
        default=[],
        help="Scene/class pair in the form scene0011_00:dishwasher. May be repeated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
