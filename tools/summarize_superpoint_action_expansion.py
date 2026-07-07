import argparse
import csv
import json
import os
import os.path as osp
from collections import Counter


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _key(row):
    return row["scene_name"], str(row["candidate_id"])


def _write_csv(path, rows, fields):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _load_review_counts(path):
    with open(path) as f:
        review_lists = json.load(f)
    return {key: len(value) for key, value in review_lists.items()}


def _format_candidate(row):
    return (
        f"{row['scene_name']} candidate{int(row['candidate_id']):04d} "
        f"{row['class_name']} "
        f"ratio={_safe_float(row['largest_cc_to_point_ratio']):.2f} "
        f"IoU={_safe_float(row['existing_mask_iou']):.2f} "
        f"conflict={_safe_float(row['conflict_overlap']):.2f}"
    )


def analyze(args):
    base_rows = _load_rows(args.base_actions_csv)
    expanded_rows = _load_rows(args.expanded_actions_csv)
    base_keys = {_key(row) for row in base_rows}
    base_by_key = {_key(row): row for row in base_rows}
    new_rows = [row for row in expanded_rows if _key(row) not in base_keys]
    changed_rows = []
    for row in expanded_rows:
        base_row = base_by_key.get(_key(row))
        if base_row is None:
            continue
        if base_row["recommended_action"] == row["recommended_action"]:
            continue
        changed = dict(row)
        changed["previous_recommended_action"] = base_row["recommended_action"]
        changed["previous_action_reason"] = base_row["action_reason"]
        changed_rows.append(changed)

    base_counts = Counter(row["recommended_action"] for row in base_rows)
    expanded_counts = Counter(row["recommended_action"] for row in expanded_rows)
    new_counts = Counter(row["recommended_action"] for row in new_rows)

    new_accept = [row for row in new_rows if row["recommended_action"] == "accept_completion"]
    new_accept_large = [
        row for row in new_accept
        if _safe_float(row["largest_cc_to_point_ratio"]) >= 2.0
    ]
    new_accept_risky = [
        row for row in new_accept
        if row["is_large_plane_class"] == "True"
        or _safe_float(row["conflict_overlap"]) >= 0.18
        or _safe_float(row["existing_mask_iou"]) < 0.30
    ]
    new_reject = [
        row for row in new_rows
        if row["recommended_action"] == "reject_or_needs_mask3d_support"
    ]
    new_manual_large = [
        row for row in new_rows
        if row["recommended_action"] == "manual_review"
        and _safe_float(row["largest_cc_to_point_ratio"]) >= 2.0
    ]

    fields = [
        "scene_name",
        "candidate_id",
        "class_name",
        "recommended_action",
        "largest_cc_to_point_ratio",
        "largest_cc_covered_by_point_ratio",
        "point_covered_by_largest_cc_ratio",
        "existing_mask_iou",
        "existing_mask_seed_coverage",
        "conflict_overlap",
        "is_large_plane_class",
        "is_small_plane_class",
        "is_soft_thin_plane_class",
        "action_reason",
    ]
    changed_fields = [
        "scene_name",
        "candidate_id",
        "class_name",
        "previous_recommended_action",
        "recommended_action",
        "largest_cc_to_point_ratio",
        "largest_cc_covered_by_point_ratio",
        "point_covered_by_largest_cc_ratio",
        "existing_mask_iou",
        "existing_mask_seed_coverage",
        "conflict_overlap",
        "is_large_plane_class",
        "is_small_plane_class",
        "is_soft_thin_plane_class",
        "previous_action_reason",
        "action_reason",
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(osp.join(args.output_dir, "new_accept_completion_candidates.csv"), new_accept, fields)
    _write_csv(osp.join(args.output_dir, "new_accept_completion_large_expansion.csv"), new_accept_large, fields)
    _write_csv(osp.join(args.output_dir, "new_reject_or_needs_mask3d_support.csv"), new_reject, fields)
    _write_csv(osp.join(args.output_dir, "new_manual_review_large_expansion.csv"), new_manual_large, fields)
    _write_csv(osp.join(args.output_dir, "changed_action_candidates.csv"), changed_rows, changed_fields)

    base_review_counts = _load_review_counts(args.base_review_lists_json)
    expanded_review_counts = _load_review_counts(args.expanded_review_lists_json)

    md_path = osp.join(args.output_dir, "expansion_review_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# {args.title}\n\n")
        f.write("This comparison uses export-only diagnostics and does not run final AP.\n\n")
        f.write("## Action counts\n\n")
        f.write("| split | candidates | accept_completion | manual_review | reject_or_needs_mask3d_support | keep_core_only |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        f.write(
            f"| {args.base_label} | {len(base_rows)} | {base_counts['accept_completion']} | "
            f"{base_counts['manual_review']} | {base_counts['reject_or_needs_mask3d_support']} | "
            f"{base_counts['keep_core_only']} |\n"
        )
        f.write(
            f"| {args.expanded_label} | {len(expanded_rows)} | {expanded_counts['accept_completion']} | "
            f"{expanded_counts['manual_review']} | {expanded_counts['reject_or_needs_mask3d_support']} | "
            f"{expanded_counts['keep_core_only']} |\n"
        )
        f.write(
            f"| {args.new_label} | {len(new_rows)} | {new_counts['accept_completion']} | "
            f"{new_counts['manual_review']} | {new_counts['reject_or_needs_mask3d_support']} | "
            f"{new_counts['keep_core_only']} |\n\n"
        )
        f.write(f"- Action changes on shared candidates: {len(changed_rows)}\n\n")

        f.write("## Review list counts\n\n")
        f.write(f"| review_list | {args.base_label} | {args.expanded_label} |\n")
        f.write("| --- | ---: | ---: |\n")
        for key in sorted(set(base_review_counts) | set(expanded_review_counts)):
            f.write(f"| {key} | {base_review_counts.get(key, 0)} | {expanded_review_counts.get(key, 0)} |\n")
        f.write("\n")

        f.write("## Main checks\n\n")
        f.write(
            f"- New accept_completion candidates: {len(new_accept)}; "
            f"large-expansion accepts among them: {len(new_accept_large)}.\n"
        )
        f.write(
            f"- New accept_completion with large-plane, conflict >= 0.18, or existing IoU < 0.30: "
            f"{len(new_accept_risky)}.\n"
        )
        f.write(
            "- The high-risk accept review list "
            f"`accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30` is "
            f"{expanded_review_counts.get('accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30', 0)} "
            f"for {args.expanded_label}.\n"
        )
        f.write(
            "- Rejections remain useful to inspect for missing reliable core, large-plane over-expansion, "
            "or generic large expansion without strong support.\n\n"
        )
        if changed_rows:
            f.write("## Action changes on shared candidates\n\n")
            for row in changed_rows:
                f.write(
                    "- "
                    f"{_format_candidate(row)}: "
                    f"{row['previous_recommended_action']} -> {row['recommended_action']}; "
                    f"{row['action_reason']}\n"
                )
            f.write("\n")

        f.write("## New accept_completion large-expansion candidates for visual review\n\n")
        if new_accept_large:
            for row in new_accept_large:
                f.write(f"- {_format_candidate(row)}\n")
        else:
            f.write("None.\n")
        f.write("\n## New reject_or_needs_mask3d_support candidates\n\n")
        if new_reject:
            for row in new_reject:
                f.write(f"- {_format_candidate(row)}: {row['action_reason']}\n")
        else:
            f.write("None.\n")
        f.write("\n## New manual_review large-expansion candidates\n\n")
        if new_manual_large:
            for row in new_manual_large:
                f.write(f"- {_format_candidate(row)}: {row['action_reason']}\n")
        else:
            f.write("None.\n")

    print(
        "[SUPERPOINT_EXPANSION] "
        f"base={len(base_rows)} expanded={len(expanded_rows)} new={len(new_rows)} "
        f"new_accept={len(new_accept)} new_accept_large={len(new_accept_large)} "
        f"new_accept_risky={len(new_accept_risky)} changed={len(changed_rows)} "
        f"output_dir={args.output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a superpoint action diagnostic expansion.")
    parser.add_argument("--base_actions_csv", required=True)
    parser.add_argument("--base_review_lists_json", required=True)
    parser.add_argument("--expanded_actions_csv", required=True)
    parser.add_argument("--expanded_review_lists_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="Superpoint action diagnostic comparison")
    parser.add_argument("--base_label", default="base")
    parser.add_argument("--expanded_label", default="expanded")
    parser.add_argument("--new_label", default="new in expanded")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
