import argparse
import csv
import os
import os.path as osp
from collections import Counter, defaultdict


FEATURE_FIELDS = [
    "existing_mask_iou",
    "largest_cc_to_point_ratio",
    "largest_cc_covered_by_point_ratio",
    "point_covered_by_largest_cc_ratio",
    "conflict_overlap",
]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _candidate_key(row):
    return row["scene_name"], str(row["candidate_id"])


def _mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _join_actions_and_labels(actions, labels):
    action_by_key = {_candidate_key(row): row for row in actions}
    joined = []
    missing = []
    for label_row in labels:
        key = _candidate_key(label_row)
        action_row = action_by_key.get(key)
        if action_row is None:
            missing.append(label_row)
            continue
        row = dict(action_row)
        row.update(
            {
                "group": label_row["group"],
                "visual_label": label_row["visual_label"],
                "short_note": label_row["short_note"],
            }
        )
        joined.append(row)
    return joined, missing


def _write_cross_table(path, rows):
    labels = sorted({row["visual_label"] for row in rows})
    actions = sorted({row["recommended_action"] for row in rows})
    counts = Counter((row["recommended_action"], row["visual_label"]) for row in rows)
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["recommended_action", *labels, "total"])
        for action in actions:
            values = [counts[(action, label)] for label in labels]
            writer.writerow([action, *values, sum(values)])
        totals = [sum(counts[(action, label)] for action in actions) for label in labels]
        writer.writerow(["total", *totals, sum(totals)])


def _write_feature_summary(path, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["visual_label"]].append(row)
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    fields = ["visual_label", "candidate_count"]
    for feature in FEATURE_FIELDS:
        fields.extend([f"{feature}_min", f"{feature}_mean", f"{feature}_max"])
    fields.extend(["large_plane_count", "small_plane_count"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for label in sorted(grouped):
            label_rows = grouped[label]
            output = {
                "visual_label": label,
                "candidate_count": len(label_rows),
                "large_plane_count": sum(row["is_large_plane_class"] == "True" for row in label_rows),
                "small_plane_count": sum(row["is_small_plane_class"] == "True" for row in label_rows),
            }
            for feature in FEATURE_FIELDS:
                values = [_safe_float(row[feature]) for row in label_rows]
                output[f"{feature}_min"] = min(values)
                output[f"{feature}_mean"] = _mean(values)
                output[f"{feature}_max"] = max(values)
            writer.writerow(output)


def _write_rule_notes(path, rows, missing):
    counts = Counter((row["recommended_action"], row["visual_label"]) for row in rows)
    manual_accept = [
        row for row in rows
        if row["recommended_action"] == "manual_review" and row["visual_label"] == "visual_accept"
    ]
    manual_uncertain = [
        row for row in rows
        if row["recommended_action"] == "manual_review" and row["visual_label"] == "uncertain"
    ]
    reject_accept = [
        row
        for row in rows
        if row["recommended_action"] == "reject_or_needs_mask3d_support"
        and row["visual_label"] == "visual_accept"
    ]
    accept_risk = [
        row
        for row in rows
        if row["recommended_action"] == "accept_completion"
        and row["visual_label"] in {"visual_reject", "uncertain"}
    ]
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("# v5 rule notes\n\n")
        f.write("Inputs are v4 actions plus 18 visual labels from the v4 review images.\n\n")
        f.write("## Label alignment\n\n")
        for (action, label), count in sorted(counts.items()):
            f.write(f"- {action} -> {label}: {count}\n")
        f.write("\n")
        f.write(f"- Missing labels in actions.csv: {len(missing)}\n")
        f.write(f"- accept_completion visual_reject/uncertain: {len(accept_risk)}\n")
        f.write(f"- reject_or_needs_mask3d_support visual_accept: {len(reject_accept)}\n")
        f.write(f"- manual_review visual_accept: {len(manual_accept)}\n")
        f.write(f"- manual_review uncertain: {len(manual_uncertain)}\n\n")
        f.write("## v5 interpretation\n\n")
        f.write("- Keep large planar classes conservative even when Mask3D IoU is high.\n")
        f.write("- Do not promote office chair or sink large expansions from manual review.\n")
        f.write("- Promote only strongly supported large non-plane completions with low conflict.\n")
        f.write("- Promote small-plane picture completion only with very strong Mask3D support.\n")
        f.write("- Leave rejected and keep_core_only cases unchanged for this 10-scene diagnostic.\n\n")
        f.write("## visual_accept candidates used by v5\n\n")
        for row in manual_accept:
            f.write(
                "- "
                f"{row['scene_name']} candidate{int(row['candidate_id']):04d} "
                f"{row['class_name']}: ratio={_safe_float(row['largest_cc_to_point_ratio']):.2f}, "
                f"IoU={_safe_float(row['existing_mask_iou']):.2f}, "
                f"point_coverage={_safe_float(row['point_covered_by_largest_cc_ratio']):.2f}, "
                f"conflict={_safe_float(row['conflict_overlap']):.2f}\n"
            )
        f.write("\n## remaining uncertain candidates\n\n")
        for row in manual_uncertain:
            f.write(
                "- "
                f"{row['scene_name']} candidate{int(row['candidate_id']):04d} "
                f"{row['class_name']}: {row['short_note']}\n"
            )


def analyze(args):
    actions = _load_csv(args.actions_csv)
    labels = _load_csv(args.visual_notes_csv)
    rows, missing = _join_actions_and_labels(actions, labels)
    _write_cross_table(osp.join(args.output_dir, "label_cross_table.csv"), rows)
    _write_feature_summary(osp.join(args.output_dir, "feature_summary_by_visual_label.csv"), rows)
    _write_rule_notes(osp.join(args.output_dir, "v5_rule_notes.md"), rows, missing)
    print(
        "[VISUAL_LABEL_ALIGNMENT] "
        f"labels={len(labels)} matched={len(rows)} missing={len(missing)} "
        f"output_dir={args.output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Align v4 diagnostic actions with visual review labels.")
    parser.add_argument("--actions_csv", required=True)
    parser.add_argument("--visual_notes_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
