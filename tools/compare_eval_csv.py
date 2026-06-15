import argparse
import csv
import json
import os
from collections import Counter, defaultdict


def _read_eval_csv(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["class"]
            rows[name] = {
                "class_id": int(row["class id"]),
                "ap": float(row["ap"]),
                "ap50": float(row["ap50"]),
                "ap25": float(row["ap25"]),
            }
    return rows


def _read_report_counts(path):
    if not path:
        return {}
    with open(path) as f:
        payload = json.load(f)
    counts = defaultdict(Counter)
    for scene_report in payload.get("scene_reports", {}).values():
        for item in scene_report.get("applied", []):
            class_name = item.get("class_name")
            if not class_name:
                continue
            source = item.get("source_kind")
            if source is None:
                source_json = item.get("source_json", "")
                source = "sam_fused" if "sam_fused" in source_json else "bpr"
            counts[class_name][source] += 1
            counts[class_name]["total"] += 1
    return counts


def _format_float(value):
    return f"{value:.3f}"


def _write_csv(path, rows):
    fieldnames = [
        "class",
        "class_id",
        "baseline_ap",
        "baseline_ap50",
        "baseline_ap25",
        "bpr_ap",
        "bpr_ap50",
        "bpr_ap25",
        "target_ap",
        "target_ap50",
        "target_ap25",
        "target_minus_baseline_ap",
        "target_minus_baseline_ap50",
        "target_minus_baseline_ap25",
        "target_minus_bpr_ap",
        "target_minus_bpr_ap50",
        "target_minus_bpr_ap25",
        "applied_total",
        "applied_sam_fused",
        "applied_bpr",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, rows, metric, limit):
    key = f"target_minus_baseline_{metric}"
    bpr_key = f"target_minus_bpr_{metric}"
    ranked = sorted(rows, key=lambda item: item[key], reverse=True)
    with open(path, "w") as f:
        f.write("# Per-class evaluation comparison\n\n")
        f.write(f"Sorted by target minus baseline `{metric}`.\n\n")
        f.write("## Top gains over baseline\n\n")
        f.write("| class | target | baseline | delta base | delta BPR | SAM props | BPR props |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in ranked[:limit]:
            f.write(
                "| {class_name} | {target} | {baseline} | {delta_base} | {delta_bpr} | {sam} | {bpr} |\n".format(
                    class_name=row["class"],
                    target=_format_float(row[f"target_{metric}"]),
                    baseline=_format_float(row[f"baseline_{metric}"]),
                    delta_base=_format_float(row[key]),
                    delta_bpr=_format_float(row[bpr_key]),
                    sam=row["applied_sam_fused"],
                    bpr=row["applied_bpr"],
                )
            )
        f.write("\n## Largest drops versus baseline\n\n")
        f.write("| class | target | baseline | delta base | delta BPR | SAM props | BPR props |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in list(reversed(ranked))[:limit]:
            f.write(
                "| {class_name} | {target} | {baseline} | {delta_base} | {delta_bpr} | {sam} | {bpr} |\n".format(
                    class_name=row["class"],
                    target=_format_float(row[f"target_{metric}"]),
                    baseline=_format_float(row[f"baseline_{metric}"]),
                    delta_base=_format_float(row[key]),
                    delta_bpr=_format_float(row[bpr_key]),
                    sam=row["applied_sam_fused"],
                    bpr=row["applied_bpr"],
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--bpr_csv", required=True)
    parser.add_argument("--target_csv", required=True)
    parser.add_argument("--target_report", default=None)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--metric", default="ap50", choices=["ap", "ap50", "ap25"])
    parser.add_argument("--limit", default=12, type=int)
    args = parser.parse_args()

    baseline = _read_eval_csv(args.baseline_csv)
    bpr = _read_eval_csv(args.bpr_csv)
    target = _read_eval_csv(args.target_csv)
    counts = _read_report_counts(args.target_report)

    rows = []
    for class_name in sorted(target):
        base = baseline[class_name]
        bpr_row = bpr[class_name]
        target_row = target[class_name]
        count = counts.get(class_name, Counter())
        row = {
            "class": class_name,
            "class_id": target_row["class_id"],
            "baseline_ap": base["ap"],
            "baseline_ap50": base["ap50"],
            "baseline_ap25": base["ap25"],
            "bpr_ap": bpr_row["ap"],
            "bpr_ap50": bpr_row["ap50"],
            "bpr_ap25": bpr_row["ap25"],
            "target_ap": target_row["ap"],
            "target_ap50": target_row["ap50"],
            "target_ap25": target_row["ap25"],
            "target_minus_baseline_ap": target_row["ap"] - base["ap"],
            "target_minus_baseline_ap50": target_row["ap50"] - base["ap50"],
            "target_minus_baseline_ap25": target_row["ap25"] - base["ap25"],
            "target_minus_bpr_ap": target_row["ap"] - bpr_row["ap"],
            "target_minus_bpr_ap50": target_row["ap50"] - bpr_row["ap50"],
            "target_minus_bpr_ap25": target_row["ap25"] - bpr_row["ap25"],
            "applied_total": int(count.get("total", 0)),
            "applied_sam_fused": int(count.get("sam_fused", 0)),
            "applied_bpr": int(count.get("bpr", 0)),
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows, args.metric, args.limit)
    print(f"Saved comparison CSV to {args.output_csv}")
    print(f"Saved comparison markdown to {args.output_md}")


if __name__ == "__main__":
    main()
