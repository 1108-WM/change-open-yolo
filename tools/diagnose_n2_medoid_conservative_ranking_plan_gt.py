#!/usr/bin/env python3
"""GT-only audit of a pre-registered frozen N2 ranking/abstention plan."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    "GT-only audit of a pre-registered no-GT N2 ranking plan; it reports coverage "
    "only and must not modify selection, masks, inference scores, or mAP."
)
BUDGETS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
LABELS = ("system_increment_oracle_iou25", "system_increment_oracle_iou50")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def scenes(path):
    result = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or duplicated")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in ("scene_list", "plan_root", "oracle_root", "output_root"):
        setattr(args, name, resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("output root is non-empty")
    all_rows = []
    selected_scenes = scenes(args.scene_list)
    for scene_index, scene in enumerate(selected_scenes):
        plan = {int(row["candidate_id"]): row for row in (json.loads(line) for line in (args.plan_root / scene / "conservative_ranking_abstention_plan.jsonl").read_text().splitlines() if line)}
        oracle = {int(row["candidate_id"]): row for row in (json.loads(line) for line in (args.oracle_root / scene / "candidate_oracle_gt.jsonl").read_text().splitlines() if line)}
        if set(plan) != set(oracle):
            raise ValueError(f"{scene}: plan/oracle candidate IDs differ")
        for candidate_id in sorted(plan):
            row = {
                "scene_name": scene, "scene_fold": scene_index % args.fold_count,
                "candidate_id": candidate_id, "component_representative": bool(plan[candidate_id]["component_representative"]),
                "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False,
            }
            for budget in BUDGETS:
                key = f"diagnostic_select_top_{str(budget).replace('.', '_')}"
                row[key] = bool(plan[candidate_id][key])
            row["system_increment_oracle_iou25"] = bool(oracle[candidate_id]["selected_by_component_system_oracle_iou25"])
            row["system_increment_oracle_iou50"] = bool(oracle[candidate_id]["selected_by_component_system_oracle_iou50"])
            all_rows.append(row)
    metrics = []
    for label in LABELS:
        for budget in BUDGETS:
            key = f"diagnostic_select_top_{str(budget).replace('.', '_')}"
            for fold in range(args.fold_count):
                rows = [row for row in all_rows if row["scene_fold"] == fold]
                selected = [row for row in rows if row[key]]
                positives = [row for row in rows if row[label]]
                captured = [row for row in selected if row[label]]
                metrics.append({
                    "label": label, "budget_fraction_per_scene": budget, "held_out_fold": fold,
                    "selected_candidate_count": len(selected), "oracle_positive_candidate_count": len(positives),
                    "captured_oracle_positive_count": len(captured),
                    "oracle_positive_recall": len(captured) / max(1, len(positives)),
                    "oracle_positive_precision": len(captured) / max(1, len(selected)),
                    "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False,
                })
    summaries = []
    for label in LABELS:
        for budget in BUDGETS:
            rows = [row for row in metrics if row["label"] == label and row["budget_fraction_per_scene"] == budget]
            summaries.append({
                "label": label, "budget_fraction_per_scene": budget,
                "held_out_fold_mean_recall": sum(row["oracle_positive_recall"] for row in rows) / len(rows),
                "held_out_fold_min_recall": min(row["oracle_positive_recall"] for row in rows),
                "held_out_fold_mean_precision": sum(row["oracle_positive_precision"] for row in rows) / len(rows),
                "held_out_fold_min_precision": min(row["oracle_positive_precision"] for row in rows),
                "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False,
            })
    args.output_root.mkdir(parents=True)
    with (args.output_root / "candidate_plan_oracle_join_gt.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_root / "fold_metrics_gt.jsonl").open("w") as handle:
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_root / "summary_metrics_gt.jsonl").open("w") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    root = {
        "diagnostic_type": "GT-only conservative N2 ranking/abstention plan audit",
        "decision_constraint": CONTRACT, "scene_count": len(selected_scenes), "candidate_count": len(all_rows),
        "fold_count": args.fold_count, "budget_fractions": BUDGETS,
        "proposal_materialization_applied": False, "ap_computed": False,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
