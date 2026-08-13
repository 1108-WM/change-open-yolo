#!/usr/bin/env python3
"""GT-only diagnostic: join frozen no-GT action features to oracle outcomes.

This tool is deliberately downstream of ``build_mv3dis_boundary_action_feature_ledger``.
It never changes that ledger and never writes proposals, scores, classes, or a
selector.  Its labels are *local fixed-target geometry deltas* relative to the
unknown=no-op baseline, not AP labels and not inference supervision ready for
the same scenes.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FEATURE_GROUPS = {
    "geometry": (
        "boundary_point_count", "candidate_proposal_superpoint_count",
        "candidate_proposal_point_count", "candidate_core_superpoint_count",
        "candidate_core_point_count", "candidate_boundary_point_fraction_of_core",
        "direct_core_neighbor_count", "direct_core_contact_count_sum",
        "direct_core_contact_ratio_sum", "direct_core_contact_ratio_max",
        "contact_weighted_boundary_distance", "contact_weighted_normal_difference",
        "contact_weighted_color_difference",
    ),
    "relative_depth_affinity": (
        "affinity_neighbor_pair_count", "affinity_defined_pair_count",
        "affinity_defined_pair_ratio", "affinity_observed_mean",
        "affinity_observed_min", "affinity_observed_max",
        "affinity_mean_minus_best_other", "affinity_mean_minus_mean_other",
        "plan_complete_edge_evidence",
    ),
    "multiview": (
        "candidate_support_view_count", "candidate_observation_count",
        "candidate_node_count", "candidate_mean_node_quality",
        "candidate_mean_consensus_rate", "candidate_mean_supported_coverage",
    ),
}
DECISION_CONSTRAINT = (
    "GT labels are post-hoc diagnostics only. They must not select actions, "
    "thresholds, scores, classes, or a selector on these scenes."
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _action_key(row):
    return (
        str(row["scene_name"]), int(row["superpoint_id"]),
        int(row["candidate_owner_proposal_id"]),
    )


def _oracle_action_map(oracle_rows):
    result = {}
    for boundary in oracle_rows:
        scene_name, superpoint_id = str(boundary["scene_name"]), int(boundary["superpoint_id"])
        for action in boundary["action_results"]:
            if action.get("action_kind") != "assign_owner":
                continue
            owner = int(action["target_owner_proposal_id"])
            key = (scene_name, superpoint_id, owner)
            if key in result:
                raise ValueError(f"duplicate oracle action: {key}")
            result[key] = action
    return result


def _label(action):
    if not action.get("locally_feasible_no_empty_proposal"):
        return "infeasible"
    deltas = [
        float(row["fixed_gt_iou_delta"])
        for row in action.get("proposal_outcomes", [])
        if bool(row.get("fixed_gt_available"))
    ]
    total = float(sum(deltas))
    if not deltas:
        state = "no_fixed_gt_target"
    elif total > 1e-12:
        state = "beneficial"
    elif total < -1e-12:
        state = "harmful"
    else:
        state = "neutral"
    return state, total, len(deltas)


def join_feature_labels(feature_rows, oracle_rows):
    """Immutable feature/GT join; returns newly allocated diagnostic rows."""
    oracle_by_key = _oracle_action_map(oracle_rows)
    output = []
    seen = set()
    for feature in feature_rows:
        if feature.get("ground_truth_usage") != "none":
            raise ValueError("source feature row is not a no-GT row")
        key = _action_key(feature)
        if key in seen:
            raise ValueError(f"duplicate feature action: {key}")
        seen.add(key)
        action = oracle_by_key.get(key)
        if action is None:
            raise ValueError(f"missing oracle action for feature row: {key}")
        label = _label(action)
        if label == "infeasible":
            state, total, count = "infeasible", None, 0
        else:
            state, total, count = label
        row = dict(feature)
        row.update({
            "gt_only_local_fixed_target_label": state,
            "gt_only_local_fixed_target_iou_delta_sum": total,
            "gt_only_fixed_target_proposal_count": int(count),
            "gt_only_fixed_target_improved_proposal_count": int(action.get("fixed_gt_iou_improved_proposal_count", 0)),
            "gt_only_fixed_target_declined_proposal_count": int(action.get("fixed_gt_iou_declined_proposal_count", 0)),
            "gt_only_iou25_upcross_proposal_count": int(action.get("fixed_gt_iou25_upcross_proposal_count", 0)),
            "gt_only_iou25_downcross_proposal_count": int(action.get("fixed_gt_iou25_downcross_proposal_count", 0)),
            "gt_only_iou50_upcross_proposal_count": int(action.get("fixed_gt_iou50_upcross_proposal_count", 0)),
            "gt_only_iou50_downcross_proposal_count": int(action.get("fixed_gt_iou50_downcross_proposal_count", 0)),
            "ground_truth_usage": "post-hoc local fixed-target action outcome label only",
            "decision_constraint": DECISION_CONSTRAINT,
        })
        output.append(row)
    if set(oracle_by_key) != seen:
        missing = len(set(oracle_by_key) - seen)
        raise ValueError(f"oracle has {missing} assign actions missing frozen features")
    return output


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "p10": float(np.quantile(values, .1)),
        "p90": float(np.quantile(values, .9)),
    }


def _as_number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def _auc_high_beneficial(beneficial, harmful):
    if not beneficial or not harmful:
        return None
    left, right = np.asarray(beneficial), np.asarray(harmful)
    return float(((left[:, None] > right[None, :]).sum() + .5 * (left[:, None] == right[None, :]).sum()) / (len(left) * len(right)))


def summarize(rows):
    labels = Counter(row["gt_only_local_fixed_target_label"] for row in rows)
    features = {}
    for group, names in FEATURE_GROUPS.items():
        for name in names:
            beneficial = [_as_number(row.get(name)) for row in rows if row["gt_only_local_fixed_target_label"] == "beneficial"]
            harmful = [_as_number(row.get(name)) for row in rows if row["gt_only_local_fixed_target_label"] == "harmful"]
            beneficial = [value for value in beneficial if value is not None]
            harmful = [value for value in harmful if value is not None]
            auc = _auc_high_beneficial(beneficial, harmful)
            features[name] = {
                "group": group,
                "beneficial": _stats(beneficial),
                "harmful": _stats(harmful),
                "high_value_toward_beneficial_auc": auc,
                "rank_effect_high_toward_beneficial": None if auc is None else float(2 * auc - 1),
                "descriptive_direction": (
                    None if auc is None else "high_toward_beneficial" if auc > .5 else
                    "low_toward_beneficial" if auc < .5 else "no_monotonic_direction"
                ),
            }
    return {
        "action_label_counts": dict(sorted(labels.items())),
        "feature_separability": features,
        "local_delta_summary": _stats([
            row["gt_only_local_fixed_target_iou_delta_sum"] for row in rows
            if row["gt_only_local_fixed_target_iou_delta_sum"] is not None
        ]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
    parser.add_argument("--oracle-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in ("feature_ledger_root", "oracle_ledger", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit("output directory already exists")
    summary_path = args.feature_ledger_root / "summary.json"
    if not summary_path.is_file():
        raise SystemExit("feature ledger summary is missing")
    feature_summary = json.loads(summary_path.read_text())
    if feature_summary.get("ground_truth_usage") != "none":
        raise ValueError("feature ledger is not no-GT")
    features = []
    for path in sorted(args.feature_ledger_root.glob("scene*/candidate_action_no_gt_features.jsonl")):
        features.extend(_read_jsonl(path))
    oracle_rows = _read_jsonl(args.oracle_ledger)
    rows = join_feature_labels(features, oracle_rows)
    payload = {
        "diagnostic_type": "GT-only separability of frozen no-GT boundary action features",
        "label_definition": "sum of affected proposals' fixed-source-GT IoU deltas for one candidate-owner action versus unknown=no-op",
        "unknown_definition": "no-op; retain exact current owner memberships",
        "proposal_materialization_applied": False,
        "ap_computed": False,
        "decision_constraint": DECISION_CONSTRAINT,
        "source_feature_ledger": str(args.feature_ledger_root),
        "source_oracle_ledger": str(args.oracle_ledger),
        "feature_row_count": len(rows),
        **summarize(rows),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "labeled_candidate_action_rows.jsonl", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
