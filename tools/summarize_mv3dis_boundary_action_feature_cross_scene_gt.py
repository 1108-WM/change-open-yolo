#!/usr/bin/env python3
"""Compare two GT-only boundary-action feature-separability summaries.

The input summaries are already post-hoc diagnostics.  This tool only reports
whether each feature's monotonic direction agrees across the fixed safety60
development set and the already-used, scene-disjoint even48 robustness set.
It does not fit a model or define an inference threshold.
"""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISION_CONSTRAINT = (
    "This is a post-hoc cross-scene diagnostic only; it cannot choose an action, "
    "threshold, score, class, or selector."
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sign(value):
    if value is None:
        return None
    return 1 if value > 0 else -1 if value < 0 else 0


def compare(left, right):
    left_features = left["feature_separability"]
    right_features = right["feature_separability"]
    if set(left_features) != set(right_features):
        raise ValueError("feature sets differ")
    rows = []
    for name in sorted(left_features):
        a, b = left_features[name], right_features[name]
        if a["group"] != b["group"]:
            raise ValueError(f"feature group differs: {name}")
        effect_a = a["rank_effect_high_toward_beneficial"]
        effect_b = b["rank_effect_high_toward_beneficial"]
        rows.append({
            "feature": name,
            "group": a["group"],
            "safety60_auc_high_toward_beneficial": a["high_value_toward_beneficial_auc"],
            "even48_auc_high_toward_beneficial": b["high_value_toward_beneficial_auc"],
            "safety60_rank_effect": effect_a,
            "even48_rank_effect": effect_b,
            "same_monotonic_direction": _sign(effect_a) == _sign(effect_b),
            "safety60_direction": a["descriptive_direction"],
            "even48_direction": b["descriptive_direction"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety60-summary", type=Path, required=True)
    parser.add_argument("--even48-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in ("safety60_summary", "even48_summary", "output"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output.exists():
        raise SystemExit("output already exists")
    safety60 = json.loads(args.safety60_summary.read_text())
    even48 = json.loads(args.even48_summary.read_text())
    rows = compare(safety60, even48)
    payload = {
        "diagnostic_type": "GT-only cross-scene direction comparison for frozen no-GT boundary-action features",
        "decision_constraint": DECISION_CONSTRAINT,
        "sets": {
            "safety60": "retrospective development set",
            "even48": "already-used scene-disjoint robustness recheck; not a new test set",
        },
        "label_definition": safety60["label_definition"],
        "label_counts": {
            "safety60": safety60["action_label_counts"],
            "even48": even48["action_label_counts"],
        },
        "local_delta_summary": {
            "safety60": safety60["local_delta_summary"],
            "even48": even48["local_delta_summary"],
        },
        "feature_count": len(rows),
        "same_monotonic_direction_count": sum(row["same_monotonic_direction"] for row in rows),
        "features": rows,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "feature_count": payload["feature_count"],
        "same_monotonic_direction_count": payload["same_monotonic_direction_count"],
        "output": str(args.output),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
