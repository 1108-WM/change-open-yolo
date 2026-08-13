#!/usr/bin/env python3
"""Audit GT transition states for frozen Z6c nested-gate OOF decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("gated_semantic_only", "gated_semantic_plus_dino")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()),
        "p01": float(np.quantile(array, 0.01)), "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)), "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("docs/diagnostics/z6c_candidate_selector_dataset_official100_20260812"),
    )
    parser.add_argument(
        "--gate-root", type=Path,
        default=Path("docs/diagnostics/z6c_nested_abstain_gate_oof_official100_20260812"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/diagnostics/z6c_nested_abstain_gate_transition_audit_official100_20260812"),
    )
    args = parser.parse_args()
    for name in ("dataset_root", "gate_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    option_targets = defaultdict(dict)
    geometry_quality = {}
    for row in _read_jsonl(args.dataset_root / "rows.jsonl"):
        key = (str(row["scene_name"]), int(row["prediction_index"]))
        option_targets[key][int(row["option_class_index"])] = float(row["label_option_ap_quality"])
        geometry_quality[key] = float(row["label_best_geometry_iou"])

    counts = {system: Counter() for system in SYSTEMS}
    deltas = {system: defaultdict(list) for system in SYSTEMS}
    by_fold = {system: defaultdict(Counter) for system in SYSTEMS}
    manifest = json.loads(
        (PROJECT_ROOT / "output/train_candidate_quality_oof_official100_v2/split_manifest.json").read_text()
    )
    fold_by_scene = {
        str(scene): int(fold["fold_index"])
        for fold in manifest["folds"] for scene in fold["validation_scenes"]
    }

    for row in _read_jsonl(args.gate_root / "oof_selections.jsonl"):
        key = (str(row["scene_name"]), int(row["prediction_index"]))
        current_class = int(row["current_class_index"])
        current_target = option_targets[key].get(current_class, 0.0)
        for system in SYSTEMS:
            decision = row["selectors"][system]
            proposed_class = int(decision["proposed_class_index"])
            proposed_target = option_targets[key].get(proposed_class, 0.0)
            true_delta = proposed_target - current_target
            transition = "improve" if true_delta > 0 else "harm" if true_delta < 0 else "neutral"
            accepted = bool(decision["accepted"] and proposed_class != current_class)
            counts[system]["prediction_count"] += 1
            counts[system][transition] += 1
            counts[system][f"{transition}__{'accepted' if accepted else 'rejected'}"] += 1
            counts[system]["accepted"] += int(accepted)
            counts[system]["accepted_true_positive"] += int(accepted and transition == "improve")
            counts[system]["accepted_harm"] += int(accepted and transition == "harm")
            counts[system]["accepted_neutral"] += int(accepted and transition == "neutral")
            by_fold[system][fold_by_scene[key[0]]][transition] += 1
            by_fold[system][fold_by_scene[key[0]]][f"{transition}__accepted"] += int(accepted)
            predicted = decision["gate_predicted_delta"]
            if predicted is not None:
                deltas[system][transition].append(float(predicted))
                deltas[system]["all_proposals"].append(float(predicted))
                if geometry_quality[key] >= 0.5:
                    deltas[system][f"{transition}__tp50_geometry"].append(float(predicted))

    systems = {}
    for system in SYSTEMS:
        c = counts[system]
        systems[system] = {
            "counts": dict(c),
            "accepted_fraction": c["accepted"] / c["prediction_count"],
            "accepted_precision_improve": c["accepted_true_positive"] / max(1, c["accepted"]),
            "improve_recall": c["improve__accepted"] / max(1, c["improve"]),
            "harm_acceptance_rate": c["harm__accepted"] / max(1, c["harm"]),
            "neutral_acceptance_rate": c["neutral__accepted"] / max(1, c["neutral"]),
            "predicted_delta_quantiles": {
                state: _quantiles(values) for state, values in sorted(deltas[system].items())
            },
            "fold_transition_counts": {
                str(index): dict(by_fold[system][index]) for index in range(5)
            },
        }
    payload = {
        "diagnostic_type": "official100 Z6c nested-gate GT transition audit",
        "systems": systems,
        "transition_contract": "improve iff proposed AP-quality target exceeds current; harm iff lower; otherwise neutral",
        "ground_truth_usage": "evaluation_and_failure_analysis_only",
        "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False,
        "inference_plan_written": False, "safety60_read": False, "even48_read": False,
        "test60_read": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
