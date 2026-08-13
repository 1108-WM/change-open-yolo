#!/usr/bin/env python3
"""Summarize isolated Z6d AP workers against the frozen current control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("improvement_gated_semantic_only", "improvement_gated_semantic_plus_dino")
METRICS = ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _delta(current: dict, control: dict) -> dict:
    return {name: float(current[name] - control[name]) for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-root", type=Path,
        default=Path("docs/diagnostics/z6d_nested_improvement_gate_oof_ap_workers_official100_20260812"),
    )
    parser.add_argument(
        "--control-summary", type=Path,
        default=Path("docs/diagnostics/z6c_candidate_selector_oof_ap_official100_20260812/summary.json"),
    )
    parser.add_argument(
        "--gate-summary", type=Path,
        default=Path("docs/diagnostics/z6d_nested_improvement_gate_oof_official100_20260812/summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/diagnostics/z6d_nested_improvement_gate_oof_ap_official100_20260812"),
    )
    args = parser.parse_args()
    for name in ("worker_root", "control_summary", "gate_summary", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    control_payload = json.loads(args.control_summary.read_text())
    control = control_payload["systems"]["current_control"]
    control_folds = {
        int(row["fold_index"]): row["systems"]["current_control"]
        for row in control_payload["folds"]
    }
    systems = {}
    folds = []
    for system in SYSTEMS:
        result = json.loads((args.worker_root / f"{system}_full" / "result.json").read_text())
        systems[system] = result["metrics"]
    for fold_index in range(5):
        fold_systems = {}
        fold_deltas = {}
        for system in SYSTEMS:
            result = json.loads(
                (args.worker_root / f"{system}_fold{fold_index}" / "result.json").read_text()
            )
            fold_systems[system] = result["metrics"]
            fold_deltas[system] = _delta(result["metrics"], control_folds[fold_index])
        folds.append({
            "fold_index": fold_index,
            "current_control": control_folds[fold_index],
            "systems": fold_systems,
            "deltas_vs_current_control": fold_deltas,
        })
    gate_summary = json.loads(args.gate_summary.read_text())
    payload = {
        "diagnostic_type": "official100 nested-cross-fitted Z6d binary improvement gate AP",
        "scene_count": 100,
        "current_control": control,
        "systems": systems,
        "deltas_vs_current_control": {
            system: _delta(systems[system], control) for system in SYSTEMS
        },
        "folds": folds,
        "positive_fold_counts_main_ap": {
            system: sum(row["deltas_vs_current_control"][system]["ap"] > 0 for row in folds)
            for system in SYSTEMS
        },
        "gate_state_counts": gate_summary["gate_state_counts"],
        "acceptance_contract": gate_summary["acceptance_contract"],
        "decision": "terminate automatic selector/gate branch; neither system exceeds control or has stable positive folds",
        "ground_truth_usage": "evaluation_only_after_nested_scene_isolated_oof_training",
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
