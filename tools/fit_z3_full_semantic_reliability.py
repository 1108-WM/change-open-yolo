#!/usr/bin/env python3
"""Fit the frozen full-official100 Z3 C_joint semantic reliability model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = list(range(41))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(
        "docs/diagnostics/z3_semantic_reliability_dataset_official100_20260811"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "pretrained/z3_full_semantic_reliability_official100_20260812"
    ))
    parser.add_argument("--random-seed", type=int, default=20260811)
    args = parser.parse_args()
    args.dataset_root = _resolve(args.dataset_root)
    args.output_dir = _resolve(args.output_dir)
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    with np.load(args.dataset_root / "dataset.npz") as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        target = np.asarray(payload["label_ap_quality"], dtype=np.float32)
    rows = [
        json.loads(line) for line in (args.dataset_root / "rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if features.shape != (len(rows), 41):
        raise ValueError("unexpected frozen Z3 dataset shape")
    weights = np.ones(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        if str(row["candidate_source"]) == "native":
            weights[index] = 1.0 / max(1.0, float(np.expm1(features[index, 5])))
    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=120, max_leaf_nodes=7,
        min_samples_leaf=50, l2_regularization=1.0, random_state=args.random_seed,
    )
    model.fit(features[:, COLUMNS], target, sample_weight=weights)
    args.output_dir.mkdir(parents=True)
    model_path = args.output_dir / "c_joint_yolo_alpha.joblib"
    joblib.dump(model, model_path)
    schema = json.loads((args.dataset_root / "feature_schema.json").read_text())
    summary = {
        "model_type": "frozen full-official100 Z3 C_joint semantic reliability",
        "training_row_count": len(rows),
        "feature_indices": COLUMNS,
        "feature_names": schema["feature_names"],
        "model": {
            "type": "HistGradientBoostingRegressor", "learning_rate": 0.05,
            "max_iter": 120, "max_leaf_nodes": 7, "min_samples_leaf": 50,
            "l2_regularization": 1.0, "random_seed": args.random_seed,
        },
        "sample_weight_contract": "native=1/exact_geometry_bound_candidate_count; track=1; pair_union=1",
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "ground_truth_usage": "official100 full-fit supervision only",
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
