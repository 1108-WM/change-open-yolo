#!/usr/bin/env python3
"""Fit the frozen full-data semantic-only selector and OOF-trained improvement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_z6c_nested_abstain_gate_oof import _base_model, _choose, _gate_feature, _groups, _read_jsonl, _resolve  # noqa: E402
from tools.train_z6d_nested_improvement_gate_oof import _make_gate  # noqa: E402


COLUMNS = list(range(25))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_dataset_official100_20260812"))
    parser.add_argument("--selector-root", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_oof_official100_20260812"))
    parser.add_argument("--output-dir", type=Path, default=Path("pretrained/z6f_full_inference_bundle_official100_20260812"))
    parser.add_argument("--random-seed", type=int, default=20260812)
    args = parser.parse_args()
    for name in ("dataset_root", "selector_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists(): raise SystemExit(f"refusing to overwrite {args.output_dir}")

    rows = _read_jsonl(args.dataset_root / "rows.jsonl")
    with np.load(args.dataset_root / "dataset.npz") as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        target = np.asarray(payload["target"], dtype=np.float32)
        weight = np.asarray(payload["sample_weight"], dtype=np.float64)
    oof_scores = np.full(len(rows), np.nan, dtype=np.float32)
    for row in _read_jsonl(args.selector_root / "oof_candidate_scores.jsonl"):
        oof_scores[int(row["row_index"])] = float(row["oof_candidate_quality"]["semantic_only"])
    if np.any(~np.isfinite(oof_scores)): raise ValueError("semantic-only OOF scores incomplete")

    gate_features, gate_targets, gate_weights = [], [], []
    for indexes in _groups(rows, np.arange(len(rows), dtype=np.int64)):
        selected, current, margin = _choose(rows, indexes, oof_scores)
        if current is not None and selected == current:
            continue
        current_target = float(target[current]) if current is not None else 0.0
        gate_features.append(_gate_feature(features, oof_scores, selected, current, margin))
        gate_targets.append(int(float(target[selected]) > current_target))
        gate_weights.append(float(sum(weight[index] for index in indexes)))
    gate_features = np.asarray(gate_features, dtype=np.float32)
    gate_targets = np.asarray(gate_targets, dtype=np.int8)
    gate_weights = np.asarray(gate_weights, dtype=np.float64)
    gate = _make_gate(args.random_seed + 1000)
    gate.fit(gate_features, gate_targets, sample_weight=gate_weights)
    print("[Z6f full bundle] improvement gate fit complete", flush=True)

    selector = _base_model(args.random_seed)
    selector.fit(features[:, COLUMNS], target, sample_weight=weight)
    print("[Z6f full bundle] semantic-only selector fit complete", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    selector_path = args.output_dir / "semantic_only_selector.joblib"
    gate_path = args.output_dir / "binary_improvement_gate.joblib"
    joblib.dump(selector, selector_path)
    joblib.dump(gate, gate_path)
    schema = json.loads((args.dataset_root / "feature_schema.json").read_text())
    payload = {
        "bundle_type": "frozen Z6f full inference bundle",
        "selector": {
            "path": selector_path.name, "feature_indices": COLUMNS,
            "feature_names": [schema["feature_names"][index] for index in COLUMNS],
            "contract": "HistGradientBoostingRegressor full official100 option rows",
        },
        "gate": {
            "path": gate_path.name, "feature_count": int(gate_features.shape[1]),
            "training_example_count": len(gate_targets),
            "positive_count": int(gate_targets.sum()), "negative_count": int((gate_targets == 0).sum()),
            "contract": "HistGradientBoostingClassifier trained on scene-isolated OOF selector proposals; accept P(improve)>0.5",
        },
        "routing_contract": "Z6f frozen router and symmetric Qwen review from z6f_vlm_selector_ap_summary_official100_20260812",
        "sha256": {
            selector_path.name: hashlib.sha256(selector_path.read_bytes()).hexdigest(),
            gate_path.name: hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        },
        "class_id_is_feature": False, "class_name_is_feature": False,
        "ground_truth_usage": "official100 full-fit supervision only; OOF proposals for gate stacking",
        "safety60_read": False, "even48_read": False, "test60_read": False,
    }
    (args.output_dir / "bundle.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
