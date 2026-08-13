#!/usr/bin/env python3
"""Train nested-cross-fitted Z6d binary improvement gates without threshold scans."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_z6c_nested_abstain_gate_oof import (  # noqa: E402
    GROUPS, _base_model, _choose, _gate_feature, _groups, _read_jsonl, _resolve,
)


def _make_gate(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=100, max_leaf_nodes=5,
        min_samples_leaf=50, l2_regularization=1.0, random_state=seed,
    )


def run(args: argparse.Namespace) -> dict:
    rows = _read_jsonl(args.dataset_root / "rows.jsonl")
    with np.load(args.dataset_root / "dataset.npz") as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        target = np.asarray(payload["target"], dtype=np.float32)
        weight = np.asarray(payload["sample_weight"], dtype=np.float64)
    outer_scores = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in GROUPS}
    for row in _read_jsonl(args.selector_root / "oof_candidate_scores.jsonl"):
        index = int(row["row_index"])
        for name in GROUPS:
            outer_scores[name][index] = float(row["oof_candidate_quality"][name])
    if any(np.any(~np.isfinite(values)) for values in outer_scores.values()):
        raise ValueError("outer selector OOF scores are incomplete")
    split = json.loads(args.split_manifest.read_text())
    gated_outputs = {name: {} for name in GROUPS}
    fold_summaries = []

    outer_specs = sorted(split["folds"], key=lambda row: int(row["fold_index"]))
    if args.outer_fold_index is not None:
        outer_specs = [row for row in outer_specs if int(row["fold_index"]) == args.outer_fold_index]
        if len(outer_specs) != 1:
            raise ValueError(f"unknown outer fold {args.outer_fold_index}")
    for outer in outer_specs:
        outer_index = int(outer["fold_index"])
        outer_train_scenes = set(outer["train_scenes"])
        outer_validation_scenes = set(outer["validation_scenes"])
        outer_train = np.asarray([
            index for index, row in enumerate(rows) if row["scene_name"] in outer_train_scenes
        ], dtype=np.int64)
        outer_validation = np.asarray([
            index for index, row in enumerate(rows) if row["scene_name"] in outer_validation_scenes
        ], dtype=np.int64)
        fold_payload = {"fold_index": outer_index, "models": {}}
        for group_name, columns in GROUPS.items():
            inner_scores = np.full(len(rows), np.nan, dtype=np.float32)
            inner_fold_count = 0
            for inner in sorted(split["folds"], key=lambda row: int(row["fold_index"])):
                inner_validation_scenes = set(inner["validation_scenes"]) & outer_train_scenes
                if not inner_validation_scenes:
                    continue
                inner_train_scenes = outer_train_scenes - inner_validation_scenes
                inner_train = np.asarray([
                    index for index in outer_train if rows[index]["scene_name"] in inner_train_scenes
                ], dtype=np.int64)
                inner_validation = np.asarray([
                    index for index in outer_train if rows[index]["scene_name"] in inner_validation_scenes
                ], dtype=np.int64)
                model = _base_model(args.random_seed + outer_index * 100 + int(inner["fold_index"]))
                model.fit(
                    features[inner_train][:, columns], target[inner_train],
                    sample_weight=weight[inner_train],
                )
                inner_scores[inner_validation] = np.clip(
                    model.predict(features[inner_validation][:, columns]), 0.0, 1.0
                )
                inner_fold_count += 1
                print(
                    f"[Z6d nested improvement] outer={outer_index} model={group_name} "
                    f"inner={inner['fold_index']} complete", flush=True,
                )
                del model
                gc.collect()
            if np.any(~np.isfinite(inner_scores[outer_train])):
                raise ValueError(f"outer fold {outer_index} {group_name}: incomplete inner OOF scores")

            gate_features, gate_targets, gate_weights = [], [], []
            for indexes in _groups(rows, outer_train):
                selected, current, margin = _choose(rows, indexes, inner_scores)
                if current is not None and selected == current:
                    continue
                current_target = float(target[current]) if current is not None else 0.0
                gate_features.append(_gate_feature(features, inner_scores, selected, current, margin))
                gate_targets.append(int(float(target[selected]) > current_target))
                gate_weights.append(float(sum(weight[index] for index in indexes)))
            gate_features = np.asarray(gate_features, dtype=np.float32)
            gate_targets = np.asarray(gate_targets, dtype=np.int8)
            gate_weights = np.asarray(gate_weights, dtype=np.float64)
            gate = _make_gate(args.random_seed + 1000 + outer_index)
            gate.fit(gate_features, gate_targets, sample_weight=gate_weights)
            print(
                f"[Z6d nested improvement] outer={outer_index} model={group_name} gate fit complete",
                flush=True,
            )

            validation_proposals = []
            pending_gate_features = []
            for indexes in _groups(rows, outer_validation):
                selected, current, margin = _choose(rows, indexes, outer_scores[group_name])
                first = rows[int(indexes[0])]
                validation_proposals.append((indexes, selected, current, margin, first))
                if current is None or selected != current:
                    pending_gate_features.append(
                        _gate_feature(features, outer_scores[group_name], selected, current, margin)
                    )
            pending_probabilities = iter(
                gate.predict_proba(np.asarray(pending_gate_features, dtype=np.float32))[:, 1].tolist()
                if pending_gate_features else []
            )

            counts = Counter()
            for indexes, selected, current, margin, first in validation_proposals:
                if current is not None and selected == current:
                    final = selected
                    improve_probability = None
                    accepted = False
                    state = "base_kept_current"
                else:
                    improve_probability = float(next(pending_probabilities))
                    accepted = improve_probability > 0.5
                    final = selected if accepted or current is None else current
                    state = "accepted_new_class" if accepted else (
                        "abstained_keep_current" if current is not None else "accepted_no_valid_current"
                    )
                key = (str(first["scene_name"]), int(first["prediction_index"]))
                gated_outputs[group_name][key] = {
                    "selected_class_index": int(rows[final]["option_class_index"]),
                    "proposed_class_index": int(rows[selected]["option_class_index"]),
                    "kept_current": bool(current is not None and final == current),
                    "accepted": bool(accepted or current is None),
                    "gate_improve_probability": improve_probability,
                    "gate_state": state,
                    "selector_margin": margin,
                }
                counts[state] += 1
            fold_payload["models"][group_name] = {
                "inner_fold_count": inner_fold_count,
                "gate_train_example_count": len(gate_targets),
                "gate_target_positive_count": int(gate_targets.sum()),
                "gate_target_negative_count": int((gate_targets == 0).sum()),
                "validation_gate_state_counts": dict(counts),
            }
            del gate, gate_features, gate_targets, gate_weights, inner_scores
            gc.collect()
        fold_summaries.append(fold_payload)
        print(f"[Z6d nested improvement] outer fold {outer_index} complete", flush=True)

    prediction_keys, seen, first_row_by_key = [], set(), {}
    for row in rows:
        key = (str(row["scene_name"]), int(row["prediction_index"]))
        if key not in seen:
            prediction_keys.append(key); seen.add(key); first_row_by_key[key] = row
    expected_output_keys = {
        key for key in prediction_keys
        if args.outer_fold_index is None or any(
            key[0] in set(spec["validation_scenes"]) for spec in outer_specs
        )
    }
    if any(set(gated_outputs[name]) != expected_output_keys for name in GROUPS):
        raise ValueError("nested gate outputs do not cover the requested selector predictions")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "oof_selections.jsonl").open("w") as handle:
        for key in prediction_keys:
            if key not in expected_output_keys:
                continue
            first = first_row_by_key[key]
            handle.write(json.dumps({
                "scene_name": key[0], "prediction_index": key[1],
                "candidate_source": str(first["candidate_source"]),
                "candidate_id": int(first["candidate_id"]),
                "semantic_evidence_node_key": str(first["semantic_evidence_node_key"]),
                "current_class_index": int(first["current_class_index"]),
                "current_class_valid": bool(first["current_class_valid"]),
                "selectors": {
                    f"improvement_gated_{name}": gated_outputs[name][key] for name in GROUPS
                },
            }, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "diagnostic_type": "official100 nested-cross-fitted Z6d binary improvement gate",
        "scene_count": len({key[0] for key in expected_output_keys}),
        "prediction_count": len(expected_output_keys), "fold_count": len(fold_summaries),
        "folds": fold_summaries,
        "gate_state_counts": {
            f"improvement_gated_{name}": dict(Counter(
                row["gate_state"] for row in gated_outputs[name].values()
            )) for name in GROUPS
        },
        "acceptance_contract": "accept proposed new class iff nested-OOF P(true AP-quality improvement) > 0.5",
        "negative_target_contract": "harm and wrong-to-wrong neutral proposals are both gate negatives",
        "base_selector_contract": "same frozen semantic-only and semantic+DINO selector models",
        "gate_model_contract": {
            "type": "HistGradientBoostingClassifier", "learning_rate": 0.05,
            "max_iter": 100, "max_leaf_nodes": 5, "min_samples_leaf": 50,
            "l2_regularization": 1.0, "random_seed": args.random_seed,
        },
        "class_id_is_feature": False, "class_name_is_feature": False,
        "ground_truth_usage": "official_train_supervision_only_with_nested_scene_isolated_oof",
        "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False,
        "inference_plan_written": False, "safety60_read": False, "even48_read": False,
        "test60_read": False,
        "params": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_dataset_official100_20260812"))
    parser.add_argument("--selector-root", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_oof_official100_20260812"))
    parser.add_argument("--split-manifest", type=Path, default=Path("output/train_candidate_quality_oof_official100_v2/split_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/diagnostics/z6d_nested_improvement_gate_oof_official100_20260812"))
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--outer-fold-index", type=int)
    args = parser.parse_args()
    for name in ("dataset_root", "selector_root", "split_manifest", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
