#!/usr/bin/env python3
"""Train fixed-split low-capacity Z6c within-candidate OOF selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_SHA256 = "aa657449965bc76164a1a1b77c7785aa705a0295eeed1307163b325f7233fe3e"
GROUPS = {
    "semantic_only": list(range(25)),
    "semantic_plus_dino": list(range(30)),
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prediction_groups(rows: list[dict]) -> list[np.ndarray]:
    groups = defaultdict(list)
    order = []
    for index, row in enumerate(rows):
        key = (str(row["scene_name"]), int(row["prediction_index"]))
        if key not in groups:
            order.append(key)
        groups[key].append(index)
    return [np.asarray(groups[key], dtype=np.int64) for key in order]


def _select_option(rows: list[dict], indexes: np.ndarray, predictions: np.ndarray) -> tuple[int, float]:
    """Max prediction with current-class then class-index deterministic ties."""
    ordered = sorted(
        (int(index) for index in indexes),
        key=lambda index: (
            -float(predictions[index]),
            -int(bool(rows[index]["option_is_current"])),
            int(rows[index]["option_class_index"]),
        ),
    )
    chosen = ordered[0]
    runner_up = float(predictions[ordered[1]]) if len(ordered) > 1 else 0.0
    return chosen, float(predictions[chosen] - runner_up)


def _selection_metrics(rows: list[dict], groups: list[np.ndarray], predictions: np.ndarray) -> dict:
    counts = Counter()
    for indexes in groups:
        chosen, _ = _select_option(rows, indexes, predictions)
        row = rows[chosen]
        eligible = float(row["label_best_geometry_iou"]) >= 0.5
        correct = bool(row["label_option_is_target_class"])
        changed = not bool(row["option_is_current"])
        counts["prediction_count"] += 1
        counts["selected_correct_count"] += int(correct)
        counts["changed_count"] += int(changed)
        counts["kept_current_count"] += int(not changed)
        counts["tp50_eligible_count"] += int(eligible)
        counts["tp50_selected_correct_count"] += int(eligible and correct)
        counts["changed_correct_count"] += int(changed and correct)
    return {
        **dict(counts),
        "kept_current_fraction": counts["kept_current_count"] / max(1, counts["prediction_count"]),
        "changed_fraction": counts["changed_count"] / max(1, counts["prediction_count"]),
        "tp50_selected_correct_fraction": counts["tp50_selected_correct_count"] / max(1, counts["tp50_eligible_count"]),
        "changed_correct_fraction": counts["changed_correct_count"] / max(1, counts["changed_count"]),
    }


def run(args: argparse.Namespace) -> dict:
    rows = _read_jsonl(args.dataset_root / "rows.jsonl")
    with np.load(args.dataset_root / "dataset.npz") as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        target = np.asarray(payload["target"], dtype=np.float32)
        tp50 = np.asarray(payload["tp50"], dtype=np.int8)
        weight = np.asarray(payload["sample_weight"], dtype=np.float64)
    schema = json.loads((args.dataset_root / "feature_schema.json").read_text())
    names = list(schema["feature_names"])
    if len(rows) != len(features) or features.shape[1] != 30 or len(names) != 30:
        raise ValueError("selector dataset dimensions disagree")
    split_sha = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError("frozen split SHA-256 mismatch")
    manifest = json.loads(args.split_manifest.read_text())
    scenes = {str(row["scene_name"]) for row in rows}
    occurrences = Counter(scene for fold in manifest["folds"] for scene in fold["validation_scenes"])
    if set(occurrences) != scenes or any(count != 1 for count in occurrences.values()):
        raise ValueError("split does not provide exactly one validation fold per scene")

    predictions = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in GROUPS}
    folds = []
    for spec in sorted(manifest["folds"], key=lambda row: int(row["fold_index"])):
        train_scenes = set(spec["train_scenes"])
        validation_scenes = set(spec["validation_scenes"])
        train = np.asarray([i for i, row in enumerate(rows) if row["scene_name"] in train_scenes], dtype=np.int64)
        validation = np.asarray([i for i, row in enumerate(rows) if row["scene_name"] in validation_scenes], dtype=np.int64)
        if not len(train) or not len(validation) or train_scenes & validation_scenes:
            raise ValueError("invalid OOF partition")
        fold_result = {
            "fold_index": int(spec["fold_index"]), "train_row_count": len(train),
            "validation_row_count": len(validation), "validation_scenes": sorted(validation_scenes),
            "models": {},
        }
        validation_groups = _prediction_groups([rows[index] for index in validation])
        # Convert group-local indexes back to the full row array.
        validation_groups = [validation[indexes] for indexes in validation_groups]
        for group_name, columns in GROUPS.items():
            model = HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=120, max_leaf_nodes=7,
                min_samples_leaf=50, l2_regularization=1.0,
                random_state=args.random_seed + int(spec["fold_index"]),
            )
            model.fit(features[train][:, columns], target[train], sample_weight=weight[train])
            values = np.clip(model.predict(features[validation][:, columns]), 0.0, 1.0)
            predictions[group_name][validation] = values
            fold_result["models"][group_name] = {
                "mae": float(mean_absolute_error(target[validation], values, sample_weight=weight[validation])),
                "tp50_roc_auc": float(roc_auc_score(tp50[validation], values, sample_weight=weight[validation])),
                "selection": _selection_metrics(rows, validation_groups, predictions[group_name]),
            }
        folds.append(fold_result)
        print(f"[Z6c selector OOF] fold {spec['fold_index']} complete", flush=True)
    if any(np.any(~np.isfinite(values)) for values in predictions.values()):
        raise AssertionError("not every selector row received an OOF prediction")

    groups = _prediction_groups(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "oof_candidate_scores.jsonl").open("w") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({
                **row,
                "oof_candidate_quality": {name: float(values[index]) for name, values in predictions.items()},
            }, ensure_ascii=False, sort_keys=True) + "\n")
    selected_rows = []
    for indexes in groups:
        first = rows[int(indexes[0])]
        output = {
            "scene_name": str(first["scene_name"]),
            "prediction_index": int(first["prediction_index"]),
            "candidate_source": str(first["candidate_source"]),
            "candidate_id": int(first["candidate_id"]),
            "semantic_evidence_node_key": str(first["semantic_evidence_node_key"]),
            "current_class_index": int(first["current_class_index"]),
            "current_class_valid": bool(first["current_class_valid"]),
            "selectors": {},
        }
        for name, values in predictions.items():
            chosen, margin = _select_option(rows, indexes, values)
            selected = rows[chosen]
            output["selectors"][name] = {
                "selected_class_index": int(selected["option_class_index"]),
                "selected_quality": float(values[chosen]),
                "selection_margin": margin,
                "kept_current": bool(selected["option_is_current"]),
            }
        selected_rows.append(output)
    with (args.output_dir / "oof_selections.jsonl").open("w") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "diagnostic_type": "official100 scene-isolated OOF within-candidate semantic selector",
        "scene_count": len(scenes), "option_row_count": len(rows),
        "prediction_count": len(groups), "fold_count": len(folds),
        "models": {
            name: {
                "feature_indices": columns, "feature_names": [names[index] for index in columns],
                "selection": _selection_metrics(rows, groups, predictions[name]),
                "weighted_mae": float(mean_absolute_error(target, predictions[name], sample_weight=weight)),
                "weighted_tp50_roc_auc": float(roc_auc_score(tp50, predictions[name], sample_weight=weight)),
            } for name, columns in GROUPS.items()
        },
        "folds": folds,
        "model_contract": {
            "type": "HistGradientBoostingRegressor", "learning_rate": 0.05,
            "max_iter": 120, "max_leaf_nodes": 7, "min_samples_leaf": 50,
            "l2_regularization": 1.0, "random_seed": args.random_seed,
        },
        "selection_contract": "OOF argmax; exact ties prefer current class then lower registered class index",
        "class_id_is_feature": False, "class_name_is_feature": False,
        "ground_truth_usage": "official_train_supervision_only_with_scene_isolated_oof",
        "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False,
        "inference_plan_written": False, "safety60_read": False, "even48_read": False,
        "test60_read": False, "split_manifest_sha256": split_sha,
        "params": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_dataset_official100_20260812"))
    parser.add_argument("--split-manifest", type=Path, default=Path("output/train_candidate_quality_oof_official100_v2/split_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_oof_official100_20260812"))
    parser.add_argument("--random-seed", type=int, default=20260812)
    args = parser.parse_args()
    for name in ("dataset_root", "split_manifest", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists(): raise SystemExit(f"refusing to overwrite {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
