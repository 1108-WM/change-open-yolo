#!/usr/bin/env python3
"""嵌套校准 official100 组件选择头的保守动作门。

外层严格使用冻结的 80/20 五折。每个外折内部再做四折交叉拟合，只用该外折
80 个训练场景选择预注册的“分位数下界 + 正收益概率”策略；随后在整个 80 场景
上重训并预测未见的 20 场景。外层验证标签从不参与策略选择。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    ACTION_TIE_PRIORITY,
    EPS,
    _sha256,
    _write_jsonl,
    configure_track_score_context,
)
from tools.train_candidate_component_action_head_oof import (  # noqa: E402
    EXPECTED_ACTION_LEDGER_SCENE_SHA256,
    EXPECTED_SPLIT_SHA256,
    MODEL_PARAMS,
    RANDOM_SEED,
    UTILITY_SCALE,
    component_balanced_weights,
    evaluate_policies,
    feature_matrix,
    load_training_rows,
    selection_metrics,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    build_seeded_scene_folds,
    load_frozen_split_manifest,
)


INNER_FOLD_COUNT = 4
MIN_INNER_PRECISION = 0.90
MIN_INNER_SELECTED_COMPONENTS = 40
POLICY_SPECS = (
    {"name": "q10_p50", "quantile": 0.10, "probability": 0.50},
    {"name": "q10_p70", "quantile": 0.10, "probability": 0.70},
    {"name": "q10_p85", "quantile": 0.10, "probability": 0.85},
    {"name": "q25_p50", "quantile": 0.25, "probability": 0.50},
    {"name": "q25_p70", "quantile": 0.25, "probability": 0.70},
    {"name": "q25_p85", "quantile": 0.25, "probability": 0.85},
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def fit_prediction_set(
    rows: list[dict], matrix: np.ndarray, train: np.ndarray, validation: np.ndarray,
    seed: int,
) -> dict[tuple[str, int, str], dict]:
    utilities = np.asarray([float(row["label_utility"]) for row in rows], dtype=np.float64)
    scaled = utilities * UTILITY_SCALE
    positive = np.asarray([int(bool(row["label_positive"])) for row in rows], dtype=np.int64)
    weights = component_balanced_weights(rows, train)
    common = {**MODEL_PARAMS, "random_state": seed}
    classifier = HistGradientBoostingClassifier(loss="log_loss", **common)
    classifier.fit(matrix[train], positive[train], sample_weight=weights[train])
    probability = classifier.predict_proba(matrix[validation])[:, 1]
    quantile_predictions = {}
    for quantile in (0.10, 0.25):
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, **common
        )
        model.fit(matrix[train], scaled[train], sample_weight=weights[train])
        quantile_predictions[quantile] = model.predict(matrix[validation]) / UTILITY_SCALE
    result = {}
    for local_index, row_index in enumerate(validation):
        row = rows[row_index]
        result[(
            str(row["scene_name"]), int(row["relation_component_id"]), str(row["action_name"])
        )] = {
            "predicted_positive_probability": float(probability[local_index]),
            "predicted_q10_utility": float(quantile_predictions[0.10][local_index]),
            "predicted_q25_utility": float(quantile_predictions[0.25][local_index]),
            "label_utility": float(row["label_utility"]),
            "label_positive": bool(row["label_positive"]),
        }
    return result


def choose_with_spec(actions: list[dict], predictions: dict, spec: dict) -> dict:
    coexist = next(row for row in actions if row["action_kind"] == "coexist")
    quantile_key = "predicted_q10_utility" if spec["quantile"] == 0.10 else "predicted_q25_utility"
    candidates = []
    for action in actions:
        if action["action_kind"] == "coexist":
            continue
        key = (
            str(action["scene_name"]), int(action["relation_component_id"]),
            str(action["action_name"]),
        )
        prediction = predictions[key]
        utility = float(prediction[quantile_key])
        probability = float(prediction["predicted_positive_probability"])
        if utility > 0.0 and probability >= float(spec["probability"]):
            candidates.append((utility, probability, action))
    if not candidates:
        return coexist
    return max(candidates, key=lambda item: (
        item[0], item[1], -ACTION_TIE_PRIORITY[str(item[2]["action_kind"])],
        str(item[2]["action_name"]),
    ))[2]


def select_components(all_actions: dict, predictions: dict, spec: dict, scene_set: set[str]) -> dict:
    return {
        key: choose_with_spec(actions, predictions, spec)
        for key, actions in all_actions.items() if key[0] in scene_set
    }


def inner_oof_predictions(
    rows: list[dict], matrix: np.ndarray, outer_train_scenes: list[str], outer_fold: int,
) -> tuple[dict, list[dict]]:
    inner_folds = build_seeded_scene_folds(
        outer_train_scenes, INNER_FOLD_COUNT, RANDOM_SEED + outer_fold
    )
    row_scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    outer_train_set = set(outer_train_scenes)
    predictions = {}
    details = []
    for inner in inner_folds:
        train_scenes = set(inner["train_scenes"])
        validation_scenes = set(inner["validation_scenes"])
        if not train_scenes <= outer_train_set or not validation_scenes <= outer_train_set:
            raise AssertionError("内部折超出外折训练场景")
        train = np.flatnonzero(np.isin(row_scenes, sorted(train_scenes)))
        validation = np.flatnonzero(np.isin(row_scenes, sorted(validation_scenes)))
        fold_predictions = fit_prediction_set(
            rows, matrix, train, validation,
            RANDOM_SEED + 100 * outer_fold + int(inner["fold_index"]),
        )
        overlap = set(predictions) & set(fold_predictions)
        if overlap:
            raise AssertionError("内部折外动作预测身份重复")
        predictions.update(fold_predictions)
        details.append({
            "inner_fold_index": int(inner["fold_index"]),
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "train_action_count": len(train),
            "validation_action_count": len(validation),
        })
    expected = {
        (str(row["scene_name"]), int(row["relation_component_id"]), str(row["action_name"]))
        for row in rows if row["scene_name"] in outer_train_set
    }
    if set(predictions) != expected:
        raise AssertionError("内部交叉拟合未完整覆盖外折训练动作")
    return predictions, details


def choose_inner_policy(all_actions: dict, predictions: dict, outer_train_scenes: list[str]) -> tuple[dict, list[dict]]:
    train_set = set(outer_train_scenes)
    action_subset = {key: rows for key, rows in all_actions.items() if key[0] in train_set}
    audits = []
    eligible = []
    for spec in POLICY_SPECS:
        selected = select_components(all_actions, predictions, spec, train_set)
        metrics = selection_metrics(selected, action_subset)
        row = {"policy_spec": dict(spec), "inner_oof_selection_metrics": metrics}
        audits.append(row)
        if (
            metrics["selected_noncoexist_component_count"] >= MIN_INNER_SELECTED_COMPONENTS
            and metrics["selected_positive_precision"] >= MIN_INNER_PRECISION
            and metrics["selected_true_utility_sum_nonadditive"] > 0.0
        ):
            eligible.append(row)
    if not eligible:
        return {"name": "always_coexist", "quantile": None, "probability": None}, audits
    best = max(eligible, key=lambda row: (
        row["inner_oof_selection_metrics"]["selected_true_utility_sum_nonadditive"],
        row["inner_oof_selection_metrics"]["selected_positive_precision"],
        -row["inner_oof_selection_metrics"]["selected_harmful_count"],
        -row["inner_oof_selection_metrics"]["selected_noncoexist_component_count"],
    ))
    return dict(best["policy_spec"]), audits


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("必须使用冻结 official100 场景与五折清单")
    action_summary_path = args.action_ledger_root / "summary.json"
    action_summary = json.loads(action_summary_path.read_text())
    if action_summary.get("scene_count") != 100 or action_summary.get("is_smoke_subset"):
        raise ValueError("动作效用账本不是完整 official100")
    if action_summary["input_provenance"]["scene_list_sha256"] != EXPECTED_ACTION_LEDGER_SCENE_SHA256:
        raise ValueError("动作效用账本场景身份不一致")
    score_context = configure_track_score_context(args, scenes)
    if action_summary.get("score_context", {}).get("track_score_mode") != args.track_score_mode:
        raise ValueError("动作效用标签与当前分数上下文不一致")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    rows, all_actions, feature_contract = load_training_rows(
        scenes, args.action_ledger_root, args.relation_feature_ledger_root
    )
    matrix, feature_names = feature_matrix(rows)
    row_scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    final_predictions = {}
    nested_selected = {}
    outer_details = []
    for outer in manifest["folds"]:
        outer_fold = int(outer["fold_index"])
        train_scenes = list(outer["train_scenes"])
        validation_scenes = list(outer["validation_scenes"])
        inner_predictions, inner_details = inner_oof_predictions(
            rows, matrix, train_scenes, outer_fold
        )
        selected_spec, policy_audits = choose_inner_policy(
            all_actions, inner_predictions, train_scenes
        )
        train = np.flatnonzero(np.isin(row_scenes, train_scenes))
        validation = np.flatnonzero(np.isin(row_scenes, validation_scenes))
        outer_predictions = fit_prediction_set(
            rows, matrix, train, validation, RANDOM_SEED + outer_fold
        )
        final_predictions.update(outer_predictions)
        if selected_spec["name"] == "always_coexist":
            fold_selected = {
                key: next(row for row in actions if row["action_kind"] == "coexist")
                for key, actions in all_actions.items() if key[0] in set(validation_scenes)
            }
        else:
            fold_selected = select_components(
                all_actions, outer_predictions, selected_spec, set(validation_scenes)
            )
        nested_selected.update(fold_selected)
        outer_details.append({
            "outer_fold_index": outer_fold,
            "selected_inner_policy": selected_spec,
            "inner_folds": inner_details,
            "inner_policy_audits": policy_audits,
            "outer_validation_selection_metrics": selection_metrics(
                fold_selected,
                {key: actions for key, actions in all_actions.items()
                 if key[0] in set(validation_scenes)},
            ),
        })
        print(
            f"[嵌套外折] {outer_fold}: 内部选择={selected_spec['name']}, "
            f"验证组件={len(fold_selected)}",
            flush=True,
        )
    if len(nested_selected) != len(all_actions):
        raise AssertionError("嵌套外折动作未完整覆盖 official100 组件")

    policies = {
        "always_coexist": {
            key: next(row for row in actions if row["action_kind"] == "coexist")
            for key, actions in all_actions.items()
        },
        "nested_calibrated": nested_selected,
    }
    # 同时报告两个固定、未嵌套选择的保守策略，便于判断嵌套校准是否真正必要。
    for spec in (POLICY_SPECS[0], POLICY_SPECS[3]):
        policies[f"fixed_{spec['name']}"] = select_components(
            all_actions, final_predictions, spec, set(scenes)
        )
    selection = {
        name: selection_metrics(selected, all_actions)
        for name, selected in policies.items()
    }
    ap_results, fold_ap = evaluate_policies(scenes, policies, args, manifest)
    baseline = ap_results["always_coexist"]["global_metrics"]
    delta = {
        name: {
            "official_ap": result["global_metrics"]["official_ap"] - baseline["official_ap"],
            "ap50": result["global_metrics"]["threshold_metrics"]["50"]["ap"] - baseline["threshold_metrics"]["50"]["ap"],
            "ap25": result["global_metrics"]["threshold_metrics"]["25"]["ap"] - baseline["threshold_metrics"]["25"]["ap"],
        }
        for name, result in ap_results.items()
    }
    prediction_rows = []
    for key, prediction in sorted(final_predictions.items()):
        prediction_rows.append({
            "scene_name": key[0],
            "relation_component_id": key[1],
            "action_name": key[2],
            **prediction,
        })
    selected_rows = [{
        "scene_name": scene,
        "relation_component_id": component_id,
        "selected_action_name": action["action_name"],
        "selected_action_kind": action["action_kind"],
        "selected_track_id": action.get("selected_track_id"),
        "label_utility_fixed_coexist_single_component": float(
            action["labels"]["delta_vs_fixed_coexist"]["official_ap"]
        ),
        "inference_action_generated": False,
    } for (scene, component_id), action in sorted(nested_selected.items())]
    summary = {
        "version": "official100_component_action_head_nested_oof_v1",
        "scene_count": 100,
        "component_count": len(all_actions),
        "feature_contract": {**feature_contract, "feature_names": feature_names},
        "score_context": score_context,
        "outer_fold_contract": "frozen official100 80/20 scene folds",
        "inner_fold_contract": {
            "fold_count": INNER_FOLD_COUNT,
            "candidate_policies": list(POLICY_SPECS),
            "minimum_inner_positive_precision": MIN_INNER_PRECISION,
            "minimum_inner_selected_components": MIN_INNER_SELECTED_COMPONENTS,
            "selection_objective": "maximize inner-OOF sum of fixed-context single-component true AP utilities after safety constraints",
            "global_ap_non_additive_warning": True,
        },
        "outer_fold_details": outer_details,
        "policy_selection_metrics": selection,
        "policy_ap_results": ap_results,
        "policy_ap_delta_vs_coexist": delta,
        "fold_policy_ap": fold_ap,
        "ground_truth_usage": "official train nested OOF supervision and offline AP only",
        "fit_all_model_exported": False,
        "inference_action_generated": False,
        "candidate_files_modified": False,
        "safety60_evaluated": False,
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "action_ledger_summary_sha256": _sha256(action_summary_path),
            "relation_feature_summary_sha256": _sha256(
                args.relation_feature_ledger_root / "summary.json"
            ),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "outer_oof_action_predictions.jsonl", prediction_rows)
        _write_jsonl(staging / "nested_selected_actions_diagnostic_only.jsonl", selected_rows)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--track-score-mode", choices=("original", "oof_quality"), default="oof_quality"
    )
    parser.add_argument("--oof-predictions", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "relation_feature_ledger_root",
        "action_ledger_root", "gt_dir", "output_root", "oof_predictions",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖：{args.output_root}")
    summary = run(args)
    print(json.dumps({
        "outer_selected_policies": [
            row["selected_inner_policy"] for row in summary["outer_fold_details"]
        ],
        "policy_selection_metrics": summary["policy_selection_metrics"],
        "policy_ap_delta_vs_coexist": summary["policy_ap_delta_vs_coexist"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
