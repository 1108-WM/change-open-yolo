#!/usr/bin/env python3
"""Z6a: GT-only class-candidate-space oracle on frozen official100 geometry.

This diagnostic asks whether the correct class of a frozen geometry node is
already present in the top-5 YOLO-World or Alpha-CLIP evidence.  Geometry,
candidate membership, and the current hybrid score contract remain unchanged.
GT is used only to select a temporary class within a pre-existing candidate
set and to run the official evaluator; no inference class or plan is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402
from tools.diagnose_z0_open_vocab_oracle_gt import (  # noqa: E402
    _load_gt,
    _one_to_one_scores,
)
from tools.evaluate_z3_semantic_reliability_oof_gt import (  # noqa: E402
    _scene_prediction,
)
from tools.evaluate_z3_yoloworld_control_group_gt import (  # noqa: E402
    _Predictions,
    _evaluate,
    _load_native,
    _read_scenes,
    _resolve,
)


VERSION = "z6a_class_candidate_space_oracle_official100_v1"
EXPECTED_SPLIT_SHA256 = "aa657449965bc76164a1a1b77c7785aa705a0295eeed1307163b325f7233fe3e"
VARIANTS = (
    "current_frozen_class",
    "yolo_top5_oracle",
    "alpha_top5_oracle",
    "yolo_alpha_union_top5_oracle",
    "full_class_oracle",
)
ORACLE_VARIANTS = VARIANTS[1:]
THRESHOLDS = (0.25, 0.50)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _top_positive_indices(values: np.ndarray, top_k: int) -> tuple[int, ...]:
    values = np.asarray(values, dtype=np.float32)
    positive = np.flatnonzero(np.isfinite(values) & (values > 0.0))
    if not len(positive):
        return ()
    order = positive[np.argsort(-values[positive], kind="stable")]
    return tuple(int(value) for value in order[:top_k])


def _candidate_sets(node_index: int, distributions: dict[str, np.ndarray], top_k: int) -> dict[str, set[int]]:
    yolo = np.maximum(
        distributions["geometry_yolo"][node_index],
        distributions["inherited_yolo"][node_index],
    )
    alpha = np.maximum(
        distributions["geometry_alpha"][node_index],
        distributions["inherited_alpha"][node_index],
    )
    yolo_top = set(_top_positive_indices(yolo, top_k))
    alpha_top = set(_top_positive_indices(alpha, top_k))
    return {
        "yolo_top5_oracle": yolo_top,
        "alpha_top5_oracle": alpha_top,
        "yolo_alpha_union_top5_oracle": yolo_top | alpha_top,
        "full_class_oracle": set(range(distributions["geometry_yolo"].shape[1])),
    }


def _load_assets(args, scenes: list[str]):
    z2_summary = json.loads((args.unified_ledger_root / "summary.json").read_text())
    if (
        z2_summary.get("ground_truth_usage") != "none"
        or z2_summary.get("candidate_mutation") is not False
        or int(z2_summary.get("node_count", -1)) != 9708
        or int(z2_summary.get("class_count", -1)) != args.class_count
    ):
        raise ValueError("Z2c unified semantic ledger contract is invalid")

    node_rows = _read_jsonl(args.unified_ledger_root / "nodes.jsonl")
    node_by_key = {str(row["semantic_evidence_node_key"]): row for row in node_rows}
    if len(node_by_key) != len(node_rows):
        raise ValueError("duplicate unified semantic node key")
    with np.load(args.unified_ledger_root / "semantic_distributions.npz") as payload:
        distributions = {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files}
    expected_arrays = {"geometry_yolo", "inherited_yolo", "geometry_alpha", "inherited_alpha"}
    if set(distributions) != expected_arrays:
        raise ValueError("unexpected Z2c distribution arrays")
    if any(array.shape != (len(node_rows), args.class_count) for array in distributions.values()):
        raise ValueError("Z2c distribution dimensions disagree")

    bindings = {}
    for row in _read_jsonl(args.z1_root / "candidate_bindings.jsonl"):
        scene = str(row["scene_name"])
        if scene not in set(scenes):
            continue
        key = (scene, str(row["candidate_source"]), int(row["candidate_id"]))
        if key in bindings:
            raise ValueError(f"duplicate Z1 binding: {key}")
        node_key = str(row["semantic_evidence_node_key"])
        if node_key not in node_by_key:
            raise ValueError(f"Z1 binding references missing Z2c node: {node_key}")
        bindings[key] = row

    oof_rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    rows_by_scene_source = defaultdict(list)
    for row in oof_rows:
        rows_by_scene_source[(str(row["scene_name"]), str(row["candidate_source"]))].append(row)
    if {scene for scene, _ in rows_by_scene_source} != set(scenes):
        raise ValueError("OOF rows do not exactly cover official100")
    return z2_summary, node_by_key, distributions, bindings, rows_by_scene_source


def _prediction_metadata(scene: str, args, bindings: dict, rows_by_scene_source) -> list[dict]:
    native = _load_native(args.stream_records_root, scene)
    rows = []
    for candidate_id in range(native["pred_masks"].shape[1]):
        binding = bindings.get((scene, "native", candidate_id))
        if binding is None:
            raise ValueError(f"{scene}: missing native Z1 binding {candidate_id}")
        rows.append({
            "candidate_source": "native",
            "candidate_id": candidate_id,
            "semantic_evidence_node_key": str(binding["semantic_evidence_node_key"]),
        })
    for source in ("track", "pair_union"):
        for oof in sorted(rows_by_scene_source[(scene, source)], key=lambda row: int(row["candidate_id"])):
            candidate_id = int(oof["candidate_id"])
            binding = bindings.get((scene, source, candidate_id))
            if binding is None:
                raise ValueError(f"{scene}: missing {source} Z1 binding {candidate_id}")
            if str(binding["semantic_evidence_node_key"]) != str(oof["semantic_evidence_node_key"]):
                raise ValueError(f"{scene}: {source} OOF/Z1 node mismatch {candidate_id}")
            rows.append({
                "candidate_source": source,
                "candidate_id": candidate_id,
                "semantic_evidence_node_key": str(binding["semantic_evidence_node_key"]),
            })
    return rows


def _best_gt_with_ids(masks: np.ndarray, gt_rows: list[tuple[int, np.ndarray]]):
    sizes = np.asarray(masks.sum(axis=0), dtype=np.int64)
    best_iou = np.zeros(masks.shape[1], dtype=np.float32)
    best_class = np.full(masks.shape[1], -1, dtype=np.int64)
    best_gt_index = np.full(masks.shape[1], -1, dtype=np.int64)
    for gt_index, (class_id, points) in enumerate(gt_rows):
        intersections = np.asarray(masks[points].sum(axis=0), dtype=np.int64)
        ious = intersections / np.maximum(1, sizes + len(points) - intersections)
        update = ious > best_iou
        best_iou[update] = ious[update]
        best_class[update] = int(class_id)
        best_gt_index[update] = gt_index
    return best_iou, best_class, best_gt_index


def _scene_bundle(
    scene: str,
    args,
    node_by_key: dict[str, dict],
    distributions: dict[str, np.ndarray],
    bindings: dict,
    rows_by_scene_source,
    inverse_class_map: dict[int, int],
) -> dict:
    baseline = _scene_prediction(
        scene,
        "pair_union",
        "C_joint_native_track_union_frozen_score",
        args,
        rows_by_scene_source,
    )
    metadata = _prediction_metadata(scene, args, bindings, rows_by_scene_source)
    if len(metadata) != baseline["pred_masks"].shape[1]:
        raise ValueError(f"{scene}: prediction metadata length mismatch")
    gt_rows = _load_gt(args.gt_instance_dir / f"{scene}.txt", args.min_region_size)[1]
    best_iou, best_class, best_gt_index = _best_gt_with_ids(baseline["pred_masks"], gt_rows)
    current_semantic = np.asarray([
        int(instance_eval.PRED_ID_TO_ID.get(int(value), -1)) for value in baseline["pred_classes"]
    ], dtype=np.int64)

    classes = {"current_frozen_class": baseline["pred_classes"].copy()}
    candidate_membership = {name: np.zeros(len(metadata), dtype=bool) for name in ORACLE_VARIANTS}
    yolo_only = np.zeros(len(metadata), dtype=bool)
    alpha_only = np.zeros(len(metadata), dtype=bool)
    class_sets_cache = {}
    for index, meta in enumerate(metadata):
        node_key = meta["semantic_evidence_node_key"]
        if node_key not in class_sets_cache:
            node_index = int(node_by_key[node_key]["node_index"])
            class_sets_cache[node_key] = _candidate_sets(node_index, distributions, args.top_k)
        target_index = inverse_class_map.get(int(best_class[index]), -1)
        if target_index < 0:
            continue
        sets = class_sets_cache[node_key]
        for variant in ORACLE_VARIANTS:
            candidate_membership[variant][index] = target_index in sets[variant]
        yolo_only[index] = target_index in sets["yolo_top5_oracle"] and target_index not in sets["alpha_top5_oracle"]
        alpha_only[index] = target_index in sets["alpha_top5_oracle"] and target_index not in sets["yolo_top5_oracle"]

    oracle_eligible = (best_iou >= args.oracle_min_iou) & (best_class >= 0)
    for variant in ORACLE_VARIANTS:
        output = baseline["pred_classes"].copy()
        selected = np.flatnonzero(oracle_eligible & candidate_membership[variant])
        for index in selected:
            output[index] = inverse_class_map[int(best_class[index])]
        classes[variant] = output

    one_to_one_scores = {}
    for variant, output_classes in classes.items():
        semantic = np.asarray([
            int(instance_eval.PRED_ID_TO_ID.get(int(value), -1)) for value in output_classes
        ], dtype=np.int64)
        one_to_one_scores[variant] = _one_to_one_scores(
            baseline["pred_masks"], semantic, gt_rows
        ).astype(np.float32)

    coverage_rows = []
    for threshold in THRESHOLDS:
        eligible = best_iou >= threshold
        current_correct = eligible & (current_semantic == best_class)
        row = {
            "scene_name": scene,
            "threshold": threshold,
            "eligible_prediction_count": int(eligible.sum()),
            "current_correct_prediction_count": int(current_correct.sum()),
            "incorrect_prediction_count": int((eligible & ~current_correct).sum()),
            "yolo_only_target_count": int((eligible & yolo_only).sum()),
            "alpha_only_target_count": int((eligible & alpha_only).sum()),
        }
        for variant in ORACLE_VARIANTS:
            contains = eligible & candidate_membership[variant]
            repair = eligible & ~current_correct & candidate_membership[variant]
            attainable = current_correct | repair
            row[f"{variant}_contains_target_count"] = int(contains.sum())
            row[f"{variant}_repairable_wrong_count"] = int(repair.sum())
            row[f"{variant}_attainable_correct_count"] = int(attainable.sum())

        unique_groups = defaultdict(list)
        for index in np.flatnonzero(eligible & (best_gt_index >= 0)):
            unique_groups[(metadata[index]["semantic_evidence_node_key"], int(best_gt_index[index]))].append(index)
        row["eligible_node_target_count"] = len(unique_groups)
        for variant in ORACLE_VARIANTS:
            row[f"{variant}_node_target_contains_count"] = sum(
                any(candidate_membership[variant][index] for index in indices)
                for indices in unique_groups.values()
            )
        coverage_rows.append(row)

    return {
        "classes": classes,
        "one_to_one_scores": one_to_one_scores,
        "coverage_rows": coverage_rows,
        "source_counts": Counter(row["candidate_source"] for row in metadata),
        "candidate_count": int(baseline["pred_masks"].shape[1]),
    }


def _aggregate_coverage(rows: list[dict]) -> dict:
    result = {}
    for threshold in THRESHOLDS:
        selected = [row for row in rows if float(row["threshold"]) == threshold]
        totals = Counter()
        for row in selected:
            for key, value in row.items():
                if key not in {"scene_name", "threshold"}:
                    totals[key] += int(value)
        eligible = totals["eligible_prediction_count"]
        incorrect = totals["incorrect_prediction_count"]
        node_targets = totals["eligible_node_target_count"]
        payload = {"counts": dict(sorted(totals.items()))}
        payload["current_correct_fraction"] = float(totals["current_correct_prediction_count"] / eligible) if eligible else None
        payload["yolo_only_target_fraction"] = float(totals["yolo_only_target_count"] / eligible) if eligible else None
        payload["alpha_only_target_fraction"] = float(totals["alpha_only_target_count"] / eligible) if eligible else None
        payload["variants"] = {}
        for variant in ORACLE_VARIANTS:
            payload["variants"][variant] = {
                "candidate_contains_target_fraction": float(totals[f"{variant}_contains_target_count"] / eligible) if eligible else None,
                "repairable_fraction_of_current_wrong": float(totals[f"{variant}_repairable_wrong_count"] / incorrect) if incorrect else None,
                "attainable_correct_fraction": float(totals[f"{variant}_attainable_correct_count"] / eligible) if eligible else None,
                "node_target_contains_fraction": float(totals[f"{variant}_node_target_contains_count"] / node_targets) if node_targets else None,
            }
        result[str(threshold)] = payload
    return result


def _delta(current: dict, baseline: dict) -> dict:
    return {
        name: float(current[name] - baseline[name])
        for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
    }


def run(args) -> dict:
    scenes = _read_scenes(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("Z6a requires the frozen official100 scene list")
    split_sha = _sha256(args.split_manifest)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError("frozen split manifest SHA-256 mismatch")
    z2_summary, node_by_key, distributions, bindings, rows_by_scene_source = _load_assets(args, scenes)
    inverse = {
        int(semantic): int(index)
        for index, semantic in instance_eval.PRED_ID_TO_ID.items() if int(semantic) >= 0
    }

    artifacts = {}
    coverage_rows = []
    source_counts = Counter()

    for index, scene in enumerate(scenes, 1):
        item = _scene_bundle(
            scene, args, node_by_key, distributions, bindings,
            rows_by_scene_source, inverse,
        )
        artifacts[scene] = {
            "classes": item["classes"],
            "one_to_one_scores": item["one_to_one_scores"],
            "candidate_count": item["candidate_count"],
        }
        coverage_rows.extend(item["coverage_rows"])
        source_counts.update(item["source_counts"])
        print(f"[Z6a coverage] {index}/100 {scene}", flush=True)

    staging = args.output_dir.parent / f".{args.output_dir.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        systems = {}
        for ranking in ("current_score", "one_to_one_score"):
            systems[ranking] = {}
            for variant in VARIANTS:
                def build(scene, variant=variant, ranking=ranking):
                    baseline = _scene_prediction(
                        scene, "pair_union", "C_joint_native_track_union_frozen_score",
                        args, rows_by_scene_source,
                    )
                    item = artifacts[scene]
                    if baseline["pred_masks"].shape[1] != item["candidate_count"]:
                        raise ValueError(f"{scene}: frozen prediction count changed during evaluation")
                    classes = item["classes"][variant]
                    scores = baseline["pred_scores"]
                    if ranking == "one_to_one_score":
                        scores = item["one_to_one_scores"][variant]
                    return {
                        "pred_masks": baseline["pred_masks"],
                        "pred_classes": classes,
                        "pred_scores": np.asarray(scores, dtype=np.float32),
                    }

                mapping = _Predictions(scenes, build)
                systems[ranking][variant] = _evaluate(
                    mapping, args.gt_instance_dir, staging / f"{ranking}__{variant}.csv"
                )
                print(
                    f"[Z6a AP] {ranking} {variant}: {systems[ranking][variant]['ap']:.6f}",
                    flush=True,
                )

        expected = json.loads(args.hybrid_control_summary.read_text())["systems"]["C_joint_native_track_union_frozen_score"]["pair_union"]
        actual = systems["current_score"]["current_frozen_class"]
        control_error = max(
            abs(float(actual[name]) - float(expected[name]))
            for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
        )
        if control_error > args.control_tolerance:
            raise RuntimeError(f"Z6a hybrid control reproduction failed: {control_error}")

        manifest = json.loads(args.split_manifest.read_text())
        validation_occurrences = Counter(
            scene for fold in manifest["folds"] for scene in fold["validation_scenes"]
        )
        if set(validation_occurrences) != set(scenes) or any(value != 1 for value in validation_occurrences.values()):
            raise ValueError("split manifest does not cover every official100 scene exactly once")
        folds = []
        for spec in sorted(manifest["folds"], key=lambda row: int(row["fold_index"])):
            fold_scenes = list(spec["validation_scenes"])
            fold_systems = {"current_score": {}}
            for ranking in fold_systems:
                for variant in VARIANTS:
                    def build_fold(scene, variant=variant, ranking=ranking):
                        baseline = _scene_prediction(
                            scene, "pair_union", "C_joint_native_track_union_frozen_score",
                            args, rows_by_scene_source,
                        )
                        item = artifacts[scene]
                        if baseline["pred_masks"].shape[1] != item["candidate_count"]:
                            raise ValueError(f"{scene}: frozen prediction count changed during fold evaluation")
                        classes = item["classes"][variant]
                        scores = baseline["pred_scores"]
                        return {
                            "pred_masks": baseline["pred_masks"],
                            "pred_classes": classes,
                            "pred_scores": np.asarray(scores, dtype=np.float32),
                        }
                    fold_systems[ranking][variant] = _evaluate(
                        _Predictions(fold_scenes, build_fold),
                        args.gt_instance_dir,
                        staging / f"fold_{spec['fold_index']}__{ranking}__{variant}.csv",
                    )
            folds.append({
                "fold_index": int(spec["fold_index"]),
                "validation_scenes": fold_scenes,
                "systems": fold_systems,
                "deltas_vs_current_frozen_class": {
                    ranking: {
                        variant: _delta(values, fold_systems[ranking]["current_frozen_class"])
                        for variant, values in fold_systems[ranking].items()
                        if variant != "current_frozen_class"
                    }
                    for ranking in fold_systems
                },
            })
            print(f"[Z6a AP] fold {spec['fold_index']} complete", flush=True)

        deltas = {
            ranking: {
                variant: _delta(values, systems[ranking]["current_frozen_class"])
                for variant, values in systems[ranking].items() if variant != "current_frozen_class"
            }
            for ranking in systems
        }
        summary = {
            "version": VERSION,
            "diagnostic_type": "Z6a frozen-geometry class candidate-space GT-only oracle",
            "scene_count": len(scenes),
            "class_count": args.class_count,
            "top_k_per_model": args.top_k,
            "candidate_count": int(sum(source_counts.values())),
            "source_candidate_counts": dict(sorted(source_counts.items())),
            "systems": systems,
            "deltas_vs_current_frozen_class": deltas,
            "coverage": _aggregate_coverage(coverage_rows),
            "folds": folds,
            "positive_fold_counts_main_ap": {
                "current_score": {
                    variant: sum(
                        fold["deltas_vs_current_frozen_class"]["current_score"][variant]["ap"] > 0
                        for fold in folds
                    )
                    for variant in ORACLE_VARIANTS
                }
            },
            "control_reproduction": {
                "reference_summary": str(args.hybrid_control_summary),
                "max_abs_error": control_error,
                "tolerance": args.control_tolerance,
                "valid": True,
            },
            "input_ledger_audit": {
                "z2c_ground_truth_usage": z2_summary["ground_truth_usage"],
                "z2c_candidate_mutation": z2_summary["candidate_mutation"],
                "z2c_node_count": z2_summary["node_count"],
                "z1_binding_count": len(bindings),
                "split_manifest_sha256": split_sha,
            },
            "ground_truth_usage": "official_train_oracle_and_evaluation_only",
            "candidate_mutation": False,
            "geometry_mutation": False,
            "score_mutation_current_score_systems": False,
            "inference_plan_written": False,
            "model_trained": False,
            "safety60_read": False,
            "even48_read": False,
            "test60_read": False,
            "contracts": {
                "candidate_set": "per geometry node, elementwise max of geometry-own and inherited evidence, then frozen top-5 per model",
                "oracle": "keep current class unless best-IoU GT class is present in the registered candidate set and IoU >= oracle_min_iou",
                "full_class_oracle": "same frozen geometry and current hybrid scores; all 198 valid prediction classes are eligible",
                "one_to_one_score": "GT-only Hungarian score diagnostic after the corresponding temporary oracle class assignment",
            },
            "params": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (staging / "coverage_by_scene.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in coverage_rows
        ))
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
        os.replace(staging, args.output_dir)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--unified-ledger-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--hybrid-control-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--oracle-min-iou", type=float, default=0.25)
    parser.add_argument("--control-tolerance", type=float, default=1e-10)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not (args.allow_gt_evaluation and args.allow_gt_diagnostics):
        raise SystemExit("Z6a requires --allow-gt-evaluation and --allow-gt-diagnostics")
    if args.class_count != 198 or args.top_k != 5:
        raise SystemExit("Z6a v1 freezes --class-count 198 and --top-k 5")
    if not 0.0 < args.oracle_min_iou <= 1.0:
        raise SystemExit("--oracle-min-iou must be in (0, 1]")
    for name in (
        "scene_list", "z1_root", "unified_ledger_root", "oof_root", "split_manifest",
        "stream_records_root", "combined_plan_root", "gt_instance_dir",
        "hybrid_control_summary", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
