#!/usr/bin/env python3
"""Build official100 supervision for the fixed Z6c within-candidate selector."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402
from tools.diagnose_z0_open_vocab_oracle_gt import _load_gt  # noqa: E402
from tools.diagnose_z6a_class_candidate_space_oracle_gt import (  # noqa: E402
    _best_gt_with_ids,
    _load_assets,
    _prediction_metadata,
)
from tools.evaluate_z3_semantic_reliability_oof_gt import _scene_prediction  # noqa: E402
from tools.evaluate_z3_yoloworld_control_group_gt import _read_scenes, _resolve  # noqa: E402


THRESHOLDS = np.arange(0.50, 0.951, 0.05, dtype=np.float32)
FEATURE_NAMES = [
    "source_native", "source_track", "source_pair_union", "log1p_point_count",
    "log1p_bound_candidate_count", "current_hybrid_score", "candidate_is_current",
    "current_in_model_union", "candidate_yolo_probability", "candidate_alpha_probability",
    "candidate_max_model_probability", "candidate_in_yolo_top5", "candidate_in_alpha_top5",
    "candidate_yolo_rank_reciprocal", "candidate_alpha_rank_reciprocal",
    "candidate_is_yolo_top1", "candidate_is_alpha_top1", "yolo_alpha_top1_disagreement",
    "yolo_margin", "alpha_margin", "yolo_entropy", "alpha_entropy",
    "geometry_yolo_alpha_js", "inherited_yolo_alpha_js", "view_count",
    "dino_available", "dino_pairwise_cosine_mean", "dino_pairwise_cosine_min",
    "dino_pairwise_cosine_std", "dino_dispersion",
]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _feature_row(source: str, current_class: int, current_score: float, node: dict, candidate: dict) -> list[float]:
    class_index = int(candidate["class_index"])
    yolo_rank = candidate.get("yolo_rank")
    alpha_rank = candidate.get("alpha_rank")
    dino_available = node["dino_embedding_state"] == "available"
    return [
        float(source == "native"), float(source == "track"), float(source == "pair_union"),
        math.log1p(int(node["point_count"])), math.log1p(int(node["bound_candidate_count"])),
        float(current_score), float(class_index == current_class),
        float(node["representative_current_in_candidate_union"]),
        float(candidate["yolo_probability"]), float(candidate["alpha_probability"]),
        float(candidate["max_model_probability"]), float(candidate["in_yolo_top5"]),
        float(candidate["in_alpha_top5"]), 0.0 if yolo_rank is None else 1.0 / int(yolo_rank),
        0.0 if alpha_rank is None else 1.0 / int(alpha_rank),
        float(class_index == node["yolo_top1_class_index"]),
        float(class_index == node["alpha_top1_class_index"]),
        float(node["yolo_alpha_top1_disagreement"]), float(node["yolo_margin"]),
        float(node["alpha_margin"]), float(node["yolo_entropy"]), float(node["alpha_entropy"]),
        float(node["geometry_yolo_alpha_js"] or 0.0),
        float(node["inherited_yolo_alpha_js"] or 0.0), int(node["view_count"]),
        float(dino_available), float(node["dino_pairwise_cosine_mean"] or 0.0),
        float(node["dino_pairwise_cosine_min"] or 0.0),
        float(node["dino_pairwise_cosine_std"] or 0.0), float(node["dino_dispersion"] or 0.0),
    ]


def _candidate_options(node: dict, current_class: int, prompts: list[str]) -> list[dict]:
    options = [dict(row) for row in node["candidate_classes"]]
    by_class = {int(row["class_index"]): row for row in options}
    if 0 <= current_class < len(prompts) and current_class not in by_class:
        options.append({
            "class_index": current_class, "class_name": prompts[current_class],
            "yolo_probability": 0.0, "alpha_probability": 0.0,
            "max_model_probability": 0.0, "in_yolo_top5": False, "in_alpha_top5": False,
            "yolo_rank": None, "alpha_rank": None,
        })
    return sorted(options, key=lambda row: int(row["class_index"]))


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("Z6c selector dataset requires official100")
    review_rows = _read_jsonl(args.review_input_root / "semantic_review_inputs.jsonl")
    review = {str(row["semantic_evidence_node_key"]): row for row in review_rows}
    review_summary = json.loads((args.review_input_root / "summary.json").read_text())
    if (
        len(review) != 9708 or review_summary.get("ground_truth_usage") != "none"
        or review_summary.get("class_mutation") is not False
    ):
        raise ValueError("Z6c review input contract is invalid")
    import yaml
    prompts = [str(value) for value in yaml.safe_load(args.config_path.read_text())["network2d"]["text_prompts"]]
    _, node_by_key, distributions, bindings, rows_by_scene_source = _load_assets(args, scenes)

    rows, features, targets, target_tp50, weights = [], [], [], [], []
    source_predictions = Counter()
    correct_option_coverage = Counter()
    skipped_predictions = Counter()
    for scene_index, scene in enumerate(scenes, 1):
        baseline = _scene_prediction(
            scene, "pair_union", "C_joint_native_track_union_frozen_score", args, rows_by_scene_source
        )
        metadata = _prediction_metadata(scene, args, bindings, rows_by_scene_source)
        if len(metadata) != baseline["pred_masks"].shape[1]:
            raise ValueError(f"{scene}: prediction metadata mismatch")
        gt_rows = _load_gt(args.gt_instance_dir / f"{scene}.txt", args.min_region_size)[1]
        best_iou, best_class, _ = _best_gt_with_ids(baseline["pred_masks"], gt_rows)
        for prediction_index, meta in enumerate(metadata):
            source = str(meta["candidate_source"])
            current_class = int(baseline["pred_classes"][prediction_index])
            current_class_valid = 0 <= current_class < 198
            node_key = str(meta["semantic_evidence_node_key"])
            node = review[node_key]
            options = _candidate_options(node, current_class, prompts)
            if not options:
                skipped_predictions["no_legal_option"] += 1
                source_predictions[source] += 1
                continue
            target_index = int(best_class[prediction_index])
            inverse = {int(semantic): int(index) for index, semantic in instance_eval.PRED_ID_TO_ID.items()}
            target_class = inverse.get(target_index, -1)
            quality = float((best_iou[prediction_index] >= THRESHOLDS).mean())
            covered = target_class in {int(option["class_index"]) for option in options}
            correct_option_coverage["eligible"] += int(best_iou[prediction_index] >= 0.5)
            correct_option_coverage["covered"] += int(best_iou[prediction_index] >= 0.5 and covered)
            group_weight = (1.0 / max(1, int(node["bound_candidate_count"]))) if source == "native" else 1.0
            group_weight /= len(options)
            for option in options:
                class_index = int(option["class_index"])
                target = quality if class_index == target_class else 0.0
                feature = _feature_row(
                    source, current_class, float(baseline["pred_scores"][prediction_index]), node, option
                )
                if len(feature) != len(FEATURE_NAMES) or not np.isfinite(feature).all():
                    raise ValueError(f"{scene}:{prediction_index}:{class_index}: invalid feature")
                rows.append({
                    "row_index": len(rows), "scene_name": scene,
                    "prediction_index": prediction_index, "candidate_source": source,
                    "candidate_id": int(meta["candidate_id"]), "semantic_evidence_node_key": node_key,
                    "current_class_index": current_class, "option_class_index": class_index,
                    "current_class_valid": current_class_valid,
                    "option_is_current": current_class_valid and class_index == current_class,
                    "label_best_geometry_iou": float(best_iou[prediction_index]),
                    "label_option_is_target_class": int(class_index == target_class),
                    "label_option_ap_quality": target,
                    "label_option_tp50": int(class_index == target_class and best_iou[prediction_index] >= 0.5),
                })
                features.append(feature); targets.append(target); target_tp50.append(rows[-1]["label_option_tp50"])
                weights.append(group_weight)
            source_predictions[source] += 1
        print(f"[Z6c selector dataset] {scene_index}/100 {scene}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_dir / "dataset.npz", features=np.asarray(features, dtype=np.float32),
        target=np.asarray(targets, dtype=np.float32), tp50=np.asarray(target_tp50, dtype=np.int8),
        sample_weight=np.asarray(weights, dtype=np.float32),
    )
    with (args.output_dir / "rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "feature_schema.json").write_text(json.dumps({
        "feature_names": FEATURE_NAMES, "class_id_is_feature": False,
        "candidate_name_is_feature": False,
    }, indent=2) + "\n")
    summary = {
        "diagnostic_type": "official100 Z6c within-candidate selector supervision dataset",
        "scene_count": len(scenes), "prediction_count": sum(source_predictions.values()),
        "option_row_count": len(rows), "source_prediction_counts": dict(source_predictions),
        "skipped_prediction_counts": dict(skipped_predictions),
        "tp50_eligible_prediction_count": int(correct_option_coverage["eligible"]),
        "tp50_target_covered_count": int(correct_option_coverage["covered"]),
        "tp50_target_covered_fraction": float(correct_option_coverage["covered"] / max(1, correct_option_coverage["eligible"])),
        "feature_count": len(FEATURE_NAMES), "class_id_is_feature": False,
        "target": "best geometry GT class receives geometry AP-quality; other registered options receive zero",
        "sample_weight": "each prediction has equal option total; native prediction additionally divided by exact-node bound count",
        "ground_truth_usage": "official_train_supervision_only", "candidate_mutation": False,
        "geometry_mutation": False, "class_mutation": False, "score_mutation": False,
        "inference_plan_written": False, "safety60_read": False, "even48_read": False,
        "test60_read": False,
        "params": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=Path("output/scannet200/scene_splits/official_train100_20260808/official_train100.txt"))
    parser.add_argument("--review-input-root", type=Path, default=Path("docs/diagnostics/z6c_semantic_review_input_official100_20260812"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--unified-ledger-root", type=Path, default=Path("docs/diagnostics/z2c_unified_semantic_node_ledger_official100_20260811"))
    parser.add_argument("--z1-root", type=Path, default=Path("docs/diagnostics/z1_yoloworld_multiview_distribution_official100_20260811_v3_frozen_support_vote"))
    parser.add_argument("--oof-root", type=Path, default=Path("docs/diagnostics/z3_semantic_reliability_oof_official100_20260811"))
    parser.add_argument("--stream-records-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/records"))
    parser.add_argument("--combined-plan-root", type=Path, default=Path("output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared/ground_truth"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_dataset_official100_20260812"))
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--min-region-size", type=int, default=100)
    args = parser.parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path): setattr(args, name, _resolve(value))
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
