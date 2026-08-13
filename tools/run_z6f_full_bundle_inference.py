#!/usr/bin/env python3
"""Run frozen full Z3 + Z6f selector/gate inference on a no-GT review ledger."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    positive = values[values > 0]
    if not len(positive):
        return 0.0, 0.0, 0.0
    ordered = np.sort(values)
    top, second = float(ordered[-1]), float(ordered[-2])
    entropy = float(-(positive * np.log(positive)).sum() / math.log(len(values)))
    return top, top - second, entropy


def _js(left: np.ndarray, right: np.ndarray) -> float:
    if left.sum() <= 0 or right.sum() <= 0:
        return 0.0
    middle = 0.5 * (left + right)
    lm, rm = left > 0, right > 0
    value = 0.5 * float((left[lm] * np.log(left[lm] / middle[lm])).sum())
    value += 0.5 * float((right[rm] * np.log(right[rm] / middle[rm])).sum())
    return value / math.log(2.0)


def _z3_feature(
    source: str, class_index: int, original_score: float, node: dict,
    geometry_yolo: np.ndarray, inherited_yolo: np.ndarray,
    geometry_alpha: np.ndarray, inherited_alpha: np.ndarray,
) -> np.ndarray:
    gy_top, gy_margin, gy_entropy = _stats(geometry_yolo)
    iy_top, iy_margin, iy_entropy = _stats(inherited_yolo)
    ga_top, ga_margin, ga_entropy = _stats(geometry_alpha)
    ia_top, ia_margin, ia_entropy = _stats(inherited_alpha)
    ga = bool(node["geometry_alpha_available"])
    ia = bool(node["inherited_alpha_available"])
    fused = inherited_yolo.copy()
    if ia:
        fused = 0.5 * inherited_yolo + 0.5 * inherited_alpha if fused.sum() > 0 else inherited_alpha.copy()
    if fused.sum() > 0:
        fused /= fused.sum()
    f_top, f_margin, f_entropy = _stats(fused)
    track_quality = float(node.get("track_quality", 0.0))
    support_views = int(node.get("track_support_view_count", 0))
    return np.asarray([
        float(source == "native"), float(source == "track"), float(source == "pair_union"),
        original_score, math.log1p(int(node["point_count"])),
        math.log1p(int(node["bound_candidate_count"])), track_quality, math.log1p(support_views),
        float(geometry_yolo.sum() > 0), float(inherited_yolo.sum() > 0),
        float(geometry_yolo[class_index]), float(inherited_yolo[class_index]),
        gy_top, gy_margin, gy_entropy, iy_top, iy_margin, iy_entropy,
        _js(geometry_yolo, inherited_yolo),
        float(geometry_yolo.sum() > 0 and class_index == int(np.argmax(geometry_yolo))),
        float(inherited_yolo.sum() > 0 and class_index == int(np.argmax(inherited_yolo))),
        float(ga), float(ia), float(geometry_alpha[class_index]), float(inherited_alpha[class_index]),
        ga_top, ga_margin, ga_entropy, ia_top, ia_margin, ia_entropy,
        _js(geometry_yolo, geometry_alpha), _js(inherited_yolo, inherited_alpha),
        _js(geometry_alpha, inherited_alpha),
        float(ga and class_index == int(np.argmax(geometry_alpha))),
        float(ia and class_index == int(np.argmax(inherited_alpha))),
        float(fused[class_index]), f_top, f_margin, f_entropy,
        float(fused.sum() > 0 and class_index == int(np.argmax(fused))),
    ], dtype=np.float32)


def _selector_feature(source: str, current_class: int, current_score: float, node: dict, option: dict) -> np.ndarray:
    yolo_rank, alpha_rank = option.get("yolo_rank"), option.get("alpha_rank")
    index = int(option["class_index"])
    dino_available = str(node["dino_embedding_state"]) == "available"
    return np.asarray([
        float(source == "native"), float(source == "track"), float(source == "pair_union"),
        math.log1p(int(node["point_count"])), math.log1p(int(node["bound_candidate_count"])),
        current_score, float(index == current_class), float(node["representative_current_in_candidate_union"]),
        float(option["yolo_probability"]), float(option["alpha_probability"]),
        float(option["max_model_probability"]), float(option["in_yolo_top5"]),
        float(option["in_alpha_top5"]), 0.0 if yolo_rank is None else 1.0 / int(yolo_rank),
        0.0 if alpha_rank is None else 1.0 / int(alpha_rank),
        float(index == node["yolo_top1_class_index"]), float(index == node["alpha_top1_class_index"]),
        float(node["yolo_alpha_top1_disagreement"]), float(node["yolo_margin"]),
        float(node["alpha_margin"]), float(node["yolo_entropy"]), float(node["alpha_entropy"]),
        float(node["geometry_yolo_alpha_js"] or 0.0), float(node["inherited_yolo_alpha_js"] or 0.0),
        float(node["view_count"]), float(dino_available),
        float(node["dino_pairwise_cosine_mean"] or 0.0),
        float(node["dino_pairwise_cosine_min"] or 0.0),
        float(node["dino_pairwise_cosine_std"] or 0.0),
        float(node["dino_dispersion"] or 0.0),
    ], dtype=np.float32)


def _gate_feature(features: np.ndarray, scores: np.ndarray, selected: int, current: int | None, margin: float) -> np.ndarray:
    current_score = float(scores[current]) if current is not None else 0.0
    selected_row = features[selected]
    return np.asarray([
        float(scores[selected]), current_score, float(scores[selected] - current_score), margin,
        *selected_row[0:6].tolist(), *selected_row[8:30].tolist(),
    ], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--unified-ledger-root", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--z3-prediction-root", type=Path, required=True)
    parser.add_argument("--z6f-bundle-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path): setattr(args, name, _resolve(value))
    if args.output_dir.exists(): raise SystemExit(f"refusing to overwrite {args.output_dir}")

    prompts = [str(x) for x in yaml.safe_load(args.config_path.read_text())["network2d"]["text_prompts"]]
    reviews = _read_jsonl(args.review_root / "semantic_review_inputs.jsonl")
    review_by_key = {str(row["semantic_evidence_node_key"]): row for row in reviews}
    if len(review_by_key) != len(reviews):
        raise ValueError("duplicate semantic review node key")
    nodes = _read_jsonl(args.unified_ledger_root / "nodes.jsonl")
    node_by_key = {str(row["semantic_evidence_node_key"]): row for row in nodes}
    if len(node_by_key) != len(nodes) or set(review_by_key) != set(node_by_key):
        raise ValueError("review and unified node ledgers do not have an exact join")
    with np.load(args.unified_ledger_root / "semantic_distributions.npz") as payload:
        distributions = {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files}
    bindings = _read_jsonl(args.z1_root / "candidate_bindings.jsonl")
    selector = joblib.load(args.z6f_bundle_root / "semantic_only_selector.joblib")
    gate = joblib.load(args.z6f_bundle_root / "binary_improvement_gate.joblib")
    z3_rows = _read_jsonl(args.z3_prediction_root / "oof_predictions.jsonl")
    z3_by_key = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])): row
        for row in z3_rows
    }
    all_binding_keys = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        for row in bindings
    }
    expected_z3_keys = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        for row in bindings
        if str(row["candidate_source"]) != "native"
        or 0 <= int(row["native_class_index"]) < 198
    }
    if len(all_binding_keys) != len(bindings) or len(z3_by_key) != len(z3_rows):
        raise ValueError("duplicate Z1 binding or Z3 prediction key")
    if set(z3_by_key) != expected_z3_keys:
        raise ValueError("Z3 predictions do not exactly cover legal Z1 candidate bindings")

    pending = []
    all_option_features = []
    counts = Counter()
    for binding in bindings:
        scene, source, cid = str(binding["scene_name"]), str(binding["candidate_source"]), int(binding["candidate_id"])
        node_key = str(binding["semantic_evidence_node_key"])
        node = node_by_key[node_key]
        review = review_by_key[node_key]
        index = int(node["node_index"])
        z3_row = z3_by_key.get((scene, source, cid))
        if z3_row is None:
            if source == "native" and not 0 <= int(binding["native_class_index"]) < 198:
                counts["omitted_invalid_native_class"] += 1
                continue
            raise AssertionError("exact legal Z3/Z1 coverage check failed unexpectedly")
        current_class = int(z3_row["class_index"])
        if not 0 <= current_class < 198:
            raise ValueError(f"{scene}:{source}:{cid}: invalid current class {current_class}")
        current_score = float(z3_row["oof_predictions"]["C_joint_yolo_alpha"])
        options = [dict(row) for row in review["candidate_classes"]]
        if current_class not in {int(row["class_index"]) for row in options}:
            options.append({
                "class_index": current_class, "class_name": prompts[current_class],
                "yolo_probability": 0.0, "alpha_probability": 0.0, "max_model_probability": 0.0,
                "in_yolo_top5": False, "in_alpha_top5": False, "yolo_rank": None, "alpha_rank": None,
            })
        options.sort(key=lambda row: int(row["class_index"]))
        features = np.stack([_selector_feature(source, current_class, current_score, review, option) for option in options])
        start = len(all_option_features)
        all_option_features.extend(features)
        pending.append({
            "scene": scene, "source": source, "candidate_id": cid, "node_key": node_key,
            "current_class": current_class, "current_score": current_score,
            "options": options, "features": features, "score_slice": slice(start, start + len(features)),
        })

    option_scores = np.clip(
        selector.predict(np.asarray(all_option_features, dtype=np.float32)[:, :25]), 0, 1
    )
    gate_features = []
    for item in pending:
        options = item["options"]
        features = item["features"]
        scores = option_scores[item["score_slice"]]
        current_class = int(item["current_class"])
        order = sorted(range(len(options)), key=lambda i: (-float(scores[i]), -int(int(options[i]["class_index"]) == current_class), int(options[i]["class_index"])))
        selected = order[0]
        current = next(i for i in order if int(options[i]["class_index"]) == current_class)
        runner = float(scores[order[1]]) if len(order) > 1 else 0.0
        margin = float(scores[selected] - runner)
        item.update({"scores": scores, "selected": selected, "current": current, "margin": margin})
        gate_features.append(_gate_feature(features, scores, selected, current, margin))

    improve_probabilities = gate.predict_proba(
        np.asarray(gate_features, dtype=np.float32)
    )[:, 1]
    outputs = []
    by_prediction = defaultdict(list)
    for item, improve in zip(pending, improve_probabilities):
        current_class = int(item["current_class"])
        proposed = int(item["options"][item["selected"]]["class_index"])
        improve = float(improve)
        accepted = proposed != current_class and improve > 0.5
        state = "accepted_new_class" if accepted else ("selector_kept_current" if proposed == current_class else "abstained_keep_current")
        counts[state] += 1
        outputs.append({
            "scene_name": item["scene"], "prediction_index": -1,
            "candidate_source": item["source"], "candidate_id": item["candidate_id"],
            "semantic_evidence_node_key": item["node_key"],
            "current_class_index": current_class, "current_class_valid": True,
            "current_hybrid_score": item["current_score"],
            "selectors": {"improvement_gated_semantic_only": {
                "selected_class_index": proposed if accepted else current_class,
                "proposed_class_index": proposed, "kept_current": not accepted,
                "accepted": accepted, "gate_improve_probability": improve,
                "gate_state": state, "selector_margin": item["margin"],
            }},
        })
        by_prediction[item["scene"]].append(outputs[-1])

    # Frozen evaluator order is native candidate ID, retained track ID, then pair-union ID.
    source_order = {"native": 0, "track": 1, "pair_union": 2}
    ordered = []
    for scene in sorted(by_prediction):
        scene_rows = sorted(by_prediction[scene], key=lambda row: (source_order[row["candidate_source"]], row["candidate_id"]))
        for prediction_index, row in enumerate(scene_rows):
            row["prediction_index"] = prediction_index
            ordered.append(row)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "oof_selections.jsonl").open("w") as handle:
        for row in ordered: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "diagnostic_type": "frozen full-bundle safety60 Z6d semantic-only inference",
        "scene_count": len(by_prediction), "prediction_count": len(ordered),
        "gate_state_counts": dict(counts), "acceptance_contract": "P(improve)>0.5",
        "ground_truth_usage": "none", "safety60_read": True,
        "candidate_mutation": False, "geometry_mutation": False, "class_mutation": False,
        "score_mutation": False, "inference_plan_written": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
