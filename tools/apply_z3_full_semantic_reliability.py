#!/usr/bin/env python3
"""Apply the frozen full-official100 Z3 model to a no-GT transfer split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--unified-ledger-root", type=Path, required=True)
    parser.add_argument("--frozen-score-plan-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path): setattr(args, name, _resolve(value))
    if args.output_dir.exists(): raise SystemExit(f"refusing to overwrite {args.output_dir}")

    from tools.build_z3_semantic_reliability_dataset_gt import _feature_row

    scenes = [line.strip() for line in args.scene_list.read_text().splitlines() if line.strip()]
    bindings = _read_jsonl(args.z1_root / "candidate_bindings.jsonl")
    track_binding_by_id = {}
    for row in bindings:
        if str(row["candidate_source"]) != "track":
            continue
        key = (str(row["scene_name"]), int(row["candidate_id"]))
        if key in track_binding_by_id:
            raise ValueError(f"duplicate track binding: {key}")
        track_binding_by_id[key] = row
    nodes = _read_jsonl(args.unified_ledger_root / "nodes.jsonl")
    node_by_key = {str(row["semantic_evidence_node_key"]): row for row in nodes}
    if len(node_by_key) != len(nodes):
        raise ValueError("duplicate unified semantic node key")
    with np.load(args.unified_ledger_root / "semantic_distributions.npz") as payload:
        distributions = {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files}
    model = joblib.load(args.model_root / "c_joint_yolo_alpha.joblib")

    score_by_key = {}
    for scene in scenes:
        for row in _read_jsonl(args.frozen_score_plan_root / scene / "frozen_score_plan.jsonl"):
            source = "native" if str(row["candidate_source"]) == "native_mask3d_yoloworld" else "track"
            score_by_key[(scene, source, int(row["candidate_id"]))] = float(row["planned_score"])
        for row in _read_jsonl(args.frozen_score_plan_root / scene / "pair_union_append_candidates.jsonl"):
            score_by_key[(scene, "pair_union", int(row["candidate_id"]))] = float(row["new_score"])

    output, feature_rows, model_row_indexes, source_counts = [], [], [], Counter()
    omitted_invalid_class = Counter()
    for row in bindings:
        scene, source, candidate_id = str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])
        node = node_by_key[str(row["semantic_evidence_node_key"])]
        index = int(node["node_index"])
        if source == "native":
            class_index = int(row["native_class_index"])
            selected = row
        else:
            iy = distributions["inherited_yolo"][index]
            ia = distributions["inherited_alpha"][index]
            if bool(node["inherited_alpha_available"]):
                fused = 0.5 * iy + 0.5 * ia if iy.sum() > 0 else ia.copy()
            else:
                fused = iy.copy()
            class_index = int(np.argmax(fused)) if fused.sum() > 0 else -1
            if source == "pair_union":
                selected_track_id = int(row["selected_track_id"])
                selected = track_binding_by_id.get((scene, selected_track_id))
                if selected is None:
                    raise ValueError(
                        f"{scene}:pair_union:{candidate_id}: missing selected track binding "
                        f"{selected_track_id}"
                    )
            else:
                selected = row
        if not 0 <= class_index < 198:
            omitted_invalid_class[source] += 1
            continue
        original_score = score_by_key[(scene, source, candidate_id)]
        track_quality = float(selected.get("track_quality", 0.0)) if source != "native" else 0.0
        support_views = int(selected.get("track_support_view_count", 0)) if source != "native" else 0
        feature = _feature_row(
            source, class_index, original_score, int(node["point_count"]),
            int(node["bound_candidate_count"]), track_quality, support_views,
            distributions["geometry_yolo"][index], distributions["inherited_yolo"][index],
            distributions["geometry_alpha"][index], distributions["inherited_alpha"][index],
            bool(node["geometry_alpha_available"]), bool(node["inherited_alpha_available"]),
        )
        prediction = original_score if source == "pair_union" else None
        output.append({
            "row_index": len(output), "scene_name": scene, "candidate_source": source,
            "candidate_id": candidate_id,
            "semantic_evidence_node_key": str(row["semantic_evidence_node_key"]),
            "class_index": class_index, "original_score": original_score,
            "oof_predictions": {"C_joint_yolo_alpha": prediction},
            "prediction_contract": (
                "frozen_original_pair_union_score" if source == "pair_union"
                else "full_official100_C_joint_yolo_alpha"
            ),
        })
        if source != "pair_union":
            feature_rows.append(feature)
            model_row_indexes.append(len(output) - 1)
        source_counts[source] += 1
    if len(output) + sum(omitted_invalid_class.values()) != len(bindings):
        raise ValueError("prediction plus invalid-class omission coverage mismatch")
    if feature_rows:
        predictions = np.clip(
            model.predict(np.asarray(feature_rows, dtype=np.float32)), 0.0, 1.0
        )
        if len(predictions) != len(model_row_indexes):
            raise ValueError("full Z3 batch prediction count mismatch")
        for output_index, prediction in zip(model_row_indexes, predictions):
            output[output_index]["oof_predictions"]["C_joint_yolo_alpha"] = float(prediction)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "oof_predictions.jsonl").open("w") as handle:
        for row in output: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "diagnostic_type": "frozen full-official100 Z3 semantic reliability transfer predictions",
        "scene_count": len({row["scene_name"] for row in output}), "row_count": len(output),
        "source_row_counts": dict(source_counts),
        "omitted_invalid_class_counts": dict(omitted_invalid_class),
        "ground_truth_usage": "none", "safety60_read": True,
        "candidate_mutation": False, "geometry_mutation": False, "class_mutation": False,
        "score_mutation": False, "inference_plan_written": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
