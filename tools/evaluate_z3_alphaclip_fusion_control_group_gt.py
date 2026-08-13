#!/usr/bin/env python3
"""Evaluate fixed Alpha-CLIP/YOLO-World Z3 fusion controls with GT only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_z3_yoloworld_control_group_gt import (  # noqa: E402
    _Predictions,
    _evaluate,
    _load_native,
    _load_track_overrides,
    _load_tracks,
    _load_union_rows,
    _load_z1,
    _points,
    _read_scenes,
    _resolve,
    _valid_class,
)


VARIANTS = (
    "alphaclip_object_top1",
    "alphaclip_context_top1",
    "frozen_object_equal_top1",
    "frozen_context_equal_top1",
    "frozen_object_agreement_abstain",
    "frozen_context_agreement_abstain",
)
EVALUATED_SOURCES = ("track_only", "native_plus_track", "pair_union")


def _softmax(values: list[float]) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _load_alpha(root: Path, scenes: list[str], class_count: int) -> dict[tuple[str, int], np.ndarray | None]:
    result = {}
    for scene in scenes:
        path = root / scene / "automatic_track_alphaclip_semantics.json"
        rows = json.loads(path.read_text())
        for row in rows:
            key = (str(row["scene_name"]), int(row["track_id"]))
            if key in result:
                raise ValueError(f"duplicate Alpha-CLIP key: {key}")
            views = row.get("views", [])
            if int(row.get("alphaclip_class_index", -1)) < 0 or not views:
                result[key] = None
                continue
            probabilities = []
            for view in views:
                logits = view.get("clip_logits", [])
                if len(logits) != class_count:
                    raise ValueError(f"{key}: expected {class_count} Alpha logits, got {len(logits)}")
                probabilities.append(_softmax(logits))
            distribution = np.stack(probabilities, axis=0).mean(axis=0)
            result[key] = (distribution / distribution.sum()).astype(np.float32)
    return result


def _dense(rows: list[dict], class_count: int) -> np.ndarray | None:
    if not rows:
        return None
    result = np.zeros(class_count, dtype=np.float32)
    for row in rows:
        result[int(row["class_index"])] = float(row["probability"])
    total = float(result.sum())
    return result / total if total > 0 else None


def _semantic_class(
    variant: str,
    frozen: np.ndarray | None,
    alpha: np.ndarray | None,
) -> int:
    if variant == "alphaclip_object_top1" or variant == "alphaclip_context_top1":
        return int(np.argmax(alpha)) if alpha is not None else -1
    if variant == "frozen_object_equal_top1" or variant == "frozen_context_equal_top1":
        if frozen is None:
            return int(np.argmax(alpha)) if alpha is not None else -1
        if alpha is None:
            return int(np.argmax(frozen))
        return int(np.argmax(0.5 * frozen + 0.5 * alpha))
    if variant == "frozen_object_agreement_abstain" or variant == "frozen_context_agreement_abstain":
        if frozen is None or alpha is None:
            return -1
        frozen_top = int(np.argmax(frozen))
        return frozen_top if frozen_top == int(np.argmax(alpha)) else -1
    raise ValueError(f"unknown Alpha fusion variant: {variant}")


def _variant_alpha_name(variant: str) -> str:
    return "object" if "object" in variant else "context"


def _candidate_class(
    scene: str,
    track_id: int,
    binding: dict,
    evidence: dict,
    variant: str,
    alpha: dict[str, dict[tuple[str, int], np.ndarray | None]],
    class_count: int,
) -> int:
    frozen_row = evidence.get("frozen_support_vote")
    frozen = _dense(frozen_row["distribution"], class_count) if frozen_row else None
    alpha_distribution = alpha[_variant_alpha_name(variant)].get((scene, track_id))
    return _semantic_class(variant, frozen, alpha_distribution)


def _scene_prediction(scene, source, variant, args, bindings, evidence, alpha):
    native = _load_native(args.stream_records_root, scene)
    point_count = native["pred_masks"].shape[0]
    tracks = _load_tracks(args.stream_records_root, scene)
    unions = _load_union_rows(args.combined_plan_root, scene)
    overrides = _load_track_overrides(args.combined_plan_root, scene)
    track_pieces = []
    track_by_id = {}
    for track_id, track in sorted(tracks.items()):
        binding = bindings.get(f"{scene}:track:{track_id}")
        if binding is None:
            raise ValueError(f"{scene}: missing Z1 track binding {track_id}")
        erow = evidence[str(binding["semantic_evidence_node_key"])]
        class_index = _candidate_class(
            scene, track_id, binding, erow, variant, alpha, args.class_count
        )
        if not _valid_class(class_index):
            continue
        points = _points(Path(track["points_path"]), point_count)
        mask = np.zeros(point_count, dtype=bool)
        mask[points] = True
        score = overrides.get(track_id, max(0.0, float(track.get("mean_node_quality", 0.0))))
        piece = (mask, class_index, score)
        track_pieces.append(piece)
        track_by_id[track_id] = piece
    track_prediction = {
        "pred_masks": np.stack([row[0] for row in track_pieces], axis=1)
        if track_pieces else np.zeros((point_count, 0), dtype=bool),
        "pred_classes": np.asarray([row[1] for row in track_pieces], dtype=np.int64),
        "pred_scores": np.asarray([row[2] for row in track_pieces], dtype=np.float32),
    }
    union_pieces = []
    for candidate_id, row in sorted(unions.items()):
        binding = bindings.get(f"{scene}:pair_union:{candidate_id}")
        if binding is None:
            raise ValueError(f"{scene}: missing Z1 pair-union binding {candidate_id}")
        selected_track_id = int(binding["selected_track_id"])
        erow = evidence[str(binding["selected_track_semantic_evidence_node_key"])]
        class_index = _candidate_class(
            scene, selected_track_id, binding, erow, variant, alpha, args.class_count
        )
        if not _valid_class(class_index):
            continue
        points = _points(Path(row["points_path"]), point_count)
        mask = np.zeros(point_count, dtype=bool)
        mask[points] = True
        union_pieces.append((mask, class_index, float(row["new_score"])))
    union_prediction = {
        "pred_masks": np.stack([row[0] for row in union_pieces], axis=1)
        if union_pieces else np.zeros((point_count, 0), dtype=bool),
        "pred_classes": np.asarray([row[1] for row in union_pieces], dtype=np.int64),
        "pred_scores": np.asarray([row[2] for row in union_pieces], dtype=np.float32),
    }
    pieces = {
        "track_only": [track_prediction],
        "native_plus_track": [native, track_prediction],
        "pair_union": [native, track_prediction, union_prediction],
    }[source]
    return {
        "pred_masks": np.concatenate([piece["pred_masks"] for piece in pieces], axis=1),
        "pred_classes": np.concatenate([piece["pred_classes"] for piece in pieces]).astype(np.int64),
        "pred_scores": np.concatenate([piece["pred_scores"] for piece in pieces]).astype(np.float32),
    }


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    bindings, evidence = _load_z1(args.z1_root, scenes)
    alpha = {
        "object": _load_alpha(args.object_center_root, scenes, args.class_count),
        "context": _load_alpha(args.limited_context_root, scenes, args.class_count),
    }
    expected_track_keys = {
        (str(row["scene_name"]), int(row["candidate_id"]))
        for row in bindings.values() if str(row["candidate_source"]) == "track"
    }
    for name, rows in alpha.items():
        if set(rows) != expected_track_keys:
            raise ValueError(f"{name} Alpha track keys disagree with Z1 track bindings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    native_mapping = _Predictions(scenes, lambda scene: _load_native(args.stream_records_root, scene))
    native_reference = _evaluate(native_mapping, args.gt_instance_dir, args.output_dir / "native_only_reference.csv")
    results = {}
    for variant in VARIANTS:
        variant_result = {"ground_truth_usage": "evaluation_only/oracle_only", "sources": {}}
        for source in EVALUATED_SOURCES:
            mapping = _Predictions(
                scenes,
                lambda scene, source=source, variant=variant: _scene_prediction(
                    scene, source, variant, args, bindings, evidence, alpha
                ),
            )
            variant_result["sources"][source] = _evaluate(
                mapping, args.gt_instance_dir, args.output_dir / f"{variant}__{source}.csv"
            )
        results[variant] = variant_result
    summary = {
        "diagnostic_type": "Z3 GT-only Alpha-CLIP/YOLO-World fixed fusion control group",
        "diagnostic_only": True,
        "ground_truth_usage": "evaluation_only/oracle_only",
        "geometry_unchanged": True,
        "candidate_membership_unchanged": True,
        "native_classes_scores_unchanged": True,
        "track_pair_base_scores_unchanged": True,
        "native_only_reference": native_reference,
        "variants": results,
        "variant_contracts": {
            "alphaclip_*_top1": "Alpha-CLIP mean multi-view distribution top1; missing Alpha abstains",
            "frozen_*_equal_top1": "0.5 frozen all-support + 0.5 Alpha; missing Alpha falls back to frozen",
            "frozen_*_agreement_abstain": "emit frozen class only when frozen and Alpha top1 agree",
            "pair_union": "uses the selected track Alpha distribution; union geometry remains unchanged",
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--object-center-root", type=Path, required=True)
    parser.add_argument("--limited-context-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not (args.allow_gt_evaluation and args.allow_gt_diagnostics):
        raise SystemExit("Z3 evaluator requires both --allow-gt-evaluation and --allow-gt-diagnostics")
    for name in (
        "scene_list", "z1_root", "object_center_root", "limited_context_root", "stream_records_root",
        "combined_plan_root", "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
