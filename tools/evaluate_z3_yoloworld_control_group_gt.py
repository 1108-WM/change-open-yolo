#!/usr/bin/env python3
"""Evaluate GT-only Z3 YOLO-World semantic aggregation control groups.

Geometry, candidate membership, and frozen base scores remain unchanged.  The
control groups only replace the semantic class used by track/pair-union
bindings with a pre-registered aggregation of the Z1 distributions.  GT is
accepted only by the explicit official evaluator; no inference ledger or
candidate cache is written.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402


SOURCE_NAMES = ("native_only", "track_only", "native_plus_track", "pair_union")
VARIANTS = (
    "frozen_current",
    "support_top1",
    "all_view_top1",
    "independent_top1",
    "agreement_abstain",
    "support_top1_prob_weighted",
    "all_view_prob_weighted",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _load_native(root: Path, scene: str) -> dict[str, np.ndarray]:
    prefix = root / scene / "native_cache" / f"{scene}_pred_"
    masks = np.asarray(np.load(str(prefix) + "masks.npy", mmap_mode="r"), dtype=bool)
    scores = np.asarray(np.load(str(prefix) + "scores.npy"), dtype=np.float32)
    classes = np.asarray(np.load(str(prefix) + "classes.npy"), dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != len(scores) or len(scores) != len(classes):
        raise ValueError(f"{scene}: native prediction dimensions disagree")
    return {"pred_masks": masks, "pred_scores": scores, "pred_classes": classes}


def _valid_class(value: int) -> bool:
    return int(value) in instance_eval.PRED_ID_TO_ID and int(instance_eval.PRED_ID_TO_ID[int(value)]) >= 0


def _load_z1(root: Path, scenes: list[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    bindings: dict[str, dict] = {}
    evidence: dict[str, dict] = {}
    expected = set(scenes)
    with (root / "semantic_evidence_nodes.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row["scene_name"]) not in expected:
                    continue
                key = str(row["semantic_evidence_node_key"])
                if key in evidence:
                    raise ValueError(f"duplicate Z1 evidence key: {key}")
                evidence[key] = row
    with (root / "candidate_bindings.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row["scene_name"]) not in expected:
                    continue
                key = f"{row['scene_name']}:{row['candidate_source']}:{row['candidate_id']}"
                if key in bindings:
                    raise ValueError(f"duplicate Z1 candidate binding: {key}")
                if str(row["semantic_evidence_node_key"]) not in evidence:
                    raise ValueError(f"missing Z1 evidence reference: {row['semantic_evidence_node_key']}")
                bindings[key] = row
    if {str(row["scene_name"]) for row in evidence.values()} != expected:
        raise ValueError("Z1 evidence scene set disagrees with requested scene list")
    return bindings, evidence


def _load_tracks(root: Path, scene: str) -> dict[int, dict]:
    rows = json.loads((root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json").read_text()).get("tracks", [])
    return {int(row["track_id"]): row for row in rows}


def _load_union_rows(plan_root: Path, scene: str) -> dict[int, dict]:
    result = {}
    with (plan_root / "pair_union_append_candidates.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["scene_name"]) == scene:
                result[int(row["candidate_id"])] = row
    return result


def _load_track_overrides(plan_root: Path, scene: str) -> dict[int, float]:
    result = {}
    with (plan_root / "champion_track_score_overrides.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["scene_name"]) == scene:
                result[int(row["candidate_id"])] = max(0.0, float(row["new_score"]))
    return result


def _points(path: Path, point_count: int) -> np.ndarray:
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if len(points) == 0 or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"invalid candidate points: {path}")
    return points


def _role_distribution(evidence: dict, role: str) -> dict:
    return evidence["distribution"]["role_distributions"][role]


def _variant_semantics(binding: dict, evidence: dict, variant: str) -> tuple[int, float]:
    source = str(binding["candidate_source"])
    if source == "track":
        frozen_class = int(binding["current_voted_class_index"])
        frozen_probability = float(evidence["frozen_support_vote"]["top1_probability"])
    elif source == "pair_union":
        frozen_class = int(binding["inherited_voted_class_index"])
        frozen_probability = float(evidence["frozen_support_vote"]["top1_probability"])
    else:
        return int(binding["native_class_index"]), 1.0
    support = _role_distribution(evidence, "track_support_view")
    independent = _role_distribution(evidence, "independent_review_view")
    all_view = evidence["distribution"]
    if variant == "frozen_current":
        return frozen_class, frozen_probability
    if variant.startswith("support_top1"):
        return int(support["top1_class_index"]), float(support["top1_probability"])
    if variant in ("all_view_top1", "all_view_prob_weighted"):
        return int(all_view["top1_class_index"]), float(all_view["top1_probability"])
    if variant == "independent_top1":
        return int(independent["top1_class_index"]), float(independent["top1_probability"])
    if variant == "agreement_abstain":
        same = (
            int(support["top1_class_index"]) >= 0
            and int(support["top1_class_index"]) == int(independent["top1_class_index"])
        )
        return (int(support["top1_class_index"]) if same else -1), float(support["top1_probability"])
    raise ValueError(f"unknown Z3 variant: {variant}")


def _scene_prediction(
    scene: str,
    source: str,
    variant: str,
    args: argparse.Namespace,
    bindings: dict[str, dict],
    evidence: dict[str, dict],
) -> dict[str, np.ndarray]:
    native = _load_native(args.stream_records_root, scene)
    point_count = native["pred_masks"].shape[0]
    tracks = _load_tracks(args.stream_records_root, scene)
    unions = _load_union_rows(args.combined_plan_root, scene)
    overrides = _load_track_overrides(args.combined_plan_root, scene)
    track_pieces = []
    track_ids = []
    for track_id, track in sorted(tracks.items()):
        binding = bindings.get(f"{scene}:track:{track_id}")
        if binding is None:
            raise ValueError(f"{scene}: missing Z1 track binding {track_id}")
        erow = evidence[str(binding["semantic_evidence_node_key"])]
        class_index, probability = _variant_semantics(binding, erow, variant)
        if not _valid_class(class_index):
            continue
        points = _points(Path(track["points_path"]), point_count)
        mask = np.zeros(point_count, dtype=bool)
        mask[points] = True
        score = overrides.get(track_id, max(0.0, float(track.get("mean_node_quality", 0.0))))
        if variant.endswith("prob_weighted"):
            score *= probability
        track_pieces.append((mask, class_index, score))
        track_ids.append(track_id)
    track_prediction = {
        "pred_masks": np.stack([row[0] for row in track_pieces], axis=1)
        if track_pieces else np.zeros((point_count, 0), dtype=bool),
        "pred_classes": np.asarray([row[1] for row in track_pieces], dtype=np.int64),
        "pred_scores": np.asarray([row[2] for row in track_pieces], dtype=np.float32),
    }
    track_by_id = {track_id: piece for track_id, piece in zip(track_ids, track_pieces)}
    union_pieces = []
    for candidate_id, row in sorted(unions.items()):
        binding = bindings.get(f"{scene}:pair_union:{candidate_id}")
        if binding is None:
            raise ValueError(f"{scene}: missing Z1 pair-union binding {candidate_id}")
        erow = evidence[str(binding["semantic_evidence_node_key"])]
        class_index, probability = _variant_semantics(binding, erow, variant)
        if not _valid_class(class_index):
            continue
        points = _points(Path(row["points_path"]), point_count)
        mask = np.zeros(point_count, dtype=bool)
        mask[points] = True
        score = float(row["new_score"])
        if variant.endswith("prob_weighted"):
            score *= probability
        union_pieces.append((mask, class_index, score))
    union_prediction = {
        "pred_masks": np.stack([row[0] for row in union_pieces], axis=1)
        if union_pieces else np.zeros((point_count, 0), dtype=bool),
        "pred_classes": np.asarray([row[1] for row in union_pieces], dtype=np.int64),
        "pred_scores": np.asarray([row[2] for row in union_pieces], dtype=np.float32),
    }
    pieces = {
        "native_only": [native],
        "track_only": [track_prediction],
        "native_plus_track": [native, track_prediction],
        "pair_union": [native, track_prediction, union_prediction],
    }[source]
    return {
        "pred_masks": np.concatenate([piece["pred_masks"] for piece in pieces], axis=1),
        "pred_classes": np.concatenate([piece["pred_classes"] for piece in pieces]).astype(np.int64),
        "pred_scores": np.concatenate([piece["pred_scores"] for piece in pieces]).astype(np.float32),
    }


class _Predictions(Mapping):
    def __init__(self, scenes, builder):
        self.scenes, self.builder = scenes, builder

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)

    def __getitem__(self, scene):
        return self.builder(scene)

    def items(self):
        for scene in self.scenes:
            yield scene, self.builder(scene)
            gc.collect()


def _evaluate(predictions: Mapping, gt_root: Path, out_csv: Path) -> dict:
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        averages, _, _, _ = instance_eval.evaluate(predictions, str(gt_root), str(out_csv), dataset="scannet200")
    return {
        "ap": float(averages["all_ap"]),
        "ap50": float(averages["all_ap_50%"]),
        "ap25": float(averages["all_ap_25%"]),
        "head_ap": float(averages["head_ap"]),
        "common_ap": float(averages["common_ap"]),
        "tail_ap": float(averages["tail_ap"]),
    }


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    bindings, evidence = _load_z1(args.z1_root, scenes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in VARIANTS:
        variant_result = {"ground_truth_usage": "evaluation_only/oracle_only", "sources": {}}
        for source in SOURCE_NAMES:
            mapping = _Predictions(
                scenes,
                lambda scene, source=source, variant=variant: _scene_prediction(
                    scene, source, variant, args, bindings, evidence
                ),
            )
            csv_path = args.output_dir / f"{variant}__{source}.csv"
            variant_result["sources"][source] = _evaluate(mapping, args.gt_instance_dir, csv_path)
        results[variant] = variant_result
    summary = {
        "diagnostic_type": "Z3 GT-only YOLO-World semantic aggregation control group",
        "diagnostic_only": True,
        "ground_truth_usage": "evaluation_only/oracle_only",
        "geometry_unchanged": True,
        "candidate_membership_unchanged": True,
        "native_classes_scores_unchanged": True,
        "track_pair_base_scores_unchanged_except_probability_weighted_variants": True,
        "variants": results,
        "variant_contracts": {
            "frozen_current": "Z0 frozen all-support track vote / selected-track inherited pair vote",
            "support_top1": "Z1 selected support-role distribution top1",
            "all_view_top1": "Z1 aggregate selected-view distribution top1",
            "independent_top1": "Z1 independent-review distribution top1",
            "agreement_abstain": "support top1 only when support and independent top1 agree; otherwise semantic abstention",
            "*_prob_weighted": "same class rule with base frozen score multiplied by selected top1 probability",
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not (args.allow_gt_evaluation and args.allow_gt_diagnostics):
        raise SystemExit("Z3 evaluator requires both --allow-gt-evaluation and --allow-gt-diagnostics")
    for name in (
        "scene_list", "z1_root", "stream_records_root", "combined_plan_root",
        "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
